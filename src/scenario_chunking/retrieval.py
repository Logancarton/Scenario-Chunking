from __future__ import annotations

import math
from time import perf_counter

from .memory import EpisodicMemory
from .models import EventKind, TruthStatus, RetrievalHit, RetrievalMode, RetrievalQuery, RetrievalStats
from .spreading import BoundedNeighborActivator, SpreadBudget


class MemoryRouter:
    """Retrieves episodic chunks using one of the experiment's controlled routing strategies."""

    def __init__(
        self, memory: EpisodicMemory, mode: RetrievalMode, *,
        top_k: int = 6, spread_budget: SpreadBudget | None = None,
        candidate_budget: int | None = None,
    ) -> None:
        self.memory = memory
        self.mode = mode
        self.top_k = top_k
        if candidate_budget is not None and candidate_budget < top_k:
            raise ValueError("candidate budget cannot be smaller than top_k")
        self.candidate_budget = candidate_budget
        self.spreader = BoundedNeighborActivator(memory, spread_budget) if mode == RetrievalMode.SPREADING else None

    def retrieve(self, query: RetrievalQuery) -> tuple[list[RetrievalHit], RetrievalStats]:
        started = perf_counter()
        stats = RetrievalStats(total_chunks=len(self.memory.chunks))
        if self.mode == RetrievalMode.FULL:
            candidate_ids = list(self.memory.chunk_order)
        else:
            candidate_ids = self._sparse_candidates(query)
        stats.candidate_chunks = len(candidate_ids)

        scored: list[RetrievalHit] = []
        for cid in candidate_ids:
            stats.header_evaluations += 1
            chunk = self.memory.chunks[cid]
            score, reasons = self._score_chunk(chunk_id=cid, query=query)
            if score <= 0:
                continue
            scored.append(RetrievalHit(chunk_id=cid, score=score, reasons=reasons))

        scored.sort(key=lambda h: (-h.score, h.chunk_id))
        hits = scored[: self.top_k]
        stats.initial_hit_ids = [h.chunk_id for h in hits]
        query_non_goal = {tag for tag in query.semantic_tags if not tag.startswith("goal:")}
        for hit in hits:
            events = self.memory.chunk_events(hit.chunk_id)
            stats.detail_loads += 1
            stats.active_detail_events += len(events)
            chunk_non_goal = {tag for tag in self.memory.chunks[hit.chunk_id].semantic_tags if not tag.startswith("goal:")}
            if query_non_goal and not (query_non_goal & chunk_non_goal):
                stats.unnecessary_activations += 1
            self.memory.activate(hit.chunk_id, query.current_time)
        if self.mode == RetrievalMode.FULL:
            # Full retrieval pays the detail-processing bill for the whole historical collection,
            # even though only top-k results are exposed to the policy.
            stats.detail_loads = len(candidate_ids)
            stats.active_detail_events = sum(len(self.memory.chunks[cid].event_ids) for cid in candidate_ids)
            stats.unnecessary_activations = sum(
                1
                for cid in candidate_ids
                if query_non_goal
                and not (query_non_goal & {tag for tag in self.memory.chunks[cid].semantic_tags if not tag.startswith("goal:")})
            )

        if self.spreader is not None:
            extra = self.spreader.expand(
                query, hits, stats,
                rank_header=lambda cid, q: self._score_chunk(chunk_id=cid, query=q),
            )
            hits = hits + extra
        stats.latency_ms = (perf_counter() - started) * 1000.0
        return hits, stats

    def _focused_candidates(self, query: RetrievalQuery) -> list[str]:
        """Candidate generation bounded independently of stored memory size.

        Prefer specific cue postings; interleave quality-preserving exemplars
        with recent observations. This is an optional experimental treatment.
        It never inspects event details or accesses environment truth.
        """
        assert self.candidate_budget is not None
        budget = self.candidate_budget
        tags = [t for t in query.semantic_tags if not t.startswith("goal:")]
        tags.sort(key=lambda tag: (len(self.memory.tag_index.get(tag, ())), tag))
        tags += ["person:" + p for p in sorted(query.person_ids)]
        selected: dict[str, None] = {}
        if tags:
            per_tag = max(2, budget // len(tags))
            for tag in tags:
                if tag.startswith("person:"):
                    # Person postings are not a separate goal-wide memory scan.
                    ids = list(sorted(self.memory.person_index.get(tag[7:], ())))[:per_tag]
                else:
                    ids = self.memory.focused_postings(tag, per_tag)
                for cid in ids:
                    if len(selected) >= budget:
                        break
                    selected[cid] = None
            if len(selected) < budget:
                for tag in tags:
                    if tag.startswith("person:"):
                        continue
                    for cid in self.memory.focused_postings(tag, budget):
                        if len(selected) >= budget:
                            break
                        selected[cid] = None
        else:
            # Goal-only queries have no discriminative category evidence.
            # Fall back to baseline semantics rather than silently truncating.
            return sorted(self.memory.goal_index.get(query.goal_id, ()))
        return list(selected)

    def _sparse_candidates(self, query: RetrievalQuery) -> list[str]:
        if self.candidate_budget is not None:
            return self._focused_candidates(query)
        candidates: set[str] = set(self.memory.goal_index.get(query.goal_id, set()))
        # Use compact routing tags; no detailed historical events are loaded here.
        for tag in query.semantic_tags:
            candidates.update(self.memory.tag_index.get(tag, set()))
        for person in query.person_ids:
            candidates.update(self.memory.person_index.get(person, set()))
        return sorted(candidates)

    def _score_chunk(self, *, chunk_id: str, query: RetrievalQuery) -> tuple[float, list[str]]:
        chunk = self.memory.chunks[chunk_id]
        header = chunk.header
        overlap = len(chunk.semantic_tags & query.semantic_tags)
        union = len(chunk.semantic_tags | query.semantic_tags) or 1
        semantic = overlap / union
        goal = 1.0 if header.goal_id == query.goal_id else 0.0
        person = 0.2 if chunk.header.person_ids & query.person_ids else 0.0
        score = 0.55 * semantic + 0.28 * goal + person + 0.12 * header.salience + 0.05 * header.self_relevance
        reasons = [f"semantic={semantic:.3f}", f"goal={goal:.1f}"]

        if self.mode == RetrievalMode.RECENCY:
            age = max(0.0, query.current_time - header.occurrence_time)
            recency = math.exp(-0.18 * age)
            score = 0.45 * score + 0.55 * recency
            reasons.append(f"fixed_recency={recency:.3f}")

        if self.mode in (RetrievalMode.STRUCTURAL, RetrievalMode.SPREADING):
            # Time remains factual structure. We use it to identify contradictions and sequence
            # relationships, not to assume that newer is universally more useful.
            contradiction_penalty = 0.0
            for key in query.contradicted_keys:
                if f"key:{key}" in chunk.semantic_tags and "outcome:success" in chunk.semantic_tags:
                    contradiction_penalty += 0.65
            sequence_bonus = 0.0
            if header.next_chunk_id:
                nxt = self.memory.chunks[header.next_chunk_id]
                if "outcome:success" in nxt.semantic_tags:
                    sequence_bonus += 0.18
            if header.previous_chunk_id:
                prev = self.memory.chunks[header.previous_chunk_id]
                if "clue:key_relation" in prev.semantic_tags:
                    sequence_bonus += 0.12
            activation = header.activation_at(query.current_time)
            activation_bonus = min(0.12, 0.04 * activation)
            score += sequence_bonus + activation_bonus - contradiction_penalty
            reasons.extend([
                f"sequence={sequence_bonus:.3f}",
                f"activation={activation_bonus:.3f}",
                f"contradiction=-{contradiction_penalty:.3f}",
            ])

        if self.mode == RetrievalMode.CHUNKED:
            reasons.append("sparse_chunked")
        elif self.mode == RetrievalMode.FULL:
            reasons.append("full_scan")
        return score, reasons


class IndexedCueReinstater:
    """Bounded header-index discovery, including memories outside initial top-K.

    Only specific entity/key indexes seed candidates. Historical detail is read
    after ranking, with strict accounting and observed-outcome provenance checks.
    """

    def __init__(self, memory: EpisodicMemory, *, max_headers: int = 18,
                 max_additional: int = 2, max_total_detail_events: int = 36) -> None:
        self.memory = memory
        self.max_headers = max_headers
        self.max_additional = max_additional
        self.max_total_detail_events = max_total_detail_events

    def expand(self, query: RetrievalQuery, actions, initial_hits: list[RetrievalHit],
               stats: RetrievalStats, *, rank_header) -> list[RetrievalHit]:
        seen = {h.chunk_id for h in initial_hits}
        if self.max_additional <= 0 or stats.active_detail_events >= self.max_total_detail_events:
            return []
        # The current problem supplies cues, not any hidden solution.
        specific_tags = sorted({f"key:{a.key_name}" for a in actions
                                if a.kind == "try_key" and a.key_name})
        candidates = set()
        for tag in specific_tags:
            candidates.update(self.memory.tag_index.get(tag, ()))
        proposals = []
        for cid in sorted(candidates):
            if cid in seen:
                continue
            if stats.reinstatement_headers_examined >= self.max_headers:
                stats.budget_exhausted = True
                break
            stats.reinstatement_headers_examined += 1
            chunk = self.memory.chunks[cid]
            if chunk.header.goal_id != query.goal_id:
                continue
            if not ("outcome:success" in chunk.semantic_tags or "outcome:failure" in chunk.semantic_tags):
                continue
            if any(f"key:{key}" in chunk.semantic_tags and "outcome:success" in chunk.semantic_tags
                   for key in query.contradicted_keys):
                continue
            score, _ = rank_header(cid, query)
            if score > 0:
                proposals.append((score, cid))
        found = []
        for score, cid in sorted(proposals, key=lambda item: (-item[0], item[1])):
            if len(found) >= self.max_additional:
                break
            cost = len(self.memory.chunks[cid].event_ids)
            if stats.active_detail_events + cost > self.max_total_detail_events:
                stats.budget_exhausted = True
                continue
            # Charge the event read even if evidence proves invalid.
            events = self.memory.chunk_events(cid)
            stats.detail_loads += 1
            stats.active_detail_events += cost
            stats.reinstatement_detail_events += cost
            attempts = {e.event_id: e for e in events
                        if e.kind == EventKind.ACTION and e.action and e.action.kind == "try_key"}
            valid = any(
                e.kind == EventKind.OUTCOME and e.truth_status == TruthStatus.OUTCOME
                and e.outcome in ("success", "failure") and e.cause_event_id in attempts
                and f"key:{attempts[e.cause_event_id].action.key_name}" in specific_tags
                for e in events
            )
            if not valid:
                continue
            hit = RetrievalHit(cid, score, ["indexed_cue", "verified_outcome"])
            found.append(hit)
            stats.reinstated_hit_ids.append(cid)
            self.memory.activate(cid, query.current_time)
        return found


def retrieval_recall_at_k(
    router: MemoryRouter,
    queries: list[tuple[RetrievalQuery, set[str]]],
) -> float:
    if not queries:
        return 0.0
    values: list[float] = []
    for query, relevant in queries:
        hits, _ = router.retrieve(query)
        retrieved = {h.chunk_id for h in hits}
        values.append(len(retrieved & relevant) / max(1, len(relevant)))
    return sum(values) / len(values)
