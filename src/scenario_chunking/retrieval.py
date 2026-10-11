from __future__ import annotations

import math
from time import perf_counter

from .memory import EpisodicMemory
from .models import EventKind, RetrievalHit, RetrievalMode, RetrievalQuery, RetrievalStats
from .spreading import BoundedNeighborActivator, SpreadBudget
from .working_memory import WorkingMemoryActivation


class MemoryRouter:
    """Retrieves episodic chunks using one of the experiment's controlled routing strategies."""

    def __init__(
        self, memory: EpisodicMemory, mode: RetrievalMode, *,
        top_k: int = 6, spread_budget: SpreadBudget | None = None,
        working_memory: WorkingMemoryActivation | None = None,
        activation_persistence: dict[str, float] | None = None,
    ) -> None:
        self.memory = memory
        self.mode = mode
        self.top_k = top_k
        self.working_memory = working_memory
        self.activation_persistence = activation_persistence if activation_persistence is not None else {}
        if mode == RetrievalMode.ACTIVATION and working_memory is None:
            raise ValueError("activation mode requires a working-memory owner")
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
            if self.mode != RetrievalMode.ACTIVATION:
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
        if self.mode == RetrievalMode.ACTIVATION and self.working_memory is not None:
            # Cue reinstatement is not meaningful-use refresh.
            for hit in hits:
                chunk = self.memory.chunks[hit.chunk_id]
                self.working_memory.cue(
                    hit.chunk_id, chunk.header.goal_id, chunk.semantic_tags,
                    query.semantic_tags, query.current_time,
                    contradicted_keys=query.contradicted_keys,
                    persistence=self.activation_persistence,
                )
        stats.latency_ms = (perf_counter() - started) * 1000.0
        return hits, stats

    def _sparse_candidates(self, query: RetrievalQuery) -> list[str]:
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

        if self.mode in (RetrievalMode.STRUCTURAL, RetrievalMode.SPREADING, RetrievalMode.ACTIVATION):
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
            if self.mode == RetrievalMode.ACTIVATION:
                activation_bonus = self.working_memory.bonus(
                    chunk_id, query.goal_id, query.current_time,
                    persistence=self.activation_persistence)
            else:
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
