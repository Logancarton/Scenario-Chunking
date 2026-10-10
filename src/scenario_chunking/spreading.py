from __future__ import annotations

from dataclasses import dataclass

from .memory import EpisodicMemory
from .models import RetrievalHit, RetrievalQuery, RetrievalStats


@dataclass(frozen=True)
class SpreadBudget:
    """Hard caps on expanding, evaluating and loading neighbor memories per retrieval."""
    max_additional_chunks: int = 2
    max_neighbor_headers: int = 18
    max_total_detail_events: int = 36
    max_hops: int = 1


class BoundedNeighborActivator:
    """One-hop, provenance-aware spreading from the unchanged initial top-K.

    No retrieved neighbor becomes another seed. Both candidate header inspections
    and detailed event reads are bounded, so long or highly connected chains stop.
    """

    def __init__(self, memory: EpisodicMemory, budget: SpreadBudget | None = None) -> None:
        self.memory = memory
        self.budget = budget or SpreadBudget()
        if min(self.budget.max_additional_chunks, self.budget.max_neighbor_headers, self.budget.max_total_detail_events) < 0:
            raise ValueError("spreading budget may not be negative")
        if self.budget.max_hops != 1:
            raise ValueError("only direct one-hop neighbors are supported")

    def expand(
        self,
        query: RetrievalQuery,
        initial_hits: list[RetrievalHit],
        stats: RetrievalStats,
        *,
        rank_header,
    ) -> list[RetrievalHit]:
        stats.initial_hit_ids = [x.chunk_id for x in initial_hits]
        initial_ids = set(stats.initial_hit_ids)
        examined: set[str] = set()
        proposed: dict[str, tuple[float, str]] = {}

        for seed in initial_hits:
            for nid, relation, evidence, confidence in self.memory.neighbors(seed.chunk_id):
                if nid in initial_ids or nid in examined:
                    continue
                if stats.neighbor_headers_examined >= self.budget.max_neighbor_headers:
                    stats.budget_exhausted = True
                    break
                examined.add(nid)
                stats.neighbor_headers_examined += 1
                chunk = self.memory.chunks[nid]
                if chunk.header.goal_id != query.goal_id:
                    continue
                if relation == "causal" and (evidence != "observed" or confidence < 0.75):
                    continue
                if any(
                    f"key:{key}" in chunk.semantic_tags and "outcome:success" in chunk.semantic_tags
                    for key in query.contradicted_keys
                ):
                    continue
                tag_overlap = len(chunk.semantic_tags & query.semantic_tags)
                # A goal tag alone is not adequate evidence of relevance.
                nong = {t for t in query.semantic_tags if not t.startswith("goal:")}
                nong_overlap = len(nong & chunk.semantic_tags)
                if not nong_overlap:
                    continue
                score, _ = rank_header(nid, query)
                relation_bonus = (
                    0.22 * confidence if relation == "causal"
                    else (0.12 if relation == "successor" else 0.08)
                )
                evidence_bonus = 0.10 if ("outcome:success" in chunk.semantic_tags and "outcome:success" in query.semantic_tags) else 0
                priority = score + relation_bonus + evidence_bonus + min(0.08, 0.02 * tag_overlap)
                justification = (
                    f"{relation} of {seed.chunk_id}; provenance={evidence}; "
                    f"goal_match; semantic_overlap={nong_overlap}; priority={priority:.3f}"
                )
                old = proposed.get(nid)
                if old is None or priority > old[0]:
                    proposed[nid] = (priority, justification)
            if stats.neighbor_headers_examined >= self.budget.max_neighbor_headers:
                break

        # Ranking does not read any detailed event. It only accesses chunk headers and routing tags.
        expanded: list[RetrievalHit] = []
        for nid, (priority, justification) in sorted(proposed.items(), key=lambda x: (-x[1][0], x[0])):
            if len(expanded) >= self.budget.max_additional_chunks:
                stats.budget_exhausted = True
                break
            cost = len(self.memory.chunks[nid].event_ids)
            if stats.active_detail_events + cost > self.budget.max_total_detail_events:
                stats.budget_exhausted = True
                continue
            expanded.append(RetrievalHit(chunk_id=nid, score=priority, reasons=["spread", justification]))
            stats.spread_hit_ids.append(nid)
            stats.spread_justifications[nid] = justification
            stats.detail_loads += 1
            stats.active_detail_events += cost
            stats.spread_detailed_events += cost
            self.memory.activate(nid, query.current_time)

        return expanded
