"""Bounded, inspectable interpretations of retrieved evidence, not a second planner.

A single historical event may inform several axes, but it counts as ONE source.
Chronology is not upgraded to causality, and missing evidence stays missing.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

from .models import Action, EventKind, TruthStatus
from .memory import EpisodicMemory
from .state import CognitiveState

AXES = ("self", "salience", "time", "sequence", "person")


@dataclass(frozen=True)
class LayerMeaning:
    axis: str
    value: float
    confidence: float
    source_event_ids: tuple[str, ...]
    temporal_valid: bool
    goal_related: bool
    provenance: str
    association: float

    def trace(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MeaningFrame:
    context: str
    action_kind: str
    meanings: tuple[LayerMeaning, ...]
    weights: dict[str, float]
    integrated: float
    independent_sources: int

    def trace(self) -> dict:
        return {
            "context": self.context,
            "action_kind": self.action_kind,
            "layers": {m.axis: m.trace() for m in self.meanings},
            "weights": dict(self.weights),
            "integrated": round(self.integrated, 6),
            "independent_sources": self.independent_sources,
        }


class MeaningInterpreter:
    """Interpret already retrieved events; never read unretrieved memory or environment truth."""

    def __init__(self, memory: EpisodicMemory, learner) -> None:
        self.memory = memory
        self.learner = learner

    @staticmethod
    def context(state: CognitiveState, action: Action) -> str:
        if action.kind == "ask_informant":
            return "social"
        if action.kind == "try_key" and action.key_name in state.contradicted_keys:
            return "contradicted"
        if action.kind == "try_key" and action.key_name == state.candidate_key:
            return "supported_clue"
        return "exploration"

    @staticmethod
    def _matches(recorded: Action, candidate: Action) -> bool:
        if recorded.kind != candidate.kind:
            return False
        # Do not transfer success of an arbitrary historical key to another key.
        return recorded.kind != "try_key" or recorded.key_name == candidate.key_name

    def evaluate(self, state: CognitiveState, action: Action, hits, *,
                 learn_layers: bool, adaptive_gate: bool) -> MeaningFrame:
        context = self.context(state, action)
        # Source ID, not number of retrieved chunks or axes, determines independence.
        sources: dict[str, tuple[dict[str, float], float, bool, str, str]] = {}
        for hit in hits:
            chunk = self.memory.chunks[hit.chunk_id]
            events = self.memory.chunk_events(hit.chunk_id)
            attempts = [e for e in events if e.kind == EventKind.ACTION and e.action
                        and self._matches(e.action, action)]
            if not attempts:
                continue
            for attempt in attempts:
                responses = [
                    e for e in events
                    if (e.kind == EventKind.OUTCOME and e.truth_status == TruthStatus.OUTCOME
                        and e.cause_event_id == attempt.event_id
                        and e.outcome in ("success", "failure"))
                    or (e.kind == EventKind.OBSERVATION and e.truth_status == TruthStatus.FACT
                        and e.informative and e.step == attempt.step
                        and action.kind != "try_key")
                ]
                for response in responses:
                    source_id = str(response.metadata.get("source_event_id") or response.event_id)
                    if source_id in sources:
                        continue
                    causal = response.kind == EventKind.OUTCOME
                    signed = (1.0 if response.outcome == "success" else -1.0) if causal else 0.35
                    contradicted = bool(action.key_name and action.key_name in state.contradicted_keys)
                    goal_related = chunk.header.goal_id == state.goal_id
                    person_matches = bool(action.person and attempt.person_id == action.person)
                    values = {
                        "self": signed if goal_related else 0.0,
                        "salience": signed * chunk.header.salience,
                        # Time records detected invalidation, not a universal recency prior.
                        "time": -1.0 if contradicted and signed > 0 else 0.0,
                        # Temporal adjacency / an informative observation is not causal proof.
                        "sequence": signed if causal else 0.0,
                        "person": signed if person_matches else 0.0,
                    }
                    conf = min(1.0, max(0.0, response.confidence))
                    if not causal:
                        conf *= 0.65  # information gain is not demonstrated goal success
                    sources[source_id] = (
                        values, conf, not contradicted, "observed_outcome" if causal else "informative_observation",
                        hit.chunk_id,
                    )

        meanings: list[LayerMeaning] = []
        for axis in AXES:
            matches = [(sid, record) for sid, record in sources.items() if record[0][axis] != 0.0]
            if not matches:
                continue  # No fabricated neutral "evidence" for a missing axis.
            raw = sum(record[0][axis] * record[1] for _, record in matches) / len(matches)
            association = self.learner.layer_association(axis, context, action.kind) if learn_layers else 0.0
            value = max(-1.0, min(1.0, raw * (1.0 + association)))
            confidence = sum(record[1] for _, record in matches) / len(matches)
            meaning = LayerMeaning(
                axis=axis, value=value, confidence=confidence,
                source_event_ids=tuple(sorted(sid for sid, _ in matches)),
                temporal_valid=all(record[2] for _, record in matches),
                goal_related=any(self.memory.chunks[record[4]].header.goal_id == state.goal_id
                                 for _, record in matches),
                provenance="observed_outcome" if all(record[3] == "observed_outcome" for _, record in matches)
                else "includes_informative_observation",
                association=association,
            )
            meanings.append(meaning)
            for _, record in matches:
                self.memory.store_layer_meaning(record[4], axis, action.label, meaning)

        if not meanings:
            return MeaningFrame(context, action.kind, (), {}, 0.0, 0)
        logits = {m.axis: (self.learner.integration_logit(context, m.axis)
                           if adaptive_gate else 0.0) for m in meanings}
        pivot = max(logits.values())
        normalizer = sum(math.exp(v - pivot) for v in logits.values())
        weights = {axis: math.exp(logit - pivot) / normalizer for axis, logit in logits.items()}
        integrated = sum(weights[m.axis] * m.value for m in meanings)
        return MeaningFrame(context, action.kind, tuple(meanings), weights, integrated, len(sources))
