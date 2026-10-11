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
    examined_event_records: int

    def trace(self) -> dict:
        return {
            "context": self.context,
            "action_kind": self.action_kind,
            "layers": {m.axis: m.trace() for m in self.meanings},
            "weights": dict(self.weights),
            "integrated": round(self.integrated, 6),
            "independent_sources": self.independent_sources,
            "examined_event_records": self.examined_event_records,
        }


@dataclass(frozen=True)
class PreparedEvidence:
    """One pass over retrieved events; indexed by candidate action identity."""
    pairs: dict[tuple[str, str | None], tuple[tuple[str, object, object], ...]]
    examined_event_records: int

    def matches(self, action: Action):
        key = (action.kind, action.key_name if action.kind == "try_key" else None)
        return self.pairs.get(key, ())


class MeaningInterpreter:
    """Interpret already retrieved events; never read unretrieved memory or environment truth."""

    def __init__(self, memory: EpisodicMemory, learner) -> None:
        self.memory = memory
        self.learner = learner

    def prepare(self, hits) -> PreparedEvidence:
        """Build one index of grounded responses without duplicating any event."""
        pairs = {}
        examined = 0
        for hit in hits:
            events = self.memory.chunk_events(hit.chunk_id)
            examined += len(events)
            attempts = []
            outcomes_by_cause = {}
            observations_by_step = {}
            positions = {}
            for index, event in enumerate(events):
                positions[event.event_id] = index
                if event.kind == EventKind.ACTION and event.action:
                    attempts.append(event)
                elif (event.kind == EventKind.OUTCOME and
                      event.truth_status == TruthStatus.OUTCOME and
                      event.outcome in ("success", "failure") and event.cause_event_id):
                    outcomes_by_cause.setdefault(event.cause_event_id, []).append(event)
                elif (event.kind == EventKind.OBSERVATION and
                      event.truth_status == TruthStatus.FACT and event.informative):
                    observations_by_step.setdefault(event.step, []).append(event)
            for attempt in attempts:
                responses = list(outcomes_by_cause.get(attempt.event_id, ()))
                if attempt.action.kind != "try_key":
                    responses.extend(observations_by_step.get(attempt.step, ()))
                responses.sort(key=lambda event: positions[event.event_id])
                key = (attempt.action.kind,
                       attempt.action.key_name if attempt.action.kind == "try_key" else None)
                pairs.setdefault(key, []).extend(
                    (hit.chunk_id, attempt, response) for response in responses
                )
        return PreparedEvidence(
            pairs={key: tuple(value) for key, value in pairs.items()},
            examined_event_records=examined,
        )

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

    def transfer_evidence(self, hits, features: frozenset[str], *, review_depth: int = 0):
        """Evaluate cross-category causal outcomes from retrieved memory only.

        A prediction, an unverified report, and an observation are NOT outcomes.
        A deeper review spends extra processing to resolve chronological conflicts,
        without writing inferred truths or accessing task-generator labels.
        """
        from collections import defaultdict
        records = defaultdict(dict)
        evaluations = 0
        for hit in hits:
            events = self.memory.chunk_events(hit.chunk_id)
            attempts = {e.event_id: e for e in events if e.kind == EventKind.ACTION}
            evaluations += len(events)
            for event in events:
                if (event.kind != EventKind.OUTCOME or
                        event.truth_status != TruthStatus.OUTCOME or
                        event.outcome not in ("success", "failure")):
                    continue
                attempt = attempts.get(event.cause_event_id)
                if attempt is None:
                    continue
                axis = attempt.action.key_name if attempt.action else None
                token = attempt.action.target if attempt.action else None
                cue = f"signal:{axis}:{token}"
                if cue not in features:
                    continue
                sid = str(event.metadata.get("source_event_id") or event.event_id)
                records[cue][sid] = (
                    event.occurrence_time,
                    1.0 if event.outcome == "success" else -1.0,
                    max(0.0, min(1.0, event.confidence)),
                    event.event_id,
                )
        category_values = {}
        provenance = {}
        for cue in sorted(f for f in features if f.startswith("signal:")):
            history = sorted(records.get(cue, {}).values(), key=lambda r: (r[0], r[3]))
            # These are alternative evidence-handling budgets, not hidden labels.
            # Only outcome events with an observed cause may influence the score.
            if review_depth == 1:
                chosen = history[-6:]
            elif review_depth >= 3:
                chosen = history[-3:]
            else:
                chosen = history
            evaluations += (len(chosen) * max(0, review_depth))
            total = sum(r[2] for r in chosen)
            value = (sum(r[1] * r[2] for r in chosen) / total) if total else 0.0
            category_values[cue] = value
            provenance[cue] = [r[3] for r in chosen]
        return {
            "score": sum(category_values.values()) / max(1, len(category_values)),
            "categories": category_values,
            "evidence_ids": sorted({eid for values in provenance.values() for eid in values}),
            "provenance": provenance,
            "evaluations": evaluations,
        }

    def evaluate(self, state: CognitiveState, action: Action, hits, *,
                 learn_layers: bool, adaptive_gate: bool,
                 prepared: PreparedEvidence | None = None) -> MeaningFrame:
        context = self.context(state, action)
        # Only the initial preparation reads events. Later candidate evaluations
        # examine already indexed evidence pairs without rescanning all events.
        owned_preparation = prepared is None
        if prepared is None:
            prepared = self.prepare(hits)
        examined_event_records = prepared.examined_event_records if owned_preparation else 0
        sources: dict[str, tuple[dict[str, float], float, bool, str, str]] = {}
        for chunk_id, attempt, response in prepared.matches(action):
            source_id = str(response.metadata.get("source_event_id") or response.event_id)
            if source_id in sources:
                continue
            chunk = self.memory.chunks[chunk_id]
            causal = response.kind == EventKind.OUTCOME
            signed = (1.0 if response.outcome == "success" else -1.0) if causal else 0.35
            contradicted = bool(action.key_name and action.key_name in state.contradicted_keys)
            goal_related = chunk.header.goal_id == state.goal_id
            person_matches = bool(action.person and attempt.person_id == action.person)
            values = {
                "self": signed if goal_related else 0.0,
                "salience": signed * chunk.header.salience,
                "time": -1.0 if contradicted and signed > 0 else 0.0,
                "sequence": signed if causal else 0.0,
                "person": signed if person_matches else 0.0,
            }
            conf = min(1.0, max(0.0, response.confidence))
            if not causal:
                conf *= 0.65
            sources[source_id] = (
                values, conf, not contradicted,
                "observed_outcome" if causal else "informative_observation",
                chunk_id,
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
            return MeaningFrame(context, action.kind, (), {}, 0.0, 0, examined_event_records)
        logits = {m.axis: (self.learner.integration_logit(context, m.axis)
                           if adaptive_gate else 0.0) for m in meanings}
        pivot = max(logits.values())
        normalizer = sum(math.exp(v - pivot) for v in logits.values())
        weights = {axis: math.exp(logit - pivot) / normalizer for axis, logit in logits.items()}
        integrated = sum(weights[m.axis] * m.value for m in meanings)
        return MeaningFrame(context, action.kind, tuple(meanings), weights, integrated, len(sources), examined_event_records)
