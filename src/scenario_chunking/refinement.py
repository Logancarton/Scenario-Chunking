"""Evidence-grounded thought continuity; a refinement record is not a new episode.

The existing five-axis interpreter supplies evidence once per candidate. This owner
compares context-only and evidence-integrated interpretations without rereading
memory. Versions reflect changes in evidence or understanding, not repeated scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from .models import Action, RetrievalHit
from .state import CognitiveState


@dataclass(frozen=True)
class ThoughtVersion:
    number: int
    preferred_action: str
    alternatives: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    memory_chunk_ids: tuple[str, ...]
    contradictions: tuple[str, ...]
    uncertainty: float
    reason: str
    signature: tuple[object, ...]
    time: float

    def trace(self) -> dict[str, object]:
        return {
            "version": self.number, "preferred_action": self.preferred_action,
            "alternatives": list(self.alternatives),
            "evidence_ids": list(self.evidence_ids),
            "memory_chunk_ids": list(self.memory_chunk_ids),
            "contradictions": list(self.contradictions),
            "uncertainty": round(self.uncertainty, 6),
            "reason": self.reason, "time": self.time,
        }


@dataclass
class ThoughtThread:
    thought_id: str
    initiating_goal: str
    initiating_episode: str
    versions: list[ThoughtVersion] = field(default_factory=list)
    outcomes: list[dict[str, object]] = field(default_factory=list)
    activation_strength: float = 1.0
    activation_last_time: float = 0.0

    def activation_at(self, now: float, decay_rate: float = 0.08) -> float:
        return self.activation_strength * math.exp(-decay_rate * max(0.0, now - self.activation_last_time))

    def status_at(self, now: float) -> str:
        value = self.activation_at(now)
        return 'focused' if value >= 1.0 else ('peripheral' if value >= 0.2 else 'dormant')


class ProgressiveRefiner:
    """At most two bounded interpretation stages per decision, with no extra detail reads.

    Stage 1 represents the current context without historical meaning scores.
    Stage 2 validates existing interpreted sources and weighs their applicability.
    The same signature never generates a redundant new version.
    """

    def __init__(self, learner, *, learning: bool = False, max_versions: int = 256,
                 max_focused: int = 2, max_active: int = 4) -> None:
        self.learner = learner
        self.learning = learning
        self.max_versions = max_versions
        self.max_focused = max_focused
        self.max_active = max_active
        if min(max_versions, max_focused, max_active) < 1 or max_focused > max_active:
            raise ValueError('thought capacity must be positive with focused <= active')
        self._active: dict[str, ThoughtThread] = {}
        self.threads: dict[tuple[str, str], ThoughtThread] = {}

    def thread(self, state: CognitiveState) -> ThoughtThread:
        key = (state.episode_id, state.goal_id)
        if key not in self.threads:
            self.threads[key] = ThoughtThread(
                thought_id=f"{state.episode_id}:{state.goal_id}",
                initiating_episode=state.episode_id,
                initiating_goal=state.goal_id,
            )
        return self.threads[key]

    def _focus(self, thread: ThoughtThread, now: float) -> None:
        # Only scan the bounded active set; dormant history remains indexed but
        # receives zero repeated processing until its specific thought is cued.
        thread.activation_strength = min(2.0, thread.activation_at(now) + 1.0)
        thread.activation_last_time = now
        self._active[thread.thought_id] = thread
        for tid, candidate in tuple(self._active.items()):
            if candidate.status_at(now) == 'dormant':
                del self._active[tid]
        focused = sorted(
            (t for t in self._active.values() if t.status_at(now) == 'focused'),
            key=lambda t: (t is not thread, -t.activation_at(now), t.thought_id),
        )
        for displaced in focused[self.max_focused:]:
            displaced.activation_strength = min(0.6, displaced.activation_at(now))
            displaced.activation_last_time = now
        active = sorted(self._active.values(),
                        key=lambda t: (t is not thread, -t.activation_at(now), t.thought_id))
        for displaced in active[self.max_active:]:
            displaced.activation_strength = 0.0
            displaced.activation_last_time = now
            del self._active[displaced.thought_id]

    @staticmethod
    def _uncertainty(ordered: list[tuple[float, Action, list[str], object]]) -> float:
        margin = ordered[0][0] - ordered[1][0] if len(ordered) > 1 else 4.0
        return 1.0 / (1.0 + max(0.0, margin))

    def refine(self, state: CognitiveState, scored, hits: list[RetrievalHit]):
        if not scored:
            raise ValueError("refinement requires candidate actions")
        thread = self.thread(state)
        self._focus(thread, state.current_time)
        # The existing selector has already evaluated all five layers. Reuse them:
        # no repeat event reads and no inference masquerading as observation.
        revised = []
        evidence_ids: set[str] = set(state.observed_evidence_ids)
        for score, action, reasons, frame in scored:
            correction = 0.0
            if frame and frame.meanings:
                for axis in frame.meanings:
                    evidence_ids.update(axis.source_event_ids)
                # An observed current contradiction invalidates transfer of an
                # old successful key. This is a consistency constraint, NOT decay.
                if action.kind == "try_key" and action.key_name in state.contradicted_keys:
                    if any(not layer.temporal_valid for layer in frame.meanings):
                        correction -= 0.24
                if self.learning and frame.independent_sources:
                    correction += self.learner.refinement_adjustment(frame.context, action.kind)
            revised.append((score + correction, action, reasons + [f"refinement={correction:.3f}"], frame))
        final_order = sorted(revised, key=lambda x: (-x[0], x[1].label))
        source_signature = tuple(
            (item[1].label,
             tuple(sorted({eid for layer in (item[3].meanings if item[3] else ())
                           for eid in layer.source_event_ids})),
             bool(item[3] and item[3].independent_sources))
            for item in sorted(revised, key=lambda x: x[1].label)
        )
        contradictions = tuple(sorted(state.contradicted_keys))
        signature = (
            state.goal_id, state.candidate_key, contradictions,
            tuple(sorted(state.observed_evidence_ids)), source_signature,
        )
        preferred = final_order[0][1].label
        alternatives = tuple(x[1].label for x in final_order[1:4])
        memory_ids = tuple(sorted({hit.chunk_id for hit in hits}))
        uncertainty = self._uncertainty(final_order)
        before_count = len(thread.versions)
        # A context-only hypothesis is an explicit initial provisional version.
        # It is not falsely presented as an observed outcome.
        if not thread.versions:
            local_order = sorted(
                scored,
                key=lambda x: (-(x[0] - next(
                    (float(reason.split("=", 1)[1]) for reason in x[2]
                     if reason.startswith("memory=")), 0.0)), x[1].label),
            )
            local_preferred = local_order[0][1].label
            initial_signature = (('initial',) if (local_preferred != preferred or
                                any(ids for _, ids, _ in source_signature)) else signature)
            thread.versions.append(ThoughtVersion(
                number=1, preferred_action=local_preferred,
                alternatives=tuple(x[1].label for x in local_order[1:4]),
                evidence_ids=tuple(sorted(state.observed_evidence_ids)),
                memory_chunk_ids=(), contradictions=contradictions,
                uncertainty=self._uncertainty(local_order),
                reason="initial_context_hypothesis", signature=initial_signature,
                time=state.current_time,
            ))
        previous = thread.versions[-1]
        # No new version from repeated activation or exploration noise alone.
        if previous.signature != signature and len(thread.versions) < self.max_versions:
            source_ids = {eid for _, ids, _ in source_signature for eid in ids}
            if preferred != previous.preferred_action:
                reason = "evidence_changed_preferred_action"
            elif contradictions != previous.contradictions:
                reason = "contradiction_changed_interpretation"
            elif source_ids - set(previous.evidence_ids):
                reason = "new_evidence_integrated"
            elif (state.candidate_key != (previous.signature[1] if len(previous.signature) > 1 else None)
                  or source_signature != (previous.signature[3] if len(previous.signature) > 3 else ())):
                reason = "context_or_evidence_reinterpreted"
            else:
                reason = "unchanged"
            if reason != "unchanged":
                thread.versions.append(ThoughtVersion(
                    number=len(thread.versions) + 1,
                    preferred_action=preferred, alternatives=alternatives,
                    evidence_ids=tuple(sorted(evidence_ids)),
                    memory_chunk_ids=memory_ids,
                    contradictions=contradictions, uncertainty=uncertainty,
                    reason=reason, signature=signature, time=state.current_time,
                ))
        # Only the latest revision is active; historical versions need no CPU.
        current = thread.versions[-1]
        return final_order, {
            "thought_id": thread.thought_id, "version": current.number,
            "activation_state": thread.status_at(state.current_time),
            "changed": len(thread.versions) != before_count,
            "reasoning_cycles": 2 if any(ids for _, ids, _ in source_signature) else 1,
            "stopped_reason": "no_new_evidence_or_stable" if len(thread.versions) == before_count else "bounded_refinement_complete",
            "current": current.trace(), "history_length": len(thread.versions),
            "meaningfully_changed_action": preferred != sorted(scored, key=lambda x: (-x[0], x[1].label))[0][1].label,
            "additional_detailed_reads": 0,
        }

    def feedback(self, state: CognitiveState, decision, *, outcome: str | None,
                 outcome_event_ids: tuple[str, ...]) -> None:
        if not decision.thought_trace:
            return
        thread = self.thread(state)
        thread.outcomes.append({
            "version": decision.thought_trace["version"],
            "action": decision.action.label, "predicted": decision.predicted_success,
            "outcome": outcome, "outcome_event_ids": list(outcome_event_ids),
        })
        # Credit only an observed success/failure of that exact attempted action.
        # An informative inspection is not a goal-success label.
        if (self.learning and outcome in ("success", "failure")
                and decision.action.kind == "try_key"
                and decision.meaning_frame and decision.meaning_frame.independent_sources):
            self.learner.learn_from_refinement(
                decision.meaning_frame.context, decision.action.kind,
                predicted=decision.predicted_success,
                outcome=(outcome == "success"),
            )
