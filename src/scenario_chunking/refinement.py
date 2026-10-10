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
    hypotheses: tuple[dict[str, object], ...] = ()
    predicted_outcomes: tuple[tuple[str, float], ...] = ()

    def trace(self) -> dict[str, object]:
        return {
            "version": self.number, "preferred_action": self.preferred_action,
            "alternatives": list(self.alternatives),
            "evidence_ids": list(self.evidence_ids),
            "memory_chunk_ids": list(self.memory_chunk_ids),
            "contradictions": list(self.contradictions),
            "uncertainty": round(self.uncertainty, 6),
            "reason": self.reason, "time": self.time,
            "hypotheses": list(self.hypotheses),
            "predicted_outcomes": dict(self.predicted_outcomes),
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

    @staticmethod
    def _hypotheses(state: CognitiveState, scored, prepared) -> list[dict[str, object]]:
        """Build falsifiable key interpretations from already-indexed causal pairs.

        Historical outcomes transfer only to the same goal, target and key.
        Observed current clues are weighted by their actual source confidence.
        No new chunk events are read, and no speculative causal edge is created.
        """
        hypotheses = []
        for _, action, _, _ in scored:
            if action.kind != "try_key" or not action.key_name:
                continue
            positives: dict[str, tuple[float, str]] = {}
            negatives: dict[str, tuple[float, str]] = {}
            for source_id, agrees, conf, provenance in state.key_observations.get(action.key_name, ()):
                (positives if agrees else negatives)[source_id] = (
                    (2.8 if agrees else 3.1) * max(0.0, min(1.0, conf)),
                    provenance,
                )
            historical: dict[str, tuple[bool, float]] = {}
            if prepared is not None:
                for _, attempt, response in prepared.matches(action):
                    if (attempt.action != action or attempt.goal_id != state.goal_id
                            or response.kind.value != "outcome"
                            or response.truth_status.value != "outcome"
                            or response.cause_event_id != attempt.event_id
                            or response.outcome not in ("success", "failure")):
                        continue
                    source_id = str(response.metadata.get("source_event_id") or response.event_id)
                    if source_id in positives or source_id in negatives:
                        continue
                    historical[source_id] = (
                        response.outcome == "success",
                        max(0.0, min(1.0, response.confidence)),
                    )
            # Bound the influence of repeated historical examples and disallow
            # mere memory quantity from becoming certainty.
            historical_support = min(1.8, sum(1.8 * c for good, c in historical.values() if good))
            historical_against = min(1.8, sum(1.8 * c for good, c in historical.values() if not good))
            log_odds = (
                sum(weight for weight, _ in positives.values())
                - sum(weight for weight, _ in negatives.values())
                + historical_support - historical_against
            )
            log_odds = max(-6.0, min(6.0, log_odds))
            probability = 1.0 / (1.0 + math.exp(-log_odds))
            supportive = tuple(sorted(set(positives) | {sid for sid, (good, _) in historical.items() if good}))
            conflicting = tuple(sorted(set(negatives) | {sid for sid, (good, _) in historical.items() if not good}))
            sources = tuple(sorted(set(supportive) | set(conflicting)))
            hypotheses.append({
                "action": action.label,
                "claim": f"{action.key_name} opens {action.target}",
                "confidence": round(probability, 6),
                "supports": supportive,
                "contradicts": conflicting,
                "sources": sources,
                "provenance": tuple(sorted({
                    tag for _, tag in (*positives.values(), *negatives.values())
                } | ({"historical_causal_outcome"} if historical else set()))),
                "uncertain": 0.30 < probability < 0.78,
            })
        return hypotheses

    def refine(self, state: CognitiveState, scored, hits: list[RetrievalHit], *, prepared=None):
        if not scored:
            raise ValueError("refinement requires candidate actions")
        thread = self.thread(state)
        self._focus(thread, state.current_time)
        direct = sorted(scored, key=lambda x: (-x[0], x[1].label))
        hypotheses = self._hypotheses(state, scored, prepared)
        known = [h for h in hypotheses if h["sources"]]
        contenders = sorted(known, key=lambda h: (-h["confidence"], h["action"]))
        desired_label = None
        decision_basis = "one_pass_adequate"
        if contenders:
            strongest = contenders[0]
            runner = contenders[1]["confidence"] if len(contenders) > 1 else 0.5
            # A key prediction can guide action when supported and discriminative.
            # Otherwise explicitly seek information rather than guess.
            if strongest["confidence"] >= 0.78 and strongest["confidence"] - runner >= 0.13:
                desired_label = strongest["action"]
                decision_basis = "evidence_supported_prediction"
            elif direct[0][1].kind == "try_key":
                investigative = [item for item in direct if item[1].kind != "try_key"]
                if investigative:
                    desired_label = investigative[0][1].label
                    decision_basis = "resolve_competing_hypotheses"
        ordered = list(direct)
        if desired_label and direct[0][1].label != desired_label:
            chosen = next((row for row in ordered if row[1].label == desired_label), None)
            if chosen:
                ordered.remove(chosen)
                ordered.insert(0, (direct[0][0] + 0.001, chosen[1],
                                   chosen[2] + [f"thought={decision_basis}"], chosen[3]))
        preferred = ordered[0][1].label
        alternatives = tuple(item[1].label for item in ordered[1:4])
        contradictions = tuple(sorted(state.contradicted_keys))
        local_observations = tuple(sorted(
            (key, eid, good, round(conf, 6))
            for key, evidence in state.key_observations.items()
            for eid, good, conf, _ in evidence
        ))
        hypothesis_signature = tuple(
            (h["action"], h["supports"], h["contradicts"], h["confidence"])
            for h in sorted(hypotheses, key=lambda h: h["action"])
        )
        signature = (state.goal_id, state.candidate_key, contradictions,
                     local_observations, hypothesis_signature)
        memory_ids = tuple(sorted({h.chunk_id for h in hits}))
        evidence_ids = tuple(sorted({sid for h in hypotheses for sid in h["sources"]}))
        supported_hypothesis = next((h for h in hypotheses if h["action"] == preferred), None)
        selected_prediction = (supported_hypothesis["confidence"]
                               if supported_hypothesis and supported_hypothesis["sources"] else None)
        history_before = len(thread.versions)
        hypotheses_record = tuple({
            "claim": h["claim"], "action": h["action"],
            "supports": list(h["supports"]), "contradicts": list(h["contradicts"]),
            "sources": list(h["sources"]),
            "confidence": h["confidence"], "provenance": list(h["provenance"]),
        } for h in hypotheses)
        predictions = tuple((h["action"], h["confidence"]) for h in hypotheses)

        if not thread.versions:
            local = sorted(
                scored, key=lambda x: (-(x[0] - next(
                    (float(reason.split("=", 1)[1]) for reason in x[2]
                     if reason.startswith("memory=")), 0.0)), x[1].label))
            thread.versions.append(ThoughtVersion(
                number=1, preferred_action=local[0][1].label,
                alternatives=tuple(x[1].label for x in local[1:4]),
                evidence_ids=tuple(sorted(state.observed_evidence_ids)),
                memory_chunk_ids=(), contradictions=contradictions,
                uncertainty=self._uncertainty(local), reason="initial_context_hypothesis",
                signature=("initial",) if known else signature,
                time=state.current_time,
            ))
        previous = thread.versions[-1]
        # Only relevant evidence or changed interpretation causes a revision;
        # stable evidence cannot create thought versions by looping or RNG.
        if previous.signature != signature and len(thread.versions) < self.max_versions:
            reason = ("contradiction_resolved_or_exposed"
                      if contradictions != previous.contradictions else
                      "revised_from_grounded_evidence")
            thread.versions.append(ThoughtVersion(
                number=len(thread.versions) + 1, preferred_action=preferred,
                alternatives=alternatives, evidence_ids=evidence_ids,
                memory_chunk_ids=memory_ids, contradictions=contradictions,
                uncertainty=(1 - max((h["confidence"] for h in known), default=0.5)),
                reason=reason, signature=signature, time=state.current_time,
                hypotheses=hypotheses_record, predicted_outcomes=predictions,
            ))
        current = thread.versions[-1]
        selected_support = list(supported_hypothesis["supports"]) if supported_hypothesis else []
        return ordered, {
            "thought_id": thread.thought_id, "version": current.number,
            "activation_state": thread.status_at(state.current_time),
            "changed": len(thread.versions) != history_before,
            "reasoning_cycles": 2 if known else 1,
            "stopped_reason": ("no_relevant_new_evidence"
                               if len(thread.versions) == history_before else
                               "evidence_revision_complete"),
            "current": current.trace(), "history_length": len(thread.versions),
            "hypotheses": list(hypotheses_record),
            "decision_basis": decision_basis,
            "prediction": selected_prediction,
            "contributing_evidence_ids": selected_support,
            "meaningfully_changed_action": preferred != direct[0][1].label,
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
