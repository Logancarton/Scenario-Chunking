from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math


def _two_zero_counts() -> list[float]:
    return [0.0, 0.0]


@dataclass
class ActionTrace:
    event_id: str
    features: frozenset[str]
    informative: bool = False
    resolved: bool = False
    produced_evidence_ids: frozenset[str] = frozenset()
    used_evidence_ids: frozenset[str] = frozenset()


class OutcomeLearner:
    """Transparent linear policy learner with causal and eligibility-trace credit assignment."""

    def __init__(self, *, learning_rate: float = 0.32, gamma: float = 0.72, enabled: bool = True, strict_credit: bool = False) -> None:
        self.learning_rate = learning_rate
        self.enabled = enabled
        self.gamma = gamma
        self.strict_credit = strict_credit
        self.weights: dict[str, float] = defaultdict(float)
        self.episode_trace: list[ActionTrace] = []
        self.credit_log: list[dict[str, object]] = []
        # Separate learned interpretation gains from contextual integration logits.
        # Historical events stay immutable; these are learned associations only.
        self.meaning_gains: dict[tuple[str, str, str], float] = defaultdict(float)
        self.gate_logits: dict[tuple[str, str], float] = defaultdict(float)
        self.meaning_updates: list[dict[str, object]] = []
        # Prediction-calibration correction, not credit for mere thought revision.
        self.refinement_bias: dict[tuple[str, str], float] = defaultdict(float)
        self.refinement_updates: list[dict[str, object]] = []
        # Source testimony is calibrated only against subsequent causal outcomes.
        # Directed prerequisites are learned from observed blocked execution,
        # never copied from the environment's private dependency graph.
        self.source_verifications: dict[tuple[str, str], list[bool]] = defaultdict(list)
        self.prerequisite_observations: dict[tuple[str, str], int] = defaultdict(int)
        # Contextual outcomes are statistics of observed causal attempts, not a
        # second memory store. Source events remain in EpisodicMemory.
        self.context_counts: dict[tuple[str, str], list[float]] = defaultdict(_two_zero_counts)
        self.context_base_counts: dict[str, list[float]] = defaultdict(_two_zero_counts)
        self.context_evidence_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.focus_value: dict[str, list[float]] = defaultdict(_two_zero_counts)
        self.inspect_value: list[float] = [1.0, 1.0]

    def verify_source(self, person: str, context: str, correct: bool) -> None:
        self.source_verifications[(person, context)].append(bool(correct))

    def source_reliability(self, person: str, context: str) -> float:
        outcomes = self.source_verifications[(person, context)]
        return (1.0 + sum(outcomes)) / (2.0 + len(outcomes))

    def observe_prerequisite(self, prerequisite: str, dependent: str) -> None:
        if prerequisite == dependent:
            return
        self.prerequisite_observations[(prerequisite, dependent)] += 1

    def prerequisite_strength(self, prerequisite: str, dependent: str) -> float:
        return min(1.0, self.prerequisite_observations[(prerequisite, dependent)] / 2.0)

    def prerequisites_for(self, dependent: str) -> dict[str, float]:
        return {before: self.prerequisite_strength(before, after)
                for before, after in self.prerequisite_observations if after == dependent}


    def contextual_base_probability(self, procedure: str) -> float:
        yes, no = self.context_base_counts[procedure]
        return (yes + 1.0) / (yes + no + 2.0)

    def contextual_probability(self, cue: str, procedure: str) -> tuple[float, float]:
        """Bounded Beta posterior over actual observed action outcomes."""
        yes, no = self.context_counts[(cue, procedure)]
        return (yes + 1.0) / (yes + no + 2.0), yes + no

    def contextual_support(self, cues: frozenset[str], procedure: str) -> dict:
        """Generalize through independently learned relational conjunctions.

        Both causal relation families contribute; surface identities are never
        memorized as a required key. Nothing here depends on private world truth.
        """
        relations = sorted(cue for cue in cues if cue.startswith("pair:"))
        scores, ids = [], []
        base = self.contextual_base_probability(procedure)
        base_log_odds = math.log(max(.05, base) / max(.05, 1 - base))
        for cue in relations:
            probability, count = self.contextual_probability(cue, procedure)
            if count:
                # Compare against the outcome's *action-specific base rate*.
                # Merely predicting common failures is not cognitive relevance.
                strength = min(1.0, count / 12.0)
                conditional_log_odds = math.log(
                    max(.05, probability) / max(.05, 1 - probability))
                gain = conditional_log_odds - base_log_odds
                attention = self.focus_value[cue]
                relevance = max(.15, min(1.0, .5 + 4 * attention[0]))
                scores.append((strength * relevance * gain,
                               strength * relevance))
                ids.extend(self.context_evidence_ids.get((cue, procedure), ()))
        normalizer = sum(weight for _, weight in scores)
        return {"score": sum(score for score, _ in scores) /
                         max(1.0, normalizer),
                "evidence_ids": sorted(set(ids)),
                "covered_relations": len(scores),
                "relation_count": len(relations)}

    def learn_context_outcome(self, cues: frozenset[str], procedure: str,
                              success: bool, *, evidence_id: str) -> None:
        """Learn from an observed action, never a supplied correct answer."""
        if not self.enabled:
            return
        baseline = self.contextual_base_probability(procedure)
        for cue in sorted(c for c in cues if c.startswith("pair:")):
            prior, sample_size = self.contextual_probability(cue, procedure)
            key = (cue, procedure)
            yes, no = self.context_counts[key]
            # Maintain reversibility: old history is still immutable in memory,
            # while effective current associations can adapt to new outcomes.
            if yes + no > 120:
                yes *= .97
                no *= .97
            self.context_counts[key] = [yes + float(success),
                                        no + float(not success)]
            ids = self.context_evidence_ids[key]
            if evidence_id not in ids:
                ids.append(evidence_id)
                del ids[:-12]
            # Causal feedback adjusts attention quality by predictive gain.
            baseline_error = (baseline - float(success)) ** 2
            pair_error = (prior - float(success)) ** 2
            value = baseline_error - pair_error
            quality = self.focus_value[cue]
            quality[0] = .96 * quality[0] + .04 * value
            quality[1] += 1
        yes, no = self.context_base_counts[procedure]
        if yes + no > 120:
            yes *= .97
            no *= .97
        self.context_base_counts[procedure] = [
            yes + float(success), no + float(not success)]

    def select_context_focus(self, cues: frozenset[str], *,
                             learned: bool, budget: int = 2) -> frozenset[str]:
        if budget < 1:
            raise ValueError("focus budget must be positive")
        relations = sorted(c for c in cues if c.startswith("pair:"))
        if not learned or not any(self.focus_value[c][1] for c in relations):
            return frozenset(relations[:budget])
        priority = lambda c: (self.focus_value[c][0], self.focus_value[c][1], c)
        return frozenset(sorted(relations, key=priority, reverse=True)[:budget])

    def observation_value(self) -> float:
        yes, no = self.inspect_value
        return yes / (yes + no)

    def learn_observation_value(self, useful: bool) -> None:
        if self.enabled:
            yes, no = self.inspect_value
            self.inspect_value = [yes * .98 + float(useful),
                                  no * .98 + float(not useful)]

    def focus_axes(self, remaining: tuple[str, ...], completed: set[str], *,
                   invalidated: set[str] | None = None, limit: int = 2) -> tuple[str, ...]:
        """Prioritize unresolved categories using observed dependency evidence.

        This is a learned *priority*, never a candidate-answer generator.
        When a relationship has been invalidated, old prerequisites cannot
        become fresh truth merely because they were learned repeatedly.
        """
        invalidated = invalidated or set()
        if not remaining or limit < 1:
            return ()
        def priority(axis: str) -> tuple[float, str]:
            prerequisites = self.prerequisites_for(axis)
            blocked = sum(value for before, value in prerequisites.items()
                          if before not in completed)
            supported = sum(value for before, value in prerequisites.items()
                            if before in completed)
            if "sequence" in invalidated:
                blocked = supported = 0.0
            return (2.0 if axis in invalidated else 0.0) + supported - blocked, axis
        return tuple(sorted(remaining, key=lambda axis: (-priority(axis)[0], axis))[:limit])

    def begin_episode(self) -> None:
        self.episode_trace = []

    def register_action(self, event_id: str, features: frozenset[str], *,
                        used_evidence_ids=()) -> None:
        self.episode_trace.append(ActionTrace(
            event_id=event_id, features=features,
            used_evidence_ids=frozenset(used_evidence_ids),
        ))

    def mark_informative(self, event_id: str, *, produced_evidence_ids=()) -> None:
        for item in reversed(self.episode_trace):
            if item.event_id == event_id:
                item.informative = True
                item.produced_evidence_ids = frozenset(produced_evidence_ids)
                return

    def expected_utility(self, features: frozenset[str]) -> float:
        if not self.enabled or not features:
            return 0.0
        # Generalized features transfer across surface names; exact key identity receives no special prior.
        learnable = [f for f in features if not f.startswith("key:") and not f.startswith("goal:")]
        if not learnable:
            return 0.0
        return sum(self.weights[f] for f in learnable) / len(learnable)

    def apply_failure(self, cause_event_id: str, reward: float = -0.35) -> None:
        target = self._find(cause_event_id)
        if target is None:
            return
        self._update(target.features, reward)
        target.resolved = True
        self.credit_log.append({"cause_event_id": cause_event_id, "credited_event_id": cause_event_id, "credit": reward, "kind": "failure"})

    def apply_success(self, cause_event_id: str, reward: float = 1.0) -> None:
        cause_index = next((i for i, x in enumerate(self.episode_trace) if x.event_id == cause_event_id), None)
        if cause_index is None:
            return
        # The causal attempt gets full credit even if feedback arrives after unrelated observations.
        cause = self.episode_trace[cause_index]
        self._update(cause.features, reward)
        cause.resolved = True
        self.credit_log.append({"cause_event_id": cause_event_id, "credited_event_id": cause_event_id, "credit": reward, "kind": "causal_success"})

        # Old policy remains a frozen historical comparison. In refinement mode,
        # only an observation demonstrably USED in the selected interpretation
        # makes its producing information action eligible. Temporal order alone
        # is insufficient; no observed causal memory edge is implied by this credit.
        distance = 1
        for item in reversed(self.episode_trace[:cause_index]):
            if not item.informative:
                continue
            if self.strict_credit and not (item.produced_evidence_ids & cause.used_evidence_ids):
                continue
            credit = reward * (self.gamma ** distance)
            self._update(item.features, credit)
            self.credit_log.append({
                "cause_event_id": cause_event_id, "credited_event_id": item.event_id,
                "source_evidence_ids": sorted(item.produced_evidence_ids & cause.used_evidence_ids),
                "credit": credit,
                "kind": "evidence_supported_predecessor" if self.strict_credit else "informative_predecessor",
            })
            distance += 1

    def _update(self, features: frozenset[str], reward: float) -> None:
        if not self.enabled:
            return
        learnable = [f for f in features if not f.startswith("key:") and not f.startswith("goal:")]
        if not learnable:
            return
        delta = self.learning_rate * reward / len(learnable)
        for feature in learnable:
            self.weights[feature] += delta

    def _find(self, event_id: str) -> ActionTrace | None:
        return next((item for item in self.episode_trace if item.event_id == event_id), None)

    def layer_association(self, axis: str, context: str, action_kind: str) -> float:
        return self.meaning_gains[(axis, context, action_kind)]

    def integration_logit(self, context: str, axis: str) -> float:
        return self.gate_logits[(context, axis)]

    def learn_from_meaning(self, frame, *, beneficial: bool,
                           learn_layers: bool, learn_gate: bool) -> None:
        """Observed immediate outcome/information gain, never hidden task truth.

        An informative inspection has evidence of information gain, NOT causal
        credit for a later successful key attempt.
        """
        if not self.enabled or not frame or not frame.meanings:
            return
        import math
        probability = 1.0 / (1.0 + math.exp(-frame.integrated))
        residual = (1.0 if beneficial else 0.0) - probability
        # Paired ablations isolate gains and gates by independently freezing them.
        if learn_gate:
            for meaning in frame.meanings:
                key = (frame.context, meaning.axis)
                gradient = frame.weights[meaning.axis] * (meaning.value - frame.integrated)
                self.gate_logits[key] = max(-2.0, min(2.0,
                    self.gate_logits[key] + 0.25 * residual * gradient))
        if learn_layers:
            for meaning in frame.meanings:
                key = (meaning.axis, frame.context, frame.action_kind)
                # Axis association only learns when that axis had evidence.
                self.meaning_gains[key] = max(-0.8, min(0.8,
                    self.meaning_gains[key] + 0.24 * residual * meaning.value))
        self.meaning_updates.append({
            "context": frame.context, "action_kind": frame.action_kind,
            "observed_benefit": beneficial, "probability": probability,
            "axes_with_evidence": [m.axis for m in frame.meanings],
            "learned_layers": learn_layers, "learned_integration": learn_gate,
        })

    def refinement_adjustment(self, context: str, action_kind: str) -> float:
        return self.refinement_bias[(context, action_kind)] if self.enabled else 0.0

    def learn_from_refinement(self, context: str, action_kind: str, *,
                              predicted: float, outcome: bool) -> None:
        if not self.enabled:
            return
        if not 0.0 <= predicted <= 1.0:
            raise ValueError('predicted probability outside [0,1]')
        # Only the causal attempt outcome; never reward a revision just for changing.
        residual = float(outcome) - predicted
        key = (context, action_kind)
        self.refinement_bias[key] = max(-0.2, min(0.2,
            self.refinement_bias[key] + 0.12 * residual))
        self.refinement_updates.append({
            'context': context, 'action_kind': action_kind,
            'predicted': predicted, 'observed_success': outcome,
            'residual': residual,
        })

    def meaning_snapshot(self) -> dict[str, dict[str, float]]:
        return {
            "layer_associations": {"|".join(k): round(v, 6) for k, v in sorted(self.meaning_gains.items())},
            "integration_logits": {"|".join(k): round(v, 6) for k, v in sorted(self.gate_logits.items())},
            "feedback_updates": len(self.meaning_updates),
        }

    def snapshot(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        return dict(sorted(self.weights.items()))
