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
        # Temporal prediction-error learning belongs to this existing learner.
        # These are aggregate outcome parameters, never replacement event facts.
        self.temporal_punctuality: dict[str, list[float]] = defaultdict(_two_zero_counts)
        self.temporal_delay: dict[str, list[float]] = defaultdict(_two_zero_counts)
        self.temporal_assessed: set[str] = set()
        self.temporal_arrivals: set[str] = set()
        # E10 sufficient statistics: a direct arrival or a right-censored interval.
        self.temporal_samples: dict[str, dict[str, tuple[float | None, float | None, float]]] = defaultdict(dict)
        self.temporal_sources: dict[tuple[str, str], list[tuple[float, bool]]] = defaultdict(list)

    @staticmethod
    def incremental_scope(cues: frozenset[str]) -> str | None:
        """Stable conceptual address, excluding incidental scenario identities."""
        domains = sorted(c for c in cues if c.startswith("pair:domain:") and "&" not in c)
        epochs = sorted(c for c in cues if c.startswith("pair:epoch:") and "&" not in c)
        if len(domains) != 1 or len(epochs) != 1:
            return None
        return domains[0] + "&" + epochs[0]

    def incremental_concept_id(self, cues: frozenset[str]) -> str:
        scope = self.incremental_scope(cues)
        if scope is None:
            raise ValueError("conceptual continuity requires a unique domain and epoch")
        return "concept:response_dispatch:" + scope

    @staticmethod
    def _incremental_axes(cues: frozenset[str]) -> list[str]:
        """Only independently observed axes; ignore markers and derived conjunctions."""
        return sorted(
            c for c in cues if c.startswith("pair:")
            and "&" not in c and not c.startswith(
                ("pair:domain:", "pair:epoch:", "pair:marker:"))
        )

    def incremental_prediction(self, cues: frozenset[str], procedure: str, *,
                               allow_interactions: bool = True) -> dict:
        """Address a continuing contextual belief, not a separate belief per episode.

        E11's domain/epoch anchor prevents unrelated regimes changing predictions.
        The historical E6 contextual estimator remains unchanged for other users.
        """
        scope = self.incremental_scope(cues)
        if scope is None:
            return self._legacy_incremental_prediction(cues, procedure)
        yes, no = self.context_counts.get((scope, procedure), (0.0, 0.0))
        support = yes + no
        base = (yes + 1.0) / (support + 2.0)
        logit_base = math.log(base / (1.0 - base))
        evidence = set(self.context_evidence_ids.get((scope, procedure), ()))
        effects = []
        if allow_interactions:
            for axis in self._incremental_axes(cues):
                key = scope + "&" + axis
                positive, negative = self.context_counts.get((key, procedure), (0.0, 0.0))
                n = positive + negative
                if n <= 0:
                    continue
                local = (positive + 1.0) / (n + 2.0)
                shrink = n / (n + 3.0)
                effect = shrink * (math.log(local / (1.0 - local)) - logit_base)
                effects.append((effect, shrink, key))
                evidence.update(self.context_evidence_ids.get((key, procedure), ()))
        # Average correlated associations: no additive vote from five co-occurring
        # axes, and no unearned certainty from generating 36 feature conjunctions.
        correction = (sum(effect for effect, _, _ in effects) /
                      max(1.0, sum(weight for _, weight, _ in effects)))
        probability = 1.0 / (1.0 + math.exp(
            -max(-4.0, min(4.0, logit_base + correction))))
        uncertainty = 1.0 / (1.0 + support / 3.0 +
                             sum(weight for _, weight, _ in effects) / 5.0)
        return {
            "probability": probability, "uncertainty": uncertainty,
            "evidence_ids": sorted(evidence),
            "covered_relations": len(effects),
            "relation_effects": {key: delta for delta, _, key in effects},
            "belief_id": self.incremental_concept_id(cues) + ":" + procedure,
            "effective_support": support,
        }

    def _legacy_incremental_prediction(self, cues: frozenset[str], procedure: str) -> dict:
        """Conditional success belief from observed attempts, not episodic lookup.

        Reuses the E6 contextual association tables and evidence provenance.
        Independent cue effects and conjunctive effects are shrunk so a single
        coincidental outcome cannot establish a universal rule.
        """
        successes, failures = self.context_base_counts.get(procedure, (0.0, 0.0))
        base_count = successes + failures
        base = (successes + 1.0) / (base_count + 2.0)
        base_logit = math.log(max(.001, base) / max(.001, 1.0 - base))
        effects = []
        evidence = set()
        for cue in sorted(x for x in cues if x.startswith("pair:")):
            yes, no = self.context_counts.get((cue, procedure), (0.0, 0.0))
            n = yes + no
            if n <= 0:
                continue
            probability = (yes + 1.0) / (n + 2.0)
            local_logit = math.log(max(.001, probability) / max(.001, 1.0 - probability))
            shrink = n / (n + (2.0 if "&" in cue else 5.0))
            # A conjunction may be more informative than a marginal, but is
            # never treated as an independent second observation.
            effects.append((shrink * (local_logit - base_logit), shrink, cue))
            evidence.update(self.context_evidence_ids.get((cue, procedure), ()))
        if effects:
            correction = sum(delta for delta, _, _ in effects) / max(
                1.0, sum(weight for _, weight, _ in effects))
        else:
            correction = 0.0
        logit = max(-4.0, min(4.0, base_logit + correction))
        probability = 1.0 / (1.0 + math.exp(-logit))
        local_count = sum(weight for _, weight, _ in effects)
        # Epistemic uncertainty declines with support, never with repetition
        # of a prediction or a thought revision.
        uncertainty = 1.0 / (1.0 + base_count / 5.0 + local_count)
        return {
            "probability": probability, "uncertainty": uncertainty,
            "evidence_ids": sorted(evidence), "covered_relations": len(effects),
            "relation_effects": {cue: delta for delta, _, cue in effects},
        }

    def incremental_feedback(self, cues: frozenset[str], procedure: str, *,
                             success: bool, evidence_id: str, weight: float = 1.0,
                             allow_interactions: bool = True) -> dict:
        """One observed outcome, at most one total unit of contextual credit.

        A domain/epoch receives half the evidence mass and its observed axes
        divide the other half. Counts and provenance remain in OutcomeLearner;
        full historical event records stay exclusively in EpisodicMemory.
        """
        if not 0.0 < weight <= 1.0:
            raise ValueError("feedback weight must lie in (0, 1]")
        scope = self.incremental_scope(cues)
        if scope is None:
            return self._legacy_incremental_feedback(
                cues, procedure, success=success, evidence_id=evidence_id, weight=weight)
        prior = self.incremental_prediction(
            cues, procedure, allow_interactions=allow_interactions)
        if not self.enabled:
            return {"updated": [], "prediction_error":
                    float(success) - prior["probability"],
                    "belief_id": prior["belief_id"], "evidence_id": evidence_id}
        axes = self._incremental_axes(cues) if allow_interactions else []
        # Only one outcome's worth of evidence is ever credited, irrespective
        # of how many attributes co-occurred. No oracle feature importance.
        shares = [(scope, weight * (0.5 if axes else 1.0))]
        shares.extend((scope + "&" + axis, weight * 0.5 / len(axes))
                      for axis in axes)
        updated = []
        for key, share in shares:
            old = self.context_counts.get((key, procedure), (0.0, 0.0))
            current = [old[0] + share * int(success),
                       old[1] + share * int(not success)]
            self.context_counts[(key, procedure)] = current
            ids = self.context_evidence_ids[(key, procedure)]
            if evidence_id not in ids:
                ids.append(evidence_id)
                del ids[:-24]
            updated.append(key)
        # This action's generic E6 base is not used to address E11 concepts:
        # an outcome in an unrelated domain must not shift another concept.
        posterior = self.incremental_prediction(
            cues, procedure, allow_interactions=allow_interactions)
        return {
            "updated": updated, "prediction_error":
                float(success) - prior["probability"],
            "belief_id": prior["belief_id"],
            "prior_probability": prior["probability"],
            "posterior_probability": posterior["probability"],
            "evidence_id": evidence_id,
            "total_credit_mass": sum(share for _, share in shares),
        }

    def _legacy_incremental_feedback(self, cues: frozenset[str], procedure: str, *,
                                     success: bool, evidence_id: str,
                                     weight: float = 1.0) -> dict:
        """One causal feedback item updates only its matching conditional keys."""
        if not 0.0 < weight <= 1.0:
            raise ValueError("feedback weight must lie in (0, 1]")
        prior = self.incremental_prediction(cues, procedure)
        if not self.enabled:
            return {"updated": [], "prediction_error": float(success) - prior["probability"]}
        before = {cue: self.contextual_probability(cue, procedure)[0]
                  for cue in cues if cue.startswith("pair:")}
        # The ordinary contextual credit owner performs the one-shot update.
        # Observation confidence scales the evidence: weighted updates are
        # supported without changing original source events.
        if weight == 1.0:
            self.learn_context_outcome(cues, procedure, success, evidence_id=evidence_id)
        else:
            base = self.contextual_base_probability(procedure)
            for cue in sorted(before):
                key = (cue, procedure)
                yes, no = self.context_counts[key]
                self.context_counts[key] = [yes + weight * int(success),
                                            no + weight * int(not success)]
                ids = self.context_evidence_ids[key]
                if evidence_id not in ids:
                    ids.append(evidence_id)
                    del ids[:-12]
                quality = self.focus_value[cue]
                posterior = before[cue]
                quality[0] = .96 * quality[0] + .04 * (
                    (base - float(success)) ** 2 -
                    (posterior - float(success)) ** 2)
                quality[1] += 1
            counts = self.context_base_counts[procedure]
            counts[0] += weight * int(success)
            counts[1] += weight * int(not success)
        updated = [cue for cue in sorted(before)
                   if self.contextual_probability(cue, procedure)[0] != before[cue]]
        return {"updated": updated, "prediction_error":
                float(success) - prior["probability"], "evidence_id": evidence_id}

    def temporal_survival(self, context: str, promised_duration: float, elapsed: float) -> float:
        """Smoothed empirical survival with censored observations still at risk."""
        if not math.isfinite(promised_duration) or promised_duration <= 0:
            raise ValueError("positive finite promised duration required")
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("finite nonnegative elapsed required")
        events = [(0.7, True), (1.5, True), (math.inf, False)]
        for arrival, lower, promise in self.temporal_samples[context].values():
            if promise > 0:
                events.append(((arrival if arrival is not None else lower) / promise,
                               arrival is not None))
        survival = 1.0
        for t in sorted(set(t for t, _, in events if t <= elapsed / promised_duration)):
            at_risk = sum(1 for when, _ in events if when >= t)
            arrived = sum(1 for when, occurred in events if when == t and occurred)
            if at_risk:
                survival *= 1.0 - arrived / at_risk
        return max(0.0, min(1.0, survival))

    def temporal_arrival_probability(self, context: str, promised_duration: float,
                                     elapsed: float, horizon: float,
                                     *, known_absent_at: float | None = None) -> float:
        """Probability of event within window, conditional on last direct absence."""
        if horizon < elapsed:
            raise ValueError("horizon precedes elapsed")
        start = max(elapsed, known_absent_at or 0.0)
        before = self.temporal_survival(context, promised_duration, start)
        after = self.temporal_survival(context, promised_duration, horizon)
        return max(0.0, min(1.0, (before - after) / max(1e-9, before)))

    def verify_temporal_source(self, person: str, context: str, *,
                               observed_at: float, correct: bool) -> None:
        if self.enabled and math.isfinite(observed_at):
            self.temporal_sources[(person, context)].append((observed_at, bool(correct)))

    def temporal_source_reliability(self, person: str, context: str,
                                    now: float, *, half_life: float = 40.0) -> float:
        yes = no = 1.0
        for when, correct in self.temporal_sources[(person, context)]:
            weight = 0.5 ** (max(0.0, now - when) / half_life)
            if correct:
                yes += weight
            else:
                no += weight
        return yes / (yes + no)


    def temporal_prediction(self, context: str, promised_duration: float) -> tuple[float, float]:
        """Return estimated actual duration and on-time probability.

        Unobserved overdue expectations supply no negative labels.
        """
        if promised_duration <= 0 or not math.isfinite(promised_duration):
            raise ValueError("promised duration must be finite and positive")
        count, total_delta = self.temporal_delay[context]
        estimated = max(0.0, promised_duration + (total_delta / count if count else 0.0))
        on_time, late = self.temporal_punctuality[context]
        return estimated, (1.0 + on_time) / (2.0 + on_time + late)

    def learn_temporal_observation(self, expectation, *, observed_at: float,
                                   returned: bool, arrival_at: float | None = None) -> None:
        """Learn punctuality once from direct evidence; learn delay on arrival.

        An initial witnessed absence can be followed by a witnessed late
        arrival without treating these as two independent punctuality trials.
        """
        if not self.enabled:
            return
        context = expectation.context
        identity = expectation.expectation_id
        # One identity, one mutable learned sample; never edit historic evidence.
        promised = expectation.due_at - expectation.started_at
        if returned and arrival_at is not None:
            self.temporal_samples[context][identity] = (
                max(0.0, arrival_at - expectation.started_at), None, promised)
        elif not returned and observed_at >= expectation.started_at:
            prior = self.temporal_samples[context].get(identity)
            if prior is None or prior[0] is None:
                lower = max(observed_at - expectation.started_at,
                            prior[1] if prior and prior[1] is not None else 0.0)
                self.temporal_samples[context][identity] = (None, lower, promised)
        if identity not in self.temporal_assessed and (
                (returned and arrival_at is not None)
                or (not returned and observed_at >= expectation.due_at)):
            if returned and arrival_at is not None and arrival_at <= expectation.due_at:
                self.temporal_punctuality[context][0] += 1.0
            else:
                self.temporal_punctuality[context][1] += 1.0
            self.temporal_assessed.add(identity)
        if returned and arrival_at is not None and identity not in self.temporal_arrivals:
            promised = expectation.due_at - expectation.started_at
            actual = arrival_at - expectation.started_at
            self.temporal_delay[context][0] += 1.0
            self.temporal_delay[context][1] += actual - promised
            self.temporal_arrivals.add(identity)


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
