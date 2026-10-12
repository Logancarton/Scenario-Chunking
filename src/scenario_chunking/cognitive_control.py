"""Process knowledge for E12 metacognitive learning (held by OutcomeLearner).

``CognitiveControl`` is data plus bounded update rules about HOW the organism
investigates and how far it trusts itself. It has no reference to memory,
explanations, the selector or authorization; ``OutcomeLearner`` is the only
owner that updates it (``process_feedback``), from the thought owner's
appraisal of past decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
def _confidence_bin(p: float) -> int:
    return min(4, int(p * 5))


def _familiarity_bin(n: int) -> int:
    return 0 if n <= 0 else (1 if n < 4 else 2)


@dataclass
class CognitiveControl:
    """Process knowledge: how this organism allocates investigation and how far
    it trusts its own predictions. Slow timescale, bounded and auditable.

    It never stores facts about the world, never edits an explanation and has
    no access to authorization. Its quantities are judgments about the
    organism's OWN cognition, revised from evidence about past examinations,
    with a hierarchical prior: judgments from OTHER scopes (each scope counted
    once) form the prior for a scope, so a scope's own evidence is never
    counted twice and one unrelated family cannot outweigh its own experience.

    * relevance[axis][scope]: the latest log Bayes factor (computed by the
      knowledge owner, model-free) that outcomes depend on the variable among
      experiences where it was examined. The belief that examining it matters
      is the posterior under the scope's prior; the prior is what other
      scopes concluded, raised by the scope's self-diagnosed missing
      information. Bounded: one outcome moves a Bayes factor only slightly.
    * drive[scope]: self-diagnosed share of outcome variability that
      always-visible variables leave unexplained (missing information).
    * trust[cell][scope]: recency-weighted track record of the organism's
      predictions per (procedure, familiarity, confidence) cell. Calibrated
      belief = track record shrunk toward the model's own claim.
    """
    prior_relevance: float = .3
    scope_prior_strength: float = 1.0
    exploration_gain: float = .5
    competence_scale: float = 8.0
    trust_strength: float = 8.0
    trust_decay: float = .98
    relevance: dict[str, dict[tuple, float]] = field(default_factory=dict)
    drive: dict[tuple, float] = field(default_factory=dict)
    trust: dict[tuple, dict[tuple, list[float]]] = field(default_factory=dict)
    adjustments: list[dict] = field(default_factory=list)
    updates: int = 0

    @staticmethod
    def _posterior(prior: float, log_bayes_factor: float) -> float:
        prior = min(1 - 1e-6, max(1e-6, prior))
        logit = math.log(prior / (1 - prior)) + max(-30.0, min(30.0, log_bayes_factor))
        return 1.0 / (1.0 + math.exp(-logit))

    def relevance_prior(self, scope: tuple, axis: str) -> float:
        """Leave-one-out pooled judgment: what other scopes concluded about this variable."""
        others = [self._posterior(self.prior_relevance, lbf)
                  for s, lbf in self.relevance.get(axis, {}).items() if s != scope]
        return ((sum(others) + self.scope_prior_strength * self.prior_relevance)
                / (len(others) + self.scope_prior_strength))

    def relevance_belief(self, scope: tuple, axis: str, *, diagnosis: bool = True) -> float:
        prior = self.relevance_prior(scope, axis)
        if diagnosis:
            prior = max(prior, self.drive.get(scope, 0.0))
        return round(self._posterior(prior, self.relevance.get(axis, {}).get(scope, 0.0)), 9)

    def activate_axes(self, scope: tuple, axes, *, mode: str, limit: int,
                      experience: int) -> frozenset[str]:
        """Allocate live computational hypotheses without hiding public evidence.

        Fixed sparse is a nonlearning control. Learned sparse uses the existing
        process relevance model and occasionally probes overlooked axes.
        """
        ordered = sorted(set(axes))
        if mode == "dense":
            return frozenset(ordered)
        if mode not in ("fixed_sparse", "learned_sparse", "adaptive_sparse") or limit < 1:
            raise ValueError("invalid sparse activation policy")
        if len(ordered) <= limit:
            return frozenset(ordered)
        if mode == "fixed_sparse":
            return frozenset(ordered[:limit])
        if mode == "adaptive_sparse":
            ranked = sorted(ordered, key=lambda a: (-self.reasoning_priority(scope, a), a))
            return frozenset(ranked[:limit])
        ranked = sorted(ordered, key=lambda a: (-self.relevance_belief(scope, a), a))
        chosen = ranked[:limit]
        diagnosed = any(a.get("scope") == list(scope) and
                        a.get("error") == "missing_information"
                        for a in self.adjustments[-8:])
        if diagnosed or experience % 7 == 0:
            skipped = [a for a in ranked if a not in chosen]
            if skipped:
                chosen[-1] = skipped[(experience // 7) % len(skipped)]
        return frozenset(chosen)

    def reasoning_priority(self, scope: tuple, axis: str, competence: float = 0.0) -> float:
        """Bounded, uncertainty-aware attention priority based on observed process history.

        A previously unexamined axis receives an optimism bonus; feedback that
        examining it harmed prediction lowers its priority. Neither the hidden
        value nor the true task rule is consulted. Historical sparse modes do
        not use this method.
        """
        history = [a["reasoning"] for a in self.adjustments
                   if tuple(a.get("scope", ())) == scope and "reasoning" in a]
        examined = [h["inspections"][axis] for h in history if axis in h["inspections"]]
        n = len(examined)
        gain = sum(max(-1.5, min(1.5, float(x.get("gain", 0.0))))
                   for x in examined) / (n + 3)
        novelty = (0.17 * (1.0 - competence) *
                   math.sqrt(math.log(2.0 + len(history)) / (1.0 + n)))
        return self.relevance_belief(scope, axis) + novelty + 0.35 * gain

    def reconsider_axis(self, scope: tuple, axes, active: frozenset[str], *,
                        adequacy: dict, competences: dict[str, float],
                        query_cost: float = 0.02) -> tuple[str, frozenset[str]] | None:
        """Suggest at most one costed, same-budget hypothesis replacement.

        This evaluates a potential reasoning investment, NOT its answer.
        The caller performs the actual belief queries and keeps final action
        selection with the sole ActionSelector.
        """
        if adequacy["status"] not in ("needs_reconsideration", "world_uncertainty"):
            return None
        omitted = sorted(set(axes) - set(active))
        if not omitted:
            return None
        axis = max(omitted, key=lambda a: (self.reasoning_priority(
            scope, a, competences.get(a, 0.0)), -len(a), a))
        uncertainty = max(0.0, min(1.0, adequacy["uncertainty"]))
        ambiguity = max(0.0, 0.25 - adequacy["margin"])
        relevance = self.relevance_belief(scope, axis)
        benefit = ((0.48 * uncertainty + 0.35 * ambiguity +
                    0.25 * self.drive.get(scope, 0.0)) *
                   (0.15 + relevance) * (1.0 - competences.get(axis, 0.0)))
        # Repeated probes that failed to change a decision or produce new
        # observations are process evidence about wasted THINKING, not world
        # evidence that the omitted variable is irrelevant.
        unproductive = sum(
            1 for entry in self.adjustments
            if tuple(entry.get("scope", ())) == scope
            for observation in [entry.get("reasoning") or {}]
            for trial in [observation.get("recovery")]
            if trial and trial["axis"] == axis
            and not (trial["changed_preferred_route"] or trial.get("led_to_inspection")))
        benefit /= 1.0 + 0.6 * unproductive
        if benefit <= query_cost * 4:
            return None
        incumbent = min(active, key=lambda a: (self.reasoning_priority(
            scope, a, competences.get(a, 0.0)), a))
        return axis, frozenset((set(active) - {incumbent}) | {axis})

    def exploration_value(self, scope: tuple, axis: str, competence: float, *,
                          diagnosis: bool = True) -> float:
        """Value of examining a variable to LEARN about it: fades with competence."""
        return round(self.exploration_gain * self.relevance_belief(scope, axis, diagnosis=diagnosis)
                     * (1.0 - competence), 9)

    def calibrate(self, scope: tuple, procedure: str, familiarity: int, p: float) -> float:
        cell = (procedure, _familiarity_bin(familiarity), _confidence_bin(p))
        table = self.trust.get(cell, {})
        yes = sum(v[0] for s, v in table.items() if s != scope)
        n = sum(v[1] for s, v in table.items() if s != scope)
        pooled = (yes + self.trust_strength * p) / (n + self.trust_strength)
        yes, n = table.get(scope, (0.0, 0.0))
        return round((yes + self.trust_strength * pooled) / (n + self.trust_strength), 9)

    def record_outcome(self, scope: tuple, procedure: str, familiarity: int, p: float,
                       success: bool) -> None:
        cell = (procedure, _familiarity_bin(familiarity), _confidence_bin(p))
        table = self.trust.setdefault(cell, {})
        yes, n = table.get(scope, (0.0, 0.0))
        table[scope] = [round(self.trust_decay * yes + float(success), 9),
                        round(self.trust_decay * n + 1.0, 9)]

    def record_relevance(self, scope: tuple, axis: str, log_bayes_factor: float) -> None:
        self.relevance.setdefault(axis, {})[scope] = round(log_bayes_factor, 9)

    def audit(self, entry: dict) -> None:
        self.updates += 1
        self.adjustments.append(entry)
        del self.adjustments[:-64]
