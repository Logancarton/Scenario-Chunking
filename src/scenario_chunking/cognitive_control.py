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
