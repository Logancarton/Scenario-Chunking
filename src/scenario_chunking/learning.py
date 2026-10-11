from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
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


@dataclass
class CausalHypothesis:
    """A defeasible conditional explanation, never an established causal fact."""
    hypothesis_id: str
    conditions: tuple[str, ...]
    predicts_success: bool
    origin: tuple[str, str]
    support: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    support_count: int = 0
    contradiction_count: int = 0

    def observe(self, evidence_id: str, success: bool) -> None:
        if success == self.predicts_success:
            self.support_count += 1
            target = self.support
        else:
            self.contradiction_count += 1
            target = self.contradictions
        target.append(evidence_id)
        del target[:-12]

    def trace(self) -> dict:
        n = self.support_count + self.contradiction_count
        confidence = (self.support_count + 1) / (n + 2)
        return {"id": self.hypothesis_id, "conditions": list(self.conditions),
                "predicts_success": self.predicts_success,
                "origin": list(self.origin), "support": list(self.support),
                "contradictions": list(self.contradictions),
                "confidence": confidence,
                "support_count": self.support_count, "contradiction_count": self.contradiction_count,
                "status": ("contradicted" if self.contradiction_count > self.support_count
                           else "supported" if self.support_count >= 2 else "tentative")}


Rule = frozenset  # frozenset[frozenset[str]]: a DNF over public literals


def _holds(rule: Rule, literals: frozenset[str]) -> bool:
    return any(term <= literals for term in rule)


def _axis(literal: str) -> str:
    return literal.split(":", 2)[1]


def describe_rule(rule: Rule) -> str:
    if not rule:
        return "never"
    return " OR ".join(" AND ".join(sorted(t)) for t in sorted(rule, key=sorted))


def _rule_size(rule: Rule) -> float:
    return sum(len(t) for t in rule) + .5 * len(rule)


def _aggregate(rows) -> dict[frozenset[str], list[int]]:
    """Sufficient statistics of (literals, success) observations, in order."""
    table: dict[frozenset[str], list[int]] = {}
    for literals, success, *_ in rows:
        counts = table.setdefault(literals, [0, 0])
        counts[0 if success else 1] += 1
    return table


def _stable_unit(text: str) -> float:
    import hashlib
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") / 2 ** 64


@dataclass
class ExplanationBank:
    """One persistent, revisable explanation of when a procedure succeeds.

    The explanation is a bounded beam of competing complete rules (small DNFs),
    each scored by the observed evidence under an explicit noise model.
    Revision edits the CURRENT best rules (specialize, generalize, add or drop
    a term); new experiences and contrasts only seed the edit material.

    Every unified-E12 mechanism below defaults OFF, which is exactly the
    original E12 learner (frozen baseline for paired comparison):

    * durable: active evidence (bounded window, provenance) is separated from
      durable knowledge (per-context sufficient statistics), so still-valid
      evidence never loses influence merely because the window rolled over.
    * change_detection: a segmented-likelihood test distinguishes a contextual
      exception (one rule still explains old and new evidence) from a regime
      change (old and recent evidence need different rules). The superseded
      regime is archived, never relabelled false, and can be reactivated.
    * directed: a confidently wrong prediction recruits error-directed edits
      for the rules that made it and opens a review that a continuing thought
      can carry to later experiences.
    * recruit: sparse k-winner recruitment of the literals used to build new
      terms ("frozen" fixed relevance, "adaptive" locally learned relevance).
    """
    evidence: list[tuple[frozenset[str], bool, str]] = field(default_factory=list)
    beam: dict[Rule, dict] = field(default_factory=lambda: {
        frozenset(): {"operator": "prior", "origin": None, "revision": 0}})
    revisions: int = 0

    noise: float = 0.1
    complexity_price: float = 0.7
    beam_width: int = 48
    evidence_budget: int = 64
    edited_rules: int = 8
    max_terms: int = 3

    durable: bool = False
    context_budget: int = 256
    contexts: dict[frozenset[str], list[int]] = field(default_factory=dict)
    forgotten: int = 0
    seen: set[str] = field(default_factory=set)
    lineage: list[dict] = field(default_factory=list)
    lineage_total: int = 0
    lineage_budget: int = 64

    change_detection: bool = False
    change_window: int = 12
    change_cost: float = 8.0
    since_change: int = 0
    # Prequential misses: was the prediction made BEFORE each observation wrong?
    prequential: list[bool] = field(default_factory=list)
    archive: list[dict] = field(default_factory=list)
    archive_budget: int = 4
    archive_forgotten: int = 0
    regime_changes: list[dict] = field(default_factory=list)

    directed: bool = False
    review_threshold: float = .3
    vocabulary: dict[str, set[str]] = field(default_factory=dict)

    recruit: str = "off"
    recruit_k: int = 3
    relevance: dict[str, float] = field(default_factory=dict)
    literal_counts: dict[str, list[int]] = field(default_factory=dict)

    search_rounds: int = 1
    work: int = 0

    def _data(self) -> dict[frozenset[str], list[int]]:
        return self.contexts if self.durable else _aggregate(self.evidence)

    def _fit(self, rule: Rule, data: dict[frozenset[str], list[int]]) -> float:
        good, bad = math.log(1 - self.noise), math.log(self.noise)
        total = 0.0
        for literals, (yes, no) in data.items():
            total += (yes * good + no * bad) if _holds(rule, literals) else (yes * bad + no * good)
        return total

    def score(self, rule: Rule) -> float:
        if self.durable:
            fit = self._fit(rule, self.contexts)
        else:
            fit = sum(math.log(1 - self.noise) if _holds(rule, c) == y else math.log(self.noise)
                      for c, y, _ in self.evidence)
        # Rounded so platform libm last-bit differences cannot reorder ties.
        return round(fit - self.complexity_price * _rule_size(rule), 9)

    def score_on(self, rule: Rule, data) -> float:
        return round(self._fit(rule, data) - self.complexity_price * _rule_size(rule), 9)

    def posterior(self) -> list[tuple[Rule, float]]:
        scored = [(rule, self.score(rule)) for rule in self.beam]
        top = max(v for _, v in scored)
        weights = [(rule, math.exp(v - top)) for rule, v in scored]
        total = sum(w for _, w in weights)
        return sorted(((r, w / total) for r, w in weights), key=lambda x: (-x[1], describe_rule(x[0])))

    def trace(self, rule: Rule, weight: float) -> dict:
        support = [eid for c, y, eid in self.evidence if _holds(rule, c) == y]
        contra = [eid for c, y, eid in self.evidence if _holds(rule, c) != y]
        trace = {"id": describe_rule(rule), "rule": sorted(sorted(t) for t in rule),
                 "weight": weight, "confidence": weight,
                 "support_count": len(support), "contradiction_count": len(contra),
                 "support": support[-12:], "contradictions": contra[-12:],
                 **{k: self.beam[rule][k] for k in ("operator", "origin", "revision")}}
        if self.durable:
            # Durable totals: still-valid evidence outside the window still counts.
            yes = sum(a if _holds(rule, c) else b for c, (a, b) in self.contexts.items())
            no = sum(b if _holds(rule, c) else a for c, (a, b) in self.contexts.items())
            trace["durable_support_count"], trace["durable_contradiction_count"] = yes, no
        return trace

    # -- bounded durable knowledge -------------------------------------------------
    def remember(self, literals: frozenset[str], success: bool, evidence_id: str) -> None:
        if not self.durable:
            return
        self.seen.add(evidence_id)
        counts = self.contexts.pop(literals, [0, 0])
        counts[0 if success else 1] += 1
        self.contexts[literals] = counts  # most recently confirmed context last
        while len(self.contexts) > self.context_budget:
            oldest = next(iter(self.contexts))
            self.forgotten += sum(self.contexts.pop(oldest))

    def known(self, evidence_id: str) -> bool:
        return evidence_id in self.seen or any(eid == evidence_id for _, _, eid in self.evidence)

    # -- sparse recruitment --------------------------------------------------------
    def recruited(self, literals: frozenset[str]) -> frozenset[str]:
        """k-winner competition over this context's literals, stable tie-break."""
        if self.recruit == "off" or len(literals) <= self.recruit_k:
            return literals
        for l in literals:
            self.relevance.setdefault(l, _stable_unit("relevance:" + l))
        ranked = sorted(literals, key=lambda l: (-self.relevance[l], _stable_unit("tie:" + l), l))
        return frozenset(ranked[:self.recruit_k])

    def adapt_relevance(self, literals: frozenset[str], success: bool) -> None:
        """Local update of the active winners' relevance toward their association."""
        base = self.literal_counts.setdefault("*", [0, 0])
        base[0 if success else 1] += 1
        for l in literals:
            counts = self.literal_counts.setdefault(l, [0, 0])
            counts[0 if success else 1] += 1
        if self.recruit != "adaptive":
            return
        total = sum(base)
        for l in literals:
            yes, no = self.literal_counts[l]
            rest_yes, rest_n = base[0] - yes, total - yes - no
            association = abs((yes + 1) / (yes + no + 2) - (rest_yes + 1) / (rest_n + 2))
            current = self.relevance.setdefault(l, _stable_unit("relevance:" + l))
            self.relevance[l] = round(current + .3 * (association - current), 9)

    # -- contextual exception versus regime change ---------------------------------
    def _refit(self, rows) -> tuple[list[Rule], int]:
        """Bounded greedy rule search on recent evidence only (2 edit rounds)."""
        data = _aggregate(rows)
        terms = {frozenset([l]) for literals, *_ in rows for l in literals} | {
            frozenset([a, b]) for literals, *_ in rows for a in literals for b in literals
            if a < b and _axis(a) != _axis(b)}
        found, scored = {frozenset()}, 0
        def ranked():
            return sorted(found, key=lambda r: (-self.score_on(r, data), describe_rule(r)))
        for _ in range(2):
            scored += len(found)
            for rule in ranked()[:4]:
                for term in terms:
                    if len(rule | {term}) <= self.max_terms:
                        found.add(rule | {term})
                for term in rule:
                    found.add(rule - {term})
                    for l in term:
                        if len(term) > 1:
                            found.add((rule - {term}) | {term - {l}})
        scored += len(found)
        return ranked()[:8], scored

    def detect_regime(self, evidence_id: str) -> dict | None:
        """Segmented-likelihood test, run only after persistent recent misfit.

        A contextual exception is absorbed by ordinary revision: one rule keeps
        explaining old and recent evidence. A regime change is declared only
        when old and recent evidence need different rules by more than
        change_cost nats. Nothing is learned from the clock or from repetition.
        """
        window = self.change_window
        if self.since_change < 2 * window or len(self.evidence) < window:
            return None
        recent_rows = self.evidence[-window:]
        recent = _aggregate(recent_rows)
        current = min(self.beam, key=lambda r: (-self.score(r), describe_rule(r)))
        # Trigger on prediction error made before each outcome, not on post-hoc
        # fit: ordinary revision can contort a rule to fit both regimes.
        errors = sum(self.prequential[-window:])
        report = {"tested": True, "recent_errors": errors, "rules_scored": len(self.beam)}
        if errors < max(3, math.ceil(.35 * window)):
            return report
        data = self._data()
        old = {}
        for c, (yes, no) in data.items():
            ry, rn = recent.get(c, (0, 0))
            if yes - ry or no - rn:
                old[c] = [yes - ry, no - rn]
        refit, refit_scored = self._refit(recent_rows)
        pool = list(dict.fromkeys([*self.beam, *refit, *(e["rule"] for e in self.archive)]))
        joint = max(self.score(r) for r in self.beam)
        old_best = max(self.score_on(r, old) for r in self.beam)
        ranked_recent = sorted(pool, key=lambda r: (-self.score_on(r, recent), describe_rule(r)))
        recent_best = self.score_on(ranked_recent[0], recent)
        gain = round(old_best + recent_best - joint, 9)
        report.update(gain=gain, rules_scored=3 * len(self.beam) + len(pool) + refit_scored)
        if gain < self.change_cost:
            return report
        restored = next((i for i, e in enumerate(self.archive)
                         if self.score_on(e["rule"], recent) >= recent_best - 1.0), None)
        # The superseded regime is remembered by what best explained ITS evidence.
        historical = min(self.beam, key=lambda r: (-self.score_on(r, old), describe_rule(r)))
        superseded = {"rule": historical, "explanation": describe_rule(historical),
                      "data": dict(list(old.items())[-self.context_budget:]),
                      "evidence_ids": [eid for *_, eid in self.evidence[:-window]][-12:],
                      "ended_revision": self.revisions, "ended_by": evidence_id,
                      "observations": sum(a + b for a, b in old.values())}
        new_data = {c: list(v) for c, v in recent.items()}
        reason, restored_entry = "regime_change", None
        if restored is not None:
            restored_entry = self.archive.pop(restored)
            reason = "regime_reactivated"
            for c, (yes, no) in restored_entry["data"].items():
                counts = new_data.setdefault(c, [0, 0])
                counts[0] += yes
                counts[1] += no
            pool.append(restored_entry["rule"])
        self.archive.append(superseded)
        while len(self.archive) > self.archive_budget:
            self.archive.pop(0)
            self.archive_forgotten += 1
        if self.durable:
            self.contexts = new_data
        self.evidence = list(recent_rows)
        kept = {r: self.beam.get(r) or {"operator": reason, "origin": evidence_id,
                                         "revision": self.revisions}
                for r in dict.fromkeys(pool)}
        self.beam = kept
        ranked = sorted(self.beam, key=lambda r: (-self.score(r), describe_rule(r)))
        self.beam = {r: self.beam[r] for r in ranked[:self.beam_width]}
        self.since_change = 0
        self.prequential = []
        event = {"reason": reason, "evidence_id": evidence_id, "revision": self.revisions,
                 "gain": gain, "superseded": describe_rule(historical),
                 "successor": describe_rule(ranked[0]),
                 "restored": restored_entry["explanation"] if restored_entry else None}
        self.regime_changes.append(event)
        del self.regime_changes[:-32]
        report.update(changed=True, **event)
        return report


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
        self.causal_hypothesis_budget = 32
        self.causal_hypotheses: dict[tuple, dict[tuple, CausalHypothesis]] = {}
        self.causal_counts: dict[tuple, list[int]] = {}
        self.causal_sources: dict[tuple, list[str]] = {}
        self.causal_seen: dict[tuple, set[str]] = {}
        self.explanations: dict[tuple, ExplanationBank] = {}
        # E12 treatment switches for newly created explanation banks; empty is
        # the original E12 learner. Set once, before any experience.
        self.explanation_options: dict[str, object] = {}
        # E10 sufficient statistics: a direct arrival or a right-censored interval.
        self.temporal_samples: dict[str, dict[str, tuple[float | None, float | None, float]]] = defaultdict(dict)
        self.temporal_sources: dict[tuple[str, str], list[tuple[float, bool]]] = defaultdict(list)

    @staticmethod
    def causal_scope(cues: frozenset[str]) -> tuple[str, ...]:
        return tuple(sorted(c for c in cues if "&" not in c and
                            c.startswith(("pair:domain:", "pair:epoch:"))))

    def causal_prediction(self, cues: frozenset[str], procedure: str) -> dict:
        key = (self.causal_scope(cues), procedure)
        yes, no = self.causal_counts.get(key, (0, 0))
        base = (yes + 1) / (yes + no + 2)
        candidates = [h for h in self.causal_hypotheses.get(key, {}).values()
                      if set(h.conditions) <= cues and h.support_count >= 2
                      and h.support_count > h.contradiction_count]
        candidates.sort(key=lambda h: (-(h.support_count + 1) /
                         (h.support_count + h.contradiction_count + 2),
                         -h.support_count, h.hypothesis_id))
        active = candidates[:4]
        probabilities, weights = [], []
        for h in active:
            n = h.support_count + h.contradiction_count
            reliability = (h.support_count + 1) / (n + 2)
            probabilities.append(reliability if h.predicts_success else 1 - reliability)
            weights.append(n / (n + 4))
        # Overlapping rules are alternatives, never independent observations.
        weight = max(weights, default=0.0)
        local = sum(p*w for p,w in zip(probabilities, weights)) / sum(weights) if active else base
        p = (1-weight)*base + weight*local
        disagreement = max(probabilities, default=base) - min(probabilities, default=base)
        conflict = max((2 * (h.contradiction_count + 1) /
                        (h.support_count + h.contradiction_count + 2) for h in active), default=0.0)
        uncertainty = (max(1 / (1 + (yes + no) / 4), 1-weight, disagreement, conflict)
                       if active else max(.5, 1 / (1 + (yes + no) / 4)))
        evidence = sorted(set(self.causal_sources.get(key, ())) |
                          {eid for h in active for eid in (*h.origin, *h.support, *h.contradictions)})
        return {"probability": p, "uncertainty": uncertainty,
                "evidence_ids": evidence,
                "hypotheses": [h.trace() for h in active],
                "scope": list(key[0]), "base_probability": base,
                "hypotheses_examined": len(self.causal_hypotheses.get(key, {}))}

    def causal_feedback(self, cues: frozenset[str], procedure: str, *,
                        success: bool, evidence_id: str, contrast=None,
                        allow_contrast: bool = True) -> dict:
        """Only this owner revises beliefs; refiner supplies a diagnostic pair.

        Candidate budget is per indexed scope/action. Source events stay in
        EpisodicMemory. No scan or activation of unrelated hypothesis banks.
        """
        key = (self.causal_scope(cues), procedure)
        prior = self.causal_prediction(cues, procedure)
        if not self.enabled or evidence_id in self.causal_seen.get(key, set()):
            return {"updated": [], "prediction_error": float(success)-prior["probability"]}
        self.causal_seen.setdefault(key, set()).add(evidence_id)
        counts = self.causal_counts.setdefault(key, [0, 0])
        counts[0 if success else 1] += 1
        sources = self.causal_sources.setdefault(key, [])
        sources.append(evidence_id)
        del sources[:-12]
        bank = self.causal_hypotheses.setdefault(key, {})
        proposed = []
        if allow_contrast and contrast is not None:
            old_cues = frozenset(contrast["cues"])
            before = {c.split(":", 2)[1]: c for c in old_cues if c.startswith("pair:") and "&" not in c}
            after = {c.split(":", 2)[1]: c for c in cues if c.startswith("pair:") and "&" not in c}
            delta = [axis for axis in after if before.get(axis) != after[axis]]
            comparable = (contrast["procedure"] == procedure and
                          self.causal_scope(old_cues) == key[0] and
                          set(before) == set(after) and len(delta) == 1 and
                          contrast["success"] != success and
                          contrast["evidence_id"] != evidence_id)
            if comparable:
                axis = delta[0]
                shared = sorted((cues & old_cues) - set(key[0]))
                # Competing marginal and context-qualified explanations; the
                # marker gets exactly the same treatment as any other feature.
                for literal, outcome in ((after[axis], success), (before[axis], contrast["success"])):
                    for condition in [(literal,)] + [tuple(sorted((literal, c))) for c in shared if "&" not in c]:
                        identity = (condition, outcome)
                        if identity not in bank and len(bank) < self.causal_hypothesis_budget:
                            h = CausalHypothesis(
                                hypothesis_id="|".join((*key[0], procedure, *condition, str(outcome))),
                                conditions=condition, predicts_success=outcome,
                                origin=(contrast["evidence_id"], evidence_id))
                            bank[identity] = h
                            proposed.append(h.hypothesis_id)
                            if set(condition) <= old_cues:
                                h.observe(contrast["evidence_id"], contrast["success"])
        revised = []
        for h in bank.values():
            if set(h.conditions) <= cues:
                h.observe(evidence_id, success)
                revised.append(h.hypothesis_id)
        return {"updated": revised, "proposed": proposed,
                "prediction_error": float(success)-prior["probability"],
                "evidence_id": evidence_id, "scope": list(key[0]),
                "hypotheses": [h.trace() for h in bank.values()]}

    @staticmethod
    def explanation_literals(cues: frozenset[str]) -> frozenset[str]:
        return frozenset(c for c in cues if c.startswith("pair:") and "&" not in c
                         and not c.startswith(("pair:domain:", "pair:epoch:")))

    def explanation_prediction(self, cues: frozenset[str], procedure: str) -> dict:
        """Read-only: posterior-weighted prediction of the scoped explanation."""
        key = (self.causal_scope(cues), procedure)
        bank = self.explanations.get(key)
        literals = self.explanation_literals(cues)
        if bank is None or not (bank.evidence or bank.contexts):
            return {"probability": .5, "uncertainty": 1.0, "evidence_ids": [],
                    "hypotheses": [], "explanation": None, "scope": list(key[0]),
                    "hypotheses_examined": 0}
        posterior = bank.posterior()
        q = round(sum(w for rule, w in posterior if _holds(rule, literals)), 9)
        p = (1 - bank.noise) * q + bank.noise * (1 - q)
        best = [bank.trace(rule, w) for rule, w in posterior[:3]]
        result = {"probability": p,
                  # Disagreement among competing explanations about THIS context.
                  "uncertainty": 4 * q * (1 - q),
                  "evidence_ids": sorted({eid for h in best[:1] for eid in (*h["support"], *h["contradictions"])}),
                  "hypotheses": best, "explanation": best[0]["id"],
                  "explains_context": _holds(posterior[0][0], literals),
                  "scope": list(key[0]), "hypotheses_examined": len(posterior)}
        if bank.change_detection:
            result["regime"] = len(bank.regime_changes)
            result["archived_regimes"] = len(bank.archive)
        return result

    def explanation_history(self, cues: frozenset[str], procedure: str) -> list[dict]:
        """Read-only: what superseded regimes predicted for this context.

        Historical relationships are retained as history, never relabelled
        false and never mixed into the current prediction.
        """
        bank = self.explanations.get((self.causal_scope(cues), procedure))
        literals = self.explanation_literals(cues)
        return [] if bank is None else [
            {"explanation": entry["explanation"], "holds": _holds(entry["rule"], literals),
             "ended_revision": entry["ended_revision"], "observations": entry["observations"]}
            for entry in bank.archive]

    def explanation_feedback(self, cues: frozenset[str], procedure: str, *,
                             success: bool, evidence_id: str, contrast=None,
                             allow_contrast: bool = True, persist: bool = True,
                             reviews=()) -> dict:
        """Revise the scoped explanation with exactly one observed outcome.

        persist=False is the explanation-structure ablation: the rule beam is
        rebuilt from this experience (evidence is still retained and scored).
        reviews are open questions carried by a continuing thought; this owner
        validates them and alone decides whether any rule changes.
        """
        if not isinstance(success, bool) or not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("explanation evidence needs a boolean outcome and an evidence id")
        key = (self.causal_scope(cues), procedure)
        prior = self.explanation_prediction(cues, procedure)
        bank = self.explanations.setdefault(key, ExplanationBank(**self.explanation_options))
        if not self.enabled or bank.known(evidence_id):
            return {"updated": [], "proposed": [], "prediction_error": float(success) - prior["probability"]}
        literals = self.explanation_literals(cues)
        bank.evidence.append((literals, success, evidence_id))
        del bank.evidence[:-bank.evidence_budget]
        bank.remember(literals, success, evidence_id)
        if bank.recruit != "off":
            bank.adapt_relevance(literals, success)
        if bank.directed:
            for l in literals:
                bank.vocabulary.setdefault(_axis(l), set()).add(l)
        bank.revisions += 1
        bank.since_change += 1
        if bank.change_detection:
            bank.prequential.append(bool(prior["hypotheses"]) and (prior["probability"] > .5) != success)
            del bank.prequential[:-bank.change_window]
        if not persist:
            bank.beam = {frozenset(): {"operator": "prior", "origin": None, "revision": bank.revisions}}
        pool = bank.recruited(literals)
        terms = {frozenset([l]) for l in pool} | {
            frozenset([a, b]) for a in pool for b in pool if a < b and _axis(a) != _axis(b)}
        seeded = set()
        if allow_contrast and contrast is not None and contrast["procedure"] == procedure:
            old = self.explanation_literals(frozenset(contrast["cues"]))
            # Compare on the contrast's own axes; interpreted meaning literals
            # (sequence, elapsed time) are not part of the public contrast.
            public = frozenset(l for l in literals if _axis(l) in {_axis(x) for x in old})
            if (self.causal_scope(frozenset(contrast["cues"])) == key[0]
                    and len(old ^ public) == 2 and contrast["success"] != success):
                # The changed condition, alone and within its successful context.
                flipped, context = (public - old, public) if success else (old - public, old)
                seeded = {flipped} | {flipped | {l} for l in context - flipped}
                terms |= seeded
        rules_scored = len(bank.beam)
        before = bank.posterior()[0][0]
        # Prediction error is computed against the prediction made BEFORE this
        # evidence; surprise = error x commitment (confidently wrong matters most).
        p = prior["probability"]
        urgency = round(abs(float(success) - p) * abs(2 * p - 1), 9)
        review, targets, review_status, rejected = None, [], {}, []
        if bank.directed:
            if urgency >= bank.review_threshold and prior["hypotheses"]:
                review = {"review_id": "review:" + evidence_id, "evidence_id": evidence_id,
                          "scope": list(key[0]), "procedure": procedure,
                          "literals": sorted(literals), "success": success,
                          "urgency": urgency, "opened_revision": bank.revisions}
                targets.append((literals, success))
            for item in reviews:
                valid = (isinstance(item, dict) and item.get("scope") == list(key[0])
                         and item.get("procedure") == procedure
                         and isinstance(item.get("success"), bool)
                         and item.get("evidence_id") != evidence_id
                         and bank.known(str(item.get("evidence_id"))))
                if valid:
                    targets.append((frozenset(item["literals"]), item["success"]))
                else:
                    rejected.append(item.get("review_id") if isinstance(item, dict) else None)
        candidates: dict[Rule, tuple[str, str]] = {}
        def propose(rule, operator):
            rule = frozenset(rule)
            if rule not in bank.beam and rule not in candidates and len(rule) <= bank.max_terms:
                candidates[rule] = ("contrast" if seeded & set(rule) else operator)
        directed_proposals = 0
        for round_index in range(bank.search_rounds):
            ranked_now = bank.posterior()
            rules_scored += len(bank.beam)
            edited = [rule for rule, _ in ranked_now[:bank.edited_rules]]
            for rule in edited:
                for term in terms:
                    propose(rule | {term}, "add_term")
                for term in rule:
                    propose(rule - {term}, "drop_term")
                    for l in term:
                        if len(term) > 1:
                            propose((rule - {term}) | {term - {l}}, "generalize")
                    for l in pool:
                        if len(term) < 2 and _axis(l) not in {_axis(x) for x in term}:
                            propose((rule - {term}) | {term | {l}}, "specialize")
            if round_index == 0 and targets:
                count = len(candidates)
                for context, outcome in targets:
                    suspects = [r for r in edited if _holds(r, context) != outcome][:4]
                    for rule in suspects:
                        if not outcome:  # predicted success: exclude this context
                            for term in rule:
                                if term <= context:
                                    propose(rule - {term}, "review_drop")
                                    if len(term) < 2:
                                        used = {_axis(x) for x in term}
                                        for axis in sorted(bank.vocabulary):
                                            if axis in used:
                                                continue
                                            for l in sorted(bank.vocabulary[axis] - context):
                                                propose((rule - {term}) | {term | {l}}, "review_exclude")
                        else:  # predicted failure: include this context
                            for term in rule:
                                missing = term - context
                                if len(missing) == 1 and len(term) > 1:
                                    propose((rule - {term}) | {term - missing}, "review_generalize")
                            for a in sorted(context):
                                propose(rule | {frozenset([a])}, "review_include")
                                for b in sorted(context):
                                    if a < b and _axis(a) != _axis(b):
                                        propose(rule | {frozenset([a, b])}, "review_include")
                directed_proposals = len(candidates) - count
            for rule, operator in candidates.items():
                if rule not in bank.beam:
                    bank.beam[rule] = {"operator": operator, "origin": evidence_id, "revision": bank.revisions}
            rules_scored += len(bank.beam)
            ranked = sorted(bank.beam, key=lambda r: (-bank.score(r), describe_rule(r)))
            bank.beam = {r: bank.beam[r] for r in ranked[:bank.beam_width]}
        regime = None
        if bank.change_detection:
            regime = bank.detect_regime(evidence_id)
            rules_scored += regime.get("rules_scored", 0) if regime else 0
        after = bank.posterior()
        rules_scored += len(bank.beam)
        # The explanation held BEFORE this evidence (prior), not after re-scoring:
        # the historical explanation_before/revised fields are computed after the
        # evidence is added and therefore miss re-scoring-only changes.
        held = prior.get("explanation") or "never"
        if describe_rule(after[0][0]) != held:
            # Append-only lineage; the superseded explanation is retained, not erased.
            bank.lineage_total += 1
            bank.lineage.append({"revision": bank.revisions, "evidence_id": evidence_id,
                                 "superseded": held,
                                 "successor": describe_rule(after[0][0]),
                                 "operator": bank.beam[after[0][0]]["operator"],
                                 "reason": (regime or {}).get("reason", "outscored_by_evidence")})
            del bank.lineage[:-bank.lineage_budget]
        best_now = after[0][0]
        if review is not None:
            review_status[review["review_id"]] = (
                "resolved" if _holds(best_now, literals) == success else "open")
        for item in reviews:
            if isinstance(item, dict) and item.get("review_id") not in rejected:
                review_status[item["review_id"]] = (
                    "resolved" if _holds(best_now, frozenset(item["literals"])) == item["success"]
                    else "open")
        data_rows = len(bank.contexts) if bank.durable else len(bank.evidence)
        return {"updated": [describe_rule(r) for r, _ in after[:3]],
                "proposed": [describe_rule(r) for r in candidates if r in bank.beam],
                "candidates_examined": len(candidates),
                "rules_scored": rules_scored, "scoring_work": rules_scored * data_rows,
                "explanation_before": describe_rule(before),
                "explanation": describe_rule(after[0][0]),
                "revised": after[0][0] != before,
                "explanation_held": held, "map_changed": describe_rule(after[0][0]) != held,
                "contrast_used": bool(seeded),
                "prediction_error": float(success) - prior["probability"],
                "surprise": urgency, "review": review, "review_status": review_status,
                "rejected_reviews": rejected, "directed_proposals": directed_proposals,
                "recruited": sorted(pool), "regime": regime,
                "evidence_id": evidence_id, "scope": list(key[0]),
                "hypotheses": [bank.trace(r, w) for r, w in after[:3]]}

    def incremental_prediction(self, cues: frozenset[str], procedure: str) -> dict:
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
                             success: bool, evidence_id: str, weight: float = 1.0) -> dict:
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
