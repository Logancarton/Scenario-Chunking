"""E12 prediction-to-decision diagnostic (evaluation harness only).

Why does better explanatory prediction not reliably yield better E12 decisions?
This module adds no cognitive owner. Every decision it measures is produced by
the existing live path, ``CausalRun.act(case, learn=False)`` on a deep copy of a
frozen learner: CognitiveState -> MemoryRouter -> MeaningInterpreter ->
OutcomeLearner/ExplanationBank -> ActionSelector (-> ProgressiveRefiner).

Interventions replace ONLY the prediction source read by the existing
MeaningInterpreter (``SubstitutedBeliefs``) on a throw-away copy, so the
existing value equation and selection authority decide. Oracle vectors are
built by the evaluator from the private world AFTER the frozen learner's own
decision and never enter learner, memory or thought state (digest-checked).

Decision quality is scored with the noiseless expected utility of each action,
EU(a) = 2 * P*(success | a, context) - 1 - cost(a), where P* = 1 - noise when
the private rule says the action works and noise otherwise. The historical
realized reward (one noisy draw of the chosen action) is reported alongside.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import statistics
from time import process_time

from .causal_experiment import (CausalRun, FAMILIES, FINAL_SEEDS, PRIMARY_EXPLORATION,
                                REPLICATION_SEEDS, behavior_action, curriculum, paired_interval)
from .environment import IncrementalDispatchWorld
from .incremental_experiment import COSTS
from .meaning import MeaningInterpreter

# IncrementalDispatchWorld.scenario flips each potential outcome with p = .09.
WORLD_NOISE = .09
LEARNERS = ("episodic_only", "strong_one_pass", "explanation_full", "unified")
# Hosts provide the frozen live path for interventions: explanation_full uses the
# refining selector path, strong_one_pass the direct (non-refining) path.
HOSTS = ("explanation_full", "strong_one_pass")
ORACLES = ("oracle_causal", "oracle_hindsight")
PARTIAL = ("explanation_full+oracle_express", "explanation_full+oracle_standard",
           "episodic_only+oracle_express", "episodic_only+oracle_standard")
PROCEDURES = ("standard", "express")
# Predefined |V(express) - V(standard)| ranges of the deciding source's own values.
MARGIN_BINS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, math.inf))
# Swap design: each learner is retrained on the other learner's ONLINE stream.
SWAPS = (("explanation_full", "episodic_only"), ("unified", "episodic_only"),
         ("episodic_only", "explanation_full"), ("strong_one_pass", "explanation_full"))
PAIRS = (("explanation_full", "episodic_only"), ("unified", "episodic_only"),
         ("explanation_full", "strong_one_pass"), ("unified", "explanation_full"))

# Development: E12 design + historical replication seeds (already used by E12).
DEV_SEEDS = (7, 23, 61, 101, 149) + REPLICATION_SEEDS
# Reanalysis: the unified-E12 final seeds where the discrepancy was observed.
# They are NOT independent of that observation and are reported separately.
HISTORICAL_SEEDS = FINAL_SEEDS
# Final: never used by E12, the unified final run, challenges or PR #8.
DIAGNOSTIC_FINAL_SEEDS = tuple(4001 + 37 * k for k in range(40))

PROTOCOL = {
    "frozen": "2026-10-11, before any final-seed run",
    "learners": list(LEARNERS),
    "streams": ["matched (fixed uniform behavior policy)", "online (own UCB choices, kappa 0.5)",
                "swap (learner retrained on another learner's online action stream)"],
    "decision": "CausalRun.act(case, learn=False) on a deepcopy of the frozen post-training run",
    "interventions": "SubstitutedBeliefs replaces only the interpreter's prediction source on the "
                     "explanation_full and strong_one_pass hosts",
    "primary_endpoints": [
        "matched: seed-paired mean expected reward, explanation_full - episodic_only",
        "online: seed-paired mean expected reward, explanation_full - episodic_only"],
    "seed_sets": {"dev": "E12 design + replication seeds; used to build this harness",
                  "historical": "unified-E12 final seeds 1301-2213; where the discrepancy was "
                                "observed; reanalysis, not independent",
                  "final": "40 seeds 4001 + 37k, run once after this protocol was committed"},
    "secondary": ["realized reward", "expected regret", "best-action accuracy",
                  "all-action Brier (realized and noiseless)", "advantage error",
                  "per-action and per-stakes calibration", "reliability bins",
                  "false-express / missed-express error types", "margin-bin analysis",
                  "decision-change taxonomy", "stakes and family decomposition",
                  "per-family learner profiles and per-family headroom",
                  "fixed always-standard / always-express / oracle references",
                  "oracle and partial-oracle interventions",
                  "online swap decomposition", "CPU, rules scored, learned-state bytes"],
    "statistics": "per-seed means; seed-paired percentile bootstrap (4000 resamples); "
                  "no multiplicity correction, secondary results are descriptive",
    "margin_bins": [list(b) for b in MARGIN_BINS],
    "stakes": "low when |EU(express)-EU(standard)| = 0.16 (both actions share a truth value)",
    "integrity": ["frozen-state digest unchanged by held-out evaluation", "replay equals historical "
                  "held-out procedure", "selector follows value argmax", "host invariance",
                  "outcome attributed to executed action", "deterministic rerun",
                  "independent-process restart"],
}


class SubstitutedBeliefs:
    """Evaluation-only prediction source read by the existing MeaningInterpreter.

    It is never a learner: it holds one fixed probability per procedure for one
    decision and is discarded with the throw-away copy that used it.
    """

    def __init__(self, probabilities: dict[str, float], uncertainty: float = 0.0) -> None:
        if set(probabilities) != set(PROCEDURES):
            raise ValueError("substituted beliefs need one probability per procedure")
        if not all(0.0 <= p <= 1.0 for p in probabilities.values()):
            raise ValueError("probabilities lie in [0, 1]")
        self.probabilities = dict(probabilities)
        self.uncertainty = uncertainty

    def _belief(self, cues, procedure: str) -> dict:
        return {"probability": self.probabilities[procedure], "uncertainty": self.uncertainty,
                "evidence_ids": [], "hypotheses": [], "hypotheses_examined": 0}

    explanation_prediction = causal_prediction = incremental_prediction = _belief


def action_value(p: float, procedure: str) -> float:
    """MeaningInterpreter.incremental_action_values with unit success/failure."""
    return p * 1.0 - (1.0 - p) * 1.0 - COSTS[procedure]


def expected_utility(truth: bool, procedure: str, noise: float = WORLD_NOISE) -> float:
    return action_value(1.0 - noise if truth else noise, procedure)


def state_digest(run: CausalRun) -> str:
    """Learner, thought, memory and selector state (evaluation integrity check)."""
    blob = pickle.dumps((run.learner, run.refiner, run.memory, run.selector.exploration_counts,
                         sorted(run.seen), run.clock.now))
    return hashlib.sha256(blob).hexdigest()


def decide(frozen: CausalRun, case: dict, beliefs: dict[str, float] | None = None) -> dict:
    """One frozen held-out decision through the live act() path on a copy."""
    run = deepcopy(frozen)
    if beliefs is not None:
        if run.treatment.learner == "episodic":
            raise ValueError("episodic valuation reads retrieved cases, not a belief source")
        run.interpreter = MeaningInterpreter(run.memory, SubstitutedBeliefs(beliefs))
    started = process_time()
    pending = run.act(case, learn=False)
    cpu = process_time() - started
    world, chosen = pending["world"], pending["chosen"]
    trace = pending["decision"].thought_trace if pending["decision"] else None
    return {"p": {a.target: pending["predictions"][a.label]["probability"] for a in pending["actions"]},
            "u": {a.target: pending["predictions"][a.label].get("uncertainty")
                  for a in pending["actions"]},
            "chosen": chosen.target, "world": world, "observed_reward": pending["observed"].reward,
            "refined": trace is not None,
            "refine_changed_heuristic": bool(trace and trace["meaningfully_changed_action"]),
            "cpu": cpu}


def score(decision: dict, world) -> dict:
    """Evaluation-only consequences of one decision (private world, after acting)."""
    p, chosen = decision["p"], decision["chosen"]
    truth = {a: world.counterfactual_truth(a) for a in PROCEDURES}
    realized = {a: world.counterfactual_reward(a) for a in PROCEDURES}
    eu = {a: expected_utility(truth[a], a) for a in PROCEDURES}
    v = {a: action_value(p[a], a) for a in PROCEDURES}
    argmax = min(PROCEDURES, key=lambda a: (-v[a], "route:" + a))
    best = max(PROCEDURES, key=lambda a: eu[a])
    p_true = {a: 1 - WORLD_NOISE if truth[a] else WORLD_NOISE for a in PROCEDURES}
    advantage, estimate = eu["express"] - eu["standard"], v["express"] - v["standard"]
    return {
        "chosen": chosen, "best": best, "correct": chosen == best,
        "ranking_correct": (estimate > 0) == (advantage > 0),
        "follows_values": chosen == argmax,
        "attributed": decision["observed_reward"] == realized[chosen],
        "p_standard": p["standard"], "p_express": p["express"],
        "u_standard": decision["u"]["standard"], "u_express": decision["u"]["express"],
        "truth_standard": truth["standard"], "truth_express": truth["express"],
        "stakes": "low" if truth["standard"] == truth["express"] else "high",
        "margin": abs(estimate), "advantage": advantage, "advantage_estimate": estimate,
        "advantage_error": abs(estimate - advantage),
        "expected_reward": eu[chosen], "expected_regret": eu[best] - eu[chosen],
        "realized_reward": realized[chosen],
        "realized_regret": max(realized.values()) - realized[chosen],
        **{f"brier_{a}": (p[a] - float(realized[a] > 0)) ** 2 for a in PROCEDURES},
        **{f"se_{a}": (p[a] - p_true[a]) ** 2 for a in PROCEDURES},
        "all_action_brier": statistics.mean((p[a] - float(realized[a] > 0)) ** 2 for a in PROCEDURES),
        "all_action_se": statistics.mean((p[a] - p_true[a]) ** 2 for a in PROCEDURES),
        "refined": decision["refined"], "refine_changed_heuristic": decision["refine_changed_heuristic"],
        "cpu": decision["cpu"]}


def oracle_vectors(world) -> dict[str, dict[str, float]]:
    """Evaluation-only upper bounds; built after the learners have acted."""
    return {"oracle_causal": {a: 1 - WORLD_NOISE if world.counterfactual_truth(a) else WORLD_NOISE
                              for a in PROCEDURES},
            "oracle_hindsight": {a: float(world.counterfactual_reward(a) > 0) for a in PROCEDURES}}


def _training_truth(row: dict) -> dict[str, bool]:
    """Evaluation-only noiseless rule for a training case (never shown to a learner)."""
    return IncrementalDispatchWorld.causal_rule(row["family"], row["factors"], row["epoch"])


def _stream_action(stream, seed: int, family: str, epoch: int, index: int) -> str | None:
    if stream == "matched":
        return behavior_action(seed, family, epoch, index)
    if stream == "online":
        return None
    return stream[(family, epoch, index)]


def run_stream(seed: int, stream, learners=LEARNERS, *, interventions: bool = True,
               stream_name: str | None = None, families=FAMILIES) -> dict:
    """Train each learner on one evidence stream in lockstep; freeze; evaluate.

    stream: "matched", "online", or {(family, epoch, index): procedure} replayed
    as the executed actions (another learner's online choices).
    """
    name = stream_name or stream
    runs = {m: CausalRun(seed, m, exploration=PRIMARY_EXPLORATION) for m in learners}
    training, decisions, integrity = [], [], []
    for family in families:
        for epoch in range(2 if family == "coordination" else 1):
            train, heldout = curriculum(seed, family, epoch)
            for mode, run in runs.items():
                for case in train:
                    behavior = _stream_action(stream, seed, family, epoch, case["index"])
                    started = process_time()
                    row = run.step_case(case, behavior=behavior)
                    target = row["action"].split(":", 1)[1]
                    truth = _training_truth(row)
                    training.append({
                        "seed": seed, "stream": name, "learner": mode, "family": family,
                        "epoch": epoch, "index": case["index"], "step": len(run.trace),
                        "action": target, "reward": row["reward"],
                        "prediction": row["prediction"], "brier": row["brier"],
                        "truth_standard": truth["standard"], "truth_express": truth["express"],
                        "rules_scored": row["rules_scored"], "scoring_work": row["scoring_work"],
                        "cpu": process_time() - started})
            digests = {m: state_digest(r) for m, r in runs.items()}
            for case in heldout:
                base = {"seed": seed, "stream": name, "family": family, "epoch": epoch,
                        "index": case["index"]}
                own, world = {}, None
                for mode, run in runs.items():
                    d = decide(run, case)
                    world = d["world"]
                    own[mode] = d["p"]
                    decisions.append({**base, "host": mode, "source": mode, "substituted": False,
                                      **score(d, world)})
                if not interventions:
                    continue
                sources = {**{m: own[m] for m in learners}, **oracle_vectors(world)}
                oracle = sources["oracle_causal"]
                for learner in ("explanation_full", "episodic_only"):
                    if learner in own:
                        sources[learner + "+oracle_express"] = {
                            "standard": own[learner]["standard"], "express": oracle["express"]}
                        sources[learner + "+oracle_standard"] = {
                            "standard": oracle["standard"], "express": own[learner]["express"]}
                for host in HOSTS:
                    if host not in runs:
                        continue
                    for source, vector in sources.items():
                        d = decide(runs[host], case, beliefs=vector)
                        decisions.append({**base, "host": host, "source": source,
                                          "substituted": True, **score(d, d["world"])})
            integrity.append({"seed": seed, "stream": name, "family": family, "epoch": epoch,
                              "frozen_state_unchanged": all(
                                  state_digest(r) == digests[m] for m, r in runs.items())})
    costs = {m: {"learned_state_bytes": len(pickle.dumps(r.learner)) + len(pickle.dumps(r.refiner)),
                 "memory_events": len(r.memory.events)}
             for m, r in runs.items()}
    return {"training": training, "decisions": decisions, "integrity": integrity, "costs": costs,
            "actions": {m: {(t["family"], t["epoch"], t["index"]): t["action"]
                            for t in training if t["learner"] == m} for m in learners}}


def run_seed(seed: int) -> dict:
    """Matched and online streams with interventions, then the online swaps."""
    matched = run_stream(seed, "matched")
    online = run_stream(seed, "online")
    swaps = []
    for learner, donor in SWAPS:
        replay = run_stream(seed, online["actions"][donor], (learner,), interventions=False,
                            stream_name=f"swap:{donor}")
        swaps.append(replay)
    parts = [matched, online, *swaps]
    return {"seed": seed,
            "training": [r for p in parts for r in p["training"]],
            "decisions": [r for p in parts for r in p["decisions"]],
            "integrity": [r for p in parts for r in p["integrity"]],
            "costs": {"matched": matched["costs"], "online": online["costs"]}}


# ---------------------------------------------------------------------------
# Analysis (evaluation only)
# ---------------------------------------------------------------------------

METRICS = ("expected_reward", "realized_reward", "expected_regret", "realized_regret", "correct",
           "ranking_correct", "all_action_brier", "all_action_se", "brier_standard",
           "brier_express", "se_standard", "se_express", "advantage_error", "margin")


def _mean(rows, key):
    values = [float(r[key]) for r in rows]
    return statistics.mean(values) if values else None


def _per_seed(rows, seeds, key):
    by = {s: [] for s in seeds}
    for r in rows:
        by[r["seed"]].append(float(r[key]))
    return [statistics.mean(by[s]) for s in seeds if by[s]]


def _auc(rows, procedure):
    """Mann-Whitney discrimination of the noiseless truth by the prediction."""
    pos = [r[f"p_{procedure}"] for r in rows if r[f"truth_{procedure}"]]
    neg = [r[f"p_{procedure}"] for r in rows if not r[f"truth_{procedure}"]]
    if not pos or not neg:
        return None
    ranked = sorted([(p, 1) for p in pos] + [(p, 0) for p in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        average = (i + j + 1) / 2
        rank_sum += average * sum(flag for _, flag in ranked[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _profile(rows) -> dict:
    out = {"n": len(rows), **{k: _mean(rows, k) for k in METRICS}}
    out["express_share"] = _mean([{"x": r["chosen"] == "express"} for r in rows], "x")
    out["auc_standard"], out["auc_express"] = _auc(rows, "standard"), _auc(rows, "express")
    for stakes in ("low", "high"):
        group = [r for r in rows if r["stakes"] == stakes]
        out["stakes_" + stakes] = {"n": len(group), **{k: _mean(group, k) for k in (
            "correct", "expected_regret", "all_action_brier", "all_action_se", "se_standard",
            "se_express", "advantage_error")}}
    for kind, chosen, best in (("false_express", "express", "standard"),
                               ("missed_express", "standard", "express")):
        group = [r for r in rows if r["chosen"] == chosen and r["best"] == best]
        out[kind] = {"rate": len(group) / max(1, len(rows)),
                     "regret_per_decision": sum(r["expected_regret"] for r in group) / max(1, len(rows))}
    out["reliability"] = {a: _reliability(rows, a) for a in PROCEDURES}
    out["by_family"] = {}
    for name in sorted({f"{r['family']}:{r['epoch']}" for r in rows}):
        group = [r for r in rows if f"{r['family']}:{r['epoch']}" == name]
        out["by_family"][name] = {"n": len(group), **{k: _mean(group, k) for k in (
            "expected_reward", "correct", "all_action_brier", "all_action_se", "se_standard",
            "se_express")}, "express_share": _mean(
                [{"x": r["chosen"] == "express"} for r in group], "x")}
    out["margin_bins"] = []
    for low, high in MARGIN_BINS:
        group = [r for r in rows if low <= r["margin"] < high]
        out["margin_bins"].append({"range": [low, high if math.isfinite(high) else None],
                                   "n": len(group), "share": len(group) / max(1, len(rows)),
                                   **{k: _mean(group, k) for k in (
                                       "correct", "expected_regret", "all_action_se",
                                       "advantage_error")}})
    return out


RELIABILITY_BINS = ((0.0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0001))


def _reliability(rows, procedure) -> list[dict]:
    """Predicted probability against the noiseless success probability."""
    out = []
    for low, high in RELIABILITY_BINS:
        group = [r for r in rows if low <= r[f"p_{procedure}"] < high]
        out.append({"range": [low, min(high, 1.0)], "n": len(group),
                    "mean_prediction": _mean(group, f"p_{procedure}"),
                    "mean_true_probability": None if not group else statistics.mean(
                        1 - WORLD_NOISE if r[f"truth_{procedure}"] else WORLD_NOISE for r in group)})
    return out


def _decomposition(rows_a, rows_b) -> dict:
    """Where the expected-reward difference A - B comes from (sums / n)."""
    keyed = {(r["seed"], r["family"], r["epoch"], r["index"]): r for r in rows_b}
    n = max(1, len(rows_a))
    by_stakes, by_family = {"low": 0.0, "high": 0.0}, {}
    for a in rows_a:
        b = keyed[(a["seed"], a["family"], a["epoch"], a["index"])]
        gain = a["expected_reward"] - b["expected_reward"]
        by_stakes[a["stakes"]] += gain / n
        name = f"{a['family']}:{a['epoch']}"
        by_family[name] = by_family.get(name, 0.0) + gain / n
    return {"by_stakes": by_stakes, "by_family": by_family}


def _paired(rows_a, rows_b, seeds, key, label, sign=1) -> dict:
    a, b = _per_seed(rows_a, seeds, key), _per_seed(rows_b, seeds, key)
    return paired_interval([sign * (x - y) for x, y in zip(a, b)], label=label)


def _taxonomy(rows_a, rows_b) -> dict:
    """Decision-change taxonomy of learner A relative to learner B (same decisions)."""
    keyed = {(r["seed"], r["family"], r["epoch"], r["index"]): r for r in rows_b}
    counts = {k: 0 for k in ("improve_change_beneficial", "improve_change_harmful",
                             "improve_same", "worsen_change_beneficial",
                             "worsen_change_harmful", "worsen_same", "equal_prediction")}
    contribution = dict.fromkeys(counts, 0.0)
    disagree, agree = [], []
    for a in rows_a:
        b = keyed[(a["seed"], a["family"], a["epoch"], a["index"])]
        delta = a["all_action_se"] - b["all_action_se"]
        changed = a["chosen"] != b["chosen"]
        gain = a["expected_reward"] - b["expected_reward"]
        (disagree if changed else agree).append({"d": delta, "stakes": a["stakes"]})
        if delta == 0:
            key = "equal_prediction"
        else:
            direction = "improve" if delta < 0 else "worsen"
            outcome = ("same" if not changed else
                       "change_beneficial" if gain > 0 else "change_harmful")
            key = direction + "_" + outcome
        counts[key] += 1
        contribution[key] += gain
    n = max(1, len(rows_a))
    improved = sum(counts[k] for k in ("improve_change_beneficial", "improve_change_harmful",
                                       "improve_same"))
    return {"n": len(rows_a), "counts": counts,
            "expected_reward_contribution": {k: v / n for k, v in contribution.items()},
            "share_of_improved_predictions_that_change_action":
                (counts["improve_change_beneficial"] + counts["improve_change_harmful"]) / max(1, improved),
            "decisions_disagree": len(disagree) / n,
            "se_difference_when_disagree": _mean(disagree, "d"),
            "se_difference_when_agree": _mean(agree, "d"),
            "disagreements_by_stakes": {s: sum(1 for x in disagree if x["stakes"] == s)
                                        for s in ("low", "high")}}


def analyze(results: list[dict]) -> dict:
    seeds = [r["seed"] for r in results]
    decisions = [d for r in results for d in r["decisions"]]
    training = [t for r in results for t in r["training"]]

    def rows(stream, host, source):
        return [d for d in decisions if d["stream"] == stream and d["host"] == host
                and d["source"] == source and not d["substituted"]]

    report = {"seeds": seeds, "protocol": PROTOCOL, "streams": {}}
    for stream in ("matched", "online"):
        section = {"learners": {m: _profile(rows(stream, m, m)) for m in LEARNERS}}
        contexts = rows(stream, LEARNERS[0], LEARNERS[0])
        section["fixed_policies"] = {
            "always_" + a: statistics.mean(expected_utility(r["truth_" + a], a) for r in contexts)
            for a in PROCEDURES}
        section["fixed_policies"]["oracle"] = statistics.mean(
            max(expected_utility(r["truth_" + a], a) for a in PROCEDURES) for r in contexts)
        section["fixed_policies"]["express_only_context_share"] = statistics.mean(
            float(r["best"] == "express") for r in contexts)
        section["fixed_policies"]["by_family"] = {}
        for name in sorted({f"{r['family']}:{r['epoch']}" for r in contexts}):
            group = [r for r in contexts if f"{r['family']}:{r['epoch']}" == name]
            section["fixed_policies"]["by_family"][name] = {
                "always_standard": statistics.mean(expected_utility(r["truth_standard"], "standard")
                                                   for r in group),
                "oracle": statistics.mean(max(expected_utility(r["truth_" + a], a) for a in PROCEDURES)
                                          for r in group),
                "express_only_context_share": statistics.mean(float(r["best"] == "express")
                                                              for r in group)}
        section["paired"] = {}
        for a, b in PAIRS:
            ra, rb = rows(stream, a, a), rows(stream, b, b)
            section["paired"][f"{a}-{b}"] = {
                "expected_reward": _paired(ra, rb, seeds, "expected_reward", f"{stream}-{a}-{b}-eu"),
                "realized_reward": _paired(ra, rb, seeds, "realized_reward", f"{stream}-{a}-{b}-r"),
                "correct": _paired(ra, rb, seeds, "correct", f"{stream}-{a}-{b}-c"),
                "all_action_brier": _paired(ra, rb, seeds, "all_action_brier", f"{stream}-{a}-{b}-b"),
                "all_action_se": _paired(ra, rb, seeds, "all_action_se", f"{stream}-{a}-{b}-se"),
                "advantage_error": _paired(ra, rb, seeds, "advantage_error", f"{stream}-{a}-{b}-ae"),
                "expected_reward_by_seed": dict(zip(seeds, (
                    x - y for x, y in zip(_per_seed(ra, seeds, "expected_reward"),
                                          _per_seed(rb, seeds, "expected_reward"))))),
                "decomposition": _decomposition(ra, rb),
                "taxonomy": _taxonomy(ra, rb)}
        # Measurement: per-seed paired SD of realized vs expected reward differences.
        a, b = PAIRS[0]
        ref, base = rows(stream, a, a), rows(stream, b, b)
        noise = {}
        for key in ("expected_reward", "realized_reward"):
            diffs = [x - y for x, y in zip(_per_seed(ref, seeds, key), _per_seed(base, seeds, key))]
            sd = statistics.stdev(diffs) if len(diffs) > 1 else None
            noise[key] = {"seed_sd": sd, "minimum_detectable_effect_80pct":
                          None if sd is None else 2.8 * sd / len(diffs) ** .5}
        section["measurement"] = noise
        # Interventions on frozen hosts: same decision, alternative prediction vectors.
        section["interventions"] = {}
        for host in HOSTS:
            sources = sorted({d["source"] for d in decisions if d["stream"] == stream
                              and d["host"] == host and d["substituted"]})
            host_rows = {s: [d for d in decisions if d["stream"] == stream and d["host"] == host
                             and d["source"] == s and d["substituted"]] for s in sources}
            section["interventions"][host] = {
                s: {k: _mean(r, k) for k in ("expected_reward", "realized_reward", "correct",
                                             "expected_regret", "all_action_se")}
                for s, r in host_rows.items()}
            if host == "explanation_full":
                reference = host_rows["explanation_full"]
                section["interventions"]["paired_vs_explanation_full"] = {
                    s: _paired(host_rows[s], reference, seeds, "expected_reward", f"{stream}-iv-{s}")
                    for s in sources if s != "explanation_full"}
        section["selector_audit"] = {
            "decisions": sum(1 for d in decisions if d["stream"] == stream),
            "not_following_value_argmax": sum(1 for d in decisions if d["stream"] == stream
                                              and not d["follows_values"]),
            "ranking_correct_but_wrong_choice": sum(1 for d in decisions if d["stream"] == stream
                                                    and d["ranking_correct"] != d["correct"]),
            "misattributed_outcomes": sum(1 for d in decisions if d["stream"] == stream
                                          and not d["attributed"]),
            "refined_decisions": sum(1 for d in decisions if d["stream"] == stream and d["refined"]),
            "refiner_differs_from_heuristic_score_order": sum(
                1 for d in decisions if d["stream"] == stream and d["refine_changed_heuristic"]),
            "host_invariance_violations": _host_violations(decisions, stream)}
        report["streams"][stream] = section
    report["online_trajectories"] = _trajectories(training, seeds)
    report["swaps"] = _swaps(decisions, seeds)
    report["costs"] = _costs(results, training, decisions)
    report["integrity"] = {
        "frozen_state_unchanged": all(i["frozen_state_unchanged"] for r in results for i in r["integrity"]),
        "checkpoints": sum(len(r["integrity"]) for r in results)}
    return report


def _host_violations(decisions, stream) -> int:
    chosen = {}
    for d in decisions:
        if d["stream"] == stream and d["substituted"]:
            chosen.setdefault((d["seed"], d["family"], d["epoch"], d["index"], d["source"]),
                              set()).add(d["chosen"])
    return sum(1 for v in chosen.values() if len(v) > 1)


def _trajectories(training, seeds) -> dict:
    out = {}
    for stream in ("matched", "online"):
        out[stream] = {}
        for m in LEARNERS:
            t = [r for r in training if r["stream"] == stream and r["learner"] == m]
            decisive = [r for r in t if not r["truth_standard"] and r["truth_express"]]
            out[stream][m] = {
                "train_express_share": _mean([{"x": r["action"] == "express"} for r in t], "x"),
                "train_reward": _mean(t, "reward"),
                "express_trials_in_express_only_contexts": len(
                    [r for r in decisive if r["action"] == "express"]) / max(1, len(seeds)),
                "express_trials_total": len([r for r in t if r["action"] == "express"]) / max(1, len(seeds)),
                "express_success_observed_share": _mean(
                    [{"x": r["reward"] > 0} for r in t if r["action"] == "express"], "x")}
    online = [r for r in training if r["stream"] == "online"]
    divergence = {}
    for a, b in PAIRS[:1]:
        firsts, differ = [], []
        for s in seeds:
            ta = {(r["family"], r["epoch"], r["index"]): r for r in online if r["seed"] == s and r["learner"] == a}
            tb = {(r["family"], r["epoch"], r["index"]): r for r in online if r["seed"] == s and r["learner"] == b}
            ordered = sorted(ta.values(), key=lambda r: r["step"])
            diff = [r["step"] for r in ordered if r["action"] != tb[(r["family"], r["epoch"], r["index"])]["action"]]
            differ.append(len(diff) / max(1, len(ordered)))
            firsts.append(diff[0] if diff else None)
        divergence[f"{a}-{b}"] = {"share_of_training_actions_differing": statistics.mean(differ),
                                  "first_divergent_step_by_seed": firsts}
    out["divergence"] = divergence
    return out


def _swaps(decisions, seeds) -> dict:
    """2x2: learner effect with evidence fixed, evidence effect with learner fixed."""
    def eu(stream, m):
        return _per_seed([d for d in decisions if d["stream"] == stream and d["host"] == m
                          and not d["substituted"]], seeds, "expected_reward")
    cells = {}
    for learner, donor in SWAPS:
        cells[f"{learner}@{donor}_stream"] = statistics.mean(eu(f"swap:{donor}", learner))
    for m in ("explanation_full", "episodic_only", "unified", "strong_one_pass"):
        cells[f"{m}@own_online"] = statistics.mean(eu("online", m))
    pairs = {
        "learner_effect_on_episodic_stream (explanation_full - episodic_only)":
            (eu("swap:episodic_only", "explanation_full"), eu("online", "episodic_only")),
        "learner_effect_on_explanation_stream (explanation_full - episodic_only)":
            (eu("online", "explanation_full"), eu("swap:explanation_full", "episodic_only")),
        "stream_effect_for_explanation_full (own - episodic stream)":
            (eu("online", "explanation_full"), eu("swap:episodic_only", "explanation_full")),
        "stream_effect_for_episodic_only (own - explanation stream)":
            (eu("online", "episodic_only"), eu("swap:explanation_full", "episodic_only"))}
    return {"expected_reward": cells,
            "paired": {k: paired_interval([x - y for x, y in zip(a, b)], label="swap-" + k)
                       for k, (a, b) in pairs.items()}}


def _costs(results, training, decisions) -> dict:
    out = {}
    for m in LEARNERS:
        t = [r for r in training if r["learner"] == m and r["stream"] in ("matched", "online")]
        d = [r for r in decisions if r["host"] == m and not r["substituted"]
             and r["stream"] in ("matched", "online")]
        out[m] = {"train_cpu_ms_per_experience": 1000 * _mean(t, "cpu"),
                  "heldout_cpu_ms_per_decision": 1000 * _mean(d, "cpu"),
                  "rules_scored_per_experience": _mean(t, "rules_scored"),
                  "scoring_work_per_experience": _mean(t, "scoring_work"),
                  "learned_state_bytes": statistics.mean(
                      r["costs"][s][m]["learned_state_bytes"] for r in results for s in ("matched", "online"))}
    return out


def _map(seeds, workers):
    if workers is None or workers == 1 or len(seeds) == 1:
        return [run_seed(s) for s in seeds]
    if workers < 1:
        raise ValueError("workers must be positive")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_seed, seeds))


def run_diagnostic(seeds=DEV_SEEDS, *, workers: int | None = None) -> tuple[dict, list[dict]]:
    results = _map(tuple(seeds), workers)
    return analyze(results), results


def _jsonable(value):
    if isinstance(value, dict):
        return {(k if isinstance(k, str) else "|".join(map(str, k)) if isinstance(k, tuple)
                 else str(k)): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_diagnostic(output="results-e12-diagnostic/dev", seeds=DEV_SEEDS,
                     workers: int | None = None) -> dict:
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    results = _map(tuple(seeds), workers)
    # Raw per-seed results first, so the frozen analysis can be re-derived.
    (path / "results.pkl").write_bytes(pickle.dumps(results))
    report = analyze(results)
    (path / "summary.json").write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n")
    with (path / "decisions.jsonl").open("w") as handle:
        for r in results:
            for d in r["decisions"]:
                handle.write(json.dumps({k: v for k, v in d.items() if k != "cpu"}, sort_keys=True) + "\n")
    with (path / "training.jsonl").open("w") as handle:
        for r in results:
            for t in r["training"]:
                handle.write(json.dumps({k: v for k, v in t.items() if k != "cpu"}, sort_keys=True) + "\n")
    return report


def reanalyze(output: str) -> dict:
    """Re-derive summary.json from the saved raw results with the current analysis."""
    path = Path(output)
    report = analyze(pickle.loads((path / "results.pkl").read_bytes()))
    (path / "summary.json").write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n")
    return report


SEED_SETS = {"dev": DEV_SEEDS, "historical": HISTORICAL_SEEDS, "final": DIAGNOSTIC_FINAL_SEEDS}

# ---------------------------------------------------------------------------
# POST-HOC EXPLORATORY analysis (added after the final run; not pre-registered)
# ---------------------------------------------------------------------------

def exploratory_calibration(dev_output: str, eval_output: str,
                            learner: str = "explanation_full") -> dict:
    """Would calibrated probabilities with UNCHANGED discrimination decide better?

    A per-action reliability map (RELIABILITY_BINS -> mean noiseless success
    probability) is fitted on the DEV seeds' held-out decisions only, then
    applied to the evaluation seeds. The map uses evaluation-only truth, so it
    bounds what a calibration fix could achieve; it is not a learner mechanism.
    Choices are the value argmax, which the live selector matched in every
    audited decision. Also reports seed-level coupling of prediction and reward
    differences against episodic_only.
    """
    dev = pickle.loads((Path(dev_output) / "results.pkl").read_bytes())
    final = pickle.loads((Path(eval_output) / "results.pkl").read_bytes())
    seeds = [r["seed"] for r in final]

    def reference(results, stream, mode):
        return [d for r in results for d in r["decisions"] if d["stream"] == stream
                and d["host"] == mode and not d["substituted"]]

    def bin_of(p):
        return next(i for i, (lo, hi) in enumerate(RELIABILITY_BINS) if lo <= p < hi)

    out = {"note": "post-hoc exploratory; fitted on dev seeds, applied once to evaluation seeds"}
    for stream in ("matched", "online"):
        fit = reference(dev, stream, learner)
        maps = {}
        for a in PROCEDURES:
            table = {}
            for d in fit:
                table.setdefault(bin_of(d[f"p_{a}"]), []).append(
                    1 - WORLD_NOISE if d[f"truth_{a}"] else WORLD_NOISE)
            maps[a] = {i: statistics.mean(v) for i, v in table.items()}
        rows = reference(final, stream, learner)
        recalibrated = []
        for d in rows:
            p = {a: maps[a].get(bin_of(d[f"p_{a}"]), d[f"p_{a}"]) for a in PROCEDURES}
            v = {a: action_value(p[a], a) for a in PROCEDURES}
            chosen = min(PROCEDURES, key=lambda a: (-v[a], "route:" + a))
            eu = {a: expected_utility(d["truth_" + a], a) for a in PROCEDURES}
            recalibrated.append({"seed": d["seed"], "expected_reward": eu[chosen],
                                 "express": chosen == "express",
                                 "false_express": chosen == "express" and d["best"] == "standard",
                                 "se": statistics.mean((p[a] - (1 - WORLD_NOISE if d["truth_" + a]
                                                                else WORLD_NOISE)) ** 2
                                                       for a in PROCEDURES)})
        episodic = reference(final, stream, "episodic_only")
        delta_eu = [x - y for x, y in zip(_per_seed(rows, seeds, "expected_reward"),
                                          _per_seed(episodic, seeds, "expected_reward"))]
        delta_se = [x - y for x, y in zip(_per_seed(rows, seeds, "all_action_se"),
                                          _per_seed(episodic, seeds, "all_action_se"))]
        delta_adv = [x - y for x, y in zip(_per_seed(rows, seeds, "advantage_error"),
                                           _per_seed(episodic, seeds, "advantage_error"))]
        out[stream] = {
            "reliability_map": {a: {str(RELIABILITY_BINS[i]): v for i, v in sorted(m.items())}
                                for a, m in maps.items()},
            "original_expected_reward": _mean(rows, "expected_reward"),
            "recalibrated_expected_reward": _mean(recalibrated, "expected_reward"),
            "original_express_share": _mean([{"x": d["chosen"] == "express"} for d in rows], "x"),
            "recalibrated_express_share": _mean([{"x": r["express"]} for r in recalibrated], "x"),
            "original_false_express": _mean([{"x": d["chosen"] == "express" and d["best"] == "standard"}
                                             for d in rows], "x"),
            "recalibrated_false_express": _mean([{"x": r["false_express"]} for r in recalibrated], "x"),
            "original_all_action_se": _mean(rows, "all_action_se"),
            "recalibrated_all_action_se": _mean(recalibrated, "se"),
            "recalibrated_minus_original": _paired(recalibrated, rows, seeds, "expected_reward",
                                                   f"explore-cal-{stream}"),
            "seed_correlation_with_episodic_delta": {
                "delta_se_vs_delta_reward": statistics.correlation(delta_se, delta_eu),
                "delta_advantage_error_vs_delta_reward": statistics.correlation(delta_adv, delta_eu)}}
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        name, seeds = "smoke", DEV_SEEDS[:2]
    elif sys.argv[1:] in (["dev"], ["final"]):
        name, seeds = sys.argv[1], SEED_SETS[sys.argv[1]]
    else:
        raise SystemExit("usage: python -m scenario_chunking.causal_diagnostic [dev|final]")
    summary = write_diagnostic("results-e12-diagnostic/" + name, seeds, workers=1)
    print(json.dumps(_jsonable({s: {"paired": {k: {m: v[m]["mean"] for m in ("expected_reward", "realized_reward")}
                                               for k, v in summary["streams"][s]["paired"].items()}}
                                for s in summary["streams"]}), indent=2))
