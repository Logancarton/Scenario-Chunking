"""Predeclared independent E12 adaptive comparison; never regenerates historical E12.

Freeze this protocol before running the confirmatory cohort. Only the
original E12 cognitive owners are used; private truth enters metrics only
after live action execution, never a selector, recovery or process learner.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import gc
import json
import pickle
import statistics
from time import process_time
import tracemalloc

from scenario_chunking import causal_metacognition as cm
from e12_activation_study import SOURCES, TARGETS

DEVELOPMENT_SEEDS = (43, 47)
FRESH_CONFIRMATORY_SEEDS = tuple(2300001 + 137 * i for i in range(12))
MODES = ("dense", "fixed_sparse", "learned_sparse", "adaptive_sparse",
         "adaptive_no_recovery", "adaptive_no_process")
TASKS = (SOURCES[0], *TARGETS)
PRETRAIN = 16
ADAPT = 12
HELDOUT = 18
# Predeclared reporting, NOT guaranteed target achievement:
REWARD_NONINFERIORITY = -0.02
MAX_ACTIVE_AXES = 2
MAX_EXTRA_REASONING_PASSES = 1


def summary(run, cases, *, learn):
    contexts = 0
    original = run.learner.explanation_beliefs
    def counted(queries, procedure):
        nonlocal contexts
        contexts += len(queries)
        return original(queries, procedure)
    run.learner.explanation_beliefs = counted
    rows, active = [], []
    gc.collect()
    tracemalloc.start()
    t = process_time()
    try:
        for case in cases:
            pending = run.act(case, learn=learn)
            active.append(len(pending["active_axes"]))
            rows.append(run.deliver(pending))
        cpu, peak = process_time() - t, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        del run.learner.explanation_beliefs
    def avg(key):
        return statistics.mean(r[key] for r in rows) if rows else 0.0
    return {
        "n": len(rows), "mean_reward": avg("expected_reward"),
        "mean_regret": avg("regret"), "deferral": avg("deferred"),
        "violations": sum(r["violations"] for r in rows),
        "decision_belief_queries": sum(r["belief_queries"] for r in rows),
        "all_belief_contexts": contexts, "cpu_seconds": cpu,
        "peak_traced_bytes": peak,
        "mean_active_axes": statistics.mean(active) if active else 0.0,
        "recovery_attempts": sum(bool(r.get("recovery")) for r in rows),
        "recovery_accepted": sum(bool((r.get("recovery") or {}).get("accepted")) for r in rows),
        "recovery_extra_queries": sum(r.get("recovery_extra_queries", 0) for r in rows),
        "inspections": sum(len(r["inspections"]) for r in rows),
        "actions": [r["action"] for r in rows],
    }


def evaluate_seed(seed, *, pretrain, adapt, heldout):
    baseline = cm.InvestigativeRun(seed, "metacognitive", exploration=0.0,
                                   policy=cm.Policy("all", "learned", diagnosis=True))
    for case in cm.training_stream(seed, SOURCES, pretrain):
        baseline.step_case(case)
    matched = pickle.dumps(baseline)
    result = {}
    for mode in MODES:
        per_task = {}
        for task in TASKS:
            run = pickle.loads(matched)
            activation = mode if mode in ("dense", "fixed_sparse", "learned_sparse") else "adaptive_sparse"
            run.policy = replace(run.policy, inspect="voi", activation=activation,
                                 activation_limit=MAX_ACTIVE_AXES,
                                 recovery_enabled=mode != "adaptive_no_recovery",
                                 process_learning=mode != "adaptive_no_process")
            run.exploration = 0.0
            if task in TARGETS:
                per_task["adapt_" + task] = summary(
                    run, cm.training_stream(seed, (task,), adapt, start=10000), learn=True)
            per_task["heldout_" + task] = summary(
                run, cm.heldout_cases(seed, task)[:heldout], learn=False)
        result[mode] = per_task
    return result


def aggregate(results):
    rows = {}
    for mode in MODES:
        for task in TASKS:
            name = "heldout_" + task
            sample = [v[mode][name] for v in results.values()]
            rows[mode + ":" + task] = {field: statistics.mean(x[field] for x in sample)
                                        for field in ("mean_reward", "mean_regret", "deferral",
                                            "violations", "decision_belief_queries",
                                            "all_belief_contexts", "cpu_seconds",
                                            "peak_traced_bytes", "mean_active_axes",
                                            "recovery_attempts", "recovery_accepted",
                                            "recovery_extra_queries", "inspections")}
    diffs = {}
    for control in MODES:
        if control == "adaptive_sparse":
            continue
        d = [statistics.mean(
            data["adaptive_sparse"]["heldout_" + task]["mean_reward"] -
            data[control]["heldout_" + task]["mean_reward"]
            for task in TARGETS) for data in results.values()]
        diffs[control] = {
            "paired_reward_differences": d,
            "seed_paired_interval": (cm.paired_interval(d, label="adaptive:" + control)
                                     if len(d) >= 5 else None)}
    return rows, diffs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirmatory", action="store_true")
    p.add_argument("--seeds", default="43,47")
    p.add_argument("--pretrain", type=int, default=PRETRAIN)
    p.add_argument("--adapt", type=int, default=ADAPT)
    p.add_argument("--heldout", type=int, default=HELDOUT)
    p.add_argument("--output", default="results-e12-adaptive.json")
    a = p.parse_args()
    if min(a.pretrain, a.adapt, a.heldout) < 1:
        p.error("all cohorts must have at least one case")
    seeds = FRESH_CONFIRMATORY_SEEDS if a.confirmatory else tuple(int(x) for x in a.seeds.split(","))
    result = {str(seed): evaluate_seed(seed, pretrain=a.pretrain, adapt=a.adapt,
                                      heldout=a.heldout) for seed in seeds}
    rows, diffs = aggregate(result)
    report = {
        "study": "e12_adaptive_reasoning",
        "status": "confirmatory" if a.confirmatory else "exploratory",
        "seeds": seeds, "pretrain": a.pretrain, "adapt": a.adapt, "heldout": a.heldout,
        "modes": MODES, "tasks": TASKS,
        "shared_pretraining": True, "on_policy_adaptation_diverges": True,
        "historical_40_seed_json_reused": False,
        "success_criteria": {"reward_noninferiority_margin": REWARD_NONINFERIORITY,
                             "safety_violations": 0,
                             "cross_family_positive_vs_fixed": "seed-paired interval excludes zero",
                             "cost_reported": "all contexts plus CPU and peak tracemalloc"},
        "limits": {"initial_active_axes": MAX_ACTIVE_AXES,
                   "extra_recovery_passes": MAX_EXTRA_REASONING_PASSES},
        "summary": rows, "paired_comparisons": diffs, "by_seed": result}
    with open(a.output, "w", encoding="utf-8") as h:
        json.dump(report, h, indent=2, sort_keys=True)
    print(json.dumps({"status": report["status"], "seeds": len(seeds),
                      "paired_comparisons": diffs, "safety_violations": sum(
                          v["violations"] for v in rows.values())}, indent=2))


if __name__ == "__main__":
    main()
