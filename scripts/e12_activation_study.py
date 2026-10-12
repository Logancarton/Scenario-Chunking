"""E12 dense / fixed-sparse / learned-sparse controlled computation study.

The common base is learned from identical public examples and inspections
BEFORE copying the learner. All arms have the same candidate routes, public
features, safety rules and hard ceiling of three hidden-variable axes.
Fixed and learned sparse use a *matched* two-axis computation budget; dense
uses the three-axis ceiling as an accuracy/reference upper-work control.

Exploratory seeds must not be presented as a confirmatory result. The separate
40-seed confirmatory cohort is fixed before any confirmatory run.
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

MODES = ("dense", "fixed_sparse", "learned_sparse")
DEV = (31, 37)
FINAL = tuple(900001 + 113 * k for k in range(40))
SOURCES = ("delivery", "repair")
TARGETS = ("relay", "clinic", "coordination")
ACCURACY_MARGIN = .02
MIN_QUERY_SAVING = .20


def measure(run, cases, *, learn: bool) -> dict:
    """Count actual explanation contexts (including hindsight and outcome audit).

    Counts every call through the ONE learner rather than trusting a synthetic
    operation counter. CPU and peak allocations are also measured directly.
    """
    contexts = 0
    original = run.learner.explanation_beliefs

    def counted(queries, procedure):
        nonlocal contexts
        contexts += len(queries)
        return original(queries, procedure)

    run.learner.explanation_beliefs = counted
    rows = []
    active_counts = []
    gc.collect()
    tracemalloc.start()
    before = process_time()
    try:
        for case in cases:
            pending = run.act(case, learn=learn)
            active_counts.append(len(pending["active_axes"]))
            rows.append(run.deliver(pending))
        cpu = process_time() - before
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        del run.learner.explanation_beliefs
    def avg(name):
        return statistics.mean(r[name] for r in rows) if rows else 0.0
    return {
        "n": len(rows), "expected_reward": avg("expected_reward"),
        "deferred": avg("deferred"),
        "violations": sum(r["violations"] for r in rows),
        "relevant_inspections": sum(r["relevant_inspections"] for r in rows),
        "irrelevant_inspections": sum(r["irrelevant_inspections"] for r in rows),
        "belief_contexts_evaluated": contexts,
        "decision_belief_queries": sum(r["belief_queries"] for r in rows),
        "mean_active_axes": statistics.mean(active_counts) if active_counts else 0.0,
        "cpu_seconds": cpu, "peak_traced_bytes": peak,
        "learning_updates": sum(bool(r["process_updated"]) for r in rows),
        "actions": [r["action"] for r in rows],
    }


def study_seed(seed: int, *, pretrain: int, adapt: int, heldout: int) -> dict:
    baseline = cm.InvestigativeRun(
        seed, "metacognitive", exploration=0.0,
        policy=cm.Policy("all", "learned", diagnosis=True))
    for case in cm.training_stream(seed, SOURCES, pretrain):
        baseline.step_case(case)
    common = pickle.dumps(baseline)
    output = {}
    for mode in MODES:
        sample = {}
        for task in (SOURCES[0], *TARGETS):
            run = pickle.loads(common)
            run.policy = replace(run.policy, inspect="voi", activation=mode, activation_limit=2)
            run.exploration = 0.0
            if task in TARGETS:
                experience = cm.training_stream(seed, (task,), adapt, start=10000)
                sample["adapt_" + task] = measure(run, experience, learn=True)
            evaluation = cm.heldout_cases(seed, task)[:heldout]
            sample["heldout_" + task] = measure(run, evaluation, learn=False)
        output[mode] = sample
    return output


def summarize(results: dict) -> dict:
    summary = {}
    for mode in MODES:
        conditions = next(iter(results.values()))[mode]
        for task in conditions:
            series = {k: statistics.mean(v[mode][task][k] for v in results.values())
                      for k in ("expected_reward", "violations", "belief_contexts_evaluated",
                                "decision_belief_queries", "cpu_seconds", "peak_traced_bytes",
                                "deferred", "relevant_inspections", "irrelevant_inspections",
                                "mean_active_axes", "learning_updates")}
            summary[mode + ":" + task] = series
    comparison = {}
    for task in next(iter(results.values()))[MODES[0]]:
        dense = summary["dense:" + task]
        for mode in MODES[1:]:
            sparse = summary[mode + ":" + task]
            comparison[mode + ":" + task] = {
                "delta_expected_reward": sparse["expected_reward"] - dense["expected_reward"],
                "fraction_fewer_belief_contexts": (
                    1.0 - sparse["belief_contexts_evaluated"] / dense["belief_contexts_evaluated"]
                    if dense["belief_contexts_evaluated"] else None),
                "fraction_less_cpu": (
                    1.0 - sparse["cpu_seconds"] / dense["cpu_seconds"]
                    if dense["cpu_seconds"] else None),
                "paired_reward_deltas": [
                    round(v[mode][task]["expected_reward"] -
                          v["dense"][task]["expected_reward"], 8)
                    for v in results.values()],
            }
            if len(results) >= 5:
                comparison[mode + ":" + task]["reward_paired_interval"] = cm.paired_interval(
                    comparison[mode + ":" + task]["paired_reward_deltas"],
                    label="e12-activation:" + mode + ":" + task)
    # Predeclared cross-family transfer: each seed contributes one averaged
    # heldout reward difference, preventing cherry-picking one favorable family.
    transfer_differences = [statistics.mean(
        row["learned_sparse"]["heldout_" + target]["expected_reward"] -
        row["fixed_sparse"]["heldout_" + target]["expected_reward"]
        for target in TARGETS) for row in results.values()]
    transfer = {"paired_deltas": transfer_differences,
                "paired_interval": (cm.paired_interval(
                    transfer_differences, label="e12-sparse-transfer")
                    if len(results) >= 5 else None)}
    return {"summary": summary, "comparisons": comparison, "transfer": transfer}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument("--seeds", default="31,37", help="development seeds (ignored with --confirmatory)")
    parser.add_argument("--pretrain", type=int, default=16)
    parser.add_argument("--adapt", type=int, default=12)
    parser.add_argument("--heldout", type=int, default=18)
    parser.add_argument("--output", default="results-e12-sparsity.json")
    args = parser.parse_args()
    if min(args.pretrain, args.adapt, args.heldout) < 1:
        parser.error("each phase requires at least one experience")
    seeds = FINAL if args.confirmatory else tuple(int(s) for s in args.seeds.split(","))
    results = {str(seed): study_seed(seed, pretrain=args.pretrain, adapt=args.adapt,
                                     heldout=args.heldout) for seed in seeds}
    report = {"study": "e12_computation_sparsity",
              "status": "confirmatory" if args.confirmatory else "exploratory",
              "seeds": seeds, "pretrain": args.pretrain, "adapt": args.adapt,
              "heldout": args.heldout,
              "source_families": SOURCES, "transfer_targets": TARGETS,
              "matched_evidence": "shared pretraining state, identical public streams; on-policy adaptation differs",
              "budget": {"dense_max_axes": 3, "sparse_max_axes": 2,
                         "same_input_information": True,
                         "sparse_budget_matched_between": ["fixed_sparse", "learned_sparse"]},
              "success_criteria": {
                  "minimum_query_saving": MIN_QUERY_SAVING,
                  "expected_reward_noninferiority_margin": ACCURACY_MARGIN,
                  "safety_violations": 0,
                  "positive_transfer_vs_fixed": "paired interval must exclude zero",
                  "overlooked_variable_recovery": "independent targeted scenario tests"},
              **summarize(results), "by_seed": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps({"status": report["status"], "seed_count": len(seeds),
                      "comparisons": report["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
