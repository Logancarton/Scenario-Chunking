"""Bounded, read-only E12 sparse disagreement replay using the saved study result.

The input JSON is REQUIRED. Never regenerate the 40-seed cohort implicitly.
Counterfactuals clone the original live InvestigativeRun before a decision and
change only its activation policy. Private task truth is used by evaluation
AFTER decisions, never by cognition. This script does not create a selector,
learner, or a new cognitive owner.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import pickle
from pathlib import Path

from scenario_chunking import causal_metacognition as cm
from e12_activation_study import MODES, SOURCES, TARGETS

TASKS = (SOURCES[0], *TARGETS)


def disagreements(report: dict) -> list[dict]:
    """Identify action disagreements without retraining or replaying any seed."""
    if report.get("study") != "e12_computation_sparsity" or not report.get("by_seed"):
        raise ValueError("expected an existing E12 sparsity study JSON with by_seed")
    n = report.get("heldout")
    if not isinstance(n, int) or n < 1:
        raise ValueError("invalid heldout count")
    rows = []
    for seed_string, arms in sorted(report["by_seed"].items(), key=lambda item: int(item[0])):
        seed = int(seed_string)
        for task in TASKS:
            key = "heldout_" + task
            original = {mode: arms[mode][key]["actions"] for mode in MODES}
            if any(len(a) != n for a in original.values()):
                raise ValueError(f"missing or truncated actions for seed={seed} task={task}")
            cases = cm.heldout_cases(seed, task)[:n]
            if len(cases) != n:
                raise ValueError("saved heldout count exceeds available cases")
            for i, case in enumerate(cases):
                fixed, learned = (original[m][i] for m in ("fixed_sparse", "learned_sparse"))
                if fixed == learned:
                    continue
                oracle = cm.oracles(case)
                if fixed not in oracle["allowed"] or learned not in oracle["allowed"]:
                    raise ValueError("saved action violates the public permissions")
                rows.append({
                    "seed": seed, "task": task, "index": i,
                    "case_index": case["index"],
                    "saved_actions": {mode: original[mode][i] for mode in MODES},
                    # This is NOT net return: saved JSON lacks per-case inspection cost.
                    "route_value_delta_before_inspection_cost":
                        round(oracle["values"][learned] - oracle["values"][fixed], 9),
                    "reference_best": oracle["best"],
                    "both_routes_prohibited": len(case["prohibited"]) == 2,
                })
    return sorted(rows, key=lambda row: (
        -abs(row["route_value_delta_before_inspection_cost"]),
        row["seed"], row["task"], row["index"]))


def _trace(pending: dict, row: dict) -> dict:
    initial = pending["analyses"][0]
    final = pending["analyses"][-1]
    return {
        "action": row["action"],
        "active_axes": list(pending["active_axes"]),
        "omitted_axes": sorted(set(pending["constraints"]["hidden"]) -
                               set(pending["active_axes"])),
        "initial_values": initial["values"],
        "initial_voi": initial["voi"],
        "initial_predictions": {route: {"probability": value["probability"],
                                       "uncertainty": value["uncertainty"]}
                                for route, value in initial["predictions"].items()
                                if route.startswith("route:")},
        "last_values": final["values"],
        "inspection_axes": row["inspections"],
        "stage_a": row["stage_a"],
        "greedy_after_information": pending["greedy"][-1],
        "expected_reward_net_of_inspection": row["expected_reward"],
        "deferred": bool(row["deferred"]),
        "belief_queries": row["belief_queries"],
        "process_updates": row["process_updated"],
    }


def replay(report: dict, selected: list[dict]) -> list[dict]:
    """Replay only selected seed/task prefixes through the actual E12 owners.

    The two observed arms are reproduced from the exact shared pretraining
    snapshot. Counterfactual dense/fixed activation is tested from the SAME
    learned-arm state immediately before the disputed case. These distinct
    comparisons must not be conflated.
    """
    by_seed: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for item in selected:
        by_seed[item["seed"]][item["task"]].add(item["index"])
    output = []
    for seed, tasks in sorted(by_seed.items()):
        baseline = cm.InvestigativeRun(
            seed, "metacognitive", exploration=0.0,
            policy=cm.Policy("all", "learned", diagnosis=True))
        for case in cm.training_stream(seed, SOURCES, report["pretrain"]):
            baseline.step_case(case)
        common = pickle.dumps(baseline)
        for task, positions in sorted(tasks.items()):
            arms = {}
            for mode in ("fixed_sparse", "learned_sparse"):
                run = pickle.loads(common)
                run.policy = replace(run.policy, inspect="voi",
                                     activation=mode, activation_limit=2)
                run.exploration = 0.0
                if task in TARGETS:
                    for case in cm.training_stream(
                            seed, (task,), report["adapt"], start=10000):
                        run.step_case(case)
                arms[mode] = run
            cases = cm.heldout_cases(seed, task)[:report["heldout"]]
            for index in range(max(positions) + 1):
                case = cases[index]
                traces = {}
                counterfactual = {}
                for mode, run in arms.items():
                    # The counterfactual changes activation only, with the
                    # same memory, process state, world case and RNG state.
                    if mode == "learned_sparse" and index in positions:
                        frozen = pickle.dumps(run)
                        for alternative in ("fixed_sparse", "dense"):
                            control = pickle.loads(frozen)
                            control.policy = replace(control.policy, activation=alternative)
                            alternative_pending = control.act(case, learn=False)
                            alternative_row = control.deliver(alternative_pending)
                            counterfactual[alternative] = _trace(
                                alternative_pending, alternative_row)
                    pending = run.act(case, learn=False)
                    actual = run.deliver(pending)
                    saved = report["by_seed"][str(seed)][mode][
                        "heldout_" + task]["actions"][index]
                    if actual["action"] != saved:
                        raise ValueError(
                            f"replay disagreement: seed={seed} task={task} case={index}"
                            f" mode={mode} saved={saved} observed={actual['action']}")
                    if index in positions:
                        traces[mode] = _trace(pending, actual)
                if index in positions:
                    learned = traces["learned_sparse"]
                    output.append({
                        "seed": seed, "task": task, "index": index,
                        "live": traces,
                        "same_learned_state_activation_counterfactuals": counterfactual,
                        "fixed_sparse_changes_action_from_learned_state":
                            counterfactual["fixed_sparse"]["action"] != learned["action"],
                        "dense_changes_action_from_learned_state":
                            counterfactual["dense"]["action"] != learned["action"],
                        "fixed_sparse_net_gain_from_learned_state": round(
                            counterfactual["fixed_sparse"]["expected_reward_net_of_inspection"] -
                            learned["expected_reward_net_of_inspection"], 9),
                        "dense_net_gain_from_learned_state": round(
                            counterfactual["dense"]["expected_reward_net_of_inspection"] -
                            learned["expected_reward_net_of_inspection"], 9),
                    })
    return output


def audit(report: dict, *, max_seeds: int = 1, max_cases: int = 6) -> dict:
    if max_seeds < 1 or max_cases < 1:
        raise ValueError("replay bounds must be positive")
    candidates = disagreements(report)
    chosen_seeds = list(dict.fromkeys(row["seed"] for row in candidates))[:max_seeds]
    chosen = [row for row in candidates if row["seed"] in chosen_seeds][:max_cases]
    return {
        "input_status": report["status"],
        "input_seed_count": len(report["by_seed"]),
        "total_action_disagreements": len(candidates),
        "selected_seed_count": len(set(item["seed"] for item in chosen)),
        "selected_case_count": len(chosen),
        "comparison_note": ("Fixed versus learned across training trajectories is "
            "observational. Only same-learned-state activation counterfactuals "
            "isolate attention; dense uses more computation. Evaluation uses "
            "private truth only after decisions. Do not infer general transfer."),
        "selected": chosen,
        "replays": replay(report, chosen),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="existing E12 result JSON; required")
    p.add_argument("--max-seeds", type=int, default=1)
    p.add_argument("--max-cases", type=int, default=6)
    p.add_argument("--output", type=Path, default=Path("results-e12-disagreement-audit.json"))
    args = p.parse_args()
    with args.input.open(encoding="utf-8") as f:
        report = json.load(f)
    result = audit(report, max_seeds=args.max_seeds, max_cases=args.max_cases)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps({k: result[k] for k in (
        "input_status", "input_seed_count", "total_action_disagreements",
        "selected_seed_count", "selected_case_count")}, indent=2))


if __name__ == "__main__":
    main()
