"""E12 pinned-main equivalence probe: full per-decision behavioral and state digests.

Run this EXACT script in separate Python processes against original main and the
candidate src directories. Work counters are compared as performance, not
behavior. No copy of the historical selector or learner is implemented.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pickle
from time import process_time
import tracemalloc

from scenario_chunking import causal_metacognition as cm


def canonical(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, (tuple, list)):
        return [canonical(x) for x in value]
    if isinstance(value, (frozenset, set)):
        return sorted((canonical(x) for x in value), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(value, dict):
        pairs = [[canonical(k), canonical(v)] for k, v in value.items()]
        return sorted(pairs, key=lambda p: json.dumps(p[0], sort_keys=True))
    if dataclasses.is_dataclass(value) or hasattr(value, "__dict__"):
        return {"type": type(value).__name__, "attrs": canonical(vars(value))}
    if callable(value):
        return getattr(value, "__qualname__", repr(value))
    raise TypeError("unexpected uncanonicalizable state type: " + type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def case_batch(seed):
    cases = cm.training_stream(seed, ("delivery", "repair", "restricted", "fleet"), 2)
    restricted = cm.training_stream(seed, ("restricted",), 1, start=50000)[0]
    restricted["prohibited"] = frozenset(("standard", "express"))
    cases.append(restricted)
    option = cm.training_stream(seed, ("fleet",), 1, start=50001)[0]
    option["horizon"] = 3
    cases.append(option)
    activity = cm.training_stream(seed, ("delivery",), 1, start=50002)[0]
    activity["activity"] = {
        "requirements": ["recipient_confirmed"],
        "steps": [{"content": "Mid-route access revoked",
                   "prohibited": ["standard", "express"]}],
        "verification": [{"requirement": "recipient_confirmed",
                          "satisfied": False, "source": "independent_receiver", "final": True}],
    }
    cases.append(activity)
    return cases


def observe(seed, mode):
    # Pin exact production-default policies. No new sparse policy is used here.
    policy = cm.Policy("voi", "learned", diagnosis=True, option=(mode == "option"))
    if mode == "all":
        policy = cm.Policy("all", "learned", diagnosis=True)
    run = cm.InvestigativeRun(seed, "metacognitive", policy=policy, exploration=.5)
    cases = case_batch(seed)
    rows = []
    for case in cases:
        pending = run.act(case)
        if pending.get("activity"):
            while pending["observed"] is None:
                run.advance_activity(pending)
        row = run.deliver(pending)
        if pending.get("activity"):
            evidence = []
            while True:
                e = run.verify_activity(pending)
                if e is None:
                    break
                evidence.append(e)
            row["verification_evidence"] = evidence
            row["goal_status_final"] = pending["state"].goal_status
            row["goal_verified"] = pending["state"].goal_complete
        # Performance counters must be measured, but cannot be equivalence keys.
        measured = {k: v for k, v in row.items()
                    if k not in ("decision_cpu", "learn_cpu", "belief_queries")}
        analyses = [{k: v for k, v in a.items() if k != "belief_queries"}
                    for a in pending["analyses"]]
        rows.append({
            "row": canonical(measured),
            "analyses": canonical(analyses),
            "chosen": pending["chosen"].label,
            "goal_status": pending["state"].goal_status,
            "goal_evidence": canonical(pending["state"].goal_evidence),
            "state": digest(pending["state"]),
            "learner": digest(vars(run.learner)),
            "memory": digest(vars(run.memory)),
            "thought": digest(run.refiner.threads),
            "selector_counts": digest(run.selector.exploration_counts),
            "selector_rng": digest(run.selector.rng.getstate()),
            "policy_rng": digest(run.policy_rng.getstate()),
            "clock": canonical((run.clock.now, run.clock.experiences)),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    tracemalloc.start()
    start = process_time()
    rows = {f"{seed}:{mode}": observe(seed, mode)
            for seed in (31, 43) for mode in ("normal", "all", "option")}
    cpu = process_time() - start
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    with open(args.output, "w") as f:
        json.dump({"rows": rows, "metrics": {"cpu_seconds": cpu,
                     "peak_traced_bytes": peak}}, f, sort_keys=True)


if __name__ == "__main__":
    main()
