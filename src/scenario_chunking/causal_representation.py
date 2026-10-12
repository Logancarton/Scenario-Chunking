"""E12 evidence-versus-representation study (evaluation harness, not a learner).

Question: when the E12 explanation learner is wrong, is it missing evidence or
unable to express the relationship? Every world below is a matched-evidence
stream (a fixed uniform behavior policy acts), so all learners receive the
identical public context, executed action and outcome. Learning curves are
read at increasing evidence quantities from ONE continuing learner, using the
read-only ``CausalRun.predict`` probe, so a checkpoint never teaches.

World types separate causes that a single accuracy number conflates:

* ``simple``       representable, one literal
* ``compound``     representable at the language edge (3 terms of 2 literals)
* ``interaction``  representable only as a combination (two-variable XOR)
* ``hidden``       the cause is never observed (missing variable)
* ``noisy``        representable, but outcome noise is 0.25 instead of 0.08
* ``parity``       outside the language (three-variable parity)
* ``proxy``        hidden cause, visible marker correlated 0.85 during training
                   and independent at test (misleading correlation)
* ``change``       the rule changes half-way through the stream

Diversity is varied independently of quantity: ``broad`` draws training
contexts from all 24 non-withheld combinations, ``narrow`` from 8 of them.
Private truths stay in the scripted world and in this evaluator.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import statistics

from .causal_experiment import TREATMENTS, CausalRun, paired_interval
from .incremental_experiment import AXES, VALUES

WORLDS = ("simple", "compound", "interaction", "hidden", "noisy", "parity", "proxy", "change")
DIVERSITY = ("broad", "narrow")
CHECKPOINTS = (16, 32, 64, 128, 256, 512)
# explanation_full is the original E12 learner, unified the selected E12 learner,
# deep_search the unified learner with 3 search rounds and a 96-rule beam
# (roughly 3x scoring work): more reasoning, same explanation language.
LEARNERS = ("explanation_full", "unified", "deep_search")
DEV_SEEDS = (11, 13, 17, 19, 29)
FINAL_SEEDS = tuple(5003 + 41 * k for k in range(20))
COSTS = {"standard": .02, "express": .18}
NOISE = .08


def _unit(*parts) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def _combos() -> list[dict]:
    combos = []
    for i in range(32):
        gray = i ^ (i >> 1)
        combos.append({axis: values[(gray >> j) & 1] for j, (axis, values) in enumerate(zip(AXES, VALUES))})
    return combos


def express_rule(world: str, f: dict, *, late: bool = False) -> bool:
    """Private noiseless rule for the express procedure (standard is urgency:low)."""
    open_, day, high = f["capacity"] == "open", f["channel"] == "day", f["urgency"] == "high"
    if world in ("simple", "hidden", "noisy", "proxy"):
        return open_
    if world == "compound":
        return (open_ and day) or (high and f["person"] == "Alex") or (
            f["delay"] == "short" and not day)
    if world == "interaction":
        return open_ != day
    if world == "parity":
        return open_ ^ day ^ high
    if world == "change":
        return (not open_) if late else open_
    raise ValueError("unknown representation world")


def truths(world: str, f: dict, *, late: bool = False) -> dict[str, bool]:
    return {"standard": f["urgency"] == "low", "express": express_rule(world, f, late=late)}


def public(world: str, f: dict) -> dict:
    """What cognition may observe: hidden worlds withhold capacity entirely."""
    return {k: v for k, v in f.items() if not (world in ("hidden", "proxy") and k == "capacity")}


def split(seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(f"e12-representation-split:{seed}")
    order = list(range(32))
    rng.shuffle(order)
    heldout, train = sorted(order[:8]), sorted(order[8:])
    return train, sorted(train[i] for i in rng.sample(range(24), 8)), heldout


def stream(world: str, seed: int, diversity: str, n: int) -> list[dict]:
    combos = _combos()
    train, narrow, _ = split(seed)
    pool = train if diversity == "broad" else narrow
    cases = []
    for i in range(n):
        f = dict(combos[pool[int(_unit(seed, world, diversity, i, "combo") * len(pool))]])
        if world == "proxy":
            agree = _unit(seed, world, diversity, i, "proxy") < .85
            f["marker"] = ("blue" if f["capacity"] == "open" else "orange") if agree else (
                "orange" if f["capacity"] == "open" else "blue")
        else:
            f["marker"] = ("blue", "orange")[int(_unit(seed, world, diversity, i, "marker") * 2)]
        late = world == "change" and i >= n // 2
        cases.append({"family": "rq", "epoch": 0, "index": i, "factors": public(world, f),
                      "stage": "train", "truths": truths(world, f, late=late),
                      "noise": .25 if world == "noisy" else NOISE,
                      "behavior": "express" if _unit(seed, world, diversity, i, "act") < .5 else "standard"})
    return cases


def probes(world: str, seed: int) -> dict[str, list[dict]]:
    """Full private contexts for evaluation; markers are independent of every cause."""
    combos = _combos()
    train, _, heldout = split(seed)
    groups = {}
    for name, idx in (("heldout", heldout), ("in_distribution", train)):
        groups[name] = [dict(combos[i], marker=m) for i in idx for m in ("blue", "orange")]
    return groups


def learner(seed: int, name: str) -> CausalRun:
    if name == "deep_search":
        run = CausalRun(seed, "deep_search", treatment=replace(TREATMENTS["unified"], search_rounds=3))
        run.learner.explanation_options["beam_width"] = 96
        return run
    return CausalRun(seed, name)


def evaluate(run: CausalRun, world: str, seed: int, *, late: bool) -> dict:
    """Read-only accuracy, calibration and decision quality against the private rule."""
    out = {}
    for group, contexts in probes(world, seed).items():
        hits, brier, regret, best = [], [], [], []
        for f in contexts:
            q = run.predict(public(world, f), family="rq")
            truth = truths(world, f, late=late)
            hits.append(float((q["express"] > .5) == truth["express"]))
            p_star = {p: (1 - NOISE) if truth[p] else NOISE for p in truth}
            brier.append((q["express"] - p_star["express"]) ** 2)
            eu = {p: 2 * p_star[p] - 1 - COSTS[p] for p in truth}
            chosen = max(("express", "standard"),
                         key=lambda p: (2 * q[p] - 1 - COSTS[p], p == "standard"))
            regret.append(max(eu.values()) - eu[chosen])
            best.append(float(eu[chosen] == max(eu.values())))
        out[group] = {"express_accuracy": statistics.mean(hits), "express_brier": statistics.mean(brier),
                      "regret": statistics.mean(regret), "best_action": statistics.mean(best)}
    bank = run.learner.explanations.get((("pair:domain:rq", "pair:epoch:0"), "express"))
    out["adequacy"] = bank.adequacy() if bank is not None and hasattr(bank, "adequacy") else None
    out["explanation"] = None
    if bank is not None:
        from .learning import describe_rule
        out["explanation"] = describe_rule(bank.posterior()[0][0])
    return out


def run_job(job: tuple) -> dict:
    world, seed, diversity, name = job
    run = learner(seed, name)
    cases = stream(world, seed, diversity, max(CHECKPOINTS))
    curve = []
    for i, case in enumerate(cases):
        run.step_case({k: v for k, v in case.items() if k != "behavior"}, behavior=case["behavior"])
        if i + 1 in CHECKPOINTS:
            late = world == "change" and i + 1 > max(CHECKPOINTS) // 2
            curve.append({"n": i + 1, **evaluate(run, world, seed, late=late)})
    work = sum(r["rules_scored"] for r in run.trace)
    return {"world": world, "seed": seed, "diversity": diversity, "learner": name,
            "curve": curve, "rules_scored": work}


def _plateau(values: list[float]) -> float:
    """Improvement over the last two evidence doublings (128 -> 512)."""
    return values[-1] - values[-3]


def run_study(*, seeds=DEV_SEEDS, worlds=WORLDS, diversities=DIVERSITY, learners=LEARNERS,
              workers: int | None = None) -> dict:
    jobs = [(w, s, d, l) for w in worlds for s in seeds for d in diversities for l in learners]
    if workers == 1:
        results = [run_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 1)) as pool:
            results = list(pool.map(run_job, jobs, chunksize=2))
    summary = {}
    for w in worlds:
        for d in diversities:
            for l in learners:
                rows = [r for r in results if (r["world"], r["diversity"], r["learner"]) == (w, d, l)]
                key = f"{w}|{d}|{l}"
                curve = {}
                for k, n in enumerate(CHECKPOINTS):
                    curve[n] = {g: {m: statistics.mean(r["curve"][k][g][m] for r in rows)
                                    for m in ("express_accuracy", "express_brier", "regret", "best_action")}
                                for g in ("heldout", "in_distribution")}
                acc = [curve[n]["heldout"]["express_accuracy"] for n in CHECKPOINTS]
                gains = [r["curve"][-1]["heldout"]["express_accuracy"] -
                         r["curve"][-3]["heldout"]["express_accuracy"] for r in rows]
                diag = [r["curve"][-1]["adequacy"] for r in rows if r["curve"][-1]["adequacy"]]
                statuses = {}
                for a in diag:
                    statuses[a["status"]] = statuses.get(a["status"], 0) + 1
                summary[key] = {"curve": curve, "late_gain_128_to_512": paired_interval(
                    gains, label=f"late-gain:{key}", resamples=2000),
                    "heldout_accuracy_512": acc[-1], "rules_scored": statistics.mean(
                        r["rules_scored"] for r in rows), "self_diagnosis_512": statuses or None}
    return {"seeds": list(seeds), "checkpoints": list(CHECKPOINTS), "worlds": list(worlds),
            "learners": list(learners), "summary": summary, "runs": results}


def write_study(output="results-e12-metacognition/representation", **kwargs) -> dict:
    report = run_study(**kwargs)
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(
        {k: v for k, v in report.items() if k != "runs"}, indent=2, sort_keys=True, default=str) + "\n")
    (path / "runs.json").write_text(json.dumps(report["runs"], indent=2, sort_keys=True, default=str) + "\n")
    return report


if __name__ == "__main__":
    import sys
    seeds = FINAL_SEEDS if sys.argv[1:] == ["final"] else DEV_SEEDS
    write_study(f"results-e12-metacognition/representation-{'final' if seeds == FINAL_SEEDS else 'dev'}",
                seeds=seeds)
