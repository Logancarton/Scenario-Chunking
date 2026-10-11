"""E12 controlled challenge streams (evaluation harness, not a learner).

Every stream is a matched-evidence protocol: a fixed, learner-independent
behavior policy chooses the executed action, so every compared treatment
receives the identical public context, executed action, observed outcome,
evidence order and chronology. Private truths stay inside each scripted world
and in the probe evaluator; they never reach a cognitive owner.

The streams separate questions the original benchmark cannot: a contextual
exception versus an unannounced regime change (abrupt, gradual, reversal),
a no-change noise control, sequence dependence, representable versus
unrepresentable structure, randomly drawn unseen structures, delayed feedback,
salient distractors and long temporal gaps.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import statistics

from .causal_experiment import CausalRun, FINAL_SEEDS, paired_interval
from .incremental_experiment import AXES, VALUES
from .learning import describe_rule

DEV_SEEDS = (11, 13, 17, 19, 29)
CHALLENGE_CONDITIONS = (
    "strong_one_pass", "explanation_full", "beam_reset", "no_identity", "no_anchor",
    "durable", "regime", "directed", "meaning", "decay", "sparse_frozen", "sparse_adaptive",
    "unified", "unified_no_identity", "unified_no_anchor", "unified_plus_directed",
    "unified_plus_sparse", "unified_plus_meaning")


def _unit(*parts) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def _combos() -> list[dict]:
    combos = []
    for i in range(32):
        gray = i ^ (i >> 1)
        combos.append({axis: values[(gray >> j) & 1] for j, (axis, values) in enumerate(zip(AXES, VALUES))})
    return combos


def _draw(seed, name, i, *, extra=()) -> dict:
    factors = {axis: values[int(_unit(seed, name, i, axis) * 2)] for axis, values in zip(AXES, VALUES)}
    factors["marker"] = ("blue", "orange")[int(_unit(seed, name, i, "marker") * 2)]
    for axis, values in extra:
        factors[axis] = values[int(_unit(seed, name, i, axis) * 2)]
    return factors


def _behavior(seed, name, i) -> str:
    return "express" if _unit(seed, name, i, "behavior") < .5 else "standard"


# Private rule fragments (evaluation side only).
def _open(f): return f["capacity"] == "open"
def _limited(f): return f["capacity"] == "limited"
def _std(f): return f["urgency"] == "low"


def build(name: str, seed: int) -> dict:
    """Return the stream, its private truth schedule and its probe groups."""
    stream, truth_at, groups, changes, delay, families = [], None, {}, [], 0, ("challenge",)
    probes = _combos()
    if name == "exception":
        n = 120
        for i in range(n):
            f = _draw(seed, name, i)
            if i < 60:
                f["channel"] = "day"   # the exception condition never appears early
            stream.append(f)
        truth_at = lambda i, f: {"standard": _std(f),
            "express": _open(f) and (i < 60 or f["channel"] == "day")}
        groups = {"original_context": lambda f: f["channel"] == "day",
                  "exception_context": lambda f: f["channel"] == "night"}
        changes = [60]
    elif name == "delayed":
        spec = build("exception", seed)
        spec.update(name="delayed", delay=6)
        return spec
    elif name in ("abrupt", "gradual", "no_change"):
        n = 180 if name != "gradual" else 140
        stream = [_draw(seed, name, i) for i in range(n)]
        def regime(i, f):
            if name == "no_change":
                return "A"
            if name == "abrupt":
                return "B" if 60 <= i < 120 else "A"
            weight = min(1.0, max(0.0, (i - 60) / 40))
            return "B" if _unit(seed, name, i, "mix") < weight else "A"
        def truth(i, f):
            r = regime(i, f)
            return {"standard": _std(f), "express": _open(f) if r == "A" else _limited(f)}
        # Probe truth uses the regime governing the CURRENT period (majority).
        def probe_truth(i, f):
            if name == "gradual":
                current = "B" if i >= 80 else "A"
            else:
                current = regime(i, f)
            return {"standard": _std(f), "express": _open(f) if current == "A" else _limited(f)}
        truth_at = truth
        groups = {"all": lambda f: True}
        changes = {"abrupt": [60, 120], "gradual": [60], "no_change": []}[name]
        return _finish(name, seed, stream, truth, probe_truth, groups, changes, delay, families, probes)
    elif name == "sequence":
        n = 140
        stream = [_draw(seed, name, i) for i in range(n)]
        behaviors = [_behavior(seed, name, i) for i in range(n)]
        # Express fails right after an express dispatch (fatigue); order matters.
        truth_at = lambda i, f: {"standard": _std(f),
            "express": _open(f) and not (i > 0 and behaviors[i - 1] == "express")}
        groups = {"all": lambda f: True}
        # A probe after step i asks about the NEXT experience (after behaviors[i]).
        return _finish(name, seed, stream, truth_at, lambda i, f: truth_at(i + 1, f),
                       groups, changes, delay, families, probes)
    elif name in ("representable", "parity"):
        n = 140
        stream = [_draw(seed, name, i) for i in range(n)]
        if name == "representable":
            express = lambda f: (_open(f) and f["channel"] == "day") or (
                f["urgency"] == "high" and f["person"] == "Alex")
        else:
            express = lambda f: (_open(f) ^ (f["channel"] == "day") ^ (f["urgency"] == "high"))
        truth_at = lambda i, f: {"standard": _std(f), "express": express(f)}
        groups = {"all": lambda f: True}
    elif name == "random_structure":
        rng = random.Random(f"e12-structure:{seed}")
        def random_dnf():
            terms = []
            for _ in range(2):
                axes = rng.sample(range(5), rng.choice((1, 2)))
                terms.append([(AXES[a], VALUES[a][rng.randrange(2)]) for a in axes])
            return terms
        rules = {"standard": random_dnf(), "express": random_dnf()}
        holds = lambda dnf, f: any(all(f[a] == v for a, v in term) for term in dnf)
        combos = _combos()
        heldout = set(rng.sample(range(32), 8))
        train = [c for k, c in enumerate(combos) if k not in heldout]
        n = 100
        for i in range(n):
            f = dict(train[int(_unit(seed, name, i, "combo") * len(train))])
            f["marker"] = ("blue", "orange")[int(_unit(seed, name, i, "marker") * 2)]
            stream.append(f)
        truth_at = lambda i, f: {p: holds(rules[p], f) for p in rules}
        held = [combos[k] for k in sorted(heldout)]
        groups = {"heldout_combinations": lambda f, held=held: any(
            all(f[a] == h[a] for a in AXES) for h in held)}
        return _finish(name, seed, stream, truth_at, truth_at, groups, [], 0, families, probes,
                       structure={p: [[f"{a}:{v}" for a, v in t] for t in d] for p, d in rules.items()})
    elif name == "distractor":
        n = 120
        extra = (("alarm", ("on", "off")),)
        stream = [_draw(seed, name, i, extra=extra) for i in range(n)]
        truth_at = lambda i, f: {"standard": _std(f), "express": _open(f) and f["channel"] == "day"}
        groups = {"all": lambda f: True}
        probes = [dict(c, alarm=a) for c in _combos() for a in ("on", "off")]
    elif name == "long_gap":
        # Two investigations alternate in blocks; between blocks, years pass.
        families = ("alpha", "beta")
        n = 160
        for i in range(n):
            f = _draw(seed, name, i)
            f["_family"] = families[(i // 10) % 2]
            f["_elapsed"] = 50000.0 if i % 10 == 0 and i else 1.0
            stream.append(f)
        truth_at = lambda i, f: {"standard": _std(f),
            "express": _open(f) if f.get("_family", "alpha") == "alpha" else f["channel"] == "day"}
        groups = {"all": lambda f: True}
    else:
        raise ValueError("unknown E12 challenge")
    return _finish(name, seed, stream, truth_at, truth_at, groups, changes, delay, families, probes)


def _finish(name, seed, stream, truth, probe_truth, groups, changes, delay, families, probes,
            structure=None) -> dict:
    cases = []
    for i, f in enumerate(stream):
        truths = truth(i, f)  # private; computed before harness-only tags are removed
        family = f.pop("_family", families[0])
        elapsed = f.pop("_elapsed", 1.0)
        cases.append({"family": family, "epoch": 0, "index": i, "factors": f,
                      "stage": "train", "elapsed": elapsed, "truths": truths})
    marker = lambda k: ("blue", "orange")[int(_unit(seed, name, "probe", k) * 2)]
    return {"name": name, "seed": seed, "cases": cases,
            "behaviors": [_behavior(seed, name, i) for i in range(len(cases))],
            "probe_truth": probe_truth, "groups": groups, "changes": changes,
            "delay": delay, "families": families, "structure": structure,
            "probes": [dict(p, marker=p.get("marker", marker(k))) for k, p in enumerate(probes)]}


def _probe(run: CausalRun, spec: dict, step: int) -> dict:
    result = {"step": step}
    for family in spec["families"]:
        scores = {g: [] for g in spec["groups"]}
        brier = []
        for factors in spec["probes"]:
            q = run.predict(factors, family=family)
            truth = spec["probe_truth"](step, dict(factors, _family=family))
            for p, value in q.items():
                if p != "express" and spec["name"] not in ("random_structure",):
                    continue  # standard's rule never changes; reported separately
                hit = float((value > .5) == truth[p])
                brier.append((value - float(truth[p])) ** 2)
                for g, member in spec["groups"].items():
                    if member(factors):
                        scores[g].append(hit)
        suffix = "" if len(spec["families"]) == 1 else ":" + family
        for g, values in scores.items():
            if values:
                result[g + suffix] = statistics.mean(values)
        result["brier" + suffix] = statistics.mean(brier)
        unrelated = [float((run.predict(f, family=family)["standard"] > .5) ==
                           spec["probe_truth"](step, dict(f, _family=family))["standard"])
                     for f in spec["probes"][:32]]
        result["unrelated_standard" + suffix] = statistics.mean(unrelated)
    return result


def _banks(run: CausalRun):
    return list(run.learner.explanations.values())


def run_challenge(job: tuple) -> dict:
    name, seed, mode = job
    spec = build(name, seed)
    run = CausalRun(seed, mode)
    rows, curve, pending = [], [], deque()
    dense = {c + k for c in spec["changes"] for k in range(0, 40)}
    snapshots = {}
    for i, case in enumerate(spec["cases"]):
        pending.append(run.act(case, behavior=spec["behaviors"][i]))
        if len(pending) > spec["delay"]:
            rows.append(run.deliver(pending.popleft()))
        if (i + 1) % 5 == 0 or i in dense or i == len(spec["cases"]) - 1:
            curve.append(_probe(run, spec, i))
        if i + 1 in (60, 120):
            snapshots[i + 1] = deepcopy(run.learner)
    while pending:
        rows.append(run.deliver(pending.popleft()))
    result = {"challenge": name, "seed": seed, "mode": mode, "curve": curve,
              "final": _probe(run, spec, len(spec["cases"]) - 1),
              "prequential_truth_agreement": statistics.mean(r["truth_agreement"] for r in rows[-60:]),
              "prequential_brier": statistics.mean(r["brier"] for r in rows[-60:]),
              "rules_scored": sum(r["rules_scored"] for r in rows),
              "scoring_work": sum(r["scoring_work"] for r in rows),
              "header_evaluations": sum(r["header_evaluations"] for r in rows),
              "detail_events": sum(r["detail_events"] for r in rows),
              "anchored_retrievals": sum(bool(r["anchor_event_ids"]) for r in rows),
              "dormant_arrivals": sum(r["attention"] == "dormant" for r in rows),
              "learned_state_bytes": len(pickle.dumps(run.learner)) + len(pickle.dumps(run.refiner)),
              "structure": spec["structure"]}
    if run.treatment.learner == "explanation":
        banks = _banks(run)
        result["explanation_changes"] = sum(b.lineage_total for b in banks)
        result["regime_changes"] = [e for b in banks for e in b.regime_changes]
        result["archived"] = [e["explanation"] for b in banks for e in b.archive]
        result["explanations"] = {
            "|".join(map(str, k)): describe_rule(b.posterior()[0][0])
            for k, b in run.learner.explanations.items()}
        result["reviews_opened"] = sum(1 for r in rows if (r["update"] or {}).get("review"))
        result["directed_proposals"] = sum((r["update"] or {}).get("directed_proposals", 0) for r in rows)
        result["unsupported_literals"] = sum(
            1 for b in banks for term in b.posterior()[0][0] for l in term
            if l.split(":", 2)[1] in ("marker", "alarm", "sequence", "gap"))
        # Historical retention: does an archived regime still predict the old truth?
        if name == "abrupt" and 120 in snapshots:
            learner = snapshots[120]
            hits = []
            for f in spec["probes"]:
                primitive = frozenset({"pair:domain:challenge", "pair:epoch:0"} |
                                      {f"pair:{a}:{v}" for a, v in f.items()})
                history = learner.explanation_history(primitive, "express")
                if history:
                    hits.append(float(history[-1]["holds"] == _open(f)))
            result["historical_regime_accuracy"] = statistics.mean(hits) if hits else None
    return result


def _metrics(results: list[dict], name: str) -> dict:
    """Challenge-specific scalar outcomes per (seed, mode) result."""
    out = {}
    for r in results:
        m = {"final_brier": r["final"].get("brier", r["final"].get("brier:alpha")),
             "prequential_truth_agreement": r["prequential_truth_agreement"],
             "prequential_brier": r["prequential_brier"],
             "rules_scored": r["rules_scored"], "scoring_work": r["scoring_work"],
             "learned_state_bytes": r["learned_state_bytes"]}
        final = r["final"]
        for k, v in final.items():
            if k not in ("step",) and not k.startswith("brier"):
                m["final_" + k] = v
        if "explanation_changes" in r:
            m["explanation_changes"] = r["explanation_changes"]
            m["regime_flags"] = len(r["regime_changes"])
            m["unsupported_literals"] = r["unsupported_literals"]
            m["reviews_opened"] = r["reviews_opened"]
        if name in ("abrupt", "gradual", "exception", "delayed"):
            changes = {"abrupt": [60, 120], "gradual": [60], "exception": [60], "delayed": [60]}[name]
            key = "exception_context" if name in ("exception", "delayed") else "all"
            for c in changes:
                after = [p for p in r["curve"] if p["step"] >= c]
                hit = next((p["step"] - c + 1 for p in after if p.get(key, 0) >= .8), None)
                m[f"adapt_steps_{c}"] = hit if hit is not None else 999
                window = [p.get(key) for p in r["curve"] if c + 40 <= p["step"] < c + 60]
                m[f"post_change_accuracy_{c}"] = statistics.mean(window) if window else None
            if name in ("exception", "delayed"):
                m["retention_original_context"] = final.get("original_context")
                early = [p.get("original_context") for p in r["curve"] if 55 <= p["step"] < 60]
                m["original_context_before_exception"] = statistics.mean(early) if early else None
        if name == "abrupt":
            m["historical_regime_accuracy"] = r.get("historical_regime_accuracy")
        if name == "long_gap":
            m["dormant_arrivals"] = r["dormant_arrivals"]
            m["anchored_retrievals"] = r["anchored_retrievals"]
            m["header_evaluations"] = r["header_evaluations"]
        out[(r["seed"], r["mode"])] = m
    return out


def run_challenges(*, names=("exception", "delayed", "abrupt", "gradual", "no_change", "sequence",
                             "representable", "parity", "random_structure", "distractor", "long_gap"),
                   seeds=DEV_SEEDS, conditions=CHALLENGE_CONDITIONS, reference="explanation_full",
                   workers: int | None = None) -> dict:
    jobs = [(n, s, m) for n in names for s in seeds for m in conditions]
    if workers == 1:
        results = [run_challenge(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 1)) as pool:
            results = list(pool.map(run_challenge, jobs, chunksize=2))
    report = {"seeds": list(seeds), "conditions": list(conditions), "reference": reference,
              "challenges": {}}
    for name in names:
        subset = [r for r in results if r["challenge"] == name]
        metrics = _metrics(subset, name)
        keys = sorted({k for v in metrics.values() for k in v})
        table = {}
        for mode in conditions:
            table[mode] = {}
            for k in keys:
                values = [metrics[(s, mode)].get(k) for s in seeds]
                values = [v for v in values if v is not None]
                table[mode][k] = statistics.mean(values) if values else None
        paired = {}
        for mode in conditions:
            if mode == reference:
                continue
            paired[mode] = {}
            for k in keys:
                a = [metrics[(s, mode)].get(k) for s in seeds]
                b = [metrics[(s, reference)].get(k) for s in seeds]
                if all(x is not None for x in a + b):
                    paired[mode][k] = paired_interval([x - y for x, y in zip(a, b)],
                                                      label=f"{name}:{mode}:{k}", resamples=2000)
        report["challenges"][name] = {"summary": table, "paired_vs_reference": paired,
                                      "runs": subset}
    return report


def write_challenges(output="results-e12-unified/challenges", **kwargs) -> dict:
    report = run_challenges(**kwargs)
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(
        {"seeds": report["seeds"], "conditions": report["conditions"], "reference": report["reference"],
         "challenges": {n: {k: v for k, v in c.items() if k != "runs"}
                        for n, c in report["challenges"].items()}},
        indent=2, sort_keys=True, default=str) + "\n")
    (path / "runs.json").write_text(json.dumps(
        {n: c["runs"] for n, c in report["challenges"].items()}, indent=2, sort_keys=True, default=str) + "\n")
    return report


if __name__ == "__main__":
    write_challenges(seeds=FINAL_SEEDS)
