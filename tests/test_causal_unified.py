"""Unified E12 behavioral proofs through the real CausalRun path.

No reward-superiority assertions: these tests prove mechanisms, isolation and
integrity. Comparative claims live in the measured benchmark reports.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
import pickle
import subprocess
import sys

import pytest

from scenario_chunking.causal_challenges import build, run_challenge
from scenario_chunking.causal_experiment import (
    CONDITIONS, TREATMENTS, CausalRun, behavior_action, curriculum, run_causal_benchmark)
from scenario_chunking.learning import ExplanationBank, OutcomeLearner, describe_rule
from scenario_chunking.meaning import MeaningInterpreter
from scenario_chunking.state import Timeline


def lits(capacity, channel, marker="blue", urgency="low"):
    return frozenset({"pair:domain:synthetic", "pair:epoch:0", "pair:capacity:" + capacity,
                      "pair:channel:" + channel, "pair:marker:" + marker,
                      "pair:urgency:" + urgency})


def feed(learner, rows, prefix, procedure="express", **kwargs):
    update = None
    for i, (cap, ch, ok) in enumerate(rows):
        update = learner.explanation_feedback(
            # Nuisance cues vary independently of the alternating causal rows.
            lits(cap, ch, "blue" if (i // 2) % 2 else "orange", "low" if (i // 4) % 2 else "high"),
            procedure, success=ok, evidence_id=f"{prefix}{i}", **kwargs)
    return update


def digest(rows):
    return hashlib.sha256(";".join(
        f'{r["action"]},{r["outcome"]},{r["prediction"]:.9f}' for r in rows).encode()).hexdigest()[:16]


@pytest.mark.parametrize("mode,expected", [
    ("explanation_full", "220e6d503c1e1cbd"), ("no_continuity", "49b686883fe1def2"),
    ("contrast_fragments", "78a044eca879c3fc"), ("episodic_only", "fc783c55ae1e3f2b")])
def test_original_e12_treatments_are_frozen_byte_for_byte(mode, expected):
    """Golden digests recorded from the c69a51b baseline before any change."""
    run = CausalRun(7, mode, exploration=.5)
    train, heldout = curriculum(7, "delivery", 0)
    rows = [run.step_case(c) for c in train] + [deepcopy(run).step_case(c, learn=False) for c in heldout]
    assert digest(rows) == expected


def test_matched_evidence_streams_are_identical_and_carry_no_private_truth():
    modes = ("strong_one_pass", "explanation_full", "no_identity", "beam_reset", "unified")
    train, _ = curriculum(23, "repair", 0)
    traces = {}
    for mode in modes:
        run = CausalRun(23, mode)
        traces[mode] = [run.step_case(c, behavior=behavior_action(23, "repair", 0, c["index"]))
                        for c in train]
    reference = [(r["action"], r["outcome"], r["outcome_event_id"], r["time"]) for r in traces[modes[0]]]
    for mode in modes[1:]:
        assert [(r["action"], r["outcome"], r["outcome_event_id"], r["time"])
                for r in traces[mode]] == reference
    for r in traces["unified"]:
        public = {l for l in r["learner_literals"]}
        assert all(l.startswith("pair:") for l in public)
        assert not any("truth" in l or "counterfactual" in l for l in public)
    # Behavior is learner-independent: no prediction can change the executed action.
    assert {behavior_action(23, "repair", 0, c["index"]) for c in train} == {"standard", "express"}


def test_unexecuted_outcomes_never_enter_learning(monkeypatch):
    from scenario_chunking.environment import IncrementalDispatchWorld
    train, _ = curriculum(7, "delivery", 0)
    a, b = CausalRun(7, "unified"), CausalRun(7, "unified")
    original = IncrementalDispatchWorld.scenario
    def flip_unexecuted(**kwargs):
        world = original(**kwargs)
        behavior = behavior_action(7, "delivery", 0, kwargs["index"] - 10000)
        other = "standard" if behavior == "express" else "express"
        world._outcomes[other] = not world._outcomes[other]
        world._truth[other] = not world._truth[other]
        return world
    for c in train:
        a.step_case(c, behavior=behavior_action(7, "delivery", 0, c["index"]))
    monkeypatch.setattr(IncrementalDispatchWorld, "scenario", flip_unexecuted)
    for c in train:
        b.step_case(c, behavior=behavior_action(7, "delivery", 0, c["index"]))
    assert pickle.dumps(a.learner) == pickle.dumps(b.learner)


def test_continuity_controls_change_only_their_own_factor():
    train, _ = curriculum(7, "delivery", 0)
    runs = {m: CausalRun(7, m, exploration=.5) for m in
            ("explanation_full", "no_identity", "no_anchor", "beam_reset", "fresh_search")}
    for run in runs.values():
        for c in train:
            run.step_case(c)
    full, no_id, no_anchor, reset = (runs[m] for m in ("explanation_full", "no_identity", "no_anchor", "beam_reset"))
    # Removing thought identity keeps the persistent explanation structure.
    assert len(no_id.refiner.threads) == 24 and len(full.refiner.threads) == 1
    assert any(v["revision"] < 20 for b in no_id.learner.explanations.values() for v in b.beam.values())
    assert not any(r["anchor_event_ids"] for r in no_id.trace)
    # Removing anchoring keeps identity and the thread's own anchor register.
    assert len(no_anchor.refiner.threads) == 1
    assert next(iter(no_anchor.refiner.threads.values())).causal_anchors
    assert not any(r["anchor_event_ids"] for r in no_anchor.trace)
    # Resetting the beam keeps identity and anchors but rebuilds the structure.
    assert len(reset.refiner.threads) == 1 and any(r["anchor_event_ids"] for r in reset.trace)
    assert all(v["revision"] == b.revisions for b in reset.learner.explanations.values()
               for v in b.beam.values())
    # Search effort is measured per experience, not assumed from a budget.
    assert all(r["rules_scored"] > 0 for run in runs.values() for r in run.trace)
    assert (sum(r["rules_scored"] for r in runs["fresh_search"].trace) >
            sum(r["rules_scored"] for r in reset.trace))


def test_durable_knowledge_survives_window_rollover_but_window_learner_forgets():
    phase1 = [("open", "day", True), ("limited", "day", False)] * 20
    phase2 = [("open", "night", False), ("limited", "night", False)] * 35   # 70 > window 64
    window, durable = OutcomeLearner(), OutcomeLearner()
    durable.explanation_options = {"durable": True}
    for learner in (window, durable):
        feed(learner, phase1, "a")
        feed(learner, phase2, "b")
    query = lits("open", "day")
    assert window.explanation_prediction(query, "express")["probability"] < .5
    assert durable.explanation_prediction(query, "express")["probability"] > .8
    bank = next(iter(durable.explanations.values()))
    assert len(bank.evidence) == bank.evidence_budget and bank.forgotten == 0
    assert describe_rule(bank.posterior()[0][0]) == "pair:capacity:open AND pair:channel:day"
    # Durable duplicate rejection outlives the active window.
    before = pickle.dumps(durable)
    durable.explanation_feedback(lits("open", "day", "orange", "high"), "express",
                                 success=True, evidence_id="a0")
    assert pickle.dumps(durable) == before


def test_contradiction_revises_only_the_responsible_explanation():
    learner = OutcomeLearner()
    learner.explanation_options = {"durable": True, "directed": True}
    feed(learner, [("open", "day", True), ("limited", "day", False)] * 6, "e")
    feed(learner, [("open", "day", True)] * 4, "s", procedure="standard")
    other_scope = frozenset(lits("open", "day") - {"pair:domain:synthetic"} | {"pair:domain:other"})
    learner.explanation_feedback(other_scope, "express", success=True, evidence_id="o1")
    untouched = {k: pickle.dumps(v) for k, v in learner.explanations.items() if k[1] == "standard"
                 or "pair:domain:other" in k[0]}
    update = learner.explanation_feedback(lits("open", "night"), "express", success=False, evidence_id="c1")
    assert update["review"] and update["surprise"] >= .3
    assert {k: pickle.dumps(v) for k, v in learner.explanations.items() if k in untouched} == untouched
    with pytest.raises(ValueError):
        learner.explanation_feedback(lits("open", "night"), "express", success="no", evidence_id="bad")
    with pytest.raises(ValueError):
        learner.explanation_feedback(lits("open", "night"), "express", success=False, evidence_id="")


def test_exception_is_refined_and_original_context_is_preserved():
    learner = OutcomeLearner()
    learner.explanation_options = {"durable": True, "change_detection": True, "directed": True}
    first = feed(learner, [("open", "day", True), ("limited", "day", False)] * 15, "a")
    assert first["explanation"] == "pair:capacity:open"
    second = feed(learner, [("open", "night", False), ("open", "day", True),
                            ("limited", "night", False)] * 10, "b")
    assert second["explanation"] == "pair:capacity:open AND pair:channel:day"
    bank = next(iter(learner.explanations.values()))
    assert not bank.regime_changes, "an exception is not a changed world"
    assert learner.explanation_prediction(lits("open", "day"), "express")["probability"] > .8
    assert learner.explanation_prediction(lits("open", "night"), "express")["probability"] < .2
    assert any(e["superseded"] == "pair:capacity:open" for e in bank.lineage)


def test_regime_change_is_inferred_archived_and_history_is_not_relabelled_false():
    learner = OutcomeLearner()
    learner.explanation_options = {"durable": True, "change_detection": True}
    feed(learner, [("open", "day", True), ("limited", "night", False),
                   ("open", "night", True), ("limited", "day", False)] * 10, "a")
    update = feed(learner, [("open", "day", False), ("limited", "night", True),
                            ("open", "night", False), ("limited", "day", True)] * 10, "b")
    bank = next(iter(learner.explanations.values()))
    assert [e["reason"] for e in bank.regime_changes] == ["regime_change"]
    assert update["explanation"] == "pair:capacity:limited"
    history = learner.explanation_history(lits("open", "day"), "express")
    assert history[-1]["explanation"] == "pair:capacity:open" and history[-1]["holds"]
    assert learner.explanation_prediction(lits("open", "day"), "express")["probability"] < .3
    # Return to the first world reactivates archived knowledge instead of relearning it.
    feed(learner, [("open", "day", True), ("limited", "night", False),
                   ("open", "night", True), ("limited", "day", False)] * 10, "c")
    assert bank.regime_changes[-1]["reason"] == "regime_reactivated"
    assert bank.regime_changes[-1]["restored"] == "pair:capacity:open"


def test_noise_alone_does_not_declare_a_regime_change():
    learner = OutcomeLearner()
    learner.explanation_options = {"durable": True, "change_detection": True}
    rows = []
    for i in range(160):
        cap = "open" if i % 2 else "limited"
        ok = (cap == "open") != (i % 11 == 0)   # ~9% label noise
        rows.append((cap, "day" if i % 3 else "night", ok))
    feed(learner, rows, "n")
    assert not next(iter(learner.explanations.values())).regime_changes


def test_review_is_opened_carried_by_identity_and_resolved_by_the_learner():
    spec = build("exception", 13)
    def run_mode(mode):
        treatment = TREATMENTS["unified_plus_directed"]
        if mode.endswith("no_identity"):
            treatment = replace(treatment, identity=False, anchoring=False)
        run = CausalRun(13, mode, treatment=treatment)
        rows = [run.step_case(c, behavior=spec["behaviors"][i]) for i, c in enumerate(spec["cases"][:90])]
        return run, rows
    run, rows = run_mode("unified_plus_directed")
    opened = [r for r in rows if r["update"].get("review")]
    assert opened, "a confident wrong prediction must open a review"
    thread = next(iter(run.refiner.threads.values()))
    carried = [r for r in rows if r["update"].get("review_status")
               and set(r["update"]["review_status"]) - {(r["update"].get("review") or {}).get("review_id")}]
    assert carried, "a continuing thought carries reviews into later experiences"
    assert any(h["status"] == "resolved" for h in thread.review_history)
    lone, lone_rows = run_mode("unified_plus_directed_no_identity")
    assert any(r["update"].get("review") for r in lone_rows)
    assert not any(set(r["update"].get("review_status") or {}) -
                   {(r["update"].get("review") or {}).get("review_id")} for r in lone_rows)
    # Foreign or malformed reviews are rejected by the owner, not applied.
    learner = OutcomeLearner()
    learner.explanation_options = {"directed": True}
    feed(learner, [("open", "day", True)] * 3, "x")
    update = learner.explanation_feedback(
        lits("open", "night"), "express", success=True, evidence_id="y",
        reviews=[{"review_id": "r1", "scope": ["pair:domain:elsewhere"], "procedure": "express",
                  "literals": [], "success": True, "evidence_id": "x0"}, "junk"])
    assert update["rejected_reviews"] == ["r1", None]


def test_clock_changes_attention_never_knowledge_and_dormant_threads_reactivate():
    train, _ = curriculum(61, "delivery", 0)
    fast, slow = CausalRun(61, "decay"), CausalRun(61, "decay")
    for i, c in enumerate(train):
        # Simulated minutes: one minute apart versus one year apart.
        fast.step_case(dict(c, elapsed=1.0), behavior=behavior_action(61, "delivery", 0, c["index"]))
        slow.step_case(dict(c, elapsed=525600.0), behavior=behavior_action(61, "delivery", 0, c["index"]))
    # Identical evidence, minutes versus years apart: identical learned knowledge.
    assert pickle.dumps(fast.learner) == pickle.dumps(slow.learner)
    assert slow.trace[-1]["attention"] == "dormant" and fast.trace[-1]["attention"] != "dormant"
    assert not any(r["anchor_event_ids"] for r in slow.trace), "stale anchors are not used"
    thread = next(iter(slow.refiner.threads.values()))
    assert thread.status_at(slow.clock.now) != "dormant", "the relevant cue reactivated it"
    # Unrelated experience never touches another investigation.
    before = (pickle.dumps(slow.learner.explanations), len(thread.versions))
    other, _ = curriculum(61, "repair", 0)
    slow.step_case(other[0], behavior="express")
    assert pickle.dumps({k: v for k, v in slow.learner.explanations.items()
                         if "pair:domain:delivery" in k[0]}) == before[0]
    assert len(thread.versions) == before[1]
    timeline = Timeline()
    timeline.advance(elapsed=5)
    with pytest.raises(ValueError):
        timeline.advance(at=1)


def test_meaning_literals_follow_chronology_and_require_a_continuing_thought():
    assert MeaningInterpreter.contextual_literals(None, 3.0) == {"pair:sequence:none"}
    seen = MeaningInterpreter.contextual_literals(
        {"procedure": "express", "outcome": "failure", "time": 1.0}, 50.0)
    assert seen == {"pair:sequence:express_failure", "pair:gap:long"}
    with pytest.raises(ValueError):
        MeaningInterpreter.contextual_literals({"procedure": "express", "outcome": "success", "time": 9.0}, 1.0)
    train, _ = curriculum(7, "delivery", 0)
    with_id = CausalRun(7, "meaning")
    without = CausalRun(7, "meaning", treatment=replace(TREATMENTS["meaning"], identity=False, anchoring=False))
    for c in train[:6]:
        b = behavior_action(7, "delivery", 0, c["index"])
        with_id.step_case(c, behavior=b)
        without.step_case(c, behavior=b)
    assert any("pair:sequence:express_success" in r["learner_literals"] or
               "pair:sequence:standard_success" in r["learner_literals"] for r in with_id.trace)
    assert all("pair:sequence:none" in r["learner_literals"] for r in without.trace)


def test_sparse_recruitment_is_bounded_deterministic_and_adapts_locally():
    frozen, adaptive = ExplanationBank(recruit="frozen"), ExplanationBank(recruit="adaptive")
    context = lits("open", "day") - {"pair:domain:synthetic", "pair:epoch:0"}
    assert len(frozen.recruited(context)) == 3
    assert frozen.recruited(context) == ExplanationBank(recruit="frozen").recruited(context)
    learner = OutcomeLearner()
    learner.explanation_options = {"recruit": "adaptive"}
    feed(learner, [("open", "day", True), ("limited", "day", False),
                   ("open", "night", True), ("limited", "night", False)] * 8, "r")
    bank = next(iter(learner.explanations.values()))
    assert bank.relevance["pair:capacity:open"] > bank.relevance["pair:channel:day"]
    assert "pair:capacity:open" in bank.recruited(context)
    update = learner.explanation_feedback(lits("open", "day", "orange", "high"), "express",
                                          success=True, evidence_id="final")
    assert len(update["recruited"]) == 3


def test_delayed_feedback_is_attributed_once_with_separate_event_and_acquisition_time():
    spec = build("delayed", 17)
    run = CausalRun(17, "unified")
    pending = [run.act(c, behavior=spec["behaviors"][i]) for i, c in enumerate(spec["cases"][:5])]
    first = run.deliver(pending[0])
    assert first["time"] == 1.0 and first["acquired_time"] == 5.0
    event = run.memory.events[first["outcome_event_id"]]
    assert event.occurrence_time == 1.0 and event.acquired_time == 5.0
    bank = next(iter(run.learner.explanations.values()))
    before = pickle.dumps(bank)
    assert run.learner.explanation_feedback(
        frozenset(first["learner_literals"]), first["action"].split(":")[1],
        success=first["outcome"] == "success", evidence_id=first["outcome_event_id"])["updated"] == []
    assert pickle.dumps(bank) == before


def test_live_runner_reaches_every_enabled_mechanism():
    report = run_causal_benchmark(seeds=(7,), families=("delivery",), protocol="matched",
                                  conditions=("explanation_full", "unified", "unified_plus_directed", "meaning",
                                              "sparse_adaptive"))
    rows = {m: [r for r in report["trials"] if r["mode"] == m and r["stage"] == "train"]
            for m in report["conditions"]}
    assert any(r["update"].get("review") for r in rows["unified_plus_directed"])
    assert any("pair:gap:short" in r["learner_literals"] for r in rows["meaning"])
    assert all(len(r["update"]["recruited"]) <= 3 for r in rows["sparse_adaptive"])
    assert all(r["update"]["recruited"] == sorted(l for l in r["learner_literals"]
               if not l.startswith(("pair:domain:", "pair:epoch:")))
               for r in rows["explanation_full"])
    assert set(report["paired_reward_deltas"]) == {"unified", "unified_plus_directed", "meaning",
                                                   "sparse_adaptive"}
    assert all("bootstrap_95_interval" in d for d in report["paired_reward_deltas"].values())
    # Regime detection is reached by the real runner on an unannounced change.
    result = run_challenge(("abrupt", 11, "unified"))
    assert result["regime_changes"] and result["historical_regime_accuracy"] is not None


def test_unified_state_restores_in_a_fresh_process_and_keeps_learning(tmp_path):
    spec = build("exception", 19)
    run = CausalRun(19, "unified")
    for i, c in enumerate(spec["cases"][:70]):
        run.step_case(c, behavior=spec["behaviors"][i])
    thread = next(iter(run.refiner.threads.values()))
    snapshot = tmp_path / "unified.pkl"
    snapshot.write_bytes(pickle.dumps(run))
    nxt = dict(spec["cases"][70])
    expected = deepcopy(run).step_case(nxt, behavior=spec["behaviors"][70])
    script = '''import pickle, sys, json
r = pickle.load(open(sys.argv[1], "rb"))
case = json.loads(sys.argv[2])
assert r.refiner.learner is r.learner and r.selector.refiner is r.refiner
before = {k: b.revisions for k, b in r.learner.explanations.items()}
row = r.step_case(case, behavior=sys.argv[3])
after = {k: b.revisions for k, b in r.learner.explanations.items()}
assert sum(after.values()) == sum(before.values()) + 1
t = next(iter(r.refiner.threads.values()))
print(json.dumps({"row": row, "thought": t.thought_id, "versions": len(t.versions)}, sort_keys=True))
'''
    output = subprocess.check_output(
        [sys.executable, "-c", script, str(snapshot), json.dumps(nxt), spec["behaviors"][70]],
        text=True, env={**os.environ, "PYTHONPATH": "src"})
    restored = json.loads(output)
    assert restored["row"] == json.loads(json.dumps(expected))
    assert restored["thought"] == thread.thought_id and restored["versions"] == len(thread.versions) + 1


def test_original_gate_conditions_are_unchanged():
    assert CONDITIONS == ("episodic_only", "strong_one_pass", "contrast_fragments",
                          "explanation_full", "no_continuity", "no_contrast")
    for mode in CONDITIONS:
        t = TREATMENTS[mode]
        assert not (t.durable or t.change or t.directed or t.meaning or t.decay)
        assert t.recruit == "off" and t.search_rounds == 1


def test_e12_working_memory_composes_with_spreading_without_a_second_selector():
    """E12 keeps the same one learner, selector, router and budget owners."""
    from scenario_chunking.models import RetrievalMode
    base = CausalRun(7, "unified")
    active = CausalRun(7, "unified_activation")
    assert base.working_memory is None
    assert base.router.working_memory is None
    assert active.working_memory is active.router.working_memory
    assert active.router.mode == base.router.mode == RetrievalMode.SPREADING
    assert active.router.top_k == base.router.top_k
    assert active.router.candidate_budget == base.router.candidate_budget
    assert active.router.spreader is not None
    assert active.router.spreader.budget == base.router.spreader.budget
    assert active.router.activation_persistence is active.learner.activation_persistence


def test_e12_working_memory_credits_only_prediction_or_used_contrast_provenance():
    """Run the real receiving path; a top-K memory is NOT meaningful use."""
    from scenario_chunking.causal_experiment import behavior_action
    run = CausalRun(23, "unified_activation", exploration=.5)
    train, _ = curriculum(23, "delivery", 0)
    first = run.step_case(train[0], behavior=behavior_action(
        23, "delivery", 0, train[0]["index"]))
    assert first["working_memory"]["meaningful_uses"] == 0
    assert not first["used_provenance"]  # fixed behavior never makes an internal choice

    rows = [run.step_case(c) for c in train[1:]]
    for row in rows:
        # Provenance must match both the actually retrieved evidence and the
        # explanation used for the selected action, never unrelated top-K hits.
        assert set(row["used_provenance"]) <= set(row["retrieved_chunks"])
        supported = set(row["predictions"][row["action"]]["evidence_ids"])
        for cid in row["used_provenance"]:
            assert supported.intersection(run.memory.chunks[cid].event_ids)
        assert row["working_memory"]["active_count"] <= run.working_memory.capacity
        assert len(row["working_memory"]["focused"]) <= run.working_memory.focused_capacity
        assert row["detail_events"] <= 16
        assert row["header_evaluations"] <= 80
    # Only actual selected predictions and causal contrasts can earn uses.
    assert run.working_memory.use_events >= sum(len(r["used_provenance"]) for r in rows)
    assert run.working_memory.trajectories
    assert run.working_memory.latest_time == run.clock.now


def test_e12_activation_matched_evidence_and_frozen_forks_do_not_leak():
    """Both learners observe the same real action outcome; heldout is read-only."""
    from scenario_chunking.causal_experiment import behavior_action
    baseline = CausalRun(61, "unified")
    active = CausalRun(61, "unified_activation")
    training, withheld = curriculum(61, "delivery", 0)
    for case in training:
        fixed = behavior_action(61, "delivery", 0, case["index"])
        left = baseline.step_case(case, behavior=fixed)
        right = active.step_case(case, behavior=fixed)
        for field in ("action", "outcome", "outcome_event_id", "time"):
            assert left[field] == right[field]
        assert right["working_memory"]["active_count"] <= 6
    before = pickle.dumps(active)
    frozen = deepcopy(active).step_case(withheld[0], learn=False)
    assert not frozen["update"]["updated"]
    assert pickle.dumps(active) == before
    assert baseline.working_memory is None
    # The learner's explanation is not a private environment outcome.
    assert not any("truth" in c or "counterfactual" in c
                   for c in frozen["learner_literals"])


def test_e12_activation_restores_live_clock_and_activation_in_fresh_process(tmp_path):
    """Cross-process restart keeps the same learner and optional WM, then acts."""
    run = CausalRun(101, "unified_activation", exploration=.5)
    train, heldout = curriculum(101, "delivery", 0)
    for case in train[:8]:
        run.step_case(case)
    path = tmp_path / "e12-wm.pkl"
    path.write_bytes(pickle.dumps((run, train[8])))
    code = (
        "import pickle,sys;"
        "p=pickle.load(open(sys.argv[1],'rb'));"
        "r,c=p;"
        "assert r.working_memory is r.router.working_memory;"
        "before=r.clock.now;"
        "row=r.step_case(c);"
        "assert r.clock.now>before;"
        "assert r.working_memory.latest_time==r.clock.now;"
        "assert row['working_memory']['active_count']<=6"
    )
    subprocess.run([sys.executable, "-c", code, str(path)], check=True)
