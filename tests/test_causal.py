"""E12 mechanism and anti-theater tests; no reward superiority assertions."""
from copy import deepcopy
import json
import os
import pickle
import subprocess
import sys

import pytest

from scenario_chunking.causal_experiment import CausalRun, curriculum, run_causal_benchmark, CONDITIONS
from scenario_chunking.learning import OutcomeLearner
from scenario_chunking.meaning import MeaningInterpreter
from scenario_chunking.memory import EpisodicMemory
from scenario_chunking.models import Action
from scenario_chunking.policy import ActionSelector
from scenario_chunking.refinement import ProgressiveRefiner
from scenario_chunking.state import CognitiveState
import random


def cues(value, nuisance="a", epoch="0"):
    return frozenset({"pair:domain:synthetic", "pair:epoch:"+epoch,
                      "pair:resource:"+value, "pair:nuisance:"+nuisance})


def pair(learner, n, *, success=True):
    old, new = cues("closed", str(n)), cues("open", str(n))
    learner.causal_feedback(old, "express", success=False, evidence_id=f"old{n}")
    return learner.causal_feedback(new, "express", success=success, evidence_id=f"new{n}",
        contrast={"cues": old, "procedure": "express", "success": False, "evidence_id": f"old{n}"})


def test_single_contrast_is_tentative_and_provenance_is_not_independent_support():
    learner = OutcomeLearner()
    update = pair(learner, 0)
    assert update["proposed"]
    assert all(h["status"] == "tentative" for h in update["hypotheses"])
    prediction = learner.causal_prediction(cues("open", "unseen"), "express")
    assert not prediction["hypotheses"]
    assert prediction["probability"] == .5
    before = pickle.dumps(learner)
    pair(learner, 0)
    assert before == pickle.dumps(learner), "replayed evidence must not teach twice"


def test_supported_hypothesis_predicts_unseen_context_then_contradiction_revises_it():
    learner = OutcomeLearner()
    for i in range(4):
        pair(learner, i)
    query = cues("open", "never-seen")
    before = learner.causal_prediction(query, "express")
    assert before["probability"] > .5
    hypotheses = before["hypotheses"]
    assert any(h["conditions"] == ["pair:resource:open"] for h in hypotheses)
    for i in range(5):
        learner.causal_feedback(cues("open", f"noise{i}"), "express",
                               success=False, evidence_id=f"failure{i}")
    after = learner.causal_prediction(query, "express")
    assert after["probability"] < before["probability"]
    bank = learner.causal_hypotheses[(learner.causal_scope(query), "express")]
    h = bank[(("pair:resource:open",), True)]
    assert h.trace()["status"] == "contradicted"
    assert len(h.support) == 4 and len(h.contradictions) == 5
    assert h.origin == ("old0", "new0")


@pytest.mark.parametrize("change", ["action", "scope", "many_axes", "same_outcome"])
def test_incomparable_pair_cannot_generate_explanation(change):
    learner = OutcomeLearner()
    old = cues("closed")
    contrast = {"cues": old, "procedure": "express", "success": False, "evidence_id": "old"}
    new = cues("open")
    if change == "action": contrast["procedure"] = "standard"
    if change == "scope": new = cues("open", epoch="1")
    if change == "many_axes": new = cues("open", "b")
    if change == "same_outcome": contrast["success"] = True
    result = learner.causal_feedback(new, "express", success=True, evidence_id="new", contrast=contrast)
    assert not result["proposed"]


def test_restart_cross_episode_identity_and_evidence_revision(tmp_path):
    run = CausalRun(7, "contrast_fragments")
    train, _ = curriculum(7, "delivery", 0)
    rows = [run.step_case(c) for c in train[:12]]
    assert len(run.refiner.threads) == 1
    assert len({r["thought"]["thought_id"] for r in rows}) == 1
    assert all(r["revision"]["thought_id"] == rows[0]["thought"]["thought_id"] for r in rows)
    thread = next(iter(run.refiner.threads.values()))
    assert len({o["episode_id"] for o in thread.outcomes}) == 12
    assert any(v.hypotheses for v in thread.versions)
    snapshot = tmp_path / "state.pkl"
    snapshot.write_bytes(pickle.dumps(run))
    expected = deepcopy(run).step_case(train[12])
    script = '''import pickle,sys,json
from scenario_chunking.causal_experiment import curriculum
r=pickle.load(open(sys.argv[1],'rb'))
assert r.mode == 'contrast_fragments'
assert r.refiner.learner is r.learner and r.selector.refiner is r.refiner
assert r.interpreter.memory is r.memory
row=r.step_case(curriculum(7,'delivery',0)[0][12])
assert len(r.refiner.threads)==1
print(json.dumps(row,sort_keys=True))
'''
    output = subprocess.check_output([sys.executable, "-c", script, str(snapshot)], text=True,
                                    env={**os.environ, "PYTHONPATH": "src"})
    assert json.loads(output) == json.loads(json.dumps(expected))


def test_continuity_and_contrast_ablations_are_separate():
    runs = {mode:CausalRun(7, mode) for mode in ("explanation_full", "no_continuity", "no_contrast")}
    train, _ = curriculum(7, "delivery", 0)
    for run in runs.values():
        for case in train: run.step_case(case)
    assert len(runs["explanation_full"].refiner.threads) == 1
    assert len(runs["no_continuity"].refiner.threads) == 24
    assert not any(r["anchor_event_ids"] for r in runs["no_continuity"].trace)
    assert any(r["anchor_event_ids"] for r in runs["explanation_full"].trace)
    assert any(r["update"].get("contrast_used") for r in runs["explanation_full"].trace)
    assert not any(r["update"].get("contrast_used") for r in runs["no_contrast"].trace)
    assert len(runs["no_contrast"].refiner.threads) == 1
    # Ablations are evidence-equal: every learner keeps its observed evidence.
    for run in runs.values():
        assert sum(len(b.evidence) for b in run.learner.explanations.values()) == 24


def test_marker_epoch_and_holdout_are_independent_by_design():
    for epoch in (0, 1):
        train, test = curriculum(23, "coordination", epoch)
        assert {c["index"] for c in train}.isdisjoint(c["index"] for c in test)
        assert len(train) == 24 and len(test) == 8
        for group in (train, test):
            assert sum(c["factors"]["marker"] == "blue" for c in group) == len(group)//2
        assert {c["factors"]["channel"] for c in train} == {"day", "night"}
    a = curriculum(23, "coordination", 0)
    b = curriculum(23, "coordination", 1)
    assert [c["index"] for c in a[1]] == [c["index"] for c in b[1]]


def test_frozen_unseen_evaluation_does_not_teach_or_reactivate_training_state():
    run = CausalRun(23, "explanation_full")
    train, test = curriculum(23, "repair", 0)
    for case in train: run.step_case(case)
    snapshot = pickle.dumps(run)
    for case in test:
        fork = deepcopy(run)
        learned = pickle.dumps(fork.learner)
        row = fork.step_case(case, learn=False)
        assert row["update"] == {"updated": []} and row["diagnosis"] is None
        assert pickle.dumps(fork.learner) == learned
        assert row["outcome_event_id"] not in row["predictions"][row["action"]]["evidence_ids"]
    assert pickle.dumps(run) == snapshot


def test_scopes_do_not_inherit_obsolete_hypotheses_and_prediction_is_read_only():
    learner = OutcomeLearner()
    for i in range(3): pair(learner, i)
    before = pickle.dumps(learner)
    p = learner.causal_prediction(cues("open", epoch="1"), "express")
    assert p["probability"] == .5 and not p["evidence_ids"]
    assert pickle.dumps(learner) == before


def test_learned_explanation_changes_actual_selector_choice_on_unseen_context():
    learner, memory = OutcomeLearner(), EpisodicMemory()
    interpreter = MeaningInterpreter(memory, learner)
    refiner = ProgressiveRefiner(learner)
    selector = ActionSelector(memory, learner, random.Random(7), refiner=refiner)
    state = CognitiveState("unseen", goal_id="response_dispatch", thought_key="test")
    actions = [Action("route", target="standard"), Action("route", target="express")]
    def decide():
        interpreted = interpreter.incremental_action_values(cues("open", "unseen"), actions,
                           costs={"standard": .02, "express": .18}, causal=True)
        return selector.select(state, actions, [], refine=True,
            transfer_scores=interpreted["values"], transfer_context={"cognitive_mode": "incremental",
            "candidate_values": interpreted["values"], "predictions": interpreted["predictions"]})
    before = decide()
    for i in range(4): pair(learner, i)
    after = decide()
    assert before.action.target == "standard" and after.action.target == "express"
    assert .5 < after.predicted_success < 1


def test_bounded_banks_routing_and_complete_measured_comparisons():
    report = run_causal_benchmark(seeds=(7,), families=("delivery",))
    assert len(report["trials"]) == 32 * len(CONDITIONS)
    assert all(v["n"] == 8 for v in report["summary"].values())
    assert all(r["header_evaluations"] <= 80 and r["detail_events"] <= 16 for r in report["trials"])
    # Two actions, each with at most one 48-rule explanation beam.
    assert all(r["hypotheses_examined"] <= 96 for r in report["trials"])
    assert all(r["selection_prediction"] == r["prediction"] for r in report["trials"])
    assert all(r["regret"] >= 0 for r in report["trials"])
    assert all(len(r["update"].get("hypotheses", [])) <= 32 for r in report["trials"])
    assert set(report["paired_reward_deltas"]) == set(CONDITIONS)-{"explanation_full"}
    assert all(r["exploration"] == 0.5 for r in report["trials"])


def test_unobserved_potential_outcomes_cannot_change_frozen_prediction(monkeypatch):
    from scenario_chunking.environment import IncrementalDispatchWorld
    run = CausalRun(7, "explanation_full")
    train, test = curriculum(7, "delivery", 0)
    for case in train: run.step_case(case)
    baseline = deepcopy(run).step_case(test[0], learn=False)
    original = IncrementalDispatchWorld.scenario
    def reversed_world(**kwargs):
        world = original(**kwargs)
        world._outcomes = {k:not v for k,v in world._outcomes.items()}
        return world
    monkeypatch.setattr(IncrementalDispatchWorld, "scenario", reversed_world)
    changed = deepcopy(run).step_case(test[0], learn=False)
    assert changed["predictions"] == baseline["predictions"]
    assert changed["action"] == baseline["action"]
    assert changed["outcome"] != baseline["outcome"]


def test_retained_evidence_is_bounded_but_contradiction_totals_survive():
    learner = OutcomeLearner()
    for i in range(40): pair(learner, i)
    bank = learner.causal_hypotheses[(learner.causal_scope(cues("open")), "express")]
    assert len(bank) <= 32
    h = bank[(("pair:resource:open",), True)]
    assert len(h.support) == 12 and h.support_count == 40
    for i in range(50):
        learner.causal_feedback(cues("open", str(i)), "express", success=False, evidence_id=f"noise:{i}")
    assert h.contradiction_count == 50 and len(h.contradictions) == 12
    assert h.trace()["status"] == "contradicted"


def explanation_cues(capacity, channel, marker):
    return frozenset({"pair:domain:synthetic", "pair:epoch:0", "pair:capacity:" + capacity,
                      "pair:channel:" + channel, "pair:marker:" + marker})


def teach(learner, rows, *, persist=True, prefix="e"):
    for i, (cap, ch, marker, ok) in enumerate(rows):
        update = learner.explanation_feedback(explanation_cues(cap, ch, marker), "express",
                                              success=ok, evidence_id=f"{prefix}{i}", persist=persist)
    return update


def test_explanation_is_formed_revised_in_place_and_predicts_an_unseen_context():
    learner = OutcomeLearner()
    # Phase 1: capacity alone seems to matter (channel is always day).
    first = teach(learner, [("open", "day", "blue", True), ("limited", "day", "orange", False)] * 3)
    assert first["explanation"] == "pair:capacity:open"
    # Phase 2: a DIFFERENT experience — open capacity fails at night.
    second = teach(learner, [("open", "night", "blue", False), ("open", "day", "orange", True)] * 3,
                   prefix="n")
    assert second["explanation"] != first["explanation"], "the same explanation was revised"
    assert second["explanation"] == "pair:capacity:open AND pair:channel:day"
    # Never-seen combination: limited at night, opposite marker.
    unseen = learner.explanation_prediction(explanation_cues("limited", "night", "orange"), "express")
    assert unseen["probability"] < .2
    seen = learner.explanation_prediction(explanation_cues("open", "day", "blue"), "express")
    assert seen["probability"] > .8
    trace = seen["hypotheses"][0]
    assert trace["support_count"] == 12 and trace["contradiction_count"] == 0
    assert set(trace["support"]) <= {f"{p}{i}" for p in "en" for i in range(6)}


def test_explanation_prediction_is_read_only_and_replay_does_not_teach_twice():
    learner = OutcomeLearner()
    teach(learner, [("open", "day", "blue", True), ("limited", "night", "orange", False)] * 2)
    before = pickle.dumps(learner)
    learner.explanation_prediction(explanation_cues("open", "night", "blue"), "express")
    assert pickle.dumps(learner) == before
    teach(learner, [("open", "day", "blue", True), ("limited", "night", "orange", False)] * 2)
    assert pickle.dumps(learner) == before, "replayed evidence ids must not teach twice"


def test_explanation_beam_and_evidence_are_bounded():
    learner = OutcomeLearner()
    rows = [("open" if i % 3 else "limited", "day" if i % 2 else "night", "blue", i % 5 != 0)
            for i in range(120)]
    teach(learner, rows)
    bank = next(iter(learner.explanations.values()))
    assert len(bank.beam) <= bank.beam_width and len(bank.evidence) == bank.evidence_budget
    assert all(len(rule) <= bank.max_terms for rule in bank.beam)


def test_continuity_ablation_keeps_evidence_but_not_the_explanation():
    kept, reset = OutcomeLearner(), OutcomeLearner()
    rows = [("open", "day", "blue", True), ("limited", "day", "orange", False),
            ("open", "night", "orange", False), ("limited", "night", "blue", False)] * 3
    teach(kept, rows)
    teach(reset, rows, persist=False)
    assert ([e[:2] for e in next(iter(kept.explanations.values())).evidence] ==
            [e[:2] for e in next(iter(reset.explanations.values())).evidence])
    revisions = [v for v in next(iter(reset.explanations.values())).beam.values()]
    assert {v["revision"] for v in revisions} == {12}, "ablated beam holds only this experience's edits"


def test_exploration_is_selector_owned_training_only_and_counted_per_scope():
    run = CausalRun(7, "episodic_only", exploration=.5)
    train, test = curriculum(7, "delivery", 0)
    for case in train: run.step_case(case)
    counts = run.selector.exploration_counts
    assert sum(counts.values()) == 24 and len({scope for scope, _ in counts}) == 1
    snapshot = dict(counts)
    fork = deepcopy(run)
    fork.step_case(test[0], learn=False)
    assert fork.selector.exploration_counts == snapshot, "held-out choices are greedy"
    greedy = CausalRun(7, "episodic_only")
    for case in train: greedy.step_case(case)
    assert not greedy.selector.exploration_counts
