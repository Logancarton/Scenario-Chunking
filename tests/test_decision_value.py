"""E10: behavioral, provenance, control, and process-continuity tests."""
import base64
import pickle
import subprocess
import sys

import pytest

from scenario_chunking.decision_value_experiment import (
    DecisionValueRun, heldout_temporal_world, run_decision_value_benchmark,
    train_temporal_evidence,
)
from scenario_chunking.environment import TemporalDecisionWorld, TemporalDelivery
from scenario_chunking.learning import OutcomeLearner
from scenario_chunking.state import CognitiveState


def test_censored_observation_does_not_create_a_missing_arrival():
    learner = OutcomeLearner()
    state = CognitiveState("t")
    ex = state.schedule_temporal(
        expectation_id="t:1", subject="Ada", context="slow",
        source_event_id="source", due_at=10, confidence=.8)
    learner.learn_temporal_observation(ex, observed_at=6, returned=False)
    assert learner.temporal_samples["slow"]["t:1"] == (None, 6, 10)
    assert learner.temporal_punctuality["slow"] == [0, 0]
    assert learner.temporal_delay["slow"] == [0, 0]
    before = learner.temporal_survival("slow", 10, 9)
    learner.learn_temporal_observation(ex, observed_at=12, returned=False)
    assert learner.temporal_samples["slow"]["t:1"][1] == 12
    assert learner.temporal_survival("slow", 10, 9) >= before
    learner.learn_temporal_observation(ex, observed_at=14,
                                       returned=True, arrival_at=13)
    assert learner.temporal_samples["slow"]["t:1"] == (13, None, 10)
    assert len(learner.temporal_samples["slow"]) == 1


def test_source_accuracy_is_dated_and_contextual():
    learner = OutcomeLearner()
    learner.verify_temporal_source("Rey", "routine", observed_at=3, correct=True)
    learner.verify_temporal_source("Rey", "routine", observed_at=4, correct=False)
    assert learner.temporal_source_reliability("Rey", "routine", 4) == pytest.approx(.5, abs=.01)
    assert learner.temporal_source_reliability("Rey", "night", 4) == .5
    assert abs(learner.temporal_source_reliability("Rey", "routine", 200) - .5) < .03


def test_private_truth_only_becomes_known_after_direct_observation():
    world = TemporalDecisionWorld((
        TemporalDelivery("Ada", "routine", 5, 2, 8, 1.0, 1.0),
        TemporalDelivery("Bea", "night", 5, 13, 9, 1.0, 1.0),
    ), reports=((1, "Ada", False, "Rey"),))
    assert all("arrival_at" not in p.metadata for p in world.promises())
    assert all("arrival_at" not in public for public in world.public_constraints().values())
    testimony = world.advance(3)[0]
    assert testimony.metadata["source_status"] == "unverified_testimony"
    assert "arrival_at" not in testimony.metadata
    sensor = world.inspect("Ada")
    assert sensor.metadata["source_status"] == "direct_observation"
    assert sensor.metadata["arrival_at"] == 2


def test_temporal_runner_uses_shared_owners_and_source_anchored_memory():
    learner, memory = train_temporal_evidence(5, 10)
    world = heldout_temporal_world(5, 1)
    run = DecisionValueRun(world, mode="predictive_value", seed=51,
                           learner=learner, memory=memory)
    source = [run.memory.events[e.source_event_id].content
              for e in run.state.timed_expectations.values()]
    assert all("promises delivery" in event for event in source)
    result = run.run()
    assert 0 < result["actions"] <= 18
    assert len(run.state.timed_expectations) == 3
    assert all(row["evidence_ids"] for row in result["trace"])
    assert result["cognitive_operations"] >= result["actions"]
    assert all(not row["thought"] or row["thought"]["additional_detailed_reads"] == 0
               for row in result["trace"])
    assert source == [run.memory.events[e.source_event_id].content
                      for e in run.state.timed_expectations.values()]


def test_no_new_evidence_and_no_new_probability_band_does_not_revise_thought():
    learner, memory = train_temporal_evidence(4, 5)
    run = DecisionValueRun(heldout_temporal_world(4, 1),
                           mode="predictive_value", seed=4,
                           learner=learner, memory=memory)
    run.step()
    before = len(run.refiner.thread(run.state).versions)
    # Re-examine at the exact same state, without executing a new action.
    actions = run._candidates()
    interpreted = run.interpreter.temporal_action_values(
        run.state, run.public, inspection_cost=run.world.inspection_cost,
        wait_cost=run.world.wait_cost, switch_reward=run.world.switch_reward)
    from scenario_chunking.models import RetrievalQuery
    hits, _ = run.router.retrieve(RetrievalQuery(
        goal_id=run.state.goal_id, semantic_tags=frozenset({"signal:temporal"}),
        current_time=run.state.world_time))
    scored = [(0.0, a, [], None) for a in actions]
    context = {"candidate_values": interpreted["values"], "presence": interpreted["presence"],
               "evidence_ids": tuple(e for values in interpreted["provenance"].values()
                                     for e in values), "thought_cost": 10.0}
    run.refiner.refine_temporal_value(run.state, scored, hits, context=context)
    new_before = len(run.refiner.thread(run.state).versions)
    _, trace = run.refiner.refine_temporal_value(run.state, scored, hits, context=context)
    assert not trace["changed"]
    assert trace["reasoning_stopped"]
    assert len(run.refiner.thread(run.state).versions) == new_before
    assert new_before >= before


def test_learning_and_thought_continuity_survive_new_python_process():
    learner, memory = train_temporal_evidence(14, 12)
    world = TemporalDecisionWorld((
        TemporalDelivery("Ada", "routine", 5, 5.5, 9, 3, .8),
        TemporalDelivery("Bea", "night", 6, 7.5, 11, 3, .8),
    ), switch_reward=.01, wait_cost=.005)
    run = DecisionValueRun(world, mode="predictive_value", seed=14,
                           learner=learner, memory=memory)
    run.step()
    assert not run.world.finished
    encoded = base64.b64encode(pickle.dumps(run)).decode("ascii")
    code = (
        "import base64,pickle,sys;"
        "r=pickle.loads(base64.b64decode(sys.argv[1]));"
        "assert len(r.state.timed_expectations)==2;"
        "assert r.learner.temporal_samples;"
        "assert r.refiner.thread(r.state).versions;"
        "assert r.state.world_time>=1;"
        "assert r.run()['actions']>=1"
    )
    subprocess.run([sys.executable, "-c", code, encoded], check=True)


def test_paired_controls_and_distractors_without_preordained_victory():
    report = run_decision_value_benchmark(
        seeds=(3, 7), training_stages=(0, 6), cases=(0, 1),
        distractors=60)
    assert report["experiment"] == 10
    assert len(report["trials"]) == 2 * 2 * 2 * 8
    assert "compute_matched_one_pass" in report["conditions"]
    assert "empty_reflection" in report["conditions"]
    for row in report["trials"]:
        assert 0 <= row["actions"] <= 18
        assert row["inspections"] <= row["actions"]
        assert all(v >= 0 for v in (row["header_evaluations"],
                                   row["detail_events"], row["cognitive_operations"]))
        assert not any("arrival_at" in str(v) for v in row["trace"])
    assert set(report["summary"]) == {"0", "6"}


def test_exhausted_opportunities_make_stopping_a_real_environment_option():
    world = heldout_temporal_world(17, 5)
    assert world.switch_reward == 0
    assert all(props["deadline"] == 0
               for props in world.public_constraints().values())
    assert world.apply(
        __import__("scenario_chunking.models", fromlist=["Action"]).Action("stop"),
        known_dependencies={}) == 0
    assert world.finished


def test_executed_outcome_creates_provenance_bound_learning_not_an_oracle_label():
    learner, memory = train_temporal_evidence(8, 10)
    world = TemporalDecisionWorld((
        TemporalDelivery("Ada", "routine", 2, .5, 8, 5, .5),
        TemporalDelivery("Bea", "routine", 3, 2, 8, 1, .5),
    ), switch_reward=0.0)
    run = DecisionValueRun(world, mode="compute_matched_one_pass",
                           seed=17, learner=learner, memory=memory)
    run.state.advance_world_time(2)
    run.world.advance(2)
    run._receive(run.world.inspect("Ada"))
    learned_before = len(run.learner.credit_log)
    run.step()
    if run.world.actions[-1].startswith("proceed:"):
        assert len(run.learner.credit_log) == learned_before + 1
        credit = run.learner.credit_log[-1]
        event = run.memory.events[credit["cause_event_id"]]
        assert event.kind.value == "action"
        assert event.action.kind == "proceed"
