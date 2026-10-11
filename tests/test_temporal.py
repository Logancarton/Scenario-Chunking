"""Experiment 9: live temporal signal path and adversarial provenance controls."""
import base64
import pickle
import subprocess
import sys

import pytest

from scenario_chunking.environment import ReturnTimelineWorld
from scenario_chunking.learning import OutcomeLearner
from scenario_chunking.state import CognitiveState
from scenario_chunking.temporal_experiment import TemporalResearchRun, run_temporal_benchmark


def _run(actual=16, *, mode="temporal", visible=True, timestamp=True):
    run = TemporalResearchRun(mode=mode)
    run.start(ReturnTimelineWorld(
        subject="Rae", promised_duration=10,
        actual_arrival_at=actual, observation_available=visible,
        timestamp_available=timestamp), "visit-1")
    return run


def test_clock_wakes_pending_thought_without_new_message_or_created_failure():
    run = _run(actual=None, visible=False)
    source_id = run.state.timed_expectations[run.expectation_id].source_event_id
    source = run.memory.events[source_id]
    assert run.trace[0]["status"] == "pending"
    assert run.advance(8) == []
    assert run.advance(10) != []
    expectation = run.state.timed_expectations[run.expectation_id]
    assert expectation.status == "due_unverified"
    assert expectation.observation_ids == []
    assert run.advance(11) == []  # no repeated thought from empty clock ticks
    assert source.content.startswith("Rae says")
    assert source.occurrence_time == 0
    assert run.trace[-1]["thought"]["reasoning_stopped"] is False


def test_witnessed_absence_updates_then_late_arrival_revises_and_learns():
    run = _run(actual=16)
    due = run.advance(12)
    assert due[0]["action"] == "check_return"
    assert due[1]["action"] == "follow_up"
    assert due[1]["status"] == "verified_absent"
    expectation = run.state.timed_expectations[run.expectation_id]
    assert expectation.last_verified_absent_at == 12
    assert run.learner.temporal_punctuality[expectation.context] == [0.0, 1.0]
    returned = run.observe_externally(run.world.check_return(16), at=16)
    assert returned["action"] == "acknowledge_return"
    assert expectation.status == "fulfilled"
    assert expectation.arrived_at == 16
    assert run.learner.temporal_punctuality[expectation.context] == [0.0, 1.0]
    estimated, ontime = run.learner.temporal_prediction(expectation.context, 10)
    assert estimated == 16.0 and ontime < .5
    assert len(expectation.observation_ids) == 2


def test_early_arrival_uses_sensor_timestamp_not_inspection_time():
    run = _run(actual=8)
    run.advance(12)
    expectation = run.state.timed_expectations[run.expectation_id]
    assert expectation.status == "fulfilled" and expectation.arrived_at == 8
    estimated, punctuality = run.learner.temporal_prediction(expectation.context, 10)
    assert estimated == 8 and punctuality > .5


def test_unknown_arrival_timestamp_never_generates_exact_duration_label():
    run = _run(actual=8, timestamp=False)
    run.advance(12)
    expectation = run.state.timed_expectations[run.expectation_id]
    assert expectation.status == "fulfilled" and expectation.arrived_at is None
    assert run.learner.temporal_delay[expectation.context] == [0.0, 0.0]
    assert run.learner.temporal_punctuality[expectation.context] == [0.0, 0.0]


def test_testimony_and_unavailable_observation_do_not_become_causal_outcomes():
    run = _run(actual=None, visible=False)
    steps = run.advance(12, reported_present=True)
    assert steps and steps[0]["action"] == "check_return"
    expectation = run.state.timed_expectations[run.expectation_id]
    assert expectation.status == "due_unverified"
    assert not expectation.observation_ids
    assert run.learner.temporal_punctuality[expectation.context] == [0.0, 0.0]


def test_time_validation_duplicate_evidence_and_stable_version_stop():
    state = CognitiveState("temporal-edge")
    with pytest.raises(ValueError):
        state.schedule_temporal(expectation_id="e", subject="A", context="c",
                                source_event_id="src", due_at=-1, confidence=.8)
    state.schedule_temporal(expectation_id="e", subject="A", context="c",
                            source_event_id="src", due_at=10, confidence=.8)
    assert state.advance_world_time(10) == ("e",)
    assert state.advance_world_time(10) == ()
    with pytest.raises(ValueError):
        state.advance_world_time(9)
    assert not state.observe_temporal("e", observation_id="rumor",
                                      observed_at=10, returned=False,
                                      direct_observation=False)
    assert state.observe_temporal("e", observation_id="direct",
                                  observed_at=10, returned=False,
                                  direct_observation=True)
    assert not state.observe_temporal("e", observation_id="direct",
                                      observed_at=10, returned=False,
                                      direct_observation=True)
    with pytest.raises(ValueError):
        state.observe_temporal("e", observation_id="future",
                               observed_at=12, returned=True,
                               direct_observation=True)

    run = _run(actual=None, visible=False)
    run.advance(12)
    versions = len(run.refiner.thread(run.state).versions)
    run._decide()  # direct re-analysis of exactly the same evidence
    assert len(run.refiner.thread(run.state).versions) == versions
    assert run.trace[-1]["thought"]["reasoning_stopped"]


def test_observed_memory_and_learned_state_survive_independent_process_restart():
    run = _run(actual=16)
    run.advance(12)
    encoded = base64.b64encode(pickle.dumps(run)).decode("ascii")
    code = (
        "import base64,pickle,sys;"
        "r=pickle.loads(base64.b64decode(sys.argv[1]));"
        "e=r.state.timed_expectations[r.expectation_id];"
        "assert e.status=='verified_absent';"
        "assert len(e.observation_ids)==1;"
        "assert r.refiner.thread(r.state).versions;"
        "assert r.memory.events[e.source_event_id].content.startswith('Rae says');"
        "assert r.learner.temporal_punctuality[e.context]==[0.0,1.0]"
    )
    subprocess.run([sys.executable, "-c", code, encoded], check=True)


def test_bounded_retrieval_avoids_irrelevant_memory_headers():
    run = _run(actual=8)
    run.memory.inject_distractors(800, start_time=2.0)
    result = run.advance(12)
    assert result[0]["action"] == "check_return"
    assert run.header_evaluations < 20
    assert result[0]["detail_events"] <= 36
    assert result[0]["thought"]["additional_detailed_reads"] == 0


def test_paired_benchmark_preserves_negative_controls_without_assuming_gain():
    results = run_temporal_benchmark(seeds=(7, 23))
    assert results["experiment"] == 9
    assert len(results["trials"]) == 40
    assert results["summary"]["temporal"]["due_wakes"] == 10
    assert results["summary"]["memory_only"]["due_wakes"] == 0
    assert not any(r["false_failure_claim"] for r in results["trials"])
    assert results["summary"]["temporal"]["observed_absence"] > 0
    assert results["summary"]["temporal"]["observed_return"] > 0
    # The identical learned-information one-pass control is allowed to tie.
    assert results["summary"]["learned_one_pass"]["due_wakes"] == 10
