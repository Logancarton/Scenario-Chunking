"""Live-path research checks. Run only after applying patch to the repository."""
from scenario_chunking.experiment import ExperimentRun
from scenario_chunking.models import RetrievalMode


def test_activation_reaches_six_live_episodes_without_changing_source_times():
    run = ExperimentRun(RetrievalMode.ACTIVATION, 7, distractor_count=40)
    result = run.run()
    assert len(result.episodes) == 6
    assert run.working_memory is not None
    assert run.working_memory.trajectories
    assert all(t["active_count"] <= 6 for t in run.working_memory.trajectories)
    assert all(len(t["focused"]) <= 2 for t in run.working_memory.trajectories)
    assert any(t["meaningful_uses"] > 0 for t in run.working_memory.trajectories)
    original_times = {
        key: (event.occurrence_time, event.acquired_time, event.storage_time)
        for key, event in run.memory.events.items()
    }
    run.working_memory.snapshot("retrieve_object_from_locked_container", run.clock + 1000)
    assert original_times == {
        key: (event.occurrence_time, event.acquired_time, event.storage_time)
        for key, event in run.memory.events.items()
    }
    assert run.working_memory.snapshot(
        "retrieve_object_from_locked_container", run.clock + 1001
    )["active_count"] == 0


def test_historical_baseline_remains_distinct_from_new_treatment():
    baseline = ExperimentRun(RetrievalMode.STRUCTURAL, 23, distractor_count=30)
    active = ExperimentRun(RetrievalMode.ACTIVATION, 23, distractor_count=30)
    b = baseline.run()
    a = active.run()
    assert len(b.episodes) == len(a.episodes) == 6
    assert baseline.working_memory is None
    assert active.working_memory is not None
    assert all(ep.retrieval_stats.active_detail_events >= 0 for ep in a.episodes)
    assert all(t["active_count"] <= 6 for t in active.working_memory.trajectories)
    assert isinstance(active.learner.activation_updates, list)
