from scenario_chunking.experiment import ExperimentRun, run_recency_negative_control
from scenario_chunking.models import RetrievalMode


def test_structural_agent_runs_full_live_path_and_learns():
    run = ExperimentRun(RetrievalMode.STRUCTURAL, 7, distractor_count=250)
    summary = run.run()
    assert all(ep.success for ep in summary.episodes)
    assert run.learner.snapshot()
    assert summary.delayed_credit_accuracy == 1.0
    assert summary.temporal_order_accuracy == 1.0
    assert summary.retrieval_recall_at_k > 0.0
    e4 = next(ep for ep in summary.episodes if ep.episode_id == "E4_distractor_load")
    assert e4.retrieval_stats.header_evaluations < e4.retrieval_stats.total_chunks * max(1, e4.actions)


def test_full_retrieval_processes_more_detail_than_sparse_structural():
    full = ExperimentRun(RetrievalMode.FULL, 23, distractor_count=300).run()
    structural = ExperimentRun(RetrievalMode.STRUCTURAL, 23, distractor_count=300).run()
    full_details = sum(ep.retrieval_stats.active_detail_events for ep in full.episodes)
    structural_details = sum(ep.retrieval_stats.active_detail_events for ep in structural.episodes)
    assert full_details > structural_details


def test_fixed_recency_has_negative_control_where_newer_is_worse():
    control = run_recency_negative_control()
    assert control["fixed_recency_worse"] is True
