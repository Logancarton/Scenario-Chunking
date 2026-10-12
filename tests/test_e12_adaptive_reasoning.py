"""Opt-in adaptive reasoning uses the real E12 owners, evidence and safety path."""
import pickle

from scenario_chunking import causal_metacognition as cm
from scenario_chunking.cognitive_control import CognitiveControl


def _adaptive(seed, **overrides):
    return cm.InvestigativeRun(
        seed, "metacognitive", exploration=0,
        policy=cm.Policy("voi", "learned", diagnosis=True,
                         activation="adaptive_sparse", activation_limit=2, **overrides))


def test_adaptive_attention_not_modulo_seven():
    control = CognitiveControl()
    scope = ("pair:domain:delivery", "pair:epoch:0")
    axes = ("capacity", "color", "weather")
    one = control.activate_axes(scope, axes, mode="adaptive_sparse", limit=2, experience=1)
    seven = control.activate_axes(scope, axes, mode="adaptive_sparse", limit=2, experience=7)
    assert one == seven
    assert len(one) == 2


def test_live_recovery_costs_real_belief_queries_without_parallel_selector():
    case = cm.training_stream(31, ("delivery",), 1)[0]
    # The costs are publicly specified, not hidden truth or forced diagnoses.
    case["inspect_costs"] = {axis: 0.0 for axis in case["inspect_costs"]}
    run = _adaptive(31)
    pending = run.act(case)
    row = run.deliver(pending)
    assert pending["reasoning_adequacy"]["status"] in (
        "world_uncertainty", "needs_reconsideration", "adequate", "negative_evidence", "safety")
    assert len(pending["active_axes"]) <= 2
    assert row["violations"] == 0
    assert row["belief_queries"] == (sum(a["belief_queries"] for a in pending["analyses"])
                                     + pending["recovery_extra_queries"])
    assert row["recovery_extra_queries"] >= 0
    assert row["recovery"] is not None  # No manually injected diagnosis.
    assert row["recovery"]["accepted"]
    assert row["recovery_extra_queries"] > 0
    assert set(row["recovery"]["after"]) <= set(case["hidden"])
    assert len(row["recovery"]["after"]) == 2
    assert "reasoning" in row["process_updated"]
    assert "reasoning" in run.learner.control.adjustments[-1]


def test_prohibited_actions_cannot_be_unlocked_by_more_thinking():
    case = cm.training_stream(47, ("restricted",), 1)[0]
    case["prohibited"] = frozenset(("standard", "express"))
    run = _adaptive(47)
    pending = run.act(case)
    row = run.deliver(pending)
    assert row["action"] == "defer"
    assert row["violations"] == 0
    assert row["recovery"] is None
    assert row["reasoning_adequacy"]["status"] == "safety"
    assert row["error"] is None
    assert run.learner.control.adjustments[-1]["error"] is None


def test_restart_preserves_learned_process_and_live_choice():
    seed = 41
    run = _adaptive(seed)
    for case in cm.training_stream(seed, ("delivery",), 4):
        run.step_case(case)
    assert any("reasoning" in a for a in run.learner.control.adjustments)
    restored = pickle.loads(pickle.dumps(run))
    case = cm.training_stream(seed, ("delivery",), 1, start=100)[0]
    pending_a, pending_b = run.act(case), restored.act(case)
    row_a, row_b = run.deliver(pending_a), restored.deliver(pending_b)
    assert pending_a["active_axes"] == pending_b["active_axes"]
    assert row_a["action"] == row_b["action"]
    assert row_a["inspections"] == row_b["inspections"]
    assert row_a["belief_queries"] == row_b["belief_queries"]
    assert row_a["recovery"] == row_b["recovery"]
    assert row_a["violations"] == row_b["violations"] == 0


def test_ablation_disables_process_feedback_without_disabling_world_learning():
    case = cm.training_stream(53, ("delivery",), 1)[0]
    run = _adaptive(53, recovery_enabled=False, process_learning=False)
    pending = run.act(case)
    row = run.deliver(pending)
    assert row["recovery"] is None
    assert row["process_updated"] == []
    assert not any("reasoning" in a for a in run.learner.control.adjustments)
    assert row["violations"] == 0
