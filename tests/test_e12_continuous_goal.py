"""E12 continuing goal assessment through the live investigation selector and world.

These tests establish integration for a synthetic opt-in activity. They do NOT
show autonomous goal invention, calibrated goal understanding, or learning-to-learn.
"""
from copy import deepcopy
import pickle

import pytest

from scenario_chunking import causal_metacognition as cm
from scenario_chunking.models import EnvironmentSignal, EventKind, TruthStatus
from scenario_chunking.state import CognitiveState


def activity_case(*, steps=(), verification=()):
    case = cm.make_case(
        31, "delivery", 8001,
        visible={"person": "Alex", "delay": "short", "urgency": "low", "channel": "day"},
        hidden={"capacity": "open", "weather": "clear", "color": "blue"},
        hazard="low", restriction="none", stage="train")
    case["truths"] = {"standard": True, "express": True}
    case["noise"] = 0.0
    case["activity"] = {
        "requirements": ["recipient_confirmed"],
        "steps": list(steps), "verification": list(verification)}
    return case


def test_mechanical_success_does_not_prove_goal_success():
    run = cm.InvestigativeRun(31, "metacognitive", exploration=0)
    case = activity_case(
        steps=[{"content": "Vehicle still travelling"}],
        verification=[{"requirement": "recipient_confirmed", "satisfied": False,
                       "source": "recipient_ack", "final": True}])
    pending = run.act(case, learn=False, behavior="standard")
    assert pending["observed"] is None
    assert pending["state"].goal_status == "active"
    unchanged = run.advance_activity(pending)
    assert unchanged["reassessed"] is False
    assert pending["activity_reassessments"] == 0
    assert run.advance_activity(pending)["phase"] == "action_outcome_pending"
    row = run.deliver(pending)
    assert row["outcome"] == "success"
    assert row["goal_status"] == "awaiting_verification"
    assert not pending["state"].goal_complete
    observed = run.verify_activity(pending)
    assert observed["goal_status"] == "failed"
    assert not pending["state"].goal_complete
    assert run.verify_activity(pending) is None


def test_changed_permission_causes_mid_execution_safe_reconsideration():
    run = cm.InvestigativeRun(31, "metacognitive", exploration=0)
    case = activity_case(steps=[
        {"content": "Road access revoked", "prohibited": ["standard", "express"]}])
    pending = run.act(case, learn=False, behavior="standard")
    result = run.advance_activity(pending)
    assert result["reassessed"] is True
    assert result["action"] == "route:defer"
    assert pending["activity_revisions"] == 1
    assert pending["state"].prohibited == frozenset({"standard", "express"})
    run.advance_activity(pending)
    row = run.deliver(pending)
    assert row["action"] == "defer"
    assert row["violations"] == 0
    assert row["activity_reassessments"] == 1


def test_later_independent_evidence_can_verify_goal():
    run = cm.InvestigativeRun(31, "metacognitive", exploration=0)
    case = activity_case(verification=[
        {"requirement": "recipient_confirmed", "satisfied": True,
         "source": "recipient_ack", "elapsed": 50}])
    pending = run.act(case, learn=False, behavior="standard")
    run.advance_activity(pending)
    run.deliver(pending)
    assert pending["state"].goal_status == "awaiting_verification"
    check = run.verify_activity(pending)
    assert check["goal_status"] == "completed_verified"
    assert pending["state"].goal_complete
    assert pending["state"].goal_evidence["recipient_confirmed"][2] == "recipient_ack"


def test_unfinished_activity_restart_has_identical_future_steps():
    run = cm.InvestigativeRun(31, "metacognitive", exploration=0)
    case = activity_case(steps=[
        {"content": "En route"}, {"content": "Route changed", "visible_updates": {"delay": "long"}}],
        verification=[{"requirement": "recipient_confirmed", "satisfied": True,
                       "source": "recipient_ack"}])
    pending = run.act(case, learn=False, behavior="standard")
    run.advance_activity(pending)
    twin_run, twin_pending = pickle.loads(pickle.dumps((run, pending)))
    for current_run, current_pending in ((run, pending), (twin_run, twin_pending)):
        current_run.advance_activity(current_pending)
        current_run.advance_activity(current_pending)
        current_run.deliver(current_pending)
    assert pending["chosen"].label == twin_pending["chosen"].label
    assert pending["activity_revisions"] == twin_pending["activity_revisions"]
    assert run.verify_activity(pending)["goal_status"] == (
        twin_run.verify_activity(twin_pending)["goal_status"])
    assert pending["state"].goal_evidence == twin_pending["state"].goal_evidence


def test_goal_evidence_requires_provenance_and_rejects_out_of_order():
    state = CognitiveState("test:1", goal_id="delivery")
    state.begin_goal_activity("delivery:1", ("recipient_confirmed",))
    def signal(value, source):
        return EnvironmentSignal(
            kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
            content="Independent recipient check", semantic_tags=frozenset(),
            metadata={"goal_check": {"requirement": "recipient_confirmed",
                                     "satisfied": value, "source": source}})
    with pytest.raises(ValueError, match="provenance"):
        state.verify_goal(signal(True, ""), event_id="bad", occurred_at=5, observed_at=5)
    assert state.verify_goal(signal(False, "recipient"), event_id="new",
                             occurred_at=5, observed_at=10) == "awaiting_verification"
    assert state.verify_goal(signal(True, "recipient"), event_id="old",
                             occurred_at=2, observed_at=11) == "awaiting_verification"
    assert not state.goal_complete


def test_live_step_case_runs_continuous_path_without_oracle_goal_label():
    run = cm.InvestigativeRun(31, "metacognitive", exploration=0)
    case = activity_case(
        steps=[{"content": "Receiver awaiting response"}],
        verification=[{"requirement": "recipient_confirmed", "satisfied": True,
                       "source": "recipient_ack"}])
    row = run.step_case(case, learn=False, behavior="standard")
    assert row["outcome"] == "success"
    assert row["goal_status"] == "awaiting_verification"
    assert row["goal_status_final"] == "completed_verified"
    assert row["goal_verified"] is True
    assert len(row["verification_evidence"]) == 1
    assert row["activity_reassessments"] == 0


def test_learning_activity_keeps_initial_variables_for_hindsight_after_restrictions_change():
    # All three variables are inspected before execution, so a later public
    # constraints refresh contains no uninspected variables. Appraisal must
    # still reason about what the organism originally did not know.
    run = cm.InvestigativeRun(
        31, "metacognitive", exploration=0,
        policy=cm.Policy("all", "learned", diagnosis=True))
    case = activity_case(
        steps=[{"content": "Route permission revoked",
                "prohibited": ["standard", "express"]}],
        verification=[{"requirement": "recipient_confirmed",
                       "satisfied": False, "source": "recipient_ack", "final": True}])
    row = run.step_case(case, learn=True)
    assert row["inspections"] == ["capacity", "color", "weather"]
    assert row["action"] == "defer" and row["violations"] == 0
    assert row["activity_revisions"] == 1
    assert row["goal_status_final"] == "failed"
    assert run.learner.control.updates > 0
