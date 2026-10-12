"""E12 sparse activation is a computation policy, not a second cognition system."""
import pickle

from scenario_chunking import causal_metacognition as cm
from scenario_chunking.cognitive_control import CognitiveControl


def test_sparsity_reduces_real_explanation_queries_under_matched_evidence():
    case = cm.training_stream(31, ("delivery",), 1)[0]
    rows = {}
    for activation in ("dense", "fixed_sparse", "learned_sparse"):
        run = cm.InvestigativeRun(
            31, "metacognitive", exploration=0,
            policy=cm.Policy("voi", activation=activation, activation_limit=2))
        pending = run.act(case, learn=False, behavior="standard")
        row = run.deliver(pending)
        rows[activation] = (pending, row)
        assert row["action"] == "standard"  # forced matched-evidence route
        assert row["violations"] == 0
        assert len(pending["analyses"]) == 1
    dense = rows["dense"][0]["analyses"][0]
    fixed = rows["fixed_sparse"][0]["analyses"][0]
    learned = rows["learned_sparse"][0]["analyses"][0]
    assert dense["belief_queries"] == 14  # two routes x (1 + 2*3 hidden axes)
    assert fixed["belief_queries"] == learned["belief_queries"] == 10
    assert rows["dense"][1]["knowledge_brier"] == rows["fixed_sparse"][1]["knowledge_brier"]
    assert rows["dense"][1]["knowledge_brier"] == rows["learned_sparse"][1]["knowledge_brier"]
    assert rows["dense"][0]["analyses"][0]["predictions"]["route:standard"] == fixed["predictions"]["route:standard"]


def test_learned_activation_revisits_omitted_axis_after_missing_information():
    control = CognitiveControl()
    scope = ("pair:domain:delivery", "pair:epoch:0")
    for axis, strength in (("capacity", 6), ("color", 5), ("weather", -4)):
        control.record_relevance(scope, axis, strength)
    axes = ("capacity", "color", "weather")
    assert control.activate_axes(scope, axes, mode="learned_sparse", limit=2, experience=1) == {
        "capacity", "color"}
    control.audit({"scope": list(scope), "error": "missing_information"})
    assert control.activate_axes(scope, axes, mode="learned_sparse", limit=2, experience=2) == {
        "capacity", "weather"}
    assert control.activate_axes(scope, axes, mode="fixed_sparse", limit=2, experience=2) == {
        "capacity", "color"}
    assert control.activate_axes(scope, axes, mode="learned_sparse", limit=2, experience=7) == {
        "capacity", "weather"}


def test_sparse_selector_still_enforces_authorization_and_restart_identity():
    case = cm.training_stream(43, ("restricted",), 1)[0]
    case["prohibited"] = frozenset(("standard", "express"))
    sparse = cm.InvestigativeRun(
        43, "metacognitive", exploration=0,
        policy=cm.Policy("voi", "learned", diagnosis=True,
                         activation="learned_sparse", activation_limit=1))
    restored = pickle.loads(pickle.dumps(sparse))
    a, b = sparse.step_case(case), restored.step_case(case)
    assert a["action"] == b["action"] == "defer"
    assert a["violations"] == b["violations"] == 0
    assert a["belief_queries"] == b["belief_queries"]
    assert sparse.learner.control.relevance == restored.learner.control.relevance
