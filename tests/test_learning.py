from scenario_chunking.learning import OutcomeLearner


def test_delayed_credit_targets_causal_action_not_recent_noise():
    learner = OutcomeLearner()
    learner.begin_episode()
    learner.register_action("action-info", frozenset({"action:inspect_storage", "role:storage"}))
    learner.mark_informative("action-info")
    learner.register_action("action-key", frozenset({"action:try_key", "role:key", "clue_supported"}))
    learner.apply_success("action-key", 1.0)
    causal = [x for x in learner.credit_log if x["kind"] == "causal_success"]
    predecessors = [x for x in learner.credit_log if x["kind"] == "informative_predecessor"]
    assert causal[-1]["credited_event_id"] == "action-key"
    assert predecessors[-1]["credited_event_id"] == "action-info"
    assert learner.expected_utility(frozenset({"action:try_key", "role:key", "clue_supported"})) > 0


def test_failed_action_reduces_future_utility():
    learner = OutcomeLearner()
    learner.begin_episode()
    features = frozenset({"action:try_key", "role:key"})
    learner.register_action("a1", features)
    before = learner.expected_utility(features)
    learner.apply_failure("a1")
    assert learner.expected_utility(features) < before
