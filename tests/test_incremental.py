"""E11 high-value behavioral, negative-control, and live-path tests."""
import base64
import pickle
import subprocess
import sys

import pytest

from scenario_chunking.environment import IncrementalDispatchWorld
from scenario_chunking.incremental_experiment import (
    AXES, CONDITIONS, FAMILIES, IncrementalRun, experience,
    run_incremental_benchmark,
)
from scenario_chunking.learning import OutcomeLearner


def test_gray_curriculum_unique_minimal_change_and_unseen_contexts():
    for family in FAMILIES:
        history = [experience(7, family, i) for i in range(32)]
        contexts = [tuple(w.factors[a] for a in AXES) for w, _ in history]
        assert len(set(contexts)) == 32
        assert all(sum(x != y for x, y in zip(before, after)) == 1
                   for before, after in zip(contexts, contexts[1:]))
        assert all(not w.executed for w, _ in history)
        assert all(not any("success" in cue or "failure" in cue
                           for cue in cues) for _, cues in history)


def test_single_exposure_causal_feedback_refines_only_matching_associations():
    learner = OutcomeLearner()
    cues = frozenset({"pair:urgency:high", "pair:delay:long",
                      "pair:delay:long&pair:urgency:high"})
    unrelated = frozenset({"pair:domain:unrelated"})
    before = learner.incremental_prediction(cues, "standard")
    learner.incremental_feedback(cues, "standard",
                                 success=False, evidence_id="observed:1")
    after = learner.incremental_prediction(cues, "standard")
    assert after["probability"] < before["probability"]
    assert after["evidence_ids"] == ["observed:1"]
    assert learner.incremental_prediction(unrelated, "express")["probability"] == .5
    assert learner.contextual_probability("pair:domain:unrelated", "standard")[1] == 0
    # Contradiction revises local confidence; it cannot erase evidence history.
    learner.incremental_feedback(cues, "standard",
                                 success=True, evidence_id="observed:2")
    latest = learner.incremental_prediction(cues, "standard")
    assert {"observed:1", "observed:2"} <= set(latest["evidence_ids"])
    assert 0 < latest["probability"] < 1
    assert len(learner.context_evidence_ids[("pair:urgency:high", "standard")]) == 2


def test_episodic_outcome_is_not_a_hidden_supervision_label():
    world, cues = experience(23, "delivery", 7)
    assert world.public_cues() <= cues
    assert not any("outcome" in c or "reward" in c for c in cues)
    from scenario_chunking.models import Action
    output = world.execute(Action("route", target="standard"))
    assert output.outcome in ("success", "failure")
    with pytest.raises(RuntimeError):
        world.execute(Action("route", target="express"))


def test_live_sequencing_action_outcome_and_provenance():
    run = IncrementalRun(7, "incremental_full")
    row = run.step("delivery", 0)
    assert row["stage"] == "early"
    assert row["action"] in ("route:standard", "route:express")
    assert 0 <= row["prediction"] <= 1
    assert run.memory.detailed_event_count == 3
    assert len(run.memory.causal_links) >= 1
    assert len(row["updated_relations"]) > 0
    observed = [e for e in run.memory.events.values() if e.outcome in ("success", "failure")]
    assert len(observed) == 1 and observed[0].cause_event_id
    assert row["belief_update"]["evidence_id"] == observed[0].event_id
    with pytest.raises(ValueError):
        run.step("delivery", 0)
    another = run.step("delivery", 1)
    assert another["family"] == "delivery"
    assert run.memory.detailed_event_count == 6
    assert run.trace[-1]["index"] == 1


def test_frozen_and_episodic_controls_do_not_generalize_beliefs():
    frozen = IncrementalRun(7, "frozen")
    episodic = IncrementalRun(7, "episodic_only")
    for index in range(6):
        frozen.step("repair", index)
        episodic.step("repair", index)
    assert all(sum(values) == 0 for values in frozen.learner.context_counts.values())
    assert all(sum(values) == 0 for values in episodic.learner.context_counts.values())
    assert len(frozen.memory.events) == len(episodic.memory.events) == 18
    assert any(r["activated_evidence"] for r in episodic.trace[1:])


def test_independent_process_restart_preserves_memory_and_conditional_beliefs():
    run = IncrementalRun(23, "incremental_full")
    run.run(families=("delivery",), length=10)
    known = len(run.memory.events)
    assert known == 30
    payload = base64.b64encode(pickle.dumps(run)).decode("ascii")
    script = (
        "import base64,pickle,sys;"
        "r=pickle.loads(base64.b64decode(sys.argv[1]));"
        "assert len(r.memory.events)==30;"
        "assert r.learner.context_counts;"
        "assert len(r.refiner.threads)==1;"
        "assert r.step('delivery',10)['index']==10;"
        "assert len(r.memory.events)==33"
    )
    subprocess.run([sys.executable, "-c", script, payload], check=True)


def test_matched_prequential_research_and_ablation_no_leakage():
    report = run_incremental_benchmark(
        seeds=(7, 23), families=("delivery", "coordination"), length=32)
    assert report["experiment"] == 11
    assert len(report["trials"]) == 2 * 2 * 32 * len(CONDITIONS)
    assert set(report["summary"]) == set(CONDITIONS)
    assert len(report["paired_seed_deltas"]) == 2
    for mode in CONDITIONS:
        assert report["summary"][mode]["heldout"]["n"] == 2 * 2 * 8
    for row in report["trials"]:
        assert 0 <= row["prediction"] <= 1
        assert 0 <= row["uncertainty"] <= 1
        assert row["brier"] >= 0 and row["regret"] >= -1e-9
        assert all(isinstance(cue, str) for cue in row["context"])
        assert row["header_evaluations"] <= 80
    # Held-out predictions are scored before each unique execution.
    assert all(r["index"] >= 24 for r in report["trials"]
               if r["stage"] == "heldout")


def test_irrelevant_marker_does_not_change_private_causal_truth():
    factors = {"person": "Jordan", "delay": "long", "urgency": "high",
               "capacity": "open", "channel": "night", "marker": "blue"}
    blue = IncrementalDispatchWorld.scenario(
        seed=23, family="delivery", index=19, factors=factors, epoch=0)
    orange = IncrementalDispatchWorld.scenario(
        seed=23, family="delivery", index=19,
        factors={**factors, "marker": "orange"}, epoch=0)
    assert blue._outcomes == orange._outcomes


def test_signaled_regime_reversal_is_not_silently_reused():
    factors = {"person": "Jordan", "delay": "short", "urgency": "high",
               "capacity": "open", "channel": "night", "marker": "blue"}
    earlier = IncrementalDispatchWorld.scenario(
        seed=7, family="coordination", index=10, factors=factors, epoch=0)
    changed = IncrementalDispatchWorld.scenario(
        seed=7, family="coordination", index=10, factors=factors, epoch=1)
    assert earlier._outcomes["express"] != changed._outcomes["express"]
    assert earlier.public_cues() != changed.public_cues()


def test_prediction_and_disabled_feedback_do_not_mutate_unrelated_beliefs():
    learner = OutcomeLearner(enabled=False)
    cues = frozenset({"pair:irrelevant:blue", "pair:person:Alex"})
    learner.incremental_prediction(cues, "standard")
    learner.incremental_feedback(cues, "standard",
                                 success=True, evidence_id="verified")
    assert not learner.context_counts and not learner.context_base_counts
    learner.enabled = True
    learner.incremental_feedback(cues, "standard",
                                 success=True, evidence_id="verified")
    before = dict(learner.context_counts)
    learner.incremental_prediction(frozenset({"pair:unseen:test"}), "express")
    assert before == dict(learner.context_counts)
    assert ("pair:unseen:test", "express") not in learner.context_counts

def test_continuous_conceptual_identity_preserves_individual_working_states():
    run = IncrementalRun(7, "incremental_full")
    rows = [run.step("delivery", index) for index in range(10)]
    assert len({row["concept_id"] for row in rows}) == 1
    assert len(run.refiner.threads) == 1
    thread = next(iter(run.refiner.threads.values()))
    assert thread.initiating_episode.endswith(":delivery:0")
    assert len(thread.outcomes) == 10
    assert all(row["thought"]["thought_id"] == rows[0]["thought"]["thought_id"]
               for row in rows)
    assert rows[0]["reused_belief_identity"] is False
    assert all(row["reused_belief_identity"] for row in rows[1:])
    assert all(row["changed_context"] for row in rows[1:])
    assert all(row["next_decision"] for row in rows[:-1])
    assert len({event.episode_id for event in run.memory.events.values()}) == 10
    assert len(run.memory.events) == 30
    assert any(row["supporting_prior_evidence"] for row in rows[1:])
    assert all(row["changed_belief"]["credit_mass"] == pytest.approx(1.0)
               for row in rows)
    assert all(len(row["updated_relations"]) <= 6 for row in rows)


def test_prior_experience_causally_changes_next_prediction():
    reused = IncrementalRun(7, "incremental_full")
    prior = reused.step("delivery", 0)
    current = reused.step("delivery", 1)
    missing = IncrementalRun(7, "incremental_full")
    without_prior = missing.step("delivery", 1)
    affected = prior["action"]
    assert (current["candidate_probabilities"][affected] !=
            without_prior["candidate_probabilities"][affected])
    assert prior["observed_outcome_id"] in set(
        current["thought"]["contributing_evidence_ids"])
    assert current["changed_belief"]["posterior"] != (
        current["changed_belief"]["prior"])
    assert all("&pair:marker:" not in key and "pair:marker:" not in key
               for key in current["updated_relations"])


def test_unrelated_prior_experience_does_not_change_next_beliefs():
    base = IncrementalRun(23, "incremental_full")
    expected = base.step("delivery", 2)
    with_unrelated = IncrementalRun(23, "incremental_full")
    with_unrelated.step("repair", 0)
    observed = with_unrelated.step("delivery", 2)
    assert observed["candidate_probabilities"] == expected["candidate_probabilities"]
    assert observed["supporting_prior_evidence"] == []
    assert observed["concept_id"] != with_unrelated.trace[0]["concept_id"]


def test_frozen_heldout_isolated_from_all_test_feedback():
    train = IncrementalRun(23, "incremental_full")
    train.run(families=("delivery",), length=24)
    frozen_snapshot = pickle.dumps(train)
    first = pickle.loads(frozen_snapshot)
    before_beliefs = dict(first.learner.context_counts)
    before_events = len(first.memory.events)
    first_row = first.step("delivery", 24, freeze=True)
    assert first_row["stage"] == "frozen_heldout"
    assert first_row["updated_relations"] == []
    assert len(first.memory.events) == before_events
    assert dict(first.learner.context_counts) == before_beliefs
    second = pickle.loads(frozen_snapshot)
    second_row = second.step("delivery", 25, freeze=True)
    assert first_row["observed_outcome_id"] is None
    assert second_row["observed_outcome_id"] is None
    assert 24 not in second.seen
    report = run_incremental_benchmark(
        seeds=(23,), families=("delivery",), length=26)
    assert len(report["frozen_trials"]) == 2 * len(CONDITIONS)
    assert all(row["updated_relations"] == [] and
               row["observed_outcome_id"] is None and
               row["stage"] == "frozen_heldout"
               for row in report["frozen_trials"])
    assert all(report["frozen_summary"][mode]["frozen_heldout"]["n"] == 2
               for mode in CONDITIONS)
    assert report["frozen_training_protocol"]["same_interventions_across_treatments"]
    assert report["frozen_training_protocol"]["each_heldout_case_from_identical_pretest_snapshot"]


def test_epoch_change_preserves_old_history_without_cross_epoch_credit():
    run = IncrementalRun(61, "incremental_full")
    first = run.step("coordination", 0)
    changed = run.step("coordination", 16)
    assert first["concept_id"] != changed["concept_id"]
    assert changed["reused_belief_identity"] is False
    assert changed["candidate_probabilities"] == {
        "route:standard": .5, "route:express": .5}
    assert first["observed_outcome_id"] not in set(
        changed["thought"]["contributing_evidence_ids"])


def test_marker_is_excluded_from_conceptual_attribution():
    from scenario_chunking.incremental_experiment import experience
    learner = OutcomeLearner()
    _, cues = experience(7, "delivery", 18)
    mirror = frozenset(
        c.replace("pair:marker:orange", "pair:marker:blue")
        for c in cues if "&" not in c)
    primitive = frozenset(c for c in cues if "&" not in c)
    learner.incremental_feedback(
        primitive, "standard", success=True, evidence_id="event:1")
    assert learner.incremental_prediction(
        primitive, "standard")["probability"] == learner.incremental_prediction(
            mirror, "standard")["probability"]
    assert sum(sum(row) for row in learner.context_counts.values()) == pytest.approx(1.0)
