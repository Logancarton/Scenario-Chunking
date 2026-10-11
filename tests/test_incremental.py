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
        "assert len(r.refiner.threads)==10;"
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
        assert row["header_evaluations"] <= 48
    # Held-out predictions are scored before each unique execution.
    assert all(r["index"] >= 24 for r in report["trials"]
               if r["stage"] == "heldout")
