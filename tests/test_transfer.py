"""Behavioral research guards: attribution, transfer, thought and persistence."""
from scenario_chunking.transfer import (
    AXES, TransferRun, TransferState, TransferWorld, run_transfer_experiment,
)
from scenario_chunking.models import Action, EventKind, RetrievalHit


def test_procedural_truth_is_seeded_and_surface_names_change():
    world = TransferWorld(7)
    assert set(world.options) == set(AXES)
    assert all(world.correct[a] in world.options[a] for a in AXES)
    assert world.correct != TransferWorld(23).correct
    assert world.scene(7, 0)["object"] != world.scene(7, 1)["object"]


def test_training_uses_observed_causal_feedback_not_unverified_reports():
    run = TransferRun(7, "extended_reflection", rounds=1, distractors=1)
    run.train()
    assert len(run.training) == 16 * len(AXES)
    for axis in AXES:
        hits, _ = run._retrieve((axis,))
        good = run.world.correct[axis]
        features = frozenset({f"signal:{axis}:{good}"})
        evidence = run.interpreter.transfer_evidence(hits, features, review_depth=3)
        for source_id in evidence["evidence_ids"]:
            assert run.memory.events[source_id].kind == EventKind.OUTCOME
        assert all(k.startswith("signal:") for k in evidence["categories"])
        assert evidence["evaluations"] > 0


def test_no_memory_or_learned_signal_cannot_fabricate_causal_evidence():
    run = TransferRun(7, "untrained_structural", rounds=1)
    hits, _ = run._retrieve(("person",))
    frame = run.interpreter.transfer_evidence(
        hits, frozenset({"signal:person:unknown"}), review_depth=3)
    assert frame["evidence_ids"] == []
    assert frame["score"] == 0


def test_ablation_and_shuffling_remove_or_misattribute_category_cues():
    state = TransferState("trial", excluded_axis="time", choices={
        "time": ("a", "b"), "self": ("c", "d"),
    })
    a = Action("integrate", target="proposal")
    state.option_features[a.label] = frozenset({"signal:time:a", "signal:self:c"})
    assert "signal:time:a" not in state.action_features(a)
    state.excluded_axis = None
    state.scramble = True
    assert "signal:time:b" in state.action_features(a)
    assert "signal:self:d" in state.action_features(a)


def test_heldout_stepwise_transfer_updates_same_memory_and_survives_reload():
    run = TransferRun(61, "extended_reflection", rounds=2, distractors=2)
    outcome = run.run()
    assert outcome["restart_preserved"]
    assert all(outcome["phase1"][axis]["observations"] == 16 for axis in AXES)
    assert len(outcome["rounds"]) == 2
    assert all(len(r["axes_completed"]) == 5 for r in outcome["rounds"])
    assert any(t["step"] == "joint_plan" for t in outcome["traces"])
    assert any(t["step"] in AXES for t in outcome["traces"])
    assert run.refiner.threads
    assert run.cost["interpretation_evaluations"] > 0
    assert len(run.memory.events) > 2 * len(run.training)


def test_matched_treatments_emit_inspectable_negative_controls(tmp_path):
    result = run_transfer_experiment(tmp_path, seeds=(7, 23), rounds=2, distractors=1)
    assert "shuffled_relationships" in result["conditions"]
    assert set(result["ablations"]) == set(AXES)
    assert result["conditions"]["extended_reflection"]["restarts_preserved"]
    assert len(result["conditions"]["extended_reflection"]["round_completion"]) == 2
    assert (tmp_path / "transfer_trials.json").is_file()
    assert (tmp_path / "transfer_summary.json").is_file()


def test_knowledge_is_retrievable_in_a_fresh_python_process(tmp_path):
    import json
    import pickle
    import subprocess
    import sys

    run = TransferRun(101, "extended_reflection", rounds=1, distractors=1)
    run.train()
    saved = tmp_path / "memory_state.pkl"
    saved.write_bytes(pickle.dumps((run.memory, run.learner)))
    # Unlike an in-process pickle round-trip, this reconstructs the memory,
    # query/router and interpretation owners in a separate Python interpreter.
    script = """
import json, pickle, sys
from scenario_chunking.meaning import MeaningInterpreter
from scenario_chunking.models import RetrievalMode, RetrievalQuery
from scenario_chunking.retrieval import MemoryRouter
memory, learner = pickle.load(open(sys.argv[1], "rb"))
axis, token = sys.argv[2], sys.argv[3]
query = RetrievalQuery("workshop_recovery", frozenset({
    "goal:workshop_recovery", f"axis:{axis}", f"signal:{axis}:{token}",
}), current_time=999)
hits, _ = MemoryRouter(memory, RetrievalMode.STRUCTURAL, top_k=16).retrieve(query)
result = MeaningInterpreter(memory, learner).transfer_evidence(
    hits, frozenset({f"signal:{axis}:{token}"}), review_depth=3)
print(json.dumps({"sources": result["evidence_ids"], "value": result["score"]}))
"""
    axis = "person"
    token = run.world.correct[axis]
    result = subprocess.run(
        [sys.executable, "-c", script, str(saved), axis, token],
        text=True, capture_output=True, check=True,
    )
    answer = json.loads(result.stdout)
    assert answer["sources"]
    assert answer["value"] > 0


def test_offline_reflection_preserves_provenance_without_fabricating_feedback():
    import pickle

    run = TransferRun(7, "extended_reflection", rounds=1, distractors=1)
    run.train()
    weights_before = run.learner.snapshot()
    events_before = len(run.memory.events)
    run._phase1b_reflect()
    assert len(run.memory.events) == events_before
    assert run.learner.snapshot() == weights_before
    assert run.refiner.threads
    assert run.memory.layer_meanings
    assert all(
        isinstance(value, dict) and
        value["status"] == "inferred_from_observed_outcomes" and
        value["evidence_ids"]
        for value in run.memory.layer_meanings.values()
    )
    restored = pickle.loads(pickle.dumps(run.memory))
    assert restored.layer_meanings == run.memory.layer_meanings


def test_revision_invalidates_old_causal_signal_without_destroying_history():
    run = TransferRun(61, "extended_reflection", rounds=4, distractors=2)
    run.train()
    axis = "time"
    obsolete = run.world.correct[axis]
    before = run.interpreter.transfer_evidence(
        run._retrieve((axis,))[0], frozenset({f"signal:{axis}:{obsolete}"}), review_depth=3)
    assert before["score"] > 0
    cutover = run.clock + 1
    after = run.interpreter.transfer_evidence(
        run._retrieve((axis,))[0], frozenset({f"signal:{axis}:{obsolete}"}),
        review_depth=3, invalidated_since={axis: cutover})
    assert after["score"] == 0.0
    assert after["evidence_ids"] == []
    assert before["evidence_ids"]
    assert all(eid in run.memory.events for eid in before["evidence_ids"])


def test_progressive_refiner_is_reached_before_transfer_action():
    run = TransferRun(61, "extended_reflection", rounds=4, distractors=1)
    result = run.run()
    plans = [t for t in result["traces"] if t["step"] == "joint_plan"]
    assert any(t.get("thought") and t["thought"].get("version") for t in plans)
    assert any(t.get("decision_basis") == "verify_invalidated_relationship"
               for t in result["traces"])
    assert any(t.get("step") == "inspect_protocol" for t in result["traces"])


def test_recovery_and_deadline_success_are_reported_separately():
    run = TransferRun(61, "full_no_reflection", rounds=4, distractors=1)
    result = run.run()
    revised = result["rounds"][2:]
    assert all("recovered_success" in t for t in revised)
    assert all("within_budget" in t for t in revised)
    assert all(t["actions"] == 1 + sum(x["step"] == "resolve" for x in [])
               or t["actions"] > 0 for t in revised)
    assert any(t["recovered_success"] for t in revised)


def test_feedback_keeps_multi_action_trace_and_causal_plan_record():
    run = TransferRun(23, "extended_reflection", rounds=2, distractors=1)
    result = run.run()
    plan_attempts = [e for e in run.memory.events.values()
                     if e.kind == EventKind.ACTION and e.action
                     and e.action.kind == "integrate"]
    assert len(plan_attempts) == 2
    assert all(any(e.cause_event_id == action.event_id
                       for e in run.memory.events.values()
                       if e.kind == EventKind.OUTCOME)
               for action in plan_attempts)
    assert any(t.get("used_evidence_ids") for t in result["traces"]
               if t.get("step") == "joint_plan" and t.get("thought"))


def test_revision_notice_without_protocol_outcome_does_not_prove_new_answer():
    run = TransferRun(61, "extended_reflection", rounds=4, distractors=0)
    run.train()
    old = run.world.correct["time"]
    timestamp = run.clock + 1
    query = run._retrieve(("time",))[0]
    evidence = run.interpreter.transfer_evidence(
        query, frozenset({f"signal:time:{old}"}),
        invalidated_since={"time": timestamp}, review_depth=3)
    assert evidence["score"] == 0
    assert not evidence["evidence_ids"]
