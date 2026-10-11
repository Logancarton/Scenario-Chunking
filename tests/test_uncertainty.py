"""Experiment 7 guards for causal provenance, live decisions, and skepticism."""
import pickle
import subprocess
import sys

from scenario_chunking.environment import AXES, UncertaintyWorld
from scenario_chunking.learning import OutcomeLearner
from scenario_chunking.models import Action, EventKind, TruthStatus
from scenario_chunking.uncertainty import (
    CONDITIONS_7, UncertaintyRun, run_uncertainty_experiment,
)


def test_four_choices_and_private_dependency_revision():
    world = UncertaintyWorld(61)
    assert set(world.options) == set(AXES)
    assert all(len(set(world.options[axis])) == 4 for axis in AXES)
    assert all(world.correct[axis] in world.options[axis] for axis in AXES)
    first_order = tuple(world.active_order)
    old_time = world.correct["time"]
    world.revise()
    assert tuple(world.active_order) != first_order
    assert world.correct["time"] != old_time
    assert world.scene(61, 0)["object"] != world.scene(61, 1)["object"]


def test_calibration_is_only_updated_by_verified_experience():
    learner = OutcomeLearner()
    assert learner.source_reliability("Aster", "day") == .5
    learner.verify_source("Aster", "day", True)
    learner.verify_source("Beryl", "day", False)
    assert learner.source_reliability("Aster", "day") > .5
    assert learner.source_reliability("Beryl", "day") < .5
    assert learner.source_reliability("Aster", "night") == .5


def test_blocked_execution_teaches_sequence_not_false_token_failure():
    run = UncertaintyRun(7, "extended_reflection", rounds=2)
    state = run._state("blocked-test")
    after = run.world.order[-1]
    missing = run.world.prerequisite(after)
    action = Action("resolve", run.world.correct[after], after)
    before = run.learner.snapshot()
    result = run._blocked(state, "blocked-test", action, missing)
    assert run.memory.events[result].outcome == "blocked"
    assert run.learner.snapshot() == before
    assert run.learner.prerequisite_strength(missing, after) > 0
    assert missing in state.current_prerequisites[after]


def test_unverified_testimony_not_promoted_to_observed_success():
    run = UncertaintyRun(61, "extended_reflection", rounds=2, distractors=1)
    scene = run.world.scene(61, 0)
    run._context = scene["context"]
    run._write_reports("new-episode", scene, 0)
    hits, _ = run._retrieve(AXES, top_k=32)
    first = run._current_reports[AXES[0]][0]
    cues = frozenset({f"signal:{AXES[0]}:{first['token']}"})
    report = run.interpreter.uncertainty_evidence(hits, cues, context=scene["context"])
    assert not report["has_causal_evidence"]
    assert report["causal_score"] == 0
    assert all(run.memory.events[e].kind == EventKind.OBSERVATION
               for e in report["report_evidence_ids"])
    assert all(run.memory.events[e].truth_status == TruthStatus.FACT
               for e in report["report_evidence_ids"])


def test_live_path_uses_same_selector_refiner_memory_and_feedback():
    run = UncertaintyRun(23, "extended_reflection", rounds=3, distractors=2)
    result = run.run()
    assert result["restart_preserved"]
    assert len(result["rounds"]) == 3
    assert run.refiner.threads
    assert any(r["training_events"] > 0 for r in [result])
    assert result["learned_prerequisites"] > 0
    assert any(t["thought"] for t in result["traces"])
    assert any(t["causal_evidence_ids"] or t["testimony_evidence_ids"]
               for t in result["traces"])
    assert any(t["outcome"] in ("blocked", "failure", "success")
               for t in result["traces"])
    assert run.learner.credit_log
    assert run.memory.events


def test_invalidation_activates_information_seeking_through_live_refiner():
    run = UncertaintyRun(61, "extended_reflection", rounds=3, distractors=1)
    result = run.run()
    after_notice = [t for t in result["traces"]
                    if t["round"] >= 1 and t["thought"]]
    assert after_notice
    assert any(t["thought"]["thought_id"] for t in after_notice)
    assert any(t["reflection_changed_action"] for t in after_notice)


def test_reliability_and_prerequisites_survive_fresh_python_process(tmp_path):
    run = UncertaintyRun(101, "extended_reflection", rounds=2, distractors=1)
    run.train()
    path = tmp_path / "e7.pkl"
    path.write_bytes(pickle.dumps((run.memory, run.learner)))
    script = """
import pickle, sys
from scenario_chunking.learning import OutcomeLearner
memory, learner = pickle.load(open(sys.argv[1], "rb"))
assert len(memory.events) > 0
assert learner.source_reliability("Aster", "day") != .5
assert learner.prerequisites_for(sys.argv[2])
print("restored")
"""
    dependent = run.world.order[-1]
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), dependent],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "restored"


def test_matched_benchmark_exports_every_baseline_and_ablation(tmp_path):
    result = run_uncertainty_experiment(
        tmp_path, seeds=(7,), rounds=2, distractors=1, action_budget=9)
    assert result["experiment"] == 7
    assert set(result["conditions"]) == set(CONDITIONS_7)
    assert set(result["ablations"]) == set(AXES)
    assert all(m["restarts_preserved"] for m in result["conditions"].values())
    assert all(len(m["by_seed"]) == 1 for m in result["conditions"].values())
    assert all(len(m["round_completion"]) == 2 for m in result["conditions"].values())
    assert (tmp_path / "uncertainty_summary.json").exists()
    assert (tmp_path / "uncertainty_trials.json").exists()
