"""Experiment 8: behavioral validity, causal generalization, restart and controls."""
from __future__ import annotations

import json
from dataclasses import replace
import pickle
import subprocess
import sys

import pytest

from scenario_chunking.environment import ContextualWorld
from scenario_chunking.learning import OutcomeLearner
from scenario_chunking.refinement import ProgressiveRefiner
from scenario_chunking.state import TransferState
from scenario_chunking.models import Action
from scenario_chunking.scaling import ScalingRun, run_scaling_experiment


def test_context_is_joint_and_heldout_outcomes_balanced():
    w = ContextualWorld(7)
    outcomes = [next(p for p in w.PROCEDURES if w.outcome(
        w.task(i, heldout=True), p)) for i in range(4)]
    assert len(set(outcomes)) == 4
    assert not (set(w.TRAIN_KEYS) & set(w.HOLDOUT_KEYS))
    for i in range(4):
        public = w.task(i, heldout=True).public()
        assert "actual_dependency" not in public
        assert "correct_action" not in public
        cues = w.relational_cues(public)
        assert any("mode=" in cue and "dependency=" in cue for cue in cues)
        assert any("material=" in cue and "phase=" in cue for cue in cues)


def test_direct_experiment8_run_uses_contextual_environment_not_legacy_runner():
    agent = ScalingRun(13, "learned_focus_one_pass", rounds=2, candidate_budget=24)
    result = agent.run(experiences=20, memory_target=10, complexity=2)
    assert result["training_experiences"] == 20
    assert result["rounds"] == 2
    assert all(task["task_id"].startswith("transfer-") for task in result["tasks"])
    assert all(len(task["traces"]) <= agent.action_budget for task in result["tasks"])


def test_cumulative_experiences_are_observed_and_preserve_previous_evidence():
    a = ScalingRun(7, "trained_association", rounds=2)
    b = ScalingRun(7, "adaptive_focus", rounds=2)
    a.add_experiences(20)
    previous = tuple(a.memory.chunk_order)
    first = [dict(record) for record in a.training_observations]
    a.add_experiences(100)
    b.add_experiences(100)
    assert a.experiences == b.experiences == 100
    assert tuple(a.memory.chunk_order[:20]) == previous
    assert a.training_observations[:20] == first
    assert a.training_observations == b.training_observations
    assert a.learner.context_counts and a.learner.source_verifications


def test_causal_abstraction_can_reverse_after_opposing_feedback():
    l = OutcomeLearner()
    cue = "pair:mode=warm|dependency=stable"
    for i in range(30):
        l.learn_context_outcome(frozenset({cue}), "isolate", True,
                                evidence_id=f"causal-{i}")
    strong = l.contextual_support(frozenset({cue}), "isolate")["score"]
    for i in range(160):
        l.learn_context_outcome(frozenset({cue}), "isolate", False,
                                evidence_id=f"contradict-{i}")
    weakened = l.contextual_support(frozenset({cue}), "isolate")["score"]
    assert strong > 0 and weakened < strong
    assert len(l.context_evidence_ids[(cue, "isolate")]) <= 12
    assert ("pair:unrelated", "isolate") not in l.context_evidence_ids


def test_evidence_based_attention_is_learned_from_predictive_value():
    l = OutcomeLearner()
    relevant = "pair:mode=warm|dependency=stable"
    irrelevant = "pair:context=day|source=Aster"
    for i in range(80):
        success = i % 2 == 0
        a = relevant if success else "pair:mode=cold|dependency=unstable"
        l.learn_context_outcome(frozenset({a, irrelevant}), "purge",
                                success, evidence_id=f"feedback-{i}")
    focus = l.select_context_focus(
        frozenset({relevant, irrelevant}), learned=True, budget=1)
    assert focus == frozenset({relevant})


def test_procedural_epoch_reversal_is_learnable_and_historical_evidence_survives():
    world = ContextualWorld(23)
    later = next(world.task(i) for i in range(500, 600)
                 if world.task(i).mode == "cold")
    assert later.protocol_epoch == "v2"
    earlier = replace(later, protocol_epoch="v1")
    assert any(world.outcome(later, p) != world.outcome(earlier, p)
               for p in world.PROCEDURES)
    learned = OutcomeLearner()
    for n in range(25):
        cues = world.relational_cues(earlier.public(),
                                    dependency=earlier.actual_dependency)
        learned.learn_context_outcome(
            cues, "isolate", world.outcome(earlier, "isolate"),
            evidence_id=f"original-{n}")
    original = dict(learned.context_counts)
    for n in range(25):
        cues = world.relational_cues(later.public(),
                                    dependency=later.actual_dependency)
        learned.learn_context_outcome(
            cues, "isolate", world.outcome(later, "isolate"),
            evidence_id=f"changed-{n}")
    assert all(key in learned.context_counts for key in original)
    assert any("epoch=v2" in cue for cue, _ in learned.context_counts)


def test_refiner_preserves_thought_identity_across_observation_and_stop():
    learned = OutcomeLearner()
    refiner = ProgressiveRefiner(learned)
    state = TransferState(episode_id="single-cognitive-thread")
    resolve = Action("resolve", "isolate", "procedure")
    other = Action("resolve", "purge", "procedure")
    inspect = Action("inspect_dependency", "pump-1", "dependency")
    scored = [(0.02, resolve, [], None), (0.01, other, [], None),
              (0.0, inspect, [], None)]
    context = {
        "cognitive_mode": "contextual", "audits": {},
        "attention_cues": frozenset({"pair:mode=warm|dependency=stable"}),
        "verified_dependency": False, "consequence": 1.0,
        "source_reliability": .5, "inspection_cost": .25,
    }
    first, trace = refiner.refine_contextual(state, scored, [], context=context)
    assert first[0][1].kind == "inspect_dependency"
    assert trace["decision_basis"] == "verify_uncertain_dependency"
    context["verified_dependency"] = True
    second, revision = refiner.refine_contextual(state, scored, [], context=context)
    assert second[0][1].kind == "resolve"
    assert revision["thought_id"] == trace["thought_id"]
    assert revision["version"] > trace["version"]


def test_one_pass_and_reflection_use_identical_learned_focus_evidence():
    a = ScalingRun(17, "adaptive_focus", rounds=1, candidate_budget=24)
    b = ScalingRun(17, "learned_focus_one_pass", rounds=1, candidate_budget=24)
    a.add_experiences(30)
    b.add_experiences(30)
    task = ContextualWorld(8017).task(0, heldout=True)
    state_a = TransferState(episode_id="equal-task")
    state_b = TransferState(episode_id="equal-task")
    a._public_observation(task, "equal-task")
    b._public_observation(task, "equal-task")
    _, trace_a = a._decide(
        task, state_a, dependency=task.reported_dependency,
        verified=False, decision_number=0)
    _, trace_b = b._decide(
        task, state_b, dependency=task.reported_dependency,
        verified=False, decision_number=0)
    assert trace_a["focus_cues"] == trace_b["focus_cues"]
    assert trace_a["retrieved_chunk_ids"] == trace_b["retrieved_chunk_ids"]
    assert trace_a["header_evaluations"] == trace_b["header_evaluations"]
    assert trace_b["thought"] is None


def test_bounded_retrieval_activates_same_live_selector_under_noise():
    a = ScalingRun(23, "adaptive_focus", rounds=2, candidate_budget=24)
    a.add_experiences(40)
    evaluated = a.heldout(0, 250, complexity=3)
    assert evaluated["stored_memories"] >= 250
    assert 0 <= evaluated["completion_rate"] <= 1
    assert evaluated["mean_header_evaluations"] <= 24
    assert len(evaluated["tasks"]) == 2
    assert all(len(t["traces"]) <= 3 for t in evaluated["tasks"])


def test_factorial_tasks_frozen_across_checkpoints_and_loads(tmp_path):
    result = run_scaling_experiment(
        tmp_path, seeds=(7,), stages=(20, 40), memory_targets=(10, 40),
        complexities=(1, 3),
        conditions=("untrained_structural", "strong_one_pass",
                    "adaptive_focus", "bounded_without_learned_focus"),
        rounds=2, candidate_budget=24,
    )
    assert result["experiment"] == 8
    assert result["design"] == "factorial_shared_heldout_context"
    records = json.loads((tmp_path / "scaling_trials.json").read_text())
    assert len(records) == 2 * 2 * 2 * 4
    by_seed = {}
    for record in records:
        assert 0 <= record["completion_rate"] <= 1
        assert record["stored_memories"] >= record["memory_target"]
        key = record["seed"], record["complexity"]
        tasks = tuple(t["task_id"] for t in record["tasks"])
        assert by_seed.setdefault(key, tasks) == tasks
    assert result["paired_focus_vs_onepass"]["20"]["bootstrap_95_interval"] is None


def test_real_restart_preserves_learned_rules_and_continues_learning(tmp_path):
    agent = ScalingRun(61, "adaptive_focus", rounds=2)
    agent.add_experiences(20)
    path = tmp_path / "agent.pkl"
    path.write_bytes(pickle.dumps(agent))
    program = """
import json, pickle, sys
with open(sys.argv[1], 'rb') as f:
    a = pickle.load(f)
before = len(a.memory.events)
previous = dict(a.learner.context_counts)
a.add_experiences(25)
assert a.experiences == 25 and len(a.memory.events) > before
assert all(k in a.learner.context_counts for k in previous)
result = a.heldout(1, 20, complexity=2)
print(json.dumps({"experiences": a.experiences, "completion": result["completion_rate"]}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", program, str(path)],
        capture_output=True, text=True, check=True)
    assert json.loads(proc.stdout)["experiences"] == 25


@pytest.mark.parametrize("stages,memory,complexities", [
    ((100, 20), (20,), (2,)),
    ((0,), (20,), (2,)),
    ((20,), (-1,), (2,)),
    ((20,), (20,), (4,)),
])
def test_invalid_factorial_design_fails(tmp_path, stages, memory, complexities):
    with pytest.raises(ValueError):
        run_scaling_experiment(
            tmp_path, seeds=(7,), stages=stages,
            memory_targets=memory, complexities=complexities,
            conditions=("adaptive_focus",),
        )
