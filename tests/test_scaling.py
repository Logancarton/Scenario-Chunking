"""Behavioral gates for progressive experience accumulation and bounded focus."""
from __future__ import annotations

import json
import pickle
import subprocess
import sys

import pytest

from scenario_chunking.scaling import ScalingRun, run_scaling_experiment


def test_training_is_incremental_and_matched_without_truth_in_policy():
    a = ScalingRun(7, "trained_association", rounds=2)
    b = ScalingRun(7, "adaptive_focus", rounds=2)
    a.add_experiences(20)
    initial_ids = tuple(a.memory.chunk_order)
    first = tuple(a.training_observations)
    a.add_experiences(100)
    b.add_experiences(100)
    assert a.experiences == b.experiences == 100
    assert len(a.training_observations) == len(b.training_observations) == 100
    assert a.training_observations == b.training_observations
    assert tuple(a.memory.chunk_order[:20]) == initial_ids
    assert a.training_observations[:20] == list(first)
    assert a.learner.source_verifications


def test_focus_limits_candidate_headers_and_does_not_evaluate_private_answers():
    run = ScalingRun(23, "adaptive_focus", rounds=2, candidate_budget=32)
    run.add_experiences(20)
    result = run.heldout(0, 250)
    assert result["stored_memories"] >= 250
    assert 0 <= result["completion_rate"] <= 1
    assert result["mean_header_evaluations"] <= 32
    assert result["focus_axis_count"] <= 2
    assert result["mean_active_details"] >= 0


def test_experience_and_memory_size_separately_controlled_and_output(tmp_path):
    payload = run_scaling_experiment(
        tmp_path, seeds=(7,), stages=(20, 40), memory_targets=(100, 220),
        conditions=("untrained_structural", "strong_one_pass", "adaptive_focus",
                    "bounded_without_learned_focus"),
        rounds=2, candidate_budget=32,
    )
    assert payload["experiment"] == 8
    assert tuple(payload["stage_results"]) == ("20", "40")
    rows = json.loads((tmp_path / "scaling_trials.json").read_text())
    assert len(rows) == 8
    for row in rows:
        assert row["stored_memories"] >= row["memory_target"]
        assert 0 <= row["completion_rate"] <= 1
    # Stage-two knowledge must not rewind to the initial checkpoint.
    second = [r for r in rows if r["stage"] == 1]
    assert all(r["training_experiences"] == 40
               for r in second if r["treatment"] != "untrained_structural")


def test_separate_process_restart_continues_feedback_learning(tmp_path):
    run = ScalingRun(61, "adaptive_focus", rounds=2)
    run.add_experiences(20)
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(run))
    script = """
import json, pickle, sys
p = sys.argv[1]
with open(p, 'rb') as f:
    a = pickle.load(f)
before = len(a.memory.events)
prior = dict(a.learner.source_verifications)
a.add_experiences(25)
assert a.experiences == 25
assert len(a.memory.events) > before
assert all(len(a.learner.source_verifications[k]) >= len(v) for k, v in prior.items())
print(json.dumps({"experiences": a.experiences, "events": len(a.memory.events)}))
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(process.stdout)["experiences"] == 25


@pytest.mark.parametrize("stages,memories", [
    ((100, 20), (200, 200)),
    ((20, 100), (100,)),
    ((0,), (100,)),
])
def test_invalid_stage_design_rejected(tmp_path, stages, memories):
    with pytest.raises(ValueError):
        run_scaling_experiment(
            tmp_path, seeds=(7,), stages=stages, memory_targets=memories,
            conditions=("adaptive_focus",),
        )
