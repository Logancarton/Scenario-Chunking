"""Diagnostic scripts must audit saved E12 evidence without fabricating a cohort."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e12_activation_study import MODES, study_seed  # noqa: E402
from e12_disagreement_audit import audit, disagreements  # noqa: E402


def _sample_report():
    return {
        "study": "e12_computation_sparsity",
        "status": "exploratory",
        "by_seed": {"31": study_seed(31, pretrain=1, adapt=1, heldout=2)},
        "pretrain": 1, "adapt": 1, "heldout": 2,
    }


def test_disagreement_audit_requires_existing_action_level_study():
    with pytest.raises(ValueError, match="existing E12"):
        disagreements({})
    report = _sample_report()
    assert all(m in report["by_seed"]["31"] for m in MODES)
    summary = audit(report, max_seeds=1, max_cases=8)
    assert summary["input_seed_count"] == 1
    assert summary["selected_case_count"] <= 8
    assert len(summary["replays"]) == summary["selected_case_count"]
    for case in summary["replays"]:
        assert case["live"]["fixed_sparse"]["action"] != case["live"]["learned_sparse"]["action"]
        assert set(case["same_learned_state_activation_counterfactuals"]) == {
            "fixed_sparse", "dense"}
        assert not case["live"]["learned_sparse"]["process_updates"]


def test_disagreement_audit_rejects_replay_divergence_instead_of_inventing_cause():
    report = _sample_report()
    modified = deepcopy(report)
    task = "heldout_delivery"
    actions = modified["by_seed"]["31"]
    fixed = actions["fixed_sparse"][task]["actions"][0]
    learned = actions["learned_sparse"][task]["actions"][0]
    actions["learned_sparse"][task]["actions"][0] = next(
        a for a in ("standard", "express", "defer") if a not in (fixed, learned))
    disagreements(modified)  # A synthetic mismatch is identifiable in the saved rows.
    with pytest.raises(ValueError, match="replay disagreement"):
        audit(modified, max_seeds=1, max_cases=100)
