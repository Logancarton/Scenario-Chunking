"""Resource safety: no implicit process pools on the E12 development machine."""
import pytest

from scenario_chunking import causal_experiment as ce
from scenario_chunking import causal_metacognition as cm
from scenario_chunking import causal_representation as cr


def forbidden_pool(*args, **kwargs):
    raise AssertionError("implicit multi-process work")


def test_causal_baseline_defaults_to_inline(monkeypatch):
    monkeypatch.setattr(ce, "ProcessPoolExecutor", forbidden_pool)
    monkeypatch.setattr(ce, "_execute", lambda job: [job])
    assert ce._map_jobs([("a",), ("b",)], None) == [[("a",)], [("b",)]]


def test_metacognition_defaults_to_inline(monkeypatch):
    monkeypatch.setattr(cm, "ProcessPoolExecutor", forbidden_pool)
    monkeypatch.setattr(cm, "run_seed", lambda job: [job])
    assert cm._map([("a",), ("b",)], None) == [[("a",)], [("b",)]]


def test_representation_defaults_to_inline(monkeypatch):
    monkeypatch.setattr(cr, "ProcessPoolExecutor", forbidden_pool)
    monkeypatch.setattr(cr, "run_job", lambda job: {"input": job})
    # Empty comparison axes still exercise the actual job dispatch, not fitting.
    result = cr.run_study(seeds=(), worlds=(), diversities=(), learners=(), workers=None)
    assert result["runs"] == []


@pytest.mark.parametrize("dispatcher", [ce._map_jobs, cm._map])
def test_invalid_worker_counts_rejected(dispatcher):
    with pytest.raises(ValueError):
        dispatcher([("a",), ("b",)], 0)
