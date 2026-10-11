"""Experiment 8: cumulative learning versus sparse focus and memory interference.

Research orchestration only. The existing environment, episodic memory, router,
learner, interpreter, selector and refiner make every live decision. Held-out
private truth is consulted solely by the environment and metric calculation.
"""
from __future__ import annotations

from collections import defaultdict
import json
import pickle
import random
import statistics
from pathlib import Path

from .environment import AXES, UncertaintyWorld
from .models import Action, EventKind, TruthStatus
from .uncertainty import UncertaintyRun

# Exactly the same training observations are replayed across trained treatments.
CONDITIONS_8 = {
    "untrained_structural": "untrained_structural",
    "trained_association": "trained_structural",
    "interpreted_no_reflection": "full_no_reflection",
    "strong_one_pass": "strong_one_pass",
    "limited_reflection": "limited_reflection",
    "extended_reflection": "extended_reflection",
    "adaptive_focus": "limited_reflection",
    "bounded_without_learned_focus": "limited_reflection",
    "larger_retrieval": "larger_retrieval",
    "shuffled_evidence": "shuffled_evidence",
}
DEFAULT_STAGES = (20, 100, 500, 2000, 10000)
DEFAULT_MEMORY = (100, 1000, 5000, 20000, 20000)


class ScalingRun(UncertaintyRun):
    """One persistent learner and one persistent memory per seed/treatment."""

    def __init__(self, seed: int, treatment: str, *, rounds: int = 4,
                 action_budget: int = 9, candidate_budget: int = 128) -> None:
        if treatment not in CONDITIONS_8:
            raise ValueError(f"unknown scaling treatment {treatment}")
        super().__init__(seed, CONDITIONS_8[treatment], rounds=rounds,
                         distractors=0, action_budget=action_budget)
        self.treatment = treatment
        self.experiences = 0
        self.candidate_budget = candidate_budget
        self.training_observations: list[dict] = []
        self._last_retrieval_stats = None
        self._focus_axes = tuple(AXES)
        if treatment in ("adaptive_focus", "bounded_without_learned_focus"):
            self.router.candidate_budget = candidate_budget

    def add_experiences(self, target: int) -> None:
        """Generate observed, feedback-bearing trials without resetting knowledge."""
        if target < self.experiences:
            raise ValueError("training checkpoints must increase")
        rng = random.Random(self.seed * 911 + 19)
        # Advance deterministic draws to maintain identical data at all checkpoints.
        for trial in range(target):
            axis = AXES[trial % len(AXES)]
            claim = self.world.reports(self.seed, trial, axis)[trial % 3]
            candidate = (claim["token"] if trial % 4 != 0
                         else self.world.options[axis][rng.randrange(4)])
            if trial < self.experiences:
                continue
            accepted = self.world.accepted(axis, candidate)
            if self.treatment != "untrained_structural":
                self.learner.begin_episode()
                self._feedback(
                    f"scale-training-{trial}", trial,
                    Action("resolve", candidate, axis), accepted,
                    source="observed_training_trial", person=claim["source"],
                )
                # A claim is checked only when the action actually tests it.
                if candidate == claim["token"]:
                    self.learner.verify_source(claim["source"],
                                               claim["context"], accepted)
                self.training_observations.append({
                    "trial": trial, "axis": axis, "candidate": candidate,
                    "source": claim["source"], "context": claim["context"],
                    "observed_success": accepted,
                })
            self.experiences += 1

    def _decision(self, state, actions, axes, *, round_no: int, step: str):
        if self.treatment == "adaptive_focus":
            remaining = tuple(axis for axis in axes if axis not in state.completed_axes)
            chosen = self.learner.focus_axes(
                remaining, state.completed_axes,
                invalidated=set(state.invalidated_since), limit=2)
            self._focus_axes = chosen or remaining
            axes = self._focus_axes
        else:
            self._focus_axes = tuple(axes)
        decision, trace = super()._decision(
            state, actions, axes, round_no=round_no, step=step)
        trace["focus_axes"] = list(self._focus_axes)
        if self._last_retrieval_stats is not None:
            stats = self._last_retrieval_stats
            trace["candidate_headers"] = stats.candidate_chunks
            trace["header_evaluations"] = stats.header_evaluations
            trace["unnecessary_activations"] = stats.unnecessary_activations
            trace["total_chunks"] = stats.total_chunks
        return decision, trace

    def _retrieve(self, axes, *, top_k=None):
        hits, stats = super()._retrieve(axes, top_k=top_k)
        self._last_retrieval_stats = stats
        return hits, stats

    def heldout(self, stage: int, memory_target: int, *,
                familiar: bool = False) -> dict:
        """Evaluation is forked so test-time success cannot leak into training."""
        agent = pickle.loads(pickle.dumps(self))
        if not familiar:
            # Different hidden mappings and novel object/person combinations.
            agent.world = UncertaintyWorld(self.seed * 7919 + 17 + stage)
        agent.source_predictions = []
        agent.traces = []
        agent._resolved_change_axes = set()
        agent._current_reports = {}
        agent._last_retrieval_stats = None
        agent.cost = defaultdict(float)
        count_before = len(agent.memory.chunks)
        target = max(count_before, memory_target)
        extra = target - count_before
        # Control for raw memory size independently of training. Add adversarial
        # same-goal noise AND unrelated content; never label it causal feedback.
        same_goal_noise = extra // 4
        for i in range(same_goal_noise):
            axis = AXES[i % len(AXES)]
            signal = agent.world.options[axis][i % 4]
            event = agent._event(
                f"scale-noise-{stage}-{i}", i,
                kind=EventKind.OBSERVATION, truth=TruthStatus.FACT,
                content="Unverified maintenance rumor",
                tags={"goal:workshop_recovery", f"axis:{axis}",
                      f"signal:{axis}:{signal}", "source:testimony"},
                informative=False, confidence=.1,
                metadata={"source_status": "unverified"},
            )
            agent.memory.create_chunk(
                events=[event], transition_reason="adversarial_noise",
                self_relevance=.08, salience=.12)
        agent.memory.inject_distractors(
            extra - same_goal_noise, start_time=agent.clock + 1,
            goal_id="unrelated_goal")
        start_cost = dict(agent.cost)
        rounds = [agent._execute_round(i) for i in range(agent.rounds)]
        deltas = {key: value - start_cost.get(key, 0)
                  for key, value in agent.cost.items()}
        trace = agent.traces
        samples = [t for t in trace if t.get("retrieved_chunk_ids") is not None]
        # Proxies: matching an available axis cue does not prove causal relevance.
        # Evaluate action success separately and NEVER equate retrieval with reasoning.
        active = [t for t in samples if t.get("detail_events") is not None]
        correct = sum(int(x["success"]) for x in rounds)
        return {
            "seed": self.seed, "treatment": self.treatment,
            "training_experiences": self.experiences if self.treatment != "untrained_structural" else 0,
            "stage": stage, "familiar": familiar,
            "memory_target": memory_target, "stored_memories": len(agent.memory.chunks),
            "rounds": len(rounds), "completed_rounds": correct,
            "completion_rate": correct / len(rounds),
            "first_action_accuracy": statistics.mean(float(r["first_action_correct"])
                                                      for r in rounds),
            "mean_actions": statistics.mean(r["actions"] for r in rounds),
            "mean_wrong_actions": statistics.mean(r["wrong_decisions"] for r in rounds),
            "mean_inspections": statistics.mean(r["verification_actions"] for r in rounds),
            "mean_active_details": statistics.mean(t["detail_events"] for t in active) if active else 0.0,
            "mean_header_evaluations": statistics.mean(
                t.get("header_evaluations", 0) for t in samples) if samples else 0.0,
            "mean_unused_activation": statistics.mean(
                t.get("unnecessary_activations", 0) for t in samples) if samples else 0.0,
            "mean_selection_ms": deltas.get("selection_ms", 0) / max(1, len(samples)),
            "mean_retrieval_ms": deltas.get("retrieval_ms", 0) / max(1, len(samples)),
            "interpretation_evaluations": deltas.get("interpretation_evaluations", 0),
            "counterfactual_evaluations": deltas.get("counterfactual_evaluations", 0),
            "reflection_action_changes": sum(bool(t.get("reflection_changed_action"))
                                             for t in trace),
            "focus_axis_count": statistics.mean(
                len(t.get("focus_axes", ())) for t in samples) if samples else 0.0,
            "round_successes": [bool(r["success"]) for r in rounds],
        }


def _paired_interval(deltas: list[float], seed: int = 8008) -> list[float] | None:
    if len(deltas) < 2:
        return None
    rng = random.Random(seed)
    sample = sorted(statistics.mean(deltas[rng.randrange(len(deltas))]
                                    for _ in deltas) for _ in range(2000))
    return [sample[49], sample[1949]]


def run_scaling_experiment(
    output_dir: str | Path, *, seeds=tuple(range(1, 31)),
    stages=DEFAULT_STAGES, memory_targets=DEFAULT_MEMORY,
    conditions=tuple(CONDITIONS_8), rounds=4,
    action_budget=9, candidate_budget=128,
) -> dict:
    """Cumulative training, stage-by-stage held-out evaluation, paired conditions."""
    if not seeds:
        raise ValueError("at least one seed required")
    if not stages or len(stages) != len(memory_targets):
        raise ValueError("each training stage requires a memory target")
    if any(n <= 0 for n in stages) or sorted(stages) != list(stages):
        raise ValueError("training stages must be positive and increasing")
    if any(n <= 0 for n in memory_targets):
        raise ValueError("memory targets must be positive")
    if any(c not in CONDITIONS_8 for c in conditions):
        raise ValueError("invalid treatment")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        agents = {c: ScalingRun(seed, c, rounds=rounds,
                                action_budget=action_budget,
                                candidate_budget=candidate_budget)
                  for c in conditions}
        for stage, (experiences, target) in enumerate(zip(stages, memory_targets)):
            for treatment, agent in agents.items():
                agent.add_experiences(experiences)
                result = agent.heldout(stage, target, familiar=False)
                runs.append(result)
    summary = {}
    for stage, experiences in enumerate(stages):
        summary[str(experiences)] = {}
        for name in conditions:
            group = [r for r in runs if r["stage"] == stage and r["treatment"] == name]
            summary[str(experiences)][name] = {
                "completion_rate": statistics.mean(r["completion_rate"] for r in group),
                "first_action_accuracy": statistics.mean(r["first_action_accuracy"] for r in group),
                "mean_actions": statistics.mean(r["mean_actions"] for r in group),
                "mean_detail_reads": statistics.mean(r["mean_active_details"] for r in group),
                "mean_header_evaluations": statistics.mean(r["mean_header_evaluations"] for r in group),
                "mean_retrieval_ms": statistics.mean(r["mean_retrieval_ms"] for r in group),
                "mean_processing_evaluations": statistics.mean(
                    r["interpretation_evaluations"] for r in group),
                "mean_memories": statistics.mean(r["stored_memories"] for r in group),
                "by_seed": [{"seed": r["seed"], "completion": r["completion_rate"],
                             "actions": r["mean_actions"]} for r in group],
            }
    paired = {}
    if "adaptive_focus" in conditions and "strong_one_pass" in conditions:
        for stage, experiences in enumerate(stages):
            a = {(r["seed"], r["stage"]): r["completion_rate"] for r in runs
                 if r["treatment"] == "adaptive_focus" and r["stage"] == stage}
            b = {(r["seed"], r["stage"]): r["completion_rate"] for r in runs
                 if r["treatment"] == "strong_one_pass" and r["stage"] == stage}
            deltas = [a[k] - b[k] for k in sorted(a)]
            paired[str(experiences)] = {"seed_deltas": deltas,
                                        "bootstrap_95_interval": _paired_interval(deltas)}
    payload = {
        "experiment": 8, "seeds": list(seeds), "stages": list(stages),
        "memory_targets": list(memory_targets), "conditions": list(conditions),
        "stage_results": summary, "paired_focus_vs_onepass": paired,
        "limitations": [
            "These are synthetic role/category worlds; no natural-language reasoning claim.",
            "Held-out worlds change mappings and scenes but reuse the same five abstract roles.",
            "Reflection depth is fixed within each inherited Experiment 7 treatment; "
            "adaptive_focus learns category priority, not an outcome-trained stopping rule.",
            "Retrieval efficiency is separate from decision accuracy; report both.",
            "Some memory targets are raised to the already-accumulated training memory size.",
            "Paired models receive the same environment distribution and experience sequence, "
            "but baseline retrieval/interpretation compute is measured, not strictly matched.",
            "Durability by separate-process restart requires an additional independent gate.",
        ],
    }
    (output / "scaling_trials.json").write_text(
        json.dumps(runs, indent=2, sort_keys=True), encoding="utf-8")
    (output / "scaling_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
