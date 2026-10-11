"""Experiment 11: prequential learning across once-seen minimally varied worlds.

All private causal truth belongs to IncrementalDispatchWorld. This file owns
the curriculum, matched controls, observational ingestion, and evaluation
only. Learning, interpretation, routing, thought, and selection use E10 owners.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import pickle
import random
import statistics

from .environment import IncrementalDispatchWorld
from .learning import OutcomeLearner
from .meaning import MeaningInterpreter
from .memory import EpisodicMemory
from .models import Event, EventKind, TruthStatus, RetrievalMode
from .policy import ActionSelector
from .refinement import ProgressiveRefiner
from .retrieval import MemoryRouter
from .state import CognitiveState

CONDITIONS = (
    "frozen", "episodic_only", "strong_one_pass",
    "incremental_full", "no_conditional_interactions",
)
FAMILIES = ("delivery", "repair", "coordination")
AXES = ("person", "delay", "urgency", "capacity", "channel")
VALUES = (
    ("Alex", "Jordan"), ("short", "long"), ("low", "high"),
    ("open", "limited"), ("day", "night"),
)
COSTS = {"standard": .02, "express": .18}


def experience(seed: int, family: str, index: int) -> tuple[IncrementalDispatchWorld, frozenset[str]]:
    """Gray-code neighbors differ in exactly one consequential public axis.

    No exact scenario repeats. Marker changes once as an irrelevant control;
    world epoch changes once as an independently observed causal shift.
    """
    if family not in FAMILIES or not 0 <= index < 32:
        raise ValueError("invalid family or distinct experience")
    gray = index ^ (index >> 1)
    factors = {name: values[(gray >> axis) & 1]
               for axis, (name, values) in enumerate(zip(AXES, VALUES))}
    factors["marker"] = "blue" if index < 16 else "orange"
    epoch = int(family == "coordination" and index >= 16)
    world = IncrementalDispatchWorld.scenario(
        seed=seed, family=family, index=index, factors=factors, epoch=epoch)
    # All one-axis and all two-axis combinations are constructed symmetrically:
    # no oracle declares which relationships are useful.
    primitive = {f"pair:{axis}:{value}" for axis, value in factors.items()}
    primitive |= {f"pair:domain:{family}", f"pair:epoch:{epoch}"}
    relations = {
        left + "&" + right
        for i, left in enumerate(sorted(primitive))
        for right in sorted(primitive)[i + 1:]
    }
    return world, frozenset(primitive | relations)


class IncrementalRun:
    """Thin live-path sequencing; each experience is ingested, acted, learned once."""

    def __init__(self, seed: int, mode: str, *, top_k: int = 12):
        if mode not in CONDITIONS:
            raise ValueError("unknown E11 baseline")
        self.seed, self.mode = seed, mode
        self.memory = EpisodicMemory()
        self.learner = OutcomeLearner(enabled=mode not in ("frozen", "episodic_only"))
        self.interpreter = MeaningInterpreter(self.memory, self.learner)
        self.router = MemoryRouter(self.memory, RetrievalMode.STRUCTURAL,
                                   top_k=top_k, candidate_budget=48)
        self.refiner = ProgressiveRefiner(self.learner)
        self.selector = ActionSelector(self.memory, self.learner,
                                       random.Random(seed), refiner=self.refiner)
        self.trace = []
        self.seen = set()

    def _event(self, state: CognitiveState, *, kind: EventKind, status: TruthStatus,
               content: str, tags: frozenset[str], action=None, outcome=None,
               cause=None, reward: float = 0.0, metadata=None) -> Event:
        event = Event(
            event_id=self.memory.next_event_id(), episode_id=state.episode_id,
            goal_id=state.goal_id, step=int(state.current_time),
            occurrence_time=state.current_time, acquired_time=state.current_time,
            storage_time=state.current_time, kind=kind, truth_status=status,
            content=content, semantic_tags=tags,
            action=action, outcome=outcome, cause_event_id=cause,
            reward=reward, metadata=metadata or {},
        )
        self.memory.create_chunk(
            events=[event], transition_reason="e11_observed_evidence",
            self_relevance=.8, salience=.7)
        return event

    def step(self, family: str, index: int) -> dict:
        identity = (family, index)
        if identity in self.seen:
            raise ValueError("cannot repeat training on a scenario")
        self.seen.add(identity)
        world, cues = experience(self.seed, family, index)
        state = CognitiveState(f"e11:{self.seed}:{family}:{index}",
                               goal_id="response_dispatch")
        clock = len(self.seen)
        state.incremental_context(cues, float(clock))
        self.learner.begin_episode()
        self._event(state, kind=EventKind.OBSERVATION, status=TruthStatus.FACT,
                    content="Received dispatch request with public situational cues",
                    tags=cues | frozenset({"goal:response_dispatch"}))
        actions = state.candidate_actions()
        hits, stats = self.router.retrieve(state.retrieval_query())
        model_cues = (frozenset(c for c in cues if "&" not in c)
                      if self.mode == "no_conditional_interactions" else cues)
        if self.mode == "episodic_only":
            interpreted = self.interpreter.incremental_episodic_values(
                hits, cues, actions, COSTS)
        else:
            interpreted = self.interpreter.incremental_action_values(
                model_cues, actions, costs=COSTS)
        values, predictions = interpreted["values"], interpreted["predictions"]
        # Same learned information and option values for full and one pass.
        decision = self.selector.select(
            state, actions, hits, refine=self.mode in (
                "incremental_full", "no_conditional_interactions"),
            transfer_scores={label: max(-.45, min(.45, v / 2))
                             for label, v in values.items()},
            transfer_context={
                "cognitive_mode": "incremental",
                "candidate_values": values,
                "predictions": predictions,
                "risk_price": .28,
            },
        )
        chosen = decision.action
        forecast = predictions[chosen.label]
        # Causal action identity is recorded before external feedback.
        attempt = self._event(
            state, kind=EventKind.ACTION, status=TruthStatus.ATTEMPT,
            content="Dispatch attempt " + chosen.label,
            tags=cues | frozenset({"goal:response_dispatch", "action:route"}),
            action=chosen)
        self.learner.register_action(
            attempt.event_id, state.action_features(chosen),
            used_evidence_ids=forecast["evidence_ids"])
        observed = world.execute(chosen)
        outcome = self._event(
            state, kind=EventKind.OUTCOME, status=TruthStatus.OUTCOME,
            content=observed.content, tags=observed.semantic_tags,
            outcome=observed.outcome, cause=attempt.event_id,
            reward=observed.reward, metadata=observed.metadata)
        state.apply_signal(observed, event_id=outcome.event_id)
        state.record_action(chosen)
        # Only observed execution feedback changes learned knowledge.
        update = self.learner.incremental_feedback(
            model_cues, chosen.target, success=observed.outcome == "success",
            evidence_id=outcome.event_id)
        if observed.outcome == "success":
            self.learner.apply_success(attempt.event_id, reward=observed.reward)
        else:
            self.learner.apply_failure(attempt.event_id, reward=observed.reward)
        probability = forecast["probability"]
        # Counterfactual private world truth is evaluation-only, after action.
        best_reward = max(world.counterfactual_reward(a.target) for a in actions)
        row = {
            "seed": self.seed, "family": family, "index": index,
            "stage": "heldout" if index >= 24 else ("early" if index < 8 else "middle" if index < 16 else "late"),
            "mode": self.mode, "context": sorted(cues),
            "retrieved_chunks": [hit.chunk_id for hit in hits],
            "activated_evidence": list(forecast["evidence_ids"]),
            "prediction": probability, "uncertainty": forecast["uncertainty"],
            "candidate_probabilities": {k: v["probability"] for k, v in predictions.items()},
            "action": chosen.label, "outcome": observed.outcome,
            "reward": observed.reward, "regret": best_reward - observed.reward,
            "brier": (probability - float(observed.outcome == "success")) ** 2,
            "belief_update": update,
            "updated_relations": update["updated"],
            "thought": decision.thought_trace,
            "detail_events": stats.active_detail_events,
            "header_evaluations": stats.header_evaluations,
            "retrieval_ms": stats.latency_ms,
            "cognitive_operations": 2 if decision.thought_trace else 1,
        }
        self.trace.append(row)
        return row

    def run(self, *, families=FAMILIES, length: int = 32) -> list[dict]:
        if not 0 <= length <= 32:
            raise ValueError("length out of range")
        for family in families:
            for index in range(length):
                self.step(family, index)
        return self.trace


def run_incremental_benchmark(*, seeds=(7, 23, 61, 101, 149),
                              families=FAMILIES, length: int = 32) -> dict:
    trials = []
    for seed in seeds:
        for mode in CONDITIONS:
            run = IncrementalRun(seed, mode)
            # Each policy sees the same distinct experiences once. Outcome
            # draws are paired per action, not generated from policy RNG.
            trials.extend(run.run(families=families, length=length))
    aggregate = {}
    for mode in CONDITIONS:
        aggregate[mode] = {}
        for stage in ("early", "middle", "late", "heldout"):
            group = [r for r in trials if r["mode"] == mode and r["stage"] == stage]
            if not group:
                continue
            aggregate[mode][stage] = {
                "n": len(group),
                "mean_reward": statistics.mean(r["reward"] for r in group),
                "success_rate": statistics.mean(r["outcome"] == "success" for r in group),
                "brier": statistics.mean(r["brier"] for r in group),
                "mean_regret": statistics.mean(r["regret"] for r in group),
                "mean_details": statistics.mean(r["detail_events"] for r in group),
                "mean_headers": statistics.mean(r["header_evaluations"] for r in group),
                "mean_cognitive_operations": statistics.mean(
                    r["cognitive_operations"] for r in group),
                "mean_retrieval_ms": statistics.mean(r["retrieval_ms"] for r in group),
                "mean_belief_updates": statistics.mean(
                    len(r["updated_relations"]) for r in group),
            }
    deltas = []
    for seed in seeds:
        sample = {
            mode: [r["reward"] for r in trials if r["seed"] == seed
                   and r["mode"] == mode and r["stage"] == "heldout"]
            for mode in ("incremental_full", "strong_one_pass",
                         "episodic_only", "no_conditional_interactions")
        }
        if sample["incremental_full"]:
            deltas.append({
                "seed": seed,
                **{other: statistics.mean(sample["incremental_full"]) -
                   statistics.mean(sample[other])
                   for other in sample if other != "incremental_full"},
            })
    return {
        "experiment": 11, "seeds": list(seeds), "families": list(families),
        "distinct_per_family": length, "conditions": list(CONDITIONS),
        "summary": aggregate, "paired_seed_deltas": deltas, "trials": trials,
        "limitation": (
            "Synthetic causal rules and selection-biased bandit feedback; "
            "a one-exposure stream does not establish human-like learning. "
            "No oracle counterfactual is provided to cognition."),
    }


def write_incremental_benchmark(output="results-e11", **kwargs) -> dict:
    payload = run_incremental_benchmark(**kwargs)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "trials"},
                   indent=2, sort_keys=True), encoding="utf-8")
    (destination / "trials.json").write_text(
        json.dumps(payload["trials"], indent=2, sort_keys=True),
        encoding="utf-8")
    return payload
