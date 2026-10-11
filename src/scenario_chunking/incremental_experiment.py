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
from .spreading import SpreadBudget
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
        self.router = MemoryRouter(
            self.memory, RetrievalMode.SPREADING, top_k=top_k,
            candidate_budget=48, spread_budget=SpreadBudget(
                max_additional_chunks=4, max_neighbor_headers=32,
                max_total_detail_events=16))
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

    def step(self, family: str, index: int, *, forced_procedure: str | None = None,
             freeze: bool = False) -> dict:
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
        if not freeze:
            self._event(state, kind=EventKind.OBSERVATION, status=TruthStatus.FACT,
                        content="Received dispatch request with public situational cues",
                        tags=cues | frozenset({"goal:response_dispatch"}))
        actions = state.candidate_actions()
        hits, stats = self.router.retrieve(state.retrieval_query())
        # The same observed primitives reach every treatment. Interaction
        # ablation is performed by the existing learner, not by hiding cues.
        model_cues = cues
        allow_interactions = self.mode != "no_conditional_interactions"
        if self.mode == "episodic_only":
            interpreted = self.interpreter.incremental_episodic_values(
                hits, cues, actions, COSTS)
        else:
            interpreted = self.interpreter.incremental_action_values(
                model_cues, actions, costs=COSTS,
                allow_interactions=allow_interactions)
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
        if forced_procedure is not None and forced_procedure not in COSTS:
            raise ValueError("unknown forced logging procedure")
        chosen = next((a for a in actions if a.target == forced_procedure),
                      None) if forced_procedure else decision.action
        forecast = predictions[chosen.label]
        attempt = None
        if not freeze:
            # Action and resulting outcome are historical causal facts.
            attempt = self._event(
                state, kind=EventKind.ACTION, status=TruthStatus.ATTEMPT,
                content="Dispatch attempt " + chosen.label,
                tags=cues | frozenset({"goal:response_dispatch", "action:route"}),
                action=chosen)
            # Do not credit 36 co-occurring features in the general policy
            # learner either; the contextual learner owns specific evidence.
            self.learner.register_action(
                attempt.event_id,
                frozenset({"goal:response_dispatch", "action:route",
                           "procedure:" + chosen.target}),
                used_evidence_ids=forecast["evidence_ids"])
        observed = world.execute(chosen)
        outcome = None
        if not freeze:
            outcome = self._event(
                state, kind=EventKind.OUTCOME, status=TruthStatus.OUTCOME,
                content=observed.content, tags=observed.semantic_tags,
                outcome=observed.outcome, cause=attempt.event_id,
                reward=observed.reward, metadata=observed.metadata)
        state.apply_signal(observed, event_id=outcome.event_id if outcome else None)
        state.record_action(chosen)
        if freeze:
            # Frozen evaluation cannot update statistics, episodic memory,
            # or the state reused by another held-out observation.
            update = {"updated": [], "prediction_error":
                      float(observed.outcome == "success") - forecast["probability"],
                      "belief_id": forecast.get("belief_id"),
                      "prior_probability": forecast["probability"],
                      "posterior_probability": forecast["probability"]}
        else:
            update = self.learner.incremental_feedback(
                model_cues, chosen.target, success=observed.outcome == "success",
                evidence_id=outcome.event_id,
                allow_interactions=allow_interactions)
            if observed.outcome == "success":
                self.learner.apply_success(attempt.event_id, reward=observed.reward)
            else:
                self.learner.apply_failure(attempt.event_id, reward=observed.reward)
            if chosen == decision.action:
                self.refiner.feedback(
                    state, decision, outcome=observed.outcome,
                    outcome_event_ids=(outcome.event_id,))
        probability = forecast["probability"]
        # Counterfactual private world truth is evaluation-only, after action.
        best_reward = max(world.counterfactual_reward(a.target) for a in actions)
        row = {
            "seed": self.seed, "family": family, "index": index,
            "stage": "frozen_heldout" if freeze else (
                "heldout" if index >= 24 else (
                    "early" if index < 8 else "middle" if index < 16 else "late")),
            "evaluation": "frozen" if freeze else "prequential",
            "forced_log_action": forced_procedure is not None,
            "policy_action": decision.action.label,
            "mode": self.mode, "context": sorted(cues),
            "retrieved_chunks": [hit.chunk_id for hit in hits],
            "activated_evidence": list(forecast["evidence_ids"]),
            "prediction": probability, "uncertainty": forecast["uncertainty"],
            "belief_id": forecast.get("belief_id"),
            "concept_id": self.learner.incremental_concept_id(cues),
            "reused_belief_identity": bool(
                decision.thought_trace and
                decision.thought_trace.get("reused_identity")),
            "unchanged_context": (decision.thought_trace or {}).get(
                "unchanged_context", []),
            "changed_context": (decision.thought_trace or {}).get(
                "changed_context", []),
            "supporting_prior_evidence": list(forecast["evidence_ids"]),
            "observed_outcome_id": outcome.event_id if outcome else None,
            "changed_belief": {
                "prior": update.get("prior_probability", probability),
                "posterior": update.get("posterior_probability", probability),
                "updated_relations": update["updated"],
                "credit_mass": update.get("total_credit_mass", 0.0)},
            "next_decision": None,
            "next_prediction": None,
            "candidate_probabilities": {k: v["probability"] for k, v in predictions.items()},
            "action": chosen.label, "outcome": observed.outcome,
            "reward": observed.reward, "regret": best_reward - observed.reward,
            "brier": (probability - float(observed.outcome == "success")) ** 2,
            "belief_update": update,
            "updated_relations": update["updated"],
            "thought": decision.thought_trace,
            "detail_events": stats.active_detail_events,
            "header_evaluations": stats.header_evaluations + stats.neighbor_headers_examined,
            "neighbor_headers_examined": stats.neighbor_headers_examined,
            "retrieval_ms": stats.latency_ms,
            "cognitive_operations": 2 if decision.thought_trace else 1,
        }
        if self.trace and self.trace[-1]["family"] == family:
            self.trace[-1]["next_decision"] = decision.action.label
            self.trace[-1]["next_prediction"] = {
                "candidate_probabilities": {
                    key: value["probability"] for key, value in predictions.items()},
                "preferred": decision.action.label,
                "new_supporting_evidence": list(
                    predictions[decision.action.label]["evidence_ids"]),
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


def _summarize_incremental(trials: list[dict], *, stages: tuple[str, ...]) -> dict:
    aggregate = {}
    for mode in CONDITIONS:
        aggregate[mode] = {}
        for stage in stages:
            group = [row for row in trials
                     if row["mode"] == mode and row["stage"] == stage]
            if not group:
                continue
            aggregate[mode][stage] = {
                "n": len(group),
                "mean_reward": statistics.mean(r["reward"] for r in group),
                "success_rate": statistics.mean(
                    r["outcome"] == "success" for r in group),
                "brier": statistics.mean(r["brier"] for r in group),
                "mean_regret": statistics.mean(r["regret"] for r in group),
                "mean_details": statistics.mean(r["detail_events"] for r in group),
                "mean_headers": statistics.mean(
                    r["header_evaluations"] for r in group),
                "mean_cognitive_operations": statistics.mean(
                    r["cognitive_operations"] for r in group),
                "mean_retrieval_ms": statistics.mean(
                    r["retrieval_ms"] for r in group),
                "mean_belief_updates": statistics.mean(
                    len(r["updated_relations"]) for r in group),
            }
    return aggregate


def _paired_incremental(trials: list[dict], seeds, stage: str) -> list[dict]:
    deltas = []
    for seed in seeds:
        sample = {
            mode: [row["reward"] for row in trials
                   if row["seed"] == seed and row["mode"] == mode
                   and row["stage"] == stage]
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
    return deltas


def run_incremental_benchmark(*, seeds=(7, 23, 61, 101, 149),
                              families=FAMILIES, length: int = 32) -> dict:
    if not 0 <= length <= 32:
        raise ValueError("length out of range")
    trials, frozen_trials = [], []
    for seed in seeds:
        for mode in CONDITIONS:
            online = IncrementalRun(seed, mode)
            trials.extend(online.run(families=families, length=length))
            if length <= 24:
                continue
            # A common logged action schedule supplies identical observed
            # training interventions across treatments. Do not use either
            # policy's preferred action to generate different training labels.
            train = IncrementalRun(seed, mode)
            for family in families:
                for index in range(24):
                    logged = "standard" if (seed + index) % 2 == 0 else "express"
                    train.step(family, index, forced_procedure=logged)
            snapshot = pickle.dumps(train)
            for family in families:
                for index in range(24, length):
                    # Every held-out case sees precisely the SAME pretest
                    # snapshot: no held-out evidence reaches later cases.
                    isolated = pickle.loads(snapshot)
                    row = isolated.step(family, index, freeze=True)
                    frozen_trials.append(row)
    return {
        "experiment": 11, "seeds": list(seeds), "families": list(families),
        "distinct_per_family": length, "conditions": list(CONDITIONS),
        "summary": _summarize_incremental(
            trials, stages=("early", "middle", "late", "heldout")),
        "paired_seed_deltas": _paired_incremental(trials, seeds, "heldout"),
        "frozen_summary": _summarize_incremental(
            frozen_trials, stages=("frozen_heldout",)),
        "frozen_paired_seed_deltas": _paired_incremental(
            frozen_trials, seeds, "frozen_heldout"),
        "trials": trials, "frozen_trials": frozen_trials,
        "frozen_training_protocol": {
            "observations_per_family": 24 if length > 24 else 0,
            "action_schedule": "standard when (seed + index) even; else express",
            "same_interventions_across_treatments": True,
            "each_heldout_case_from_identical_pretest_snapshot": True,
            "heldout_feedback_updates": False,
            "budget": {"top_k": 12, "candidate_headers": 48,
                       "spreading_neighbors": 4,
                       "maximum_thought_stages": 2},
        },
        "limitation": (
            "Synthetic, noisy private causal rules. Online prequential policies "
            "see selection-biased action feedback; frozen controls instead have "
            "identical logged interventions, not identical learned inferences. "
            "Online heldout choices adapt but frozen heldout choices cannot. "
            "One-pass spends fewer reasoning stages under the same maximum "
            "two-stage budget; its compute is not exactly equal. "
            "No private counterfactual truth is given to cognition."),
    }


def write_incremental_benchmark(output="results-e11", **kwargs) -> dict:
    payload = run_incremental_benchmark(**kwargs)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps({k: v for k, v in payload.items()
                    if k not in ("trials", "frozen_trials")},
                   indent=2, sort_keys=True), encoding="utf-8")
    (destination / "trials.json").write_text(
        json.dumps(payload["trials"], indent=2, sort_keys=True),
        encoding="utf-8")
    (destination / "frozen_trials.json").write_text(
        json.dumps(payload["frozen_trials"], indent=2, sort_keys=True),
        encoding="utf-8")
    if payload["seeds"] and payload["families"]:
        # Inspectable sequence of 10 minimally changing live experiences.
        trace = [row for row in payload["trials"]
                 if row["seed"] == payload["seeds"][0]
                 and row["family"] == payload["families"][0]
                 and row["mode"] == "incremental_full"][:10]
        (destination / "continuity_trace.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True),
            encoding="utf-8")
    return payload
