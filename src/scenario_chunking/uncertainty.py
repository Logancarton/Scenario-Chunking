"""Experiment 7: uncertain evidence, costly choices, and changed dependencies.

This is research orchestration, not a parallel cognitive runtime. Every
decision uses existing memory, meaning, learner, selector and refiner owners.
"""
from __future__ import annotations

from collections import defaultdict
import json
import pickle
import random
import statistics
from pathlib import Path
from time import perf_counter

from .environment import AXES, UncertaintyWorld
from .models import Action, EventKind, TruthStatus
from .policy import ActionSelector
from .refinement import ProgressiveRefiner
from .transfer import TransferRun

CONDITIONS_7 = (
    "untrained_structural", "trained_structural", "full_no_reflection",
    "limited_reflection", "extended_reflection", "strong_one_pass",
    "larger_retrieval", "shuffled_evidence", "misleading_reports",
)


class UncertaintyRun(TransferRun):
    """Run one matched seed against the same private task generator."""

    def __init__(self, seed: int, condition: str, *, disabled_axis: str | None = None,
                 rounds: int = 4, distractors: int = 4, action_budget: int = 9) -> None:
        if condition not in CONDITIONS_7:
            raise ValueError(f"unknown Experiment 7 condition {condition}")
        parent_condition = condition if condition in (
            "untrained_structural", "trained_structural", "full_no_reflection",
            "limited_reflection", "extended_reflection", "strong_one_pass",
        ) else "full_no_reflection"
        super().__init__(seed, parent_condition, disabled_axis=disabled_axis,
                         rounds=rounds, distractors=distractors)
        self.condition = condition
        self.world = UncertaintyWorld(seed)
        self.action_budget = action_budget
        self.router.top_k = (64 if condition == "larger_retrieval" else
                             32 if condition == "strong_one_pass" else 24)
        self.source_checks = []
        self._current_reports = {}
        self._resolved_change_axes = set()

    def train(self) -> None:
        """Same causal experiences for all trained treatments.

        A report is verified only *after* an observed action outcome. The
        sequence owner learns dependencies from blocked attempts, not world.order.
        """
        if self.condition == "untrained_structural":
            return
        super().train()
        rng = random.Random(self.seed * 131 + 43)
        for context in ("day", "night"):
            for i in range(36):
                axis = AXES[i % len(AXES)]
                actual = self.world.correct[axis]
                choices = self.world.options[axis]
                person = ("Aster", "Beryl", "Cinder")[i % 3]
                reliability = {"Aster": .84, "Beryl": .22,
                               "Cinder": .75 if context == "day" else .27}[person]
                claimed = (actual if rng.random() < reliability
                           else rng.choice([x for x in choices if x != actual]))
                # Training knows the outcome only by taking the action.
                self.learner.begin_episode()
                self._feedback(
                    f"source-verification-{context}-{i}", i,
                    Action("resolve", actual, axis), True,
                    source="measured_training_trial",
                )
                self.learner.verify_source(person, context, claimed == actual)
                self.source_checks.append({
                    "person": person, "context": context,
                    "claimed": claimed, "observed": actual,
                    "correct": claimed == actual,
                })
        # Causal, observed blocked attempts establish ordered dependencies.
        for attempt_round in range(10):
            completed = set()
            remaining = list(AXES)
            rng.shuffle(remaining)
            while remaining:
                made_progress = False
                for axis in list(remaining):
                    status, missing = self.world.attempt(
                        axis, self.world.correct[axis], completed)
                    if status == "blocked":
                        # The workshop instrument explicitly identifies the
                        # blocking predecessor in its observed failure.
                        assert missing is not None
                        self.learner.observe_prerequisite(missing, axis)
                    else:
                        completed.add(axis)
                        remaining.remove(axis)
                        made_progress = True
                if not made_progress:
                    raise RuntimeError("dependency graph did not progress")

    def _phase1b_reflect(self) -> None:
        # Uses the original thought owner and its provenance-bearing inference.
        if self.condition in ("limited_reflection", "extended_reflection"):
            super()._phase1b_reflect()

    def _write_reports(self, episode: str, scene: dict, round_no: int) -> None:
        self._current_reports = {}
        for axis in AXES:
            reports = self.world.reports(self.seed, round_no, axis)
            self._current_reports[axis] = reports
            for item in reports:
                reported = item["token"]
                if self.condition == "misleading_reports":
                    choices = self.world.options[axis]
                    reported = choices[(choices.index(reported) + 1) % len(choices)]
                event = self._event(
                    episode, round_no, kind=EventKind.OBSERVATION,
                    truth=TruthStatus.FACT,
                    content=f"{item['source']} reports {reported} for {axis}",
                    tags={"goal:workshop_recovery", f"axis:{axis}",
                          f"signal:{axis}:{reported}", "source:testimony"},
                    person=item["source"], confidence=0.5,
                    metadata={
                        "source_status": "testimony",
                        "reported_axis": axis, "reported_token": reported,
                        "source_name": item["source"], "context": scene["context"],
                    },
                )
                self.memory.create_chunk(
                    events=[event], transition_reason="unverified_testimony",
                    self_relevance=0.7, salience=0.65,
                )

    def _revision_notice(self, round_no: int) -> None:
        if round_no != self.rounds // 2 or self.world.revised:
            return
        self.world.revise()
        for axis in ("time", "sequence"):
            notice = self._event(
                f"revision-e7-{round_no}", round_no,
                kind=EventKind.OBSERVATION, truth=TruthStatus.FACT,
                content=f"Inspection bulletin: {axis} procedures may have changed",
                tags={"goal:workshop_recovery", f"axis:{axis}",
                      "revision:announced"},
                metadata={"change_notice": True, "changed_axis": axis},
            )
            self.memory.create_chunk(
                events=[notice], transition_reason="observed_change_warning",
                self_relevance=1.0, salience=1.0,
            )

    def _candidate_actions(self, state, scene: dict) -> list[Action]:
        actions = []
        for axis in AXES:
            if axis in state.completed_axes:
                continue
            options = self.world.options[axis]  # public candidate inventory only
            for index, token in enumerate(options):
                action = Action("resolve", token, axis)
                cue_token = (options[(index + 1) % len(options)]
                             if self.condition == "shuffled_evidence" else token)
                state.option_features[action.label] = frozenset(
                    {f"signal:{axis}:{cue_token}"})
                actions.append(action)
        for axis in sorted(state.invalidated_since):
            if axis in self._resolved_change_axes:
                continue
            kind = "inspect_dependencies" if axis == "sequence" else "inspect_protocol"
            action = Action(kind, scene["object"], axis)
            state.option_features[action.label] = frozenset()
            actions.append(action)
        return actions

    def _decision(self, state, actions, axes, *, round_no: int, step: str):
        hits, stats = self._retrieve(axes)
        depth = {
            "untrained_structural": -1, "trained_structural": -1,
            "full_no_reflection": 0, "limited_reflection": 1,
            "extended_reflection": 3, "strong_one_pass": 0,
            "larger_retrieval": 0, "shuffled_evidence": 0,
            "misleading_reports": 0,
        }[self.condition]
        started = perf_counter()
        evidence, audits = {}, {}
        for action in actions:
            if depth < 0:
                break
            audit = self.interpreter.uncertainty_evidence(
                hits, state.action_features(action),
                context=self._context, review_depth=depth,
                invalidated_since=state.invalidated_since,
            )
            audits[action.label] = audit
            evidence[action.label] = .85 * audit["score"]
            self.cost["interpretation_evaluations"] += audit["evaluations"]
        rng_seed = self.seed * 991 + round_no * 97 + len(state.attempted_actions) * 13 + len(step)
        selector = ActionSelector(
            self.memory, self.learner, random.Random(rng_seed),
            refiner=self.refiner if depth > 0 else None,
        )
        decision = selector.select(
            state, actions, hits, transfer_scores=evidence if depth >= 0 else None,
            refine=depth > 0,
            transfer_context={"audits": audits,
                              "resolved_change_axes": self._resolved_change_axes},
        )
        shadow = None
        if depth > 0:
            onepass = {}
            for action in actions:
                audit = self.interpreter.uncertainty_evidence(
                    hits, state.action_features(action), context=self._context,
                    review_depth=0, invalidated_since=state.invalidated_since)
                onepass[action.label] = .85 * audit["score"]
                self.cost["counterfactual_evaluations"] += audit["evaluations"]
            shadow = ActionSelector(
                self.memory, self.learner, random.Random(rng_seed),
            ).select(state, actions, hits, transfer_scores=onepass).action.label
        latency = (perf_counter() - started) * 1000
        self.cost["decisions"] += 1
        self.cost["selection_ms"] += latency
        audit = audits.get(decision.action.label, {})
        return decision, {
            "round": round_no, "step": step,
            "action": decision.action.label, "prediction": decision.predicted_success,
            "retrieved_chunk_ids": [h.chunk_id for h in hits],
            "detail_events": stats.active_detail_events,
            "provenance": audit.get("provenance", {}),
            "testimony_evidence_ids": audit.get("report_evidence_ids", []),
            "source_reliability": audit.get("source_reliability", {}),
            "contradictory_axes": audit.get("contradictory_axes", []),
            "causal_evidence_ids": sorted(set(audit.get("evidence_ids", [])) -
                                          set(audit.get("report_evidence_ids", []))),
            "used_evidence_ids": (decision.thought_trace or {}).get(
                "contributing_evidence_ids", audit.get("evidence_ids", [])),
            "one_pass_counterfactual_action": shadow,
            "reflection_changed_action": bool(shadow and shadow != decision.action.label),
            "thought": decision.thought_trace,
            "decision_basis": (decision.thought_trace or {}).get("decision_basis"),
            "selection_ms": latency, "outcome": None,
        }

    def _inspect(self, state, episode: str, action: Action, decision) -> str:
        axis = action.key_name
        observation_action = self._event(
            episode, len(state.attempted_actions), kind=EventKind.ACTION,
            truth=TruthStatus.ATTEMPT,
            content=f"Requested instrument check for {axis}",
            tags={"goal:workshop_recovery", f"axis:{axis}"}, action=action)
        self.learner.register_action(observation_action.event_id,
                                     state.policy_features(action))
        if axis == "sequence":
            observed = {
                name: self.world.prerequisite(name) for name in AXES
            }  # Only instrument execution exposes this; selector never sees world.
            for dependent, before in observed.items():
                if before:
                    state.current_prerequisites.setdefault(dependent, set()).add(before)
            details = observed
            tags = {"goal:workshop_recovery", "axis:sequence",
                    "source:current_protocol"}
        else:
            token = self.world.correct[axis]
            details = {"reported_axis": axis, "reported_token": token,
                       "source_status": "current_protocol"}
            tags = {"goal:workshop_recovery", f"axis:{axis}",
                    f"signal:{axis}:{token}", "source:current_protocol"}
        observation = self._event(
            episode, len(state.attempted_actions), kind=EventKind.OBSERVATION,
            truth=TruthStatus.FACT,
            content=f"Instrument check returned current {axis} procedure",
            tags=tags, confidence=.98, cause=observation_action.event_id,
            metadata=details,
        )
        self.memory.create_chunk(
            events=[observation_action, observation],
            transition_reason="current_procedure_observed",
            self_relevance=1.0, salience=1.0)
        self.learner.mark_informative(
            observation_action.event_id, produced_evidence_ids=(observation.event_id,))
        if decision.thought_trace is not None:
            self.refiner.feedback(state, decision, outcome=None,
                                  outcome_event_ids=(observation.event_id,))
        self._resolved_change_axes.add(axis)
        return observation.event_id

    def _blocked(self, state, episode: str, action: Action, missing: str) -> str:
        attempt = self._event(
            episode, len(state.attempted_actions), kind=EventKind.ACTION,
            truth=TruthStatus.ATTEMPT,
            content=f"Attempted {action.label}; prerequisite not ready",
            tags={"goal:workshop_recovery", f"axis:{action.key_name}"},
            action=action)
        result = self._event(
            episode, len(state.attempted_actions), kind=EventKind.OUTCOME,
            truth=TruthStatus.OUTCOME, outcome="blocked",
            content=f"Instrument reports {missing} must finish first",
            tags={"goal:workshop_recovery", "outcome:blocked",
                  f"axis:{action.key_name}", f"prerequisite:{missing}"},
            cause=attempt.event_id,
            metadata={"observed_blocker": missing, "dependent": action.key_name})
        self.memory.create_chunk(
            events=[attempt, result], transition_reason="causal_blocked_attempt",
            self_relevance=1.0, salience=.95)
        # Blocked means no evidence that the CHOSEN TOKEN is wrong. Credit
        # only the observed ordering relation, never a false token failure.
        self.learner.observe_prerequisite(missing, action.key_name)
        state.current_prerequisites.setdefault(action.key_name, set()).add(missing)
        return result.event_id

    def _execute_round(self, round_no: int) -> dict:
        self._revision_notice(round_no)
        scene = self.world.scene(self.seed, round_no)
        self._context = scene["context"]
        episode = f"uncertainty-{round_no}-{scene['object']}"
        state = self._state(episode)
        self._resolved_change_axes = set()
        self.learner.begin_episode()
        self._write_reports(episode, scene, round_no)
        mistakes = inspections = blocked = attempts = 0
        first_action_correct = None
        predictions = []
        actions_taken = 0
        while len(state.completed_axes) < len(AXES) and actions_taken < self.action_budget:
            state.current_time = self.clock
            actions = self._candidate_actions(state, scene)
            decision, trace = self._decision(
                state, actions, AXES, round_no=round_no, step="adaptive_choice")
            chosen = decision.action
            if chosen.kind.startswith("inspect_"):
                result_id = self._inspect(state, episode, chosen, decision)
                trace["outcome"] = "information_observed"
                trace["observation_ids"] = [result_id]
                inspections += 1
            else:
                status, missing = self.world.attempt(
                    chosen.key_name, chosen.target, state.completed_axes)
                if status == "blocked":
                    result_id = self._blocked(state, episode, chosen, missing)
                    blocked += 1
                    mistakes += 1
                else:
                    _, result_id = self._feedback(
                        episode, actions_taken, chosen, status == "success",
                        source=scene["object"], person=scene["person"],
                        state=state, used_evidence_ids=trace["used_evidence_ids"],
                    )
                    if status == "success":
                        state.completed_axes.add(chosen.key_name)
                        for item in self._current_reports[chosen.key_name]:
                            if self.condition == "untrained_structural":
                                continue
                            reported = item["token"]
                            if self.condition == "misleading_reports":
                                choices = self.world.options[chosen.key_name]
                                reported = choices[(choices.index(reported) + 1) % len(choices)]
                            self.learner.verify_source(
                                item["source"], item["context"],
                                reported == chosen.target)
                    else:
                        mistakes += 1
                trace["outcome"] = status
                predictions.append((decision.predicted_success, status == "success"))
                if first_action_correct is None:
                    first_action_correct = status == "success"
                if decision.thought_trace is not None:
                    self.refiner.feedback(
                        state, decision, outcome=(
                            "success" if status == "success" else "failure"),
                        outcome_event_ids=(result_id,),
                    )
                attempts += 1
            self.traces.append(trace)
            state.record_action(chosen)
            actions_taken += 1
        recovered = len(state.completed_axes) == len(AXES)
        return {
            "round": round_no, "scene": scene,
            "success": recovered and actions_taken <= self.action_budget,
            "recovered_success": recovered,
            "initial_plan_correct": bool(first_action_correct),
            "first_action_correct": bool(first_action_correct),
            "correct_order": recovered,
            "verification_actions": inspections, "blocked_actions": blocked,
            "axes_completed": sorted(state.completed_axes),
            "within_budget": actions_taken <= self.action_budget,
            "action_budget": self.action_budget, "actions": actions_taken,
            "wrong_decisions": mistakes, "retries": max(0, attempts - len(state.completed_axes)),
            "predictions": predictions,
        }

    def run(self) -> dict:
        self.train()
        self._phase1b_reflect()
        knowledge = dict(self.learner.snapshot())
        sources = dict(self.learner.source_verifications)
        prerequisites = dict(self.learner.prerequisite_observations)
        before_events = len(self.memory.events)
        state = pickle.dumps((self.memory, self.learner, self.refiner))
        self.memory, self.learner, self.refiner = pickle.loads(state)
        from .meaning import MeaningInterpreter
        from .retrieval import MemoryRouter
        from .models import RetrievalMode
        self.interpreter = MeaningInterpreter(self.memory, self.learner)
        self.router = MemoryRouter(
            self.memory, RetrievalMode.STRUCTURAL, top_k=self.router.top_k)
        restart_preserved = (
            len(self.memory.events) == before_events and
            self.learner.snapshot() == knowledge and
            dict(self.learner.source_verifications) == sources and
            dict(self.learner.prerequisite_observations) == prerequisites)
        rounds = [self._execute_round(i) for i in range(self.rounds)]
        trials = [t for r in rounds for t in r["predictions"]]
        return {
            "seed": self.seed, "condition": self.condition,
            "disabled_axis": self.disabled_axis,
            "rounds": rounds, "traces": self.traces, "cost": dict(self.cost),
            "restart_preserved": restart_preserved,
            "brier": statistics.mean((p - float(y)) ** 2 for p, y in trials) if trials else None,
            "source_samples": len(self.source_checks),
            "learned_prerequisites": len(prerequisites),
            "training_events": before_events,
        }


def run_uncertainty_experiment(output_dir: str | Path, *,
                               seeds=(7, 23, 61, 101, 149),
                               rounds: int = 4, distractors: int = 4,
                               action_budget: int = 9) -> dict:
    """Matched design; reports and hidden truth are generated identically per seed."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = [
        UncertaintyRun(seed, condition, rounds=rounds, distractors=distractors,
                       action_budget=action_budget).run()
        for seed in seeds for condition in CONDITIONS_7
    ]
    ablations = [
        UncertaintyRun(seed, "extended_reflection", disabled_axis=axis,
                       rounds=rounds, distractors=distractors,
                       action_budget=action_budget).run()
        for seed in seeds for axis in AXES
    ]

    def aggregate(group):
        trials = [r for run in group for r in run["rounds"]]
        if not trials:
            raise ValueError("no Experiment 7 trials")
        return {
            "completion_rate": statistics.mean(float(t["success"]) for t in trials),
            "first_action_accuracy": statistics.mean(float(t["first_action_correct"]) for t in trials),
            "correct_dependency_execution_rate": statistics.mean(
                float(t["correct_order"]) for t in trials),
            "mean_actions": statistics.mean(t["actions"] for t in trials),
            "mean_wrong_decisions": statistics.mean(t["wrong_decisions"] for t in trials),
            "mean_blocked_actions": statistics.mean(t["blocked_actions"] for t in trials),
            "mean_information_actions": statistics.mean(t["verification_actions"] for t in trials),
            "mean_brier": statistics.mean(
                r["brier"] for r in group if r["brier"] is not None),
            "mean_active_detail_events": statistics.mean(
                r["cost"].get("detail_events", 0) for r in group),
            "mean_processing_evaluations": statistics.mean(
                r["cost"].get("interpretation_evaluations", 0) +
                r["cost"].get("offline_reflection_evaluations", 0)
                for r in group),
            "mean_counterfactual_evaluations": statistics.mean(
                r["cost"].get("counterfactual_evaluations", 0) for r in group),
            "mean_latency_ms": statistics.mean(
                r["cost"].get("selection_ms", 0) + r["cost"].get("retrieval_ms", 0)
                for r in group),
            "reflection_action_changes": sum(
                bool(t.get("reflection_changed_action"))
                for run in group for t in run["traces"]),
            "restarts_preserved": all(r["restart_preserved"] for r in group),
            "round_completion": [
                statistics.mean(float(r["rounds"][i]["success"]) for r in group)
                for i in range(rounds)],
            "by_seed": [
                {"seed": r["seed"],
                 "completion_rate": statistics.mean(float(t["success"]) for t in r["rounds"]),
                 "wrong_decisions": sum(t["wrong_decisions"] for t in r["rounds"])}
                for r in group],
        }

    summary = {name: aggregate([r for r in runs if r["condition"] == name])
               for name in CONDITIONS_7}
    ablation_summary = {
        axis: aggregate([r for r in ablations if r["disabled_axis"] == axis])
        for axis in AXES
    }
    reflect = summary["extended_reflection"]
    onepass = summary["strong_one_pass"]
    no_reflect = summary["full_no_reflection"]
    verdict = {
        "gain_over_interpretation": reflect["completion_rate"] -
                                    no_reflect["completion_rate"],
        "gain_over_strong_one_pass": reflect["completion_rate"] -
                                     onepass["completion_rate"],
        "action_savings_vs_one_pass": onepass["mean_actions"] - reflect["mean_actions"],
        "extra_processing_vs_one_pass": reflect["mean_processing_evaluations"] -
                                         onepass["mean_processing_evaluations"],
        "reflection_has_verified_advantage": (
            reflect["completion_rate"] > onepass["completion_rate"] and
            reflect["completion_rate"] > no_reflect["completion_rate"]),
    }
    payload = {
        "experiment": 7, "seeds": list(seeds), "rounds": rounds,
        "action_budget": action_budget,
        "conditions": summary, "ablations": ablation_summary,
        "verdict": verdict,
        "architecture": (
            "UncertaintyWorld -> EpisodicMemory/MemoryRouter -> "
            "MeaningInterpreter -> ActionSelector/ProgressiveRefiner -> "
            "environment feedback -> OutcomeLearner -> durable memory"
        ),
        "limitations": [
            "Role tokens are shared between training and testing; only surface "
            "objects, people, context, report correctness and dependencies change.",
            "Training dependencies are learned from explicit blocked-predecessor "
            "feedback, not inferred from latent environmental dynamics.",
            "Refinement and one-pass comparisons have measured but not identical "
            "computation budgets; do not claim compute-matched superiority.",
            "In-process pickle round-trip is not an independent process restart.",
            "Small synthetic world; no general reasoning or natural-language claim.",
        ],
    }
    (output / "uncertainty_trials.json").write_text(
        json.dumps({"runs": runs, "ablations": ablations}, indent=2, sort_keys=True),
        encoding="utf-8")
    (output / "uncertainty_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
