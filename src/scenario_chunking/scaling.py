"""Experiment 8: controlled transfer from accumulating contextual experiences.

An orchestration harness, not a second selector/runtime/learner. Hidden world
outcomes never enter cognitive state except through executed actions and
inspection observations. Tests are frozen across training checkpoints.
"""
from __future__ import annotations

from collections import defaultdict
import json
import pickle
import random
import statistics
from pathlib import Path
from time import perf_counter

from .environment import ContextualWorld
from .models import Action, EventKind, TruthStatus, RetrievalQuery
from .state import TransferState
from .policy import ActionSelector
from .transfer import TransferRun

CONDITIONS_8 = {
    "untrained_structural": "untrained_structural",
    "trained_association": "trained_structural",
    "interpreted_no_reflection": "full_no_reflection",
    "strong_one_pass": "strong_one_pass",
    "limited_reflection": "limited_reflection",
    "extended_reflection": "extended_reflection",
    "adaptive_focus": "limited_reflection",
    "learned_focus_one_pass": "full_no_reflection",
    "bounded_without_learned_focus": "limited_reflection",
    "larger_retrieval": "larger_retrieval",
    "shuffled_evidence": "shuffled_evidence",
}
DEFAULT_STAGES = (20, 100, 500, 2000, 10000)
DEFAULT_MEMORY = (100, 1000, 5000, 20000)
DEFAULT_COMPLEXITIES = (1, 2, 3)


class ScalingRun(TransferRun):
    """The existing cognitive owners operate in one fully contextual task."""

    def __init__(self, seed: int, treatment: str, *, rounds: int = 4,
                 action_budget: int = 3, candidate_budget: int = 48) -> None:
        if treatment not in CONDITIONS_8:
            raise ValueError(f"unknown Experiment 8 treatment: {treatment}")
        if candidate_budget < 8:
            raise ValueError("candidate budget must allow causal comparisons")
        parent_condition = CONDITIONS_8[treatment]
        if parent_condition in ("larger_retrieval", "shuffled_evidence"):
            parent_condition = "full_no_reflection"
        super().__init__(seed, parent_condition, rounds=rounds, distractors=0)
        self.action_budget = action_budget
        self.world = ContextualWorld(seed)
        self.treatment = treatment
        self.experiences = 0
        self.training_observations = []
        self.candidate_budget = candidate_budget
        self.router.top_k = min(16, candidate_budget)
        self.router.candidate_budget = (
            min(4 * candidate_budget, 256) if treatment == "larger_retrieval"
            else candidate_budget)
        # Identical action affordances, training data and public contexts across
        # all trained treatments. Only the cognition mechanism is ablated.

    def run(self, *, experiences: int = 100, memory_target: int = 100,
            complexity: int = 2) -> dict:
        """Run one verified E8 research path, never an inherited E6/E7 runner."""
        self.add_experiences(experiences)
        return self.heldout(0, memory_target, complexity=complexity)

    def _attempt(self, task, procedure: str, *, episode: str, step: int,
                 dependency: str, context_cues: frozenset[str],
                 used_evidence_ids=()) -> tuple[bool, str]:
        """A real executed action produces the only supervised causal target."""
        action = Action("resolve", procedure, "procedure")
        tags = set(context_cues) | {"goal:workshop_recovery", "action:choose"}
        event = self._event(
            episode, step, kind=EventKind.ACTION, truth=TruthStatus.ATTEMPT,
            content=f"Attempted procedure {procedure}", tags=tags,
            action=action, person=task.source,
            metadata={"reported_dependency": dependency})
        accepted = self.world.outcome(task, procedure)
        outcome = self._event(
            episode, step, kind=EventKind.OUTCOME, truth=TruthStatus.OUTCOME,
            content=f"Instrument observed procedure {'success' if accepted else 'failure'}",
            tags=tags | {f"outcome:{'success' if accepted else 'failure'}"},
            outcome="success" if accepted else "failure",
            cause=event.event_id,
            metadata={"tested_procedure": procedure})
        self.memory.create_chunk(
            events=[event, outcome], transition_reason="observed_contextual_feedback",
            self_relevance=task.self_relevance, salience=task.consequence)
        self.learner.register_action(
            event.event_id, frozenset({"action:contextual_trial"}),
            used_evidence_ids=used_evidence_ids)
        # Keep the existing action-credit owner active; contextual statistics
        # are an additional interpretation of that very same observed outcome.
        if accepted:
            self.learner.apply_success(event.event_id)
        else:
            self.learner.apply_failure(event.event_id, -1.0)
        self.learner.learn_context_outcome(
            context_cues, procedure, accepted, evidence_id=outcome.event_id)
        return accepted, outcome.event_id

    def add_experiences(self, target: int) -> None:
        if target < self.experiences:
            raise ValueError("experience checkpoints cannot rewind")
        for index in range(self.experiences, target):
            task = self.world.task(index, heldout=False, complexity=2)
            reported = task.public()
            # A subset of probes chooses costly observation before acting.
            # No correct action is ever supplied to the learner or policy.
            inspected = (index % 4 == 0)
            dependency = task.reported_dependency
            if inspected:
                observation = self.world.inspect_dependency(task)
                dependency = observation["dependency"]
                if self.treatment != "untrained_structural":
                    self.learner.verify_source(
                        task.source, task.context,
                        dependency == task.reported_dependency)
                    self.learner.learn_observation_value(
                        dependency != task.reported_dependency)
            cues = self.world.relational_cues(reported, dependency=dependency)
            procedure = self.world.PROCEDURES[
                (index + index // len(self.world.TRAIN_KEYS) + self.seed) % 4]
            if self.treatment != "untrained_structural":
                self.learner.begin_episode()
                success, outcome_id = self._attempt(
                    task, procedure, episode=f"training-{index}",
                    step=index, dependency=dependency, context_cues=cues)
                self.training_observations.append({
                    "index": index, "public_context": reported,
                    "inspected": inspected, "procedure": procedure,
                    "observed_success": success,
                    "outcome_event_id": outcome_id,
                })
            self.experiences += 1

    def _add_memory_load(self, count: int, *, target_seed: int) -> None:
        """Additional memories, independently controlled from experience count."""
        if count < 0:
            raise ValueError("memory load cannot be negative")
        misleading = count // 4
        for index in range(misleading):
            public = self.world.task(index + 300000 + target_seed).public()
            cues = self.world.relational_cues(public)
            event = self._event(
                f"low-trust-{target_seed}-{index}", index,
                kind=EventKind.OBSERVATION, truth=TruthStatus.FACT,
                content="Unverified third-party maintenance claim",
                tags=set(cues) | {"goal:workshop_recovery", "source:unverified"},
                informative=False, confidence=.1,
                metadata={"source_status": "unverified"},
            )
            self.memory.create_chunk(
                events=[event], transition_reason="distracting_unverified_claim",
                self_relevance=.1, salience=.12)
        self.memory.inject_distractors(
            count - misleading, start_time=self.clock + 1,
            goal_id="unrelated_goal")
        self.clock += 2 + .001 * (count - misleading)

    def _public_observation(self, task, episode: str) -> None:
        public = task.public()
        tags = set(self.world.relational_cues(public)) | {
            "goal:workshop_recovery", "source:testimony"}
        event = self._event(
            episode, 0, kind=EventKind.OBSERVATION, truth=TruthStatus.FACT,
            content=f"{task.source} reports dependency {task.reported_dependency}",
            tags=tags, person=task.source, confidence=.5,
            metadata={"source_status": "testimony",
                      "reported_dependency": task.reported_dependency,
                      "source": task.source, "context": task.context,
                      "revision_notice": task.revision_notice})
        self.memory.create_chunk(
            events=[event], transition_reason="unverified_current_report",
            self_relevance=task.self_relevance, salience=task.consequence)

    def _decide(self, task, state: TransferState, *,
                dependency: str, verified: bool, decision_number: int) -> tuple:
        public = task.public()
        cues = self.world.relational_cues(public, dependency=dependency)
        learned_focus = self.treatment in ("adaptive_focus", "learned_focus_one_pass")
        if self.treatment in ("adaptive_focus", "learned_focus_one_pass",
                               "bounded_without_learned_focus"):
            focus = self.learner.select_context_focus(
                cues, learned=learned_focus, budget=2) if learned_focus else frozenset(
                c for c in cues if c.startswith("pair:"))
        else:
            focus = frozenset(c for c in cues if c.startswith("pair:"))
        query_tags = focus | frozenset({
            "goal:workshop_recovery", f"mode:{public['mode']}",
            f"phase:{public['phase']}", f"material:{public['material']}",
            f"dependency:{dependency}", f"protocol:{public['protocol_epoch']}"})
        if self.treatment == "shuffled_evidence":
            query_tags = frozenset({
                "goal:workshop_recovery", "pair:misaligned:untrusted"})
        query = RetrievalQuery(goal_id="workshop_recovery",
                               semantic_tags=query_tags, current_time=self.clock)
        hits, stats = self.router.retrieve(query)
        # Evaluation-only audit: this does not add cognitive retrieval cost
        # or enter ActionSelector. A relevant memory must be verified feedback
        # with overlapping causal relationship evidence.
        eligible = set()
        for cue in focus:
            eligible.update(self.memory.tag_index.get(cue, ()))
        relevant = {cid for cid in eligible if
                    self.memory.chunks[cid].transition_reason ==
                    "observed_contextual_feedback"}
        hit_ids = {h.chunk_id for h in hits}
        relevant_hits = len(relevant & hit_ids)
        retrieval_precision = relevant_hits / len(hits) if hits else 0.0
        retrieval_recall = relevant_hits / len(relevant) if relevant else None
        actions = [Action("resolve", procedure, "procedure")
                   for procedure in self.world.PROCEDURES]
        if task.profile != "simple" and not verified:
            actions.append(Action("inspect_dependency", task.device_name, "dependency"))
        for action in actions:
            state.option_features[action.label] = cues | frozenset({
                "goal:workshop_recovery", "action:choose", "role:workshop"})
        state.current_time = self.clock
        alternatives = tuple(v for v in self.world.VALUES if v != dependency)
        if len(alternatives) != 1:
            raise ValueError("context requires exactly one alternative dependency")
        alternate_cues = self.world.relational_cues(
            public, dependency=alternatives[0])
        alternate_focus = frozenset(
            cue.replace("dependency=" + dependency,
                        "dependency=" + alternatives[0])
            for cue in focus)
        started = perf_counter()
        audits, scores = {}, {}
        for action in actions:
            if action.kind == "inspect_dependency":
                # This information option is available to every treatment;
                # reflection can deliberately prefer it when evidence warrants.
                score = .12 * (1.0 - self.learner.source_reliability(
                    task.source, task.context)) * task.consequence
                scores[action.label] = score
                audits[action.label] = {"evidence_ids": [], "evaluations": 0}
                continue
            active = focus if learned_focus else cues
            audit = self.interpreter.contextual_evidence(
                hits, state.action_features(action), procedure=action.target,
                source=task.source, source_context=task.context,
                verified=verified, attention_cues=active,
                alternative_attention_cues=(
                    alternate_focus if learned_focus else alternate_cues))
            audits[action.label] = audit
            scores[action.label] = .85 * audit["score"]
        # A trained one-pass control uses the same retrieval and evidence budget
        # as adaptive focus, except for intentionally larger-retrieval ablation.
        refine = self.treatment in (
            "limited_reflection", "extended_reflection", "adaptive_focus",
            "bounded_without_learned_focus")
        selector = ActionSelector(
            self.memory, self.learner,
            random.Random(self.seed * 7001 + decision_number * 37),
            refiner=self.refiner if refine else None)
        decision = selector.select(
            state, actions, hits, transfer_scores=scores, refine=refine,
            transfer_context={
                "cognitive_mode": "contextual", "audits": audits,
                "attention_cues": focus,
                "verified_dependency": verified,
                "consequence": task.consequence,
                "source_reliability": self.learner.source_reliability(
                    task.source, task.context),
                "inspection_cost": .25,
                "revision_notice": task.revision_notice})
        elapsed = (perf_counter() - started) * 1000.0
        evaluations = sum(record.get("evaluations", 0) for record in audits.values())
        return decision, {
            "action": decision.action.label, "predicted_success": decision.predicted_success,
            "retrieved_chunk_ids": [h.chunk_id for h in hits],
            "causal_retrieval_precision": retrieval_precision,
            "causal_retrieval_recall": retrieval_recall,
            "missed_causal_memories": max(0, len(relevant) - relevant_hits),
            "candidate_headers": stats.candidate_chunks,
            "header_evaluations": stats.header_evaluations,
            "detail_events": stats.active_detail_events,
            "unnecessary_activations": stats.unnecessary_activations,
            "retrieval_ms": stats.latency_ms, "selection_ms": elapsed,
            "interpretation_evaluations": evaluations,
            "thought": decision.thought_trace,
            "focus_cues": sorted(focus),
            "evidence_ids": audits.get(decision.action.label, {}).get("evidence_ids", []),
        }

    def _solve(self, task, *, action_budget: int) -> dict:
        episode = "heldout-" + task.task_id
        state = TransferState(episode_id=episode)
        self._public_observation(task, episode)
        self.learner.begin_episode()
        dependency = task.reported_dependency
        verified = False
        attempts = 0
        success = False
        first_procedure_correct = None
        steps = []
        for n in range(action_budget):
            decision, trace = self._decide(
                task, state, dependency=dependency, verified=verified,
                decision_number=n + 1000 * int(task.task_id.rsplit("-", 1)[-1]))
            chosen = decision.action
            if chosen.kind == "inspect_dependency":
                observed = self.world.inspect_dependency(task)
                dependency = observed["dependency"]
                verified = True
                observation = self._event(
                    episode, n, kind=EventKind.OBSERVATION, truth=TruthStatus.FACT,
                    content="Instrument directly confirms dependency state",
                    tags=set(self.world.relational_cues(task.public(), dependency=dependency))
                         | {"goal:workshop_recovery", "source:current_protocol"},
                    metadata={"source_status": "current_protocol",
                              "observed_dependency": dependency})
                self.memory.create_chunk(
                    events=[observation], transition_reason="observed_dependency",
                    self_relevance=task.self_relevance, salience=1.0)
                if self.treatment != "untrained_structural":
                    self.learner.verify_source(
                        task.source, task.context,
                        dependency == task.reported_dependency)
                    self.learner.learn_observation_value(
                        dependency != task.reported_dependency)
                trace["outcome"] = "investigated"
                trace["observation_ids"] = [observation.event_id]
                if decision.thought_trace is not None:
                    self.refiner.feedback(
                        state, decision, outcome=None,
                        outcome_event_ids=(observation.event_id,))
            else:
                cues = self.world.relational_cues(task.public(), dependency=dependency)
                success, event_id = self._attempt(
                    task, chosen.target, episode=episode, step=n,
                    dependency=dependency, context_cues=cues,
                    used_evidence_ids=trace["evidence_ids"])
                if first_procedure_correct is None:
                    first_procedure_correct = success
                attempts += 1
                trace["outcome"] = "success" if success else "failure"
                if decision.thought_trace is not None:
                    self.refiner.feedback(
                        state, decision, outcome=trace["outcome"],
                        outcome_event_ids=(event_id,))
            steps.append(trace)
            self.traces.append(trace)
            state.record_action(chosen)
            if success:
                break
        return {
            "task_id": task.task_id, "completed": success,
            "first_procedure_correct": bool(first_procedure_correct),
            "total_actions": len(steps),
            "attempts": attempts,
            "inspections": sum(step["outcome"] == "investigated" for step in steps),
            "thought_versions": len(self.refiner.thread(state).versions)
                                if self.treatment in (
                                  "limited_reflection", "extended_reflection",
                                  "adaptive_focus", "bounded_without_learned_focus") else 0,
            "traces": steps,
        }

    def heldout(self, stage: int, memory_target: int, *,
                familiar: bool = False, complexity: int = 2,
                task_seed: int = 8008) -> dict:
        """Freeze evaluation tasks independently of training checkpoints."""
        agent = pickle.loads(pickle.dumps(self))
        agent.traces = []
        agent._add_memory_load(memory_target, target_seed=task_seed)
        starting_memories = len(agent.memory.chunks)
        # The held-out world is independent of training history. Its private
        # causal mechanism is stable; never transfer a surface answer table.
        test_world = ContextualWorld(self.seed + task_seed)
        agent.world = test_world
        outcomes = []
        for n in range(agent.rounds):
            task = test_world.task(n, heldout=not familiar, complexity=complexity)
            outcomes.append(agent._solve(task, action_budget=agent.action_budget))
        steps = [step for row in outcomes for step in row["traces"]]
        avg = lambda key: (statistics.mean(float(s[key]) for s in steps)
                           if steps else 0.0)
        correct = [float(row["completed"]) for row in outcomes]
        return {
            "seed": self.seed, "treatment": self.treatment,
            "training_experiences": self.experiences if self.treatment !=
                                    "untrained_structural" else 0,
            "stage": stage, "complexity": complexity, "familiar": familiar,
            "memory_target": memory_target, "stored_memories": starting_memories,
            "stored_after_trials": len(agent.memory.chunks),
            "rounds": len(outcomes), "completed_rounds": int(sum(correct)),
            "completion_rate": statistics.mean(correct),
            "first_action_accuracy": statistics.mean(
                float(o["first_procedure_correct"]) for o in outcomes),
            "mean_actions": statistics.mean(o["total_actions"] for o in outcomes),
            "mean_inspections": statistics.mean(o["inspections"] for o in outcomes),
            "mean_active_details": avg("detail_events"),
            "mean_header_evaluations": avg("header_evaluations"),
            "mean_unused_activation": avg("unnecessary_activations"),
            "mean_causal_retrieval_precision": avg("causal_retrieval_precision"),
            "mean_causal_retrieval_recall": statistics.mean(
                t["causal_retrieval_recall"] for t in steps if
                t["causal_retrieval_recall"] is not None)
                if any(t["causal_retrieval_recall"] is not None for t in steps)
                else None,
            "mean_missed_causal_memories": avg("missed_causal_memories"),
            "mean_retrieval_ms": avg("retrieval_ms"),
            "mean_selection_ms": avg("selection_ms"),
            "mean_processing_evaluations": avg("interpretation_evaluations"),
            "mean_thought_versions": statistics.mean(
                o["thought_versions"] for o in outcomes),
            "round_successes": [bool(o["completed"]) for o in outcomes],
            "tasks": outcomes,
        }


def _paired_interval(deltas: list[float]) -> list[float] | None:
    if len(deltas) < 2:
        return None
    rng = random.Random(8008)
    draws = sorted(statistics.mean(deltas[rng.randrange(len(deltas))]
                                   for _ in deltas) for _ in range(2000))
    return [draws[49], draws[1949]]


def run_scaling_experiment(
    output_dir: str | Path, *, seeds=tuple(range(1, 31)),
    stages=DEFAULT_STAGES, memory_targets=DEFAULT_MEMORY,
    conditions=tuple(CONDITIONS_8), complexities=DEFAULT_COMPLEXITIES,
    rounds=4, action_budget=3, candidate_budget=48,
) -> dict:
    """Factorial design: hold out task bank independently of all three factors."""
    if not seeds or not stages or any(v < 1 for v in stages):
        raise ValueError("positive seeds and training stages required")
    if tuple(sorted(set(stages))) != tuple(stages):
        raise ValueError("training stages must increase")
    if not memory_targets or any(v < 0 for v in memory_targets):
        raise ValueError("memory load must be nonnegative")
    if not complexities or any(v not in (1, 2, 3) for v in complexities):
        raise ValueError("invalid task complexities")
    if not conditions or any(c not in CONDITIONS_8 for c in conditions):
        raise ValueError("unknown cognitive treatment")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trials = []
    for seed in seeds:
        agents = {name: ScalingRun(seed, name, rounds=rounds,
                                   action_budget=action_budget,
                                   candidate_budget=candidate_budget)
                  for name in conditions}
        for stage, checkpoint in enumerate(stages):
            for treatment, agent in agents.items():
                agent.add_experiences(checkpoint)
                for load in memory_targets:
                    for complexity in complexities:
                        trials.append(agent.heldout(
                            stage, load, complexity=complexity, task_seed=8008))
    rows = {}
    for checkpoint in stages:
        rows[str(checkpoint)] = {}
        for name in conditions:
            matched = [r for r in trials if r["training_experiences"] ==
                       (0 if name == "untrained_structural" else checkpoint)
                       and r["treatment"] == name and r["stage"] ==
                       stages.index(checkpoint)]
            rows[str(checkpoint)][name] = {
                "completion_rate": statistics.mean(r["completion_rate"] for r in matched),
                "first_action_accuracy": statistics.mean(
                    r["first_action_accuracy"] for r in matched),
                "mean_actions": statistics.mean(r["mean_actions"] for r in matched),
                "mean_detail_reads": statistics.mean(r["mean_active_details"] for r in matched),
                "mean_header_evaluations": statistics.mean(
                    r["mean_header_evaluations"] for r in matched),
                "mean_retrieval_ms": statistics.mean(r["mean_retrieval_ms"] for r in matched),
                "mean_causal_retrieval_precision": statistics.mean(
                    r["mean_causal_retrieval_precision"] for r in matched),
                "mean_causal_retrieval_recall": statistics.mean(
                    r["mean_causal_retrieval_recall"] for r in matched
                    if r["mean_causal_retrieval_recall"] is not None)
                    if any(r["mean_causal_retrieval_recall"] is not None for r in matched)
                    else None,
                "mean_processing_evaluations": statistics.mean(
                    r["mean_processing_evaluations"] for r in matched),
                "mean_memories": statistics.mean(r["stored_memories"] for r in matched),
                "by_seed": [{"seed": r["seed"], "complexity": r["complexity"],
                             "distractors": r["memory_target"],
                             "completion": r["completion_rate"]}
                            for r in matched],
            }
    paired = {}
    if {"adaptive_focus", "strong_one_pass"}.issubset(conditions):
        for stage, checkpoint in enumerate(stages):
            a = { (r["seed"], r["complexity"], r["memory_target"]): r
                  for r in trials if r["stage"] == stage and r["treatment"] == "adaptive_focus"}
            b = { (r["seed"], r["complexity"], r["memory_target"]): r
                  for r in trials if r["stage"] == stage and r["treatment"] == "strong_one_pass"}
            deltas = defaultdict(list)
            for key in sorted(a):
                deltas[key[0]].append(
                    a[key]["completion_rate"] - b[key]["completion_rate"])
            seed_deltas = {seed: statistics.mean(values)
                           for seed, values in sorted(deltas.items())}
            paired[str(checkpoint)] = {
                "by_seed": seed_deltas,
                "bootstrap_95_interval": _paired_interval(list(seed_deltas.values())),
            }
    # Isolate reflection itself: same learned focus and retrieval budget.
    reflection_paired = {}
    if {"adaptive_focus", "learned_focus_one_pass"}.issubset(conditions):
        for stage, checkpoint in enumerate(stages):
            a = {(r["seed"], r["complexity"], r["memory_target"]): r
                 for r in trials if r["stage"] == stage and
                 r["treatment"] == "adaptive_focus"}
            b = {(r["seed"], r["complexity"], r["memory_target"]): r
                 for r in trials if r["stage"] == stage and
                 r["treatment"] == "learned_focus_one_pass"}
            by_seed = defaultdict(list)
            for key in sorted(a):
                by_seed[key[0]].append(
                    a[key]["completion_rate"] - b[key]["completion_rate"])
            seed_values = {seed: statistics.mean(values)
                           for seed, values in sorted(by_seed.items())}
            reflection_paired[str(checkpoint)] = {
                "by_seed": seed_values,
                "bootstrap_95_interval": _paired_interval(list(seed_values.values())),
            }
    payload = {
        "experiment": 8, "design": "factorial_shared_heldout_context",
        "seeds": list(seeds), "stages": list(stages),
        "memory_targets": list(memory_targets),
        "complexities": list(complexities), "conditions": list(conditions),
        "stage_results": rows, "paired_focus_vs_onepass": paired,
        "paired_reflection_only": reflection_paired,
        "limitations": [
            "Procedural contextual reasoning only: no natural-language general cognition claim.",
            "World causal rule is fixed across scenarios but synthetic and small.",
            "Source reliability is validated only when instrument observations permit it.",
            "Evidence and retrieval budgets are bounded; full instruction-level compute matching is not established.",
            "Metric confidence intervals must resample independent seeds, not correlated task cells.",
            "Focused retrieval may miss relevant causal evidence; explicit recall checks remain needed.",
            "Learning curves at 10,000 experiences and 20,000 memories require a separate full run.",
        ],
    }
    (out / "scaling_trials.json").write_text(
        json.dumps(trials, indent=2, sort_keys=True), encoding="utf-8")
    (out / "scaling_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
