"""Experiment 6: controlled, causal cross-category transfer, not a production runtime.

Five categories are randomized latent associations; neither retrieval nor selector
receives the answer key. Existing episodic memory, router, meaning interpreter,
outcome learner, action selector and thought-thread owner are reused.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import copy
import json
import pickle
import random
import statistics
from pathlib import Path
from time import perf_counter

from .learning import OutcomeLearner
from .meaning import MeaningInterpreter
from .memory import EpisodicMemory
from .models import Action, Event, EventKind, RetrievalMode, RetrievalQuery, TruthStatus
from .policy import ActionSelector
from .refinement import ProgressiveRefiner
from .retrieval import MemoryRouter

AXES = ("self", "salience", "time", "sequence", "person")
CONDITIONS = (
    "untrained_structural", "trained_structural", "full_no_reflection",
    "limited_reflection", "extended_reflection", "strong_one_pass",
    "shuffled_relationships",
)
SYMBOLS = (
    "amber", "birch", "cobalt", "delta", "elm", "fennel", "garnet", "harbor",
    "indigo", "jasper", "kelp", "lumen", "marble", "nectar", "opal", "piper",
    "quill", "raven", "sable", "tulip", "umber", "violet", "willow", "zircon",
)
OBJECTS = ("compressor", "lifter", "conveyor", "generator", "sorter", "mixer")
PEOPLE = ("Orin", "Sela", "Amin", "Jora", "Nemi", "Tavi", "Lora")


class TransferWorld:
    """Private procedural truth; no reference to this object from the policy."""

    def __init__(self, seed: int) -> None:
        rng = random.Random(seed * 101 + 29)
        selected = rng.sample(SYMBOLS, 2 * len(AXES))
        self.options = {axis: tuple(selected[2*i:2*i+2]) for i, axis in enumerate(AXES)}
        self.correct = {axis: rng.choice(self.options[axis]) for axis in AXES}
        self.original_time = next(x for x in self.options["time"] if x != self.correct["time"])

    def accepted(self, axis: str, token: str, *, historical: bool = False) -> bool:
        if axis not in AXES:
            raise ValueError("unknown axis")
        expected = self.original_time if historical and axis == "time" else self.correct[axis]
        return token == expected

    def scene(self, seed: int, round_no: int) -> dict:
        rng = random.Random(seed * 1009 + round_no * 83)
        return {
            "object": rng.choice(OBJECTS) + "-" + str(seed + round_no),
            "person": rng.choice(PEOPLE) + "-" + str(round_no),
            "order": rng.sample(list(AXES), len(AXES)),
        }


@dataclass
class TransferState:
    episode_id: str
    goal_id: str = "workshop_recovery"
    current_time: float = 0.0
    option_features: dict[str, frozenset[str]] = field(default_factory=dict)
    attempted_actions: list[str] = field(default_factory=list)
    excluded_axis: str | None = None
    scramble: bool = False
    choices: dict[str, tuple[str, str]] = field(default_factory=dict)

    def action_features(self, action: Action) -> frozenset[str]:
        cues = self.option_features.get(action.label, frozenset())
        if self.excluded_axis:
            cues = frozenset(x for x in cues if not x.startswith(f"signal:{self.excluded_axis}:"))
        if self.scramble:
            swapped = set()
            for feature in cues:
                parts = feature.split(":", 2)
                if len(parts) == 3 and parts[0] == "signal" and parts[1] in self.choices:
                    pair = self.choices[parts[1]]
                    swapped.add(f"signal:{parts[1]}:{pair[1] if parts[2] == pair[0] else pair[0]}")
                else:
                    swapped.add(feature)
            cues = frozenset(swapped)
        return cues | frozenset({"action:choose", "role:workshop"})

    def record_action(self, action: Action) -> None:
        self.attempted_actions.append(action.label)


class TransferRun:
    def __init__(self, seed: int, condition: str, *, disabled_axis: str | None = None,
                 rounds: int = 4, distractors: int = 4) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition}")
        self.seed, self.condition, self.rounds = seed, condition, rounds
        self.world = TransferWorld(seed)
        self.memory = EpisodicMemory()
        self.learner = OutcomeLearner()
        self.interpreter = MeaningInterpreter(self.memory, self.learner)
        self.refiner = ProgressiveRefiner(self.learner, learning=False)
        self.router = MemoryRouter(self.memory, RetrievalMode.STRUCTURAL, top_k=32 if condition == "strong_one_pass" else 16)
        self.clock = 0.0
        self.disabled_axis = disabled_axis
        self.distractors = distractors
        self.training = []
        self.traces = []
        self.cost = defaultdict(float)

    def _event(self, episode: str, step: int, *, kind: EventKind, truth: TruthStatus,
               content: str, tags: set[str], action: Action | None = None,
               outcome: str | None = None, cause: str | None = None,
               informative: bool = True, confidence: float = 1.0,
               person: str | None = None, metadata: dict | None = None) -> Event:
        self.clock += 1.0
        return Event(
            event_id=self.memory.next_event_id(), episode_id=episode, goal_id="workshop_recovery",
            step=step, occurrence_time=self.clock, acquired_time=self.clock,
            storage_time=self.clock, kind=kind, truth_status=truth,
            content=content, semantic_tags=frozenset(tags), action=action,
            outcome=outcome, cause_event_id=cause, informative=informative,
            confidence=confidence, person_id=person, metadata=metadata or {},
        )

    def _feedback(self, episode: str, step: int, action: Action, accepted: bool, *,
                  source: str, historical: bool = False, person: str | None = None,
                  state: TransferState | None = None) -> tuple[str, str]:
        axis, token = action.key_name, action.target
        assert axis in AXES and token is not None
        tags = {"goal:workshop_recovery", f"axis:{axis}", f"signal:{axis}:{token}"}
        attempt = self._event(episode, step, kind=EventKind.ACTION, truth=TruthStatus.ATTEMPT,
                              content=f"{source} tried signal {token} under {axis}",
                              tags=tags, action=action, person=person)
        if state is None:
            features = frozenset({f"signal:{axis}:{token}", "action:choose", "role:workshop"})
        else:
            features = state.action_features(action)
        self.learner.begin_episode()
        self.learner.register_action(attempt.event_id, features)
        result = self._event(
            episode, step, kind=EventKind.OUTCOME, truth=TruthStatus.OUTCOME,
            content=f"Observed {source} outcome for {axis}",
            tags=tags | {f"outcome:{'success' if accepted else 'failure'}"},
            outcome="success" if accepted else "failure", cause=attempt.event_id,
            metadata={"axis": axis, "source": source, "historical": historical},
        )
        if accepted:
            self.learner.apply_success(attempt.event_id)
        else:
            self.learner.apply_failure(attempt.event_id, -1.0)
        self.memory.create_chunk(
            events=[attempt, result], transition_reason="causal_feedback",
            self_relevance=0.9, salience=0.95 if accepted else 0.8,
        )
        return attempt.event_id, result.event_id

    def train(self) -> None:
        if self.condition == "untrained_structural":
            return
        rng = random.Random(self.seed * 41 + 13)
        # Procedural balanced exploration, NOT an answer table handed to the policy.
        # The learner observes only chosen actions and their causal feedback.
        for axis in AXES:
            pair = self.world.options[axis]
            order = list(pair) * 8
            rng.shuffle(order)
            for i, token in enumerate(order):
                historical = axis == "time" and i < 6
                accepted = self.world.accepted(axis, token, historical=historical)
                source = rng.choice(("instrument", "technician", "inspection"))
                aid, oid = self._feedback(
                    f"learning-{axis}-{i}", i, Action("resolve", token, axis),
                    accepted, source=source, historical=historical,
                    person=rng.choice(PEOPLE),
                )
                self.training.append({
                    "axis": axis, "token": token, "historical": historical,
                    "source": source, "cause_id": aid, "outcome_id": oid, "observed": accepted,
                })
                if i in (1, 6):
                    # Missing/inconsistent hearsay is stored as observation,
                    # never promoted to validated causal knowledge.
                    rumor = self._event(
                        f"noise-{axis}-{i}", i, kind=EventKind.OBSERVATION,
                        truth=TruthStatus.FACT, content="A person offers an unverified conflicting interpretation",
                        tags={"goal:workshop_recovery", f"axis:{axis}", f"signal:{axis}:{pair[i % 2]}"},
                        confidence=0.25, informative=False, person=rng.choice(PEOPLE),
                        metadata={"source_status": "unverified"},
                    )
                    self.memory.create_chunk(events=[rumor], transition_reason="hearsay",
                                             self_relevance=0.2, salience=1.0)
        for i in range(self.distractors):
            axis = rng.choice(AXES)
            noise = self._event(
                f"distractor-{i}", i, kind=EventKind.OBSERVATION, truth=TruthStatus.FACT,
                content="Recent high-salience but irrelevant maintenance anecdote",
                tags={"goal:workshop_recovery", f"axis:{axis}", "topic:unrelated"},
                informative=False, confidence=0.1,
            )
            self.memory.create_chunk(events=[noise], transition_reason="distractor",
                                     self_relevance=0.05, salience=1.0)

    def _phase1b_reflect(self) -> None:
        """Bounded offline thought, with zero new observations or outcome credit.

        Consolidated hypotheses remain tagged as inferred, never facts. They may
        fill a later retrieval miss but cannot override new causal evidence.
        """
        depth = {"limited_reflection": 1, "extended_reflection": 3,
                 "shuffled_relationships": 3}.get(self.condition, 0)
        if not depth:
            return
        for axis in AXES:
            for token in self.world.options[axis]:
                cue = f"signal:{axis}:{token}"
                hits, _ = self._retrieve((axis,))
                record = self.interpreter.transfer_evidence(
                    hits, frozenset({cue}), review_depth=depth)
                self.cost["offline_reflection_evaluations"] += record["evaluations"]
                if not record["evidence_ids"]:
                    continue
                for hit in hits:
                    if cue in self.memory.chunks[hit.chunk_id].semantic_tags:
                        self.memory.store_layer_meaning(hit.chunk_id, axis, cue, {
                            "status": "inferred_from_observed_outcomes",
                            "score": record["score"],
                            "evidence_ids": record["evidence_ids"],
                            "depth": depth,
                        })
                        break
                reflection_state = TransferState(
                    episode_id="training-reflection", current_time=self.clock)
                self.refiner.record_transfer(
                    reflection_state, preferred=cue,
                    evidence_ids=record["evidence_ids"], review_depth=depth)

    def _consolidated_score(self, cue: str) -> float | None:
        values = [meaning for (_, _, name), meaning in self.memory.layer_meanings.items()
                  if name == cue and isinstance(meaning, dict)
                  and meaning.get("status") == "inferred_from_observed_outcomes"]
        return float(values[-1]["score"]) if values else None

    def _retrieve(self, axes: tuple[str, ...], *, top_k: int | None = None):
        tags = {"goal:workshop_recovery"}
        for axis in axes:
            tags.add(f"axis:{axis}")
            tags.update(f"signal:{axis}:{t}" for t in self.world.options[axis])
        query = RetrievalQuery(
            goal_id="workshop_recovery", semantic_tags=frozenset(tags),
            current_time=self.clock,
        )
        if top_k is None:
            hits, stats = self.router.retrieve(query)
        else:
            hits, stats = MemoryRouter(self.memory, RetrievalMode.STRUCTURAL, top_k=top_k).retrieve(query)
        self.cost["header_evaluations"] += stats.header_evaluations
        self.cost["detail_events"] += stats.active_detail_events
        self.cost["retrieval_ms"] += stats.latency_ms
        return hits, stats

    def _decision(self, state: TransferState, actions: list[Action], axes: tuple[str, ...],
                  *, round_no: int, step: str):
        hits, stats = self._retrieve(axes)
        depth = {
            "untrained_structural": -1, "trained_structural": -1,
            "full_no_reflection": 0, "limited_reflection": 1,
            "extended_reflection": 3, "strong_one_pass": 0,
            "shuffled_relationships": 3,
        }[self.condition]
        started = perf_counter()
        evidence = {}
        audits = {}
        for action in actions:
            if depth < 0:
                break
            record = self.interpreter.transfer_evidence(
                hits, state.action_features(action), review_depth=depth)
            audits[action.label] = record
            estimate = record["score"]
            if not record["evidence_ids"]:
                # Only when detailed retrieval misses all sources may dormant,
                # provenance-tagged consolidation provide a weak hypothesis.
                backup = [self._consolidated_score(cue) for cue in
                          state.action_features(action) if cue.startswith("signal:")]
                backup = [value for value in backup if value is not None]
                if backup:
                    estimate = sum(backup) / len(backup) * 0.5
            evidence[action.label] = 0.85 * estimate
            self.cost["interpretation_evaluations"] += record["evaluations"]
        # Every choice gets the same exploration noise in all conditions.
        rng_seed = self.seed * 991 + round_no * 97 + len(state.attempted_actions) * 13 + len(step)
        selector = ActionSelector(self.memory, self.learner, random.Random(rng_seed))
        decision = selector.select(state, actions, hits, transfer_scores=evidence if depth >= 0 else None)
        shadow = None
        if depth > 0:
            # Genuine paired one-pass counterfactual: same evidence and RNG.
            onepass = {}
            for action in actions:
                audit = self.interpreter.transfer_evidence(
                    hits, state.action_features(action), review_depth=0)
                onepass[action.label] = 0.85 * audit["score"]
                self.cost["counterfactual_evaluations"] += audit["evaluations"]
            shadow = ActionSelector(self.memory, self.learner, random.Random(rng_seed)).select(
                state, actions, hits, transfer_scores=onepass).action.label
            self.refiner.record_transfer(
                state, preferred=decision.action.label,
                evidence_ids=audits[decision.action.label]["evidence_ids"],
                alternatives=[a.label for a in actions if a != decision.action],
                review_depth=depth,
            )
        latency = (perf_counter() - started) * 1000.0
        self.cost["decisions"] += 1
        self.cost["selection_ms"] += latency
        relevant = audits.get(decision.action.label, {})
        return decision, {
            "round": round_no, "step": step, "action": decision.action.label,
            "predicted_success": decision.predicted_success,
            "score": decision.score, "retrieved_chunk_ids": [h.chunk_id for h in hits],
            "detail_events": stats.active_detail_events,
            "categories": relevant.get("categories", {}),
            "provenance": relevant.get("provenance", {}),
            "contributing_categories": sum(abs(x) > 1e-12 for x in relevant.get("categories", {}).values()),
            "one_pass_counterfactual_action": shadow,
            "reflection_changed_action": bool(shadow and shadow != decision.action.label),
            "review_depth": max(depth, 0),
            "selection_ms": latency,
            "outcome": None,
        }

    def _state(self, episode: str) -> TransferState:
        return TransferState(
            episode_id=episode, current_time=self.clock,
            excluded_axis=self.disabled_axis,
            scramble=self.condition == "shuffled_relationships",
            choices=self.world.options,
        )

    def _record_step(self, record: dict, success: bool) -> None:
        record["outcome"] = "success" if success else "failure"
        self.traces.append(record)

    def _execute_round(self, round_no: int) -> dict:
        # Matched across treatments: halfway through evaluation, the previously
        # effective temporal relation becomes obsolete. The notice reveals only
        # that a revision happened, NOT the new valid option.
        if round_no == self.rounds // 2:
            self.world.correct["time"] = self.world.original_time
            notice = self._event(
                f"revision-{round_no}", round_no,
                kind=EventKind.OBSERVATION, truth=TruthStatus.FACT,
                content="The workshop protocol has changed; previous temporal advice may be obsolete",
                tags={"goal:workshop_recovery", "axis:time", "revision:announced"},
                informative=True, metadata={"change_notice": True},
            )
            self.memory.create_chunk(
                events=[notice], transition_reason="revision_notice",
                self_relevance=1.0, salience=1.0,
            )
        scene = self.world.scene(self.seed, round_no)
        episode = f"novel-{round_no}-{scene['object']}"
        state = self._state(episode)
        # The visible proposals are randomized neutral labels. Their correctness
        # is never encoded in candidate labels or memory queries.
        rng = random.Random(self.seed * 83 + round_no * 43)
        # The environment supplies one feasible and three infeasible plans.
        # Every visible proposal is an opaque combination of five neutral cues,
        # not a correctness label. The solver has no access to world.correct.
        plans = [dict(self.world.correct)]
        for error_count in (1, 2, 3):
            proposal = dict(self.world.correct)
            for axis in rng.sample(list(AXES), error_count):
                proposal[axis] = next(token for token in self.world.options[axis]
                                      if token != self.world.correct[axis])
            plans.append(proposal)
        rng.shuffle(plans)
        actions = []
        for idx, proposal in enumerate(plans):
            action = Action("integrate", target=f"proposal-{idx}-{scene['object']}")
            actions.append(action)
            state.option_features[action.label] = frozenset(
                f"signal:{axis}:{token}" for axis, token in proposal.items())
        decision, record = self._decision(state, actions, AXES, round_no=round_no, step="joint_plan")
        chosen = plans[actions.index(decision.action)]
        plan_correct = all(self.world.accepted(axis, token) for axis, token in chosen.items())
        record["plan_correct"] = plan_correct
        record["surface_object"] = scene["object"]
        record["surface_person"] = scene["person"]
        self._record_step(record, plan_correct)
        wrong, retries, actions_count = 0, 0, 1
        # Stepwise independent actions with causal feedback. The environment
        # requires all five dependencies; no hidden truth is exposed to policy.
        completed = {}
        for axis in scene["order"]:
            candidates = [Action("resolve", target=token, key_name=axis)
                          for token in self.world.options[axis]]
            for action in candidates:
                state.option_features[action.label] = frozenset({f"signal:{axis}:{action.target}"})
            for attempt in range(2):
                state.current_time = self.clock
                selected, step_record = self._decision(
                    state, candidates, (axis,), round_no=round_no, step=axis)
                success = self.world.accepted(axis, selected.action.target)
                self._record_step(step_record, success)
                state.record_action(selected.action)
                self._feedback(episode, actions_count, selected.action, success,
                               source=scene["object"], person=scene["person"], state=state)
                actions_count += 1
                if success:
                    completed[axis] = selected.action.target
                    break
                wrong += 1
                retries += 1
                candidates = [a for a in candidates if a != selected.action]
                if not candidates:
                    break
        # Phase-2 corrective plan is derived from observed, causally confirmed
        # subtask outcomes; no direct read of generator solution.
        if not plan_correct and len(completed) == len(AXES):
            plan_correct = all(
                self.world.accepted(axis, token) for axis, token in completed.items())
            actions_count += 1
            self.traces.append({
                "round": round_no, "step": "revised_plan",
                "action": "integrate:observed_subtasks",
                "outcome": "success" if plan_correct else "failure",
                "evidence_ids": [e.event_id for e in self.memory.events.values()
                                 if e.episode_id == episode and e.kind == EventKind.OUTCOME],
            })
        # A finite work window makes wasted probes consequential; otherwise
        # every condition could eventually brute-force every binary decision.
        budget = 6
        return {
            "round": round_no, "scene": scene, "initial_plan_correct": record["plan_correct"],
            "success": plan_correct and len(completed) == len(AXES) and actions_count <= budget,
            "axes_completed": sorted(completed), "action_budget": budget,
            "actions": actions_count, "wrong_decisions": wrong + int(not record["plan_correct"]),
            "retries": retries, "correct_order": len(completed) == len(AXES),
        }

    def run(self) -> dict:
        self.train()
        self._phase1b_reflect()
        phase1 = {}
        recall = {}
        for axis in AXES:
            features = [frozenset({f"signal:{axis}:{token}", "action:choose", "role:workshop"})
                        for token in self.world.options[axis]]
            preferences = [self.learner.expected_utility(x) for x in features]
            expected = self.world.correct[axis]
            chosen = self.world.options[axis][preferences.index(max(preferences))]
            phase1[axis] = {
                "learned_correct_preference": chosen == expected,
                "preference_margin": round(max(preferences) - min(preferences), 6),
                "observations": sum(t["axis"] == axis for t in self.training),
            }
            hits, _ = self._retrieve((axis,))
            recall[axis] = any(
                e.kind == EventKind.OUTCOME and e.outcome == "success"
                and f"signal:{axis}:{expected}" in e.semantic_tags
                for hit in hits for e in self.memory.chunk_events(hit.chunk_id))
        # Pickle is a process-independent serialization round-trip; reconstruct
        # owners from state after reload rather than retaining old object pointers.
        before = (copy.deepcopy(self.memory), copy.deepcopy(self.learner))
        memory_bytes = pickle.dumps((self.memory, self.learner, self.refiner))
        restored_memory, restored_learner, restored_refiner = pickle.loads(memory_bytes)
        preserved = (
            len(restored_memory.events) == len(before[0].events) and
            restored_learner.snapshot() == before[1].snapshot() and
            all(restored_memory.tag_index.get(k) == before[0].tag_index.get(k)
                for k in before[0].tag_index)
        )
        self.memory, self.learner = restored_memory, restored_learner
        self.router = MemoryRouter(self.memory, RetrievalMode.STRUCTURAL, top_k=self.router.top_k)
        self.interpreter = MeaningInterpreter(self.memory, self.learner)
        self.refiner = restored_refiner
        rounds = [self._execute_round(i) for i in range(self.rounds)]
        probabilities = [
            (t["predicted_success"], float(t["outcome"] == "success"))
            for t in self.traces if t.get("predicted_success") is not None
        ]
        return {
            "seed": self.seed, "condition": self.condition, "disabled_axis": self.disabled_axis,
            "phase1": phase1, "recall": recall, "restart_preserved": preserved,
            "rounds": rounds, "traces": self.traces,
            "cost": dict(self.cost),
            "brier": statistics.mean((p - y)**2 for p, y in probabilities) if probabilities else None,
            "training_count": len(self.training), "stored_event_count": len(self.memory.events),
        }


def run_transfer_experiment(output_dir: str | Path, *,
                            seeds=(7, 23, 61, 101, 149),
                            rounds: int = 4, distractors: int = 4) -> dict:
    """Reproducible matched treatment, category-ablation and counterfactual runs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = [TransferRun(seed, condition, rounds=rounds, distractors=distractors).run()
            for seed in seeds for condition in CONDITIONS]
    ablations = [
        TransferRun(seed, "extended_reflection", disabled_axis=axis,
                    rounds=rounds, distractors=distractors).run()
        for seed in seeds for axis in AXES
    ]

    def aggregate(group):
        trials = [r for run in group for r in run["rounds"]]
        return {
            "completion_rate": statistics.mean(float(t["success"]) for t in trials),
            "initial_plan_accuracy": statistics.mean(float(t["initial_plan_correct"]) for t in trials),
            "mean_actions": statistics.mean(t["actions"] for t in trials),
            "mean_wrong_decisions": statistics.mean(t["wrong_decisions"] for t in trials),
            "mean_brier": statistics.mean(r["brier"] for r in group if r["brier"] is not None),
            "mean_detail_events": statistics.mean(r["cost"].get("detail_events", 0) for r in group),
            "mean_interpretation_evaluations": statistics.mean(r["cost"].get("interpretation_evaluations", 0) for r in group),
            "mean_offline_reflection_evaluations": statistics.mean(r["cost"].get("offline_reflection_evaluations", 0) for r in group),
            "mean_counterfactual_evaluations": statistics.mean(r["cost"].get("counterfactual_evaluations", 0) for r in group),
            "mean_latency_ms": statistics.mean(r["cost"].get("selection_ms", 0) + r["cost"].get("retrieval_ms", 0) for r in group),
            "reflection_action_changes": sum(
                bool(t.get("reflection_changed_action")) for r in group for t in r["traces"]),
            "restarts_preserved": all(r["restart_preserved"] for r in group),
            "learning_accuracy_by_category": {
                axis: statistics.mean(float(r["phase1"][axis]["learned_correct_preference"])
                                      for r in group) for axis in AXES
            },
            "recall_by_category": {
                axis: statistics.mean(float(r["recall"][axis]) for r in group) for axis in AXES
            },
            "round_completion": [
                statistics.mean(float(r["rounds"][i]["success"]) for r in group)
                for i in range(rounds)
            ],
        }

    summary = {condition: aggregate([r for r in runs if r["condition"] == condition])
               for condition in CONDITIONS}
    category_ablation = {axis: aggregate([r for r in ablations if r["disabled_axis"] == axis])
                         for axis in AXES}
    main = summary["extended_reflection"]
    control = summary["full_no_reflection"]
    saved_actions = control["mean_actions"] - main["mean_actions"]
    extra_processing = (
        main["mean_interpretation_evaluations"] +
        main["mean_offline_reflection_evaluations"] -
        control["mean_interpretation_evaluations"] -
        control["mean_offline_reflection_evaluations"]
    )
    verdict = {
        "reflection_completion_gain": main["completion_rate"] - control["completion_rate"],
        "reflection_action_savings": saved_actions,
        "extra_interpretation_evaluations": extra_processing,
        "better_than_strong_one_pass": main["completion_rate"] > summary["strong_one_pass"]["completion_rate"],
        "compounding_completion_gain": (
            main["round_completion"][-1] - main["round_completion"][0]
        ),
        "reflection_cost_justified": (
            main["completion_rate"] > control["completion_rate"] or
            (main["completion_rate"] == control["completion_rate"] and saved_actions > 0)
        ) and extra_processing >= 0,
    }
    payload = {
        "experiment": 6, "seeds": list(seeds), "rounds": rounds,
        "architecture": "existing MemoryRouter/EpisodicMemory/MeaningInterpreter/OutcomeLearner/ActionSelector/ProgressiveRefiner",
        "scope": "narrow synthetic causal association transfer; not general intelligence",
        "conditions": summary, "ablations": category_ablation, "verdict": verdict,
        "limitations": [
            "Synthetic binary associations, not open-world causal discovery",
            "Five independent prerequisite choices plus a joint proposal; ordering is presented as task structure",
            "A serialization round-trip is not a separate operating-system process restart",
            "Latency is platform-specific; event counts are compute proxies",
            "Counterfactual decisions on matched state do not establish counterfactual future trajectories",
        ],
    }
    (output / "transfer_trials.json").write_text(
        json.dumps({"runs": runs, "ablations": ablations}, indent=2, sort_keys=True), encoding="utf-8")
    (output / "transfer_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
