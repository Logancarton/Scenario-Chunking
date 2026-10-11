"""Experiment 10: paired predictive temporal decision-value research.

The runner only receives public promises, sensed observations, observed
dependency updates and action feedback. The environment owns private truth.
Learning and decision authority remain with their existing cognitive owners.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import random
import statistics
from time import perf_counter

from .environment import TemporalDecisionWorld, TemporalDelivery
from .learning import OutcomeLearner
from .meaning import MeaningInterpreter
from .memory import EpisodicMemory
from .models import Action, EnvironmentSignal, Event, EventKind, RetrievalMode, RetrievalQuery, TruthStatus
from .policy import ActionSelector
from .refinement import ProgressiveRefiner
from .retrieval import MemoryRouter
from .state import CognitiveState

CONDITIONS = (
    "untrained", "fixed_clock", "learned_prediction",
    "learned_focus_one_pass", "time_triggered",
    "predictive_value", "compute_matched_one_pass", "empty_reflection",
)


def _store(memory: EpisodicMemory, state: CognitiveState, signal: EnvironmentSignal,
           *, action: Action | None = None, cause: str | None = None) -> Event:
    event = Event(
        event_id=memory.next_event_id(), episode_id=state.episode_id,
        goal_id=state.goal_id, step=int(state.current_time),
        occurrence_time=state.world_time, acquired_time=state.world_time,
        storage_time=state.world_time, kind=signal.kind,
        truth_status=signal.truth_status, content=signal.content,
        semantic_tags=frozenset(set(signal.semantic_tags) | {"signal:temporal", "goal:temporal_decisions"}),
        person_id=signal.person_id, informative=signal.informative,
        confidence=signal.confidence, reward=signal.reward,
        action=action, cause_event_id=cause, outcome=signal.outcome,
        metadata=dict(signal.metadata))
    memory.create_chunk(
        events=[event], transition_reason="temporal_decision_evidence",
        self_relevance=.9, salience=.75)
    if action is None:
        state.apply_signal(signal, event_id=event.event_id)
        state.observed_evidence_ids.append(event.event_id)
    return event


def _experience(seed: int, index: int) -> tuple[TemporalDecisionWorld, str]:
    """Independent deterministic training worlds; no held-out trajectories."""
    rng = random.Random(seed * 100003 + index * 73 + 21)
    contexts = ("routine", "night", "urgent")
    context = contexts[index % len(contexts)]
    promised = rng.choice((4.0, 6.0, 9.0))
    multiplier = {"routine": 0.9, "night": 1.4, "urgent": 0.75}[context]
    duration = max(0.5, rng.gauss(promised * multiplier, promised * .26))
    other = max(.5, rng.gauss(promised, promised * .35))
    deliveries = (
        TemporalDelivery("Ada", context, promised, duration, promised + 4, 1.2, 1.3),
        TemporalDelivery("Bea", "routine", promised, other, promised + 4, 1.0, 1.0),
    )
    return TemporalDecisionWorld(deliveries), context


def train_temporal_evidence(seed: int, count: int, *,
                            distractors: int = 0) -> tuple[OutcomeLearner, EpisodicMemory]:
    """Feed direct instrument evidence to the SAME OutcomeLearner and memory."""
    learner, memory = OutcomeLearner(), EpisodicMemory()
    for i in range(count):
        world, context = _experience(seed, i)
        state = CognitiveState(f"training:{seed}:{i}", goal_id="temporal_decisions")
        promise = world.promises()[0]
        source = _store(memory, state, promise)
        d = float(world.public_constraints()["Ada"]["promised"])
        state.schedule_temporal(
            expectation_id=f"train:{seed}:{i}", subject="Ada", context=context,
            source_event_id=source.event_id, due_at=d, confidence=.75)
        # A direct negative inspection is a lower bound, never a failed arrival.
        checked_at = max(1.0, d * .65)
        world.advance(checked_at)
        state.advance_world_time(checked_at)
        check = world.inspect("Ada")
        record = _store(memory, state, check)
        seen = state.observe_temporal(
            f"train:{seed}:{i}", observation_id=record.event_id,
            observed_at=checked_at, returned=bool(check.metadata["returned"]),
            direct_observation=True, arrival_at=check.metadata.get("arrival_at"))
        if seen:
            learner.learn_temporal_observation(
                state.timed_expectations[f"train:{seed}:{i}"],
                observed_at=checked_at, returned=bool(check.metadata["returned"]),
                arrival_at=check.metadata.get("arrival_at"))
        # Follow-up direct checks after censored visits, no oracle timestamps.
        if not check.metadata["returned"]:
            later = d * 2.5
            world.advance(later)
            state.advance_world_time(later)
            later_signal = world.inspect("Ada")
            later_event = _store(memory, state, later_signal)
            accepted = state.observe_temporal(
                f"train:{seed}:{i}", observation_id=later_event.event_id,
                observed_at=later, returned=bool(later_signal.metadata["returned"]),
                direct_observation=True, arrival_at=later_signal.metadata.get("arrival_at"))
            if accepted:
                learner.learn_temporal_observation(
                    state.timed_expectations[f"train:{seed}:{i}"],
                    observed_at=later, returned=bool(later_signal.metadata["returned"]),
                    arrival_at=later_signal.metadata.get("arrival_at"))
        # Source accuracy is updated ONLY after an observed comparison.
        speaker = ("Rey", "Sol")[i % 2]
        report = random.Random(seed * 17 + i).random() < (.8 if speaker == "Rey" else .35)
        actual_present = bool(check.metadata["returned"])
        learner.verify_temporal_source(speaker, context, observed_at=checked_at,
                                       correct=(report == actual_present))
    if distractors:
        memory.inject_distractors(distractors, start_time=0.0)
    return learner, memory


def heldout_temporal_world(seed: int, case: int) -> TemporalDecisionWorld:
    """New temporal/causal combinations: evolving source reports and task edges."""
    rng = random.Random(seed * 65537 + case * 997 + 901)
    durations = []
    for idx, person in enumerate(("Ada", "Bea", "Cy")):
        context = ("routine", "night", "urgent")[(case + idx) % 3]
        promised = float(rng.choice((3, 5, 7, 11)))  # includes unseen promise spans
        scale = {"routine": .95, "night": 1.35, "urgent": .8}[context]
        if case % 4 == 3 and idx == 1:
            scale *= 1.8  # unseen causal/temporal combination
        duration = max(.5, rng.gauss(promised * scale, promised * .30))
        if case % 5 == 4 and idx == 2:
            duration = None  # unobserved within episode
        deadline = promised + rng.choice((1., 3., 5.))
        durations.append(TemporalDelivery(
            person, context, promised, duration, deadline,
            rng.choice((.9, 1.3, 1.8)), rng.choice((.7, 1.2, 2.0))))
    speaker = ("Rey", "Sol")[case % 2]
    reported_person = durations[case % len(durations)]
    when = float(rng.choice((1, 2, 3)))
    actual_at_report = (reported_person.arrival_at is not None
                        and reported_person.arrival_at <= when)
    reliability = (.8 if speaker == "Rey" else .35)
    reported = (actual_at_report if rng.random() < reliability
                else not actual_at_report)
    change = ((float(rng.choice((2, 4, 6))), "Ada", "Bea")
              if case % 3 == 0 else None)
    down = frozenset({"Bea"}) if case % 7 == 0 else frozenset()
    return TemporalDecisionWorld(
        tuple(durations), inspection_cost=rng.choice((.08, .16, .28)),
        wait_cost=rng.choice((.025, .065, .12)),
        switch_reward=rng.choice((.12, .32, .52)),
        sensor_down=down,
        reports=((when, reported_person.person, reported, speaker),),
        dependency_change=change)


class DecisionValueRun:
    """Thin experimental sequencing; no duplicate inference or action authority."""

    def __init__(self, world: TemporalDecisionWorld, *, mode: str, seed: int,
                 learner: OutcomeLearner, memory: EpisodicMemory,
                 thought_cost: float = .025):
        if mode not in CONDITIONS:
            raise ValueError("unknown experimental control")
        self.world, self.mode = world, mode
        self.learner, self.memory = learner, memory
        self.state = CognitiveState(f"heldout:{seed}", goal_id="temporal_decisions")
        self.public = world.public_constraints()
        self.router = MemoryRouter(memory, RetrievalMode.STRUCTURAL, top_k=6)
        self.refiner = ProgressiveRefiner(learner)
        self.interpreter = MeaningInterpreter(memory, learner)
        self.selector = ActionSelector(memory, learner, random.Random(seed),
                                       refiner=self.refiner)
        self.thought_cost = thought_cost
        self.history: list[dict] = []
        self.total_details = 0
        self.total_headers = 0
        self.latency_ms = 0.0
        self.computation = 0
        self.changes = 0
        self.inspections = 0
        self.missed_opportunity = 0
        for signal in world.promises():
            event = _store(memory, self.state, signal)
            person = signal.person_id
            props = self.public[person]
            self.state.schedule_temporal(
                expectation_id="delivery:" + person, subject=person,
                context=props["context"], source_event_id=event.event_id,
                due_at=props["promised"], confidence=.75)

    def _receive(self, signal: EnvironmentSignal) -> None:
        event = _store(self.memory, self.state, signal)
        if signal.metadata.get("source_status") == "direct_observation":
            eid = "delivery:" + signal.person_id
            accepted = self.state.observe_temporal(
                eid, observation_id=event.event_id,
                observed_at=self.state.world_time,
                returned=bool(signal.metadata.get("returned")),
                direct_observation=True,
                arrival_at=signal.metadata.get("arrival_at"))
            if accepted:
                self.learner.learn_temporal_observation(
                    self.state.timed_expectations[eid],
                    observed_at=self.state.world_time,
                    returned=bool(signal.metadata.get("returned")),
                    arrival_at=signal.metadata.get("arrival_at"))
                for p, report in self.state.beliefs.get("temporal_reports", {}).items():
                    if p == signal.person_id:
                        claimed, speaker, _, reported_at = report
                        actual_at_report = (
                            signal.metadata.get("arrival_at") is not None and
                            signal.metadata["arrival_at"] <= reported_at)
                        # Witnessed ABSENCE at a later time validates a negative
                        # historical report; presence without timestamp does not.
                        if signal.metadata.get("returned") and signal.metadata.get("arrival_at") is None:
                            continue
                        self.learner.verify_temporal_source(
                            speaker, self.public[p]["context"],
                            observed_at=self.state.world_time,
                            correct=(claimed == actual_at_report))

    def _candidates(self) -> list[Action]:
        actions = [Action("wait"), Action("switch"), Action("stop")]
        for person, props in sorted(self.public.items()):
            if self.state.world_time <= props["deadline"]:
                actions.extend((Action("check", target=person),
                                Action("proceed", target=person)))
        return actions

    def _scores(self, actions: list[Action], values: dict[str, float]) -> dict[str, float]:
        scale = max(1.0, max(abs(v) for v in values.values()))
        return {a.label: max(-.45, min(.45, values.get(a.label, -2.0) / scale * .45))
                for a in actions}

    def step(self) -> dict:
        now = self.state.world_time
        for signal in self.world.advance(now):
            self._receive(signal)
        actions = self._candidates()
        query = RetrievalQuery(
            goal_id=self.state.goal_id,
            semantic_tags=frozenset({"goal:temporal_decisions", "signal:temporal"}),
            person_ids=frozenset(self.public),
            current_time=now)
        start = perf_counter()
        hits, stats = self.router.retrieve(query)
        interpreted = self.interpreter.temporal_action_values(
            self.state, self.public,
            inspection_cost=self.world.inspection_cost,
            wait_cost=self.world.wait_cost,
            switch_reward=self.world.switch_reward)
        values = interpreted["values"]
        immediate = dict(values)
        for action in actions:
            if action.kind == "check":
                immediate[action.label] = -self.world.inspection_cost
            if action.kind == "wait":
                immediate[action.label] = -self.world.wait_cost
        if self.mode == "fixed_clock":
            due = [p for p in sorted(self.public)
                   if self.state.timed_expectations["delivery:" + p].status == "due_unverified"]
            fulfilled = [p for p in sorted(self.public)
                         if self.state.timed_expectations["delivery:" + p].status == "fulfilled"]
            absent_to_recheck = [
                p for p in sorted(self.public)
                if (self.state.timed_expectations["delivery:" + p].status == "verified_absent"
                    and self.state.world_time -
                        self.state.timed_expectations["delivery:" + p].last_verified_absent_at >= 2
                    and self.state.world_time <= self.public[p]["deadline"])
            ]
            target = ("proceed:" + fulfilled[0] if fulfilled else
                      "check:" + due[0] if due else
                      "check:" + absent_to_recheck[0] if absent_to_recheck else
                      "switch" if all(self.state.world_time > p["deadline"]
                                      for p in self.public.values())
                      else "wait")
            immediate = {a.label: (2.0 if a.label == target else -2.0) for a in actions}
        if self.mode in ("learned_focus_one_pass", "compute_matched_one_pass"):
            immediate = values
        if self.mode == "untrained":
            # Fresh unspecialized learner (same observable evidence).
            blank = MeaningInterpreter(self.memory, OutcomeLearner())
            immediate = blank.temporal_action_values(
                self.state, self.public, inspection_cost=self.world.inspection_cost,
                wait_cost=self.world.wait_cost, switch_reward=self.world.switch_reward)["values"]
        if self.mode == "learned_prediction":
            immediate = {**immediate, **{k: v for k, v in values.items()
                                          if k.startswith("proceed:")}}
        provenance = tuple(dict.fromkeys(
            evidence for p in sorted(interpreted["provenance"])
            for evidence in interpreted["provenance"][p]))
        refine = self.mode in ("predictive_value", "time_triggered")
        decision = self.selector.select(
            self.state, actions, hits, transfer_scores=self._scores(actions, immediate),
            refine=refine, transfer_context={
                "cognitive_mode": "temporal_value",
                "candidate_values": values,
                "evidence_ids": provenance,
                "presence": interpreted["presence"],
                "thought_cost": self.thought_cost,
                "clock_rule": self.mode == "time_triggered",
            })
        if self.mode in ("compute_matched_one_pass", "empty_reflection"):
            # Actually recompute on the SAME evidence; never add fake votes.
            self.interpreter.temporal_action_values(
                self.state, self.public,
                inspection_cost=self.world.inspection_cost,
                wait_cost=self.world.wait_cost,
                switch_reward=self.world.switch_reward)
            self.computation += 2
        elif refine and decision.thought_trace:
            self.computation += decision.thought_trace["reasoning_cycles"]
        else:
            self.computation += 1
        self.changes += int(bool(decision.thought_trace and
                                  decision.thought_trace["meaningfully_changed_action"]))
        self.total_headers += stats.header_evaluations
        self.total_details += stats.active_detail_events
        self.latency_ms += (perf_counter() - start) * 1000
        chosen = decision.action
        # Environment execution returns only observable feedback, not hidden times.
        delta = self.world.apply(chosen, known_dependencies={
            p: dep for p, (dep, _) in
            self.state.beliefs.get("temporal_dependencies", {}).items()})
        if chosen.kind == "check":
            self.inspections += 1
            self._receive(self.world.inspect(chosen.target))
        if chosen.kind not in ("wait",):
            self.state.record_action(chosen)
        if chosen.kind == "proceed":
            outcome = EnvironmentSignal(
                kind=EventKind.OUTCOME, truth_status=TruthStatus.OUTCOME,
                content="Executed dependent work; outcome observed",
                semantic_tags=frozenset({"signal:temporal", "action:proceed"}),
                outcome="success" if delta > 0 else "failure",
                reward=delta, informative=True)
            _store(self.memory, self.state, outcome)
        row = {
            "time": now, "action": chosen.label, "reward": delta,
            "presence_prediction": interpreted["presence"],
            "evidence_ids": list(provenance),
            "chosen_utility": values.get(chosen.label),
            "changed_by_refinement": bool(decision.thought_trace and
                                          decision.thought_trace["meaningfully_changed_action"]),
            "thought": decision.thought_trace,
            "retrieval_headers": stats.header_evaluations,
            "retrieval_details": stats.active_detail_events,
        }
        self.history.append(row)
        if not self.world.finished:
            next_time = now + 1.0
            self.state.advance_world_time(next_time)
        return row

    def run(self, *, max_steps: int = 18) -> dict:
        for _ in range(max_steps):
            if self.world.finished:
                break
            self.step()
        if not self.world.finished:
            self.world.finished = True
            self.missed_opportunity += 1
        return {
            "mode": self.mode, "reward": self.world.reward,
            "success": any(row["action"].startswith("proceed:")
                           and row["reward"] > 0 for row in self.history),
            "actions": len(self.history), "inspections": self.inspections,
            "unnecessary_actions": sum(row["action"].startswith("check:")
                                      and self.world.reward < 0
                                      for row in self.history),
            "missed_opportunities": self.missed_opportunity,
            "refinement_action_changes": self.changes,
            "cognitive_operations": self.computation,
            "header_evaluations": self.total_headers,
            "detail_events": self.total_details,
            "latency_ms": self.latency_ms,
            "thought_versions": len(self.refiner.thread(self.state).versions)
                                if self.mode in ("predictive_value", "time_triggered") else 0,
            "trace": self.history,
        }


def run_decision_value_benchmark(*, seeds=tuple(range(30)),
                                 training_stages=(0, 20, 100),
                                 cases=tuple(range(6)),
                                 distractors: int = 0) -> dict:
    trials: list[dict] = []
    for seed in seeds:
        for stage in training_stages:
            trained, learned_memory = train_temporal_evidence(
                seed, stage, distractors=distractors)
            for case in cases:
                for mode in CONDITIONS:
                    # All policies get same history and same hidden held-out
                    # world, but post-decision observations remain policy-specific.
                    learner = (OutcomeLearner() if mode == "untrained"
                               else deepcopy(trained))
                    world = heldout_temporal_world(seed, case)
                    run = DecisionValueRun(
                        world, mode=mode, seed=seed * 101 + case,
                        learner=learner, memory=deepcopy(learned_memory))
                    outcome = run.run()
                    outcome.update(seed=seed, stage=stage, case=case)
                    trials.append(outcome)
    summary = {}
    for stage in training_stages:
        summary[str(stage)] = {}
        for mode in CONDITIONS:
            group = [t for t in trials if t["stage"] == stage and t["mode"] == mode]
            summary[str(stage)][mode] = {
                "completion_rate": statistics.mean(t["success"] for t in group),
                "mean_reward": statistics.mean(t["reward"] for t in group),
                "mean_actions": statistics.mean(t["actions"] for t in group),
                "mean_inspections": statistics.mean(t["inspections"] for t in group),
                "mean_cognitive_operations": statistics.mean(t["cognitive_operations"] for t in group),
                "mean_header_evaluations": statistics.mean(t["header_evaluations"] for t in group),
                "mean_details": statistics.mean(t["detail_events"] for t in group),
                "mean_latency_ms": statistics.mean(t["latency_ms"] for t in group),
                "refinement_changes": sum(t["refinement_action_changes"] for t in group),
            }
    return {"experiment": 10, "seeds": list(seeds),
            "training_stages": list(training_stages),
            "cases": list(cases), "conditions": list(CONDITIONS),
            "trials": trials, "summary": summary,
            "limitation": ("Synthetic reward is not human cognition. "
                           "Paired end-to-end rewards do not isolate causal "
                           "effects of individual thought revisions.")}


def write_decision_value_benchmark(output="results-e10", **kwargs) -> dict:
    payload = run_decision_value_benchmark(**kwargs)
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "decision_value_summary.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "trials"},
                   indent=2, sort_keys=True), encoding="utf-8")
    (dest / "decision_value_trials.json").write_text(
        json.dumps(payload["trials"], indent=2, sort_keys=True), encoding="utf-8")
    return payload
