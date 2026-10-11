"""Experiment 9 orchestration only; temporal cognition stays with existing owners.

This deterministic world deliberately separates clock passage, testimony and
verified observations. No hidden arrival time is sent to the policy. Baselines
run with the same learner, memory route and observable action affordances.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import pickle
import random
import statistics
from time import perf_counter

from .environment import ReturnTimelineWorld
from .learning import OutcomeLearner
from .memory import EpisodicMemory
from .models import Action, Event, EventKind, RetrievalMode, RetrievalQuery, TruthStatus
from .policy import ActionSelector
from .refinement import ProgressiveRefiner
from .retrieval import MemoryRouter
from .state import CognitiveState


class TemporalResearchRun:
    def __init__(self, *, mode: str = "temporal", seed: int = 7,
                 learner: OutcomeLearner | None = None) -> None:
        if mode not in ("temporal", "learned_one_pass", "clock_only", "memory_only"):
            raise ValueError("unsupported temporal control")
        self.mode = mode
        self.rng = random.Random(seed)
        self.learner = learner if learner is not None else OutcomeLearner()
        self.memory = EpisodicMemory()
        self.router = MemoryRouter(self.memory, RetrievalMode.STRUCTURAL, top_k=6)
        self.refiner = ProgressiveRefiner(self.learner)
        self.state: CognitiveState | None = None
        self.world: ReturnTimelineWorld | None = None
        self.expectation_id = ""
        self.trace: list[dict] = []
        self.decisions = 0
        self.header_evaluations = 0
        self.detail_events = 0
        self.elapsed_ms = 0.0

    def _store(self, signal, *, prediction: bool = False) -> Event:
        assert self.state is not None
        now = self.state.world_time
        event = Event(
            event_id=self.memory.next_event_id(),
            episode_id=self.state.episode_id, goal_id=self.state.goal_id,
            step=len(self.trace), occurrence_time=now, acquired_time=now,
            storage_time=now,
            kind=EventKind.PREDICTION if prediction else signal.kind,
            truth_status=TruthStatus.PREDICTION if prediction else signal.truth_status,
            content=signal.content,
            semantic_tags=frozenset(signal.semantic_tags | {f"goal:{self.state.goal_id}"}),
            person_id=signal.person_id,
            informative=signal.informative, confidence=signal.confidence,
            metadata=dict(signal.metadata),
        )
        self.memory.create_chunk(
            events=[event],
            transition_reason="temporal_prediction" if prediction else "observed_temporal_signal",
            self_relevance=.8, salience=.7,
        )
        if not prediction:
            self.state.apply_signal(signal, event_id=event.event_id)
            self.state.observed_evidence_ids.append(event.event_id)
        return event

    def start(self, world: ReturnTimelineWorld, episode_id: str,
              *, context: str = "ordinary_return") -> dict:
        if self.state is not None:
            raise ValueError("start one episode per runner")
        self.world = world
        self.state = CognitiveState(
            episode_id=episode_id, goal_id="temporal:" + episode_id,
        )
        source = self._store(world.promise())
        self.expectation_id = "return:" + episode_id
        expectation = self.state.schedule_temporal(
            expectation_id=self.expectation_id, subject=world.subject,
            context=context, source_event_id=source.event_id,
            due_at=world.promised_duration, confidence=.9,
        )
        prediction_signal = world.promise()
        prediction_signal.content = "Prediction: return may occur near the promised deadline"
        prediction_signal.metadata = {
            "expectation_id": expectation.expectation_id,
            "source_event_id": source.event_id, "due_at": expectation.due_at,
        }
        self._store(prediction_signal, prediction=True)
        return self._decide()

    def _actions(self) -> list[Action]:
        assert self.world is not None
        return [Action(kind, target=self.world.subject) for kind in (
            "wait", "check_return", "follow_up", "acknowledge_return")]

    def _baseline_scores(self) -> dict[str, float]:
        assert self.state is not None
        expectation = self.state.timed_expectations[self.expectation_id]
        if self.mode == "clock_only":
            preferred = ("check_return" if expectation.status == "due_unverified"
                         else "wait" if expectation.status == "pending"
                         else "follow_up" if expectation.status == "verified_absent"
                         else "acknowledge_return")
        else:
            promised = expectation.due_at - expectation.started_at
            _, punctuality = self.learner.temporal_prediction(
                expectation.context, promised)
            expected_information_value = (
                (1.0 - punctuality) * expectation.confidence *
                (1.0 + min(1.0, max(0.0, self.state.world_time - expectation.due_at) /
                           max(1.0, promised))))
            preferred = (
                "wait" if expectation.status == "pending" else
                ("check_return" if expected_information_value > .25 else "wait")
                if expectation.status == "due_unverified" else
                "follow_up" if expectation.status == "verified_absent" else
                "acknowledge_return"
            )
        return {action.label: (0.45 if action.kind == preferred else -0.45)
                for action in self._actions()}

    def _decide(self) -> dict:
        assert self.state is not None
        query = RetrievalQuery(
            goal_id=self.state.goal_id,
            semantic_tags=frozenset({f"goal:{self.state.goal_id}",
                                     "signal:temporal"}),
            person_ids=frozenset({self.world.subject}),
            current_time=self.state.world_time,
        )
        started = perf_counter()
        hits, stats = self.router.retrieve(query)
        # The action selector uses the same retrieval and option set for both
        # the one-pass and refinement conditions; only refinement is switched.
        selector = ActionSelector(self.memory, self.learner, self.rng,
                                  refiner=self.refiner if self.mode == "temporal" else None)
        decision = selector.select(
            self.state, self._actions(), hits,
            transfer_scores=self._baseline_scores(),
            refine=self.mode == "temporal",
            transfer_context={
                "cognitive_mode": "temporal",
                "expectation_id": self.expectation_id,
                "consequence": 1.0,
                "inspection_cost": .25,
            })
        elapsed = (perf_counter() - started) * 1000
        self.decisions += 1
        self.header_evaluations += stats.header_evaluations
        self.detail_events += stats.active_detail_events
        self.elapsed_ms += elapsed
        expectation = self.state.timed_expectations[self.expectation_id]
        record = {
            "now": self.state.world_time, "processing_tick": self.state.current_time,
            "status": expectation.status, "action": decision.action.kind,
            "source_event_id": expectation.source_event_id,
            "observation_ids": list(expectation.observation_ids),
            "retrieval": [hit.chunk_id for hit in hits],
            "header_evaluations": stats.header_evaluations,
            "detail_events": stats.active_detail_events,
            "thought": decision.thought_trace,
        }
        self.trace.append(record)
        return record

    def advance(self, now: float, *, reported_present: bool | None = None) -> list[dict]:
        """Advance fake world time; only due expectations cause a cognitive wake.

        A report is stored as historical testimony, not upgraded to direct
        outcome evidence or permission for a new reasoning cycle.
        """
        assert self.state is not None and self.world is not None
        due = self.state.advance_world_time(now)
        if reported_present is not None:
            self._store(self.world.testimony(reported_present=reported_present))
        if not due or self.mode == "memory_only":
            return []
        initial = self._decide()
        result = [initial]
        if initial["action"] == "check_return":
            signal = self.world.check_return(now)
            event = self._store(signal)
            meta = signal.metadata
            direct = meta.get("source_status") == "direct_observation"
            accepted = self.state.observe_temporal(
                self.expectation_id, observation_id=event.event_id,
                observed_at=now, returned=bool(meta.get("returned", False)),
                direct_observation=direct, arrival_at=meta.get("arrival_at"),
            )
            if accepted:
                expectation = self.state.timed_expectations[self.expectation_id]
                self.learner.learn_temporal_observation(
                    expectation, observed_at=now,
                    returned=bool(meta["returned"]),
                    arrival_at=meta.get("arrival_at"),
                )
                result.append(self._decide())
        return result

    def observe_externally(self, signal, *, at: float) -> dict | None:
        """A new direct observation can wake a dormant thread after its due check."""
        assert self.state is not None
        self.state.advance_world_time(at)
        event = self._store(signal)
        direct = signal.metadata.get("source_status") == "direct_observation"
        if not direct:
            return None
        meta = signal.metadata
        accepted = self.state.observe_temporal(
            self.expectation_id, observation_id=event.event_id,
            observed_at=at, returned=bool(meta.get("returned")),
            direct_observation=True, arrival_at=meta.get("arrival_at"),
        )
        if not accepted:
            return None
        expectation = self.state.timed_expectations[self.expectation_id]
        self.learner.learn_temporal_observation(
            expectation, observed_at=at, returned=bool(meta["returned"]),
            arrival_at=meta.get("arrival_at"))
        return self._decide()


def run_temporal_benchmark(*, seeds=(7, 23, 61, 101, 149)) -> dict:
    """Paired observational scenarios; negative results are expected and reported.

    A prior training example is identical for each treatment. Future synthetic
    hidden times never enter state, policy or memory until directly observed.
    """
    cases = (
        ("early", 8.0, True, True, None),
        ("late", 16.0, True, True, None),
        ("missing", None, True, True, None),
        ("sensor_down", None, False, True, False),
        ("unknown_arrival_time", 8.0, True, False, True),
    )
    records = []
    for seed in seeds:
        for mode in ("temporal", "learned_one_pass", "clock_only", "memory_only"):
            learner = OutcomeLearner()
            # Matched evidence history across controls, learned through the
            # same owner. Context "ordinary_return" has one early and one late.
            for i, actual in enumerate((8.0, 16.0)):
                training = TemporalResearchRun(mode="temporal", seed=seed + i, learner=learner)
                training.start(ReturnTimelineWorld("training-person", 10, actual), f"train-{seed}-{i}")
                training.advance(11.0)
                if actual > 11:
                    training.observe_externally(
                        training.world.check_return(actual), at=actual)
            for case, actual, available, timestamp, rumor in cases:
                world = ReturnTimelineWorld(
                    subject=f"person-{seed}", promised_duration=10,
                    actual_arrival_at=actual, observation_available=available,
                    timestamp_available=timestamp)
                # Held-out feedback must never contaminate another trial.
                # Fork only after the identical training observations.
                frozen_learner = pickle.loads(pickle.dumps(learner))
                run = TemporalResearchRun(mode=mode, seed=seed, learner=frozen_learner)
                run.start(world, f"heldout-{seed}-{case}")
                idle = run.advance(5.0)
                assert not idle
                steps = run.advance(12.0, reported_present=rumor)
                expectation = run.state.timed_expectations[run.expectation_id]
                records.append({
                    "seed": seed, "mode": mode, "case": case,
                    "status": expectation.status,
                    "due_triggered": bool(steps),
                    "first_due_action": steps[0]["action"] if steps else None,
                    "final_action": steps[-1]["action"] if steps else None,
                    "evidence_count": len(expectation.observation_ids),
                    "false_failure_claim": (
                        expectation.status == "verified_absent"
                        and not expectation.observation_ids),
                    "decisions": run.decisions,
                    "header_evaluations": run.header_evaluations,
                    "detail_events": run.detail_events,
                    "latency_ms": run.elapsed_ms,
                    "thought_versions": len(run.refiner.thread(run.state).versions)
                                        if mode == "temporal" else 0,
                })
    summary = {}
    for mode in ("temporal", "learned_one_pass", "clock_only", "memory_only"):
        group = [r for r in records if r["mode"] == mode]
        summary[mode] = {
            "due_wakes": sum(r["due_triggered"] for r in group),
            "observed_absence": sum(r["status"] == "verified_absent" for r in group),
            "observed_return": sum(r["status"] == "fulfilled" for r in group),
            "false_failure_claims": sum(r["false_failure_claim"] for r in group),
            "mean_decisions": statistics.mean(r["decisions"] for r in group),
            "mean_header_evaluations": statistics.mean(r["header_evaluations"] for r in group),
            "mean_detail_events": statistics.mean(r["detail_events"] for r in group),
            "mean_latency_ms": statistics.mean(r["latency_ms"] for r in group),
        }
    return {"experiment": 9, "seeds": list(seeds), "cases": [c[0] for c in cases],
            "trials": records, "summary": summary,
            "limitation": "synthetic direct checks, not demonstrated superiority to matched one-pass"}


def write_temporal_benchmark(output: str = "results-temporal") -> dict:
    payload = run_temporal_benchmark()
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "temporal_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
