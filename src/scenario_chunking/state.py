from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
import math

from .models import Action, EnvironmentSignal, EventKind, RetrievalQuery


@dataclass
class TimedExpectation:
    """Revisable temporal belief, not an historical observation.

    Due means verification is worth considering; it never asserts that the
    expected event failed to occur. Source evidence remains in EpisodicMemory.
    """
    expectation_id: str
    subject: str
    context: str
    source_event_id: str
    started_at: float
    due_at: float
    confidence: float
    status: str = "pending"
    observation_ids: list[str] = field(default_factory=list)
    arrived_at: float | None = None
    last_verified_absent_at: float | None = None

    @property
    def due_unverified(self) -> bool:
        return self.status == "due_unverified"


@dataclass
class Timeline:
    """Deterministic, injectable simulated clock for a continuing learner.

    Elapsed time and the number of processed experiences are different
    quantities: two consecutive experiences may be minutes or years apart.
    Advancing the clock never teaches anything; it only changes what is recent.
    """
    now: float = 0.0
    experiences: int = 0

    def advance(self, *, elapsed: float | None = None, at: float | None = None) -> float:
        if (elapsed is None) == (at is None):
            raise ValueError("advance by elapsed time or to an absolute time, not both")
        target = self.now + float(elapsed) if elapsed is not None else float(at)
        if not math.isfinite(target) or target < self.now:
            raise ValueError("simulated time is finite and monotonic")
        self.now = target
        self.experiences += 1
        return self.now


@dataclass
class CognitiveState:
    episode_id: str
    goal_id: str = "retrieve_object_from_locked_container"
    current_time: float = 0.0  # Existing processing/episode time; not wall time.
    world_time: float = 0.0  # Externally advanced, monotonic elapsed-world clock.
    timed_expectations: dict[str, TimedExpectation] = field(default_factory=dict)
    due_queue: list[tuple[float, str]] = field(default_factory=list)
    container_name: str = "container"
    storage_name: str = "storage"
    work_surface_name: str = "bench"
    person_name: str = "person"
    visible_keys: list[str] = field(default_factory=list)
    candidate_key: str | None = None
    contradicted_keys: set[str] = field(default_factory=set)
    inspected: set[str] = field(default_factory=set)
    asked_people: set[str] = field(default_factory=set)
    attempted_actions: list[str] = field(default_factory=list)
    active_memory: list[str] = field(default_factory=list)
    observed_evidence_ids: list[str] = field(default_factory=list)
    # Source-specific current observations: never confuse a key cue with truth.
    key_observations: dict[str, list[tuple[str, bool, float, str]]] = field(default_factory=dict)
    beliefs: dict[str, object] = field(default_factory=dict)
    goal_complete: bool = False
    # Opt-in E12 continuing activity. The goal owner is this existing state,
    # not an action outcome or a new planner.
    activity_id: str | None = None
    goal_requirements: frozenset[str] = frozenset()
    goal_evidence: dict[str, tuple[bool, float, str, str]] = field(default_factory=dict)
    goal_status: str = "active"
    execution_status: str = "not_started"
    progress_observations: list[dict] = field(default_factory=list)
    track_outcome_contradictions: bool = False  # Historical scorer remains a frozen comparison

    # Cross-experience E11 context; events are stored in the existing memory.
    context_cues: frozenset[str] = frozenset()
    # Explicit task identity; observations keep their distinct episode IDs.
    thought_key: str | None = None
    # E12 investigation. Hard constraints come only from a public authorization
    # signal; they can be added, never lifted, and no learning owner holds a
    # reference to them. Risk and preference are valuation, not permission.
    prohibited: frozenset[str] = frozenset()
    investigation: dict | None = None
    revealed: dict[str, str] = field(default_factory=dict)

    def authorize(self, prohibited) -> None:
        """Apply a public authorization restriction (monotone: only tightens)."""
        self.prohibited = self.prohibited | frozenset(prohibited)

    def permissible(self, action: Action) -> bool:
        return not (action.kind == "route" and action.target in self.prohibited)

    def investigation_context(self, constraints: dict) -> None:
        """Public decision constraints: hidden variables, domains, stakes, deferral."""
        self.investigation = {"hidden": dict(constraints["hidden"]),
                              "domains": dict(constraints["domains"]),
                              "stakes": {p: dict(v) for p, v in constraints["stakes"].items()},
                              "defer_value": float(constraints["defer_value"]),
                              "budget": int(constraints.get("budget", len(constraints["hidden"])))}
        self.authorize(constraints["prohibited"])

    def reveal(self, axis: str, value: str) -> None:
        if self.investigation is None or axis not in self.investigation["hidden"] or axis in self.revealed:
            raise ValueError("only an unrevealed hidden variable can be revealed")
        self.revealed[axis] = value

    def schedule_temporal(self, *, expectation_id: str, subject: str, context: str,
                          source_event_id: str, due_at: float, confidence: float) -> TimedExpectation:
        if (not expectation_id or not source_event_id or not subject or not context
                or expectation_id in self.timed_expectations):
            raise ValueError("a temporal expectation needs unique identity and sourced context")
        if (not math.isfinite(due_at) or due_at <= self.world_time
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("deadline must be in the future with bounded confidence")
        expectation = TimedExpectation(
            expectation_id, subject, context, source_event_id,
            self.world_time, due_at, confidence,
        )
        self.timed_expectations[expectation_id] = expectation
        heappush(self.due_queue, (due_at, expectation_id))
        return expectation

    def advance_world_time(self, now: float) -> tuple[str, ...]:
        """Run only due expectations; zero scans of unrelated or dormant history."""
        if not math.isfinite(now) or now < self.world_time:
            raise ValueError("world clock must be finite and monotonic")
        self.world_time = now
        due: list[str] = []
        while self.due_queue and self.due_queue[0][0] <= now:
            _, eid = heappop(self.due_queue)
            expectation = self.timed_expectations[eid]
            if expectation.status == "pending":
                expectation.status = "due_unverified"
                due.append(eid)
        return tuple(due)

    def observe_temporal(self, expectation_id: str, *, observation_id: str,
                         observed_at: float, returned: bool,
                         direct_observation: bool, arrival_at: float | None = None) -> bool:
        """Only a directly observed state changes temporal truth assessment.

        A testimony or absence of incoming messages is never counted as an
        observed non-return. A person may still arrive after an observed delay.
        """
        expectation = self.timed_expectations[expectation_id]
        if not observation_id or not math.isfinite(observed_at):
            raise ValueError("an observation requires identity and event time")
        if observed_at > self.world_time or observed_at < expectation.started_at:
            raise ValueError("observation lies outside the experienced clock")
        if arrival_at is not None and (not returned or
                not math.isfinite(arrival_at) or
                not expectation.started_at <= arrival_at <= observed_at):
            raise ValueError("arrival timestamp requires a witnessed, bounded arrival")
        if not direct_observation:
            return False
        if observation_id in expectation.observation_ids:
            return False
        expectation.observation_ids.append(observation_id)
        if returned:
            expectation.status = "fulfilled"
            expectation.arrived_at = arrival_at
        elif observed_at >= expectation.due_at and expectation.status != "fulfilled":
            expectation.status = "verified_absent"
            expectation.last_verified_absent_at = observed_at
        return True

    def apply_signal(self, signal: EnvironmentSignal, *, event_id: str | None = None) -> None:
        self.current_time += 1.0
        meta = signal.metadata
        if "container_name" in meta:
            self.container_name = str(meta["container_name"])
        if "storage_name" in meta:
            self.storage_name = str(meta["storage_name"])
        if "work_surface_name" in meta:
            self.work_surface_name = str(meta["work_surface_name"])
        if "person_name" in meta:
            self.person_name = str(meta["person_name"])
        if "visible_keys" in meta:
            self.visible_keys = list(meta["visible_keys"])
        if "candidate_key" in meta and signal.kind == EventKind.OBSERVATION:
            key = str(meta["candidate_key"])
            self.candidate_key = key
            self.beliefs["candidate_key_confidence"] = float(signal.confidence)
            if event_id:
                self.key_observations.setdefault(key, []).append(
                    (event_id, True, float(signal.confidence), "observed_clue"))
        if "contradicts_key" in meta and signal.kind == EventKind.OBSERVATION:
            key = str(meta["contradicts_key"])
            self.contradicted_keys.add(key)
            if event_id:
                self.key_observations.setdefault(key, []).append(
                    (event_id, False, float(signal.confidence), "observed_contradiction"))
            if self.candidate_key == key:
                self.candidate_key = None
        # A failed observed attempt is evidence against that key in THIS episode.
        # Outcome tags are not accepted unless the signal is an actual causal outcome.
        if (signal.kind == EventKind.OUTCOME and signal.truth_status.value == "outcome"
                and signal.cause_event_id and signal.outcome in ("failure", "success")):
            keys = sorted(tag[4:] for tag in signal.semantic_tags if tag.startswith("key:"))
            if len(keys) == 1:
                key = keys[0]
                if event_id:
                    self.key_observations.setdefault(key, []).append(
                        (event_id, signal.outcome == "success", float(signal.confidence),
                         "causal_attempt_outcome"))
                if signal.outcome == "failure" and self.track_outcome_contradictions:
                    self.contradicted_keys.add(key)
                    if self.candidate_key == key:
                        self.candidate_key = None
        # Testimony changes a revisable belief, not the truth of a pending event.
        if meta.get("source_status") == "unverified_testimony" and event_id:
            reports = self.beliefs.setdefault("temporal_reports", {})
            reports[str(meta.get("about", ""))] = (
                bool(meta.get("reported_present")), str(signal.person_id),
                event_id, float(meta.get("reported_at", self.world_time)))
        if "dependency_change" in meta and event_id:
            dependencies = self.beliefs.setdefault("temporal_dependencies", {})
            dependencies[str(meta["dependency_change"])] = (
                str(meta["required"]), event_id)
        if "inspected" in meta:
            self.inspected.add(str(meta["inspected"]))
        if "asked_person" in meta:
            self.asked_people.add(str(meta["asked_person"]))
        # Legacy worlds have no explicit goal criteria. Opt-in continuing
        # activities NEVER equate mechanical success with verified goal success.
        if signal.outcome == "success" and not self.goal_requirements:
            self.goal_complete = True

    def begin_goal_activity(self, activity_id: str, requirements) -> None:
        """Keep a goal open until independent, sourced observations cover its criteria."""
        required = frozenset(requirements)
        if self.activity_id is not None or not activity_id or not required:
            raise ValueError("an activity requires one unique identity and nonempty criteria")
        self.activity_id = str(activity_id)
        self.goal_requirements = required
        self.goal_status = "active"
        self.goal_complete = False
        self.execution_status = "active"

    def observe_activity_progress(self, *, event_id: str, occurred_at: float,
                                  observed_at: float, description: str) -> None:
        if self.activity_id is None:
            raise ValueError("no continuing activity")
        if len(self.progress_observations) >= 64:
            self.progress_observations.pop(0)
        self.progress_observations.append({
            "event_id": event_id, "occurred_at": float(occurred_at),
            "observed_at": float(observed_at), "description": description})

    def verify_goal(self, signal: EnvironmentSignal, *, event_id: str,
                    occurred_at: float, observed_at: float) -> str:
        """Assess source evidence, never an evaluator's oracle success label."""
        if self.activity_id is None or not event_id:
            raise ValueError("no identified activity or evidence")
        if signal.kind != EventKind.OBSERVATION or signal.truth_status.value != "fact":
            raise ValueError("goal verification requires a factual observation")
        check = signal.metadata.get("goal_check")
        if not isinstance(check, dict):
            raise ValueError("goal verification requires sourced criteria")
        requirement, source = check.get("requirement"), check.get("source")
        if requirement not in self.goal_requirements or not isinstance(source, str) or not source:
            raise ValueError("unknown requirement or missing evidence provenance")
        if type(check.get("satisfied")) is not bool:
            raise ValueError("goal observation must be a measured boolean")
        old = self.goal_evidence.get(requirement)
        if old and old[1] > occurred_at:
            return self.goal_status  # late acquisition is not necessarily newer truth
        if old and old[1] == occurred_at and old[0] != check["satisfied"]:
            self.goal_status, self.goal_complete = "uncertain", False
            return self.goal_status
        self.goal_evidence[requirement] = (
            check["satisfied"], float(occurred_at), str(source), event_id)
        if all(k in self.goal_evidence and self.goal_evidence[k][0]
               for k in self.goal_requirements):
            self.goal_status = "completed_verified"
        elif check["satisfied"]:
            self.goal_status = "partially_complete"
        elif check.get("final", False):
            self.goal_status = "failed"
        else:
            self.goal_status = "awaiting_verification"
        self.goal_complete = self.goal_status == "completed_verified"
        return self.goal_status

    def record_action(self, action: Action) -> None:
        self.attempted_actions.append(action.label)

    def incremental_context(self, cues: frozenset[str], timestamp: float) -> None:
        # timestamp is simulated event time (a Timeline value), not a count.
        self.context_cues = cues
        self.current_time = timestamp

    def candidate_actions(self) -> list[Action]:
        if self.goal_id == "response_dispatch":
            return [Action("route", target="standard"),
                    Action("route", target="express")]
        if self.goal_id == "investigative_dispatch":
            # Every viable action is a candidate; permission is enforced by the
            # selector, never by omitting an action from evaluation.
            info = self.investigation or {"hidden": {}, "budget": 0}
            spent = len(self.revealed)
            inspect = [Action("inspect", target=axis) for axis in sorted(info["hidden"])
                       if axis not in self.revealed and spent < info["budget"]]
            return [Action("route", target=p) for p in ("standard", "express", "defer")] + inspect
        actions: list[Action] = []
        if self.storage_name not in self.inspected:
            actions.append(Action("inspect_storage", target=self.storage_name))
        if self.work_surface_name not in self.inspected:
            actions.append(Action("inspect_work_surface", target=self.work_surface_name))
        if self.container_name not in self.inspected:
            actions.append(Action("inspect_container", target=self.container_name))
        if self.person_name not in self.asked_people:
            actions.append(Action("ask_informant", person=self.person_name))

        ordered_keys = list(self.visible_keys)
        if self.candidate_key and self.candidate_key in ordered_keys:
            ordered_keys.remove(self.candidate_key)
            ordered_keys.insert(0, self.candidate_key)
        for key in ordered_keys:
            actions.append(Action("try_key", target=self.container_name, key_name=key))
        return actions

    def action_features(self, action: Action) -> frozenset[str]:
        if self.goal_id in ("response_dispatch", "investigative_dispatch"):
            return self.context_cues | frozenset({
                "goal:" + self.goal_id, "action:" + action.kind,
                "procedure:" + (action.target or action.kind),
            })
        if self.goal_id == "temporal_decisions":
            signals = {"goal:temporal_decisions", f"action:{action.kind}"}
            if action.target:
                signals.add(f"person:{action.target}")
                expectation = self.timed_expectations.get("delivery:" + action.target)
                if expectation:
                    signals.add(f"state:{expectation.status}")
                    signals.add(f"context:{expectation.context}")
            return frozenset(signals)
        features = {
            f"goal:{self.goal_id}",
            f"action:{action.kind}",
            "role:locked_container",
        }
        if action.kind == "inspect_storage":
            features.add("role:storage")
        elif action.kind == "inspect_work_surface":
            features.add("role:work_surface")
        elif action.kind == "ask_informant":
            features.add("role:informant")
        elif action.kind == "try_key":
            features.add("role:key")
            if action.key_name:
                features.add(f"key:{action.key_name}")
            if self.candidate_key and action.key_name == self.candidate_key:
                features.add("clue_supported")
            if action.key_name in self.contradicted_keys:
                features.add("contradicted_key")
        return frozenset(features)

    def retrieval_query(self) -> RetrievalQuery:
        if self.goal_id in ("response_dispatch", "investigative_dispatch"):
            return RetrievalQuery(
                goal_id=self.goal_id,
                semantic_tags=frozenset(cue for cue in self.context_cues if "&" not in cue) |
                frozenset({"goal:" + self.goal_id, "procedure:standard", "procedure:express"}),
                current_time=self.current_time,
                desired_event_kinds=frozenset({EventKind.ACTION, EventKind.OUTCOME}),
            )
        tags = {
            f"goal:{self.goal_id}",
            "role:locked_container",
        }
        if self.candidate_key:
            tags.add("clue_supported")
            tags.add(f"key:{self.candidate_key}")
        if self.inspected:
            tags.update(f"inspected_role:{self._role_for_name(name)}" for name in self.inspected)
        return RetrievalQuery(
            goal_id=self.goal_id,
            semantic_tags=frozenset(tags),
            person_ids=frozenset({self.person_name}) if self.person_name else frozenset(),
            current_time=self.current_time,
            contradicted_keys=frozenset(self.contradicted_keys),
            desired_event_kinds=frozenset({EventKind.ACTION, EventKind.OUTCOME, EventKind.OBSERVATION}),
        )

    def _role_for_name(self, name: str) -> str:
        if name == self.storage_name:
            return "storage"
        if name == self.work_surface_name:
            return "work_surface"
        if name == self.container_name:
            return "locked_container"
        return "entity"


# Research-only state owner; validity is reconstructed from observed memory.
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
    invalidated_since: dict[str, float] = field(default_factory=dict)
    completed_axes: set[str] = field(default_factory=set)
    current_prerequisites: dict[str, set[str]] = field(default_factory=dict)

    def policy_features(self, action: Action) -> frozenset[str]:
        """Namespace new beliefs after an observed validity change.

        Previously learned success is historical evidence, not a current
        policy prior. New feedback gets distinct epoch-scoped associations.
        """
        features = set(self.action_features(action))
        for axis, epoch in self.invalidated_since.items():
            for cue in tuple(features):
                prefix = f"signal:{axis}:"
                if cue.startswith(prefix):
                    features.remove(cue)
                    features.add(f"epoch:{axis}:{int(epoch)}:{cue[len(prefix):]}")
        return frozenset(features)

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
