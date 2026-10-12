from __future__ import annotations

from dataclasses import dataclass
import random

from .models import Action, EnvironmentSignal, EventKind, TruthStatus


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    container_name: str
    storage_name: str
    work_surface_name: str
    person_name: str
    keys: tuple[str, ...]
    correct_key: str
    clue_source: str
    clue_text: str
    changed_from_key: str | None = None
    delayed_feedback_events: int = 0
    reported_key: str | None = None
    report_confidence: float | None = None


class LockedWorkshop:
    """Small deterministic environment. Ground truth never enters the policy or retrieval query."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.spec: EpisodeSpec | None = None
        self.drawer_seen = False
        self.bench_seen = False
        self.container_seen = False
        self.person_asked = False
        self.opened = False

    def reset(self, spec: EpisodeSpec) -> list[EnvironmentSignal]:
        self.spec = spec
        self.drawer_seen = self.bench_seen = self.container_seen = self.person_asked = self.opened = False
        return [
            EnvironmentSignal(
                kind=EventKind.OBSERVATION,
                truth_status=TruthStatus.FACT,
                content=(
                    f"A {spec.container_name} is locked. Several keys are visible: "
                    + ", ".join(spec.keys)
                    + f". Nearby are a {spec.storage_name}, {spec.work_surface_name}, and {spec.person_name}."
                ),
                semantic_tags=frozenset({
                    "goal:retrieve_object_from_locked_container",
                    "role:locked_container",
                    "role:key",
                    "role:storage",
                    "role:work_surface",
                    "role:informant",
                }),
                informative=True,
                metadata={
                    "container_name": spec.container_name,
                    "storage_name": spec.storage_name,
                    "work_surface_name": spec.work_surface_name,
                    "person_name": spec.person_name,
                    "visible_keys": list(spec.keys),
                },
            )
        ]

    def step(self, action: Action, *, cause_event_id: str) -> list[EnvironmentSignal]:
        if self.spec is None:
            raise RuntimeError("environment must be reset first")
        spec = self.spec

        if action.kind == "inspect_storage":
            self.drawer_seen = True
            if spec.clue_source == "storage":
                reported_key = spec.reported_key or spec.correct_key
                tags = {"role:storage", "clue:key_relation", f"key:{reported_key}"}
                metadata = {"inspected": spec.storage_name, "candidate_key": reported_key}
                if spec.changed_from_key:
                    tags.add(f"contradicts_key:{spec.changed_from_key}")
                    metadata["contradicts_key"] = spec.changed_from_key
                return [EnvironmentSignal(
                    kind=EventKind.OBSERVATION,
                    truth_status=TruthStatus.FACT,
                    content=spec.clue_text,
                    semantic_tags=frozenset(tags),
                    informative=True,
                    confidence=spec.report_confidence if spec.report_confidence is not None else 0.96,
                    metadata=metadata,
                )]
            return [EnvironmentSignal(
                kind=EventKind.OBSERVATION,
                truth_status=TruthStatus.FACT,
                content=f"The {spec.storage_name} contains unrelated screws and paper scraps.",
                semantic_tags=frozenset({"role:storage", "observation:irrelevant"}),
                informative=False,
                metadata={"inspected": spec.storage_name},
            )]

        if action.kind == "inspect_work_surface":
            self.bench_seen = True
            if spec.clue_source == "work_surface":
                reported_key = spec.reported_key or spec.correct_key
                return [EnvironmentSignal(
                    kind=EventKind.OBSERVATION,
                    truth_status=TruthStatus.FACT,
                    content=spec.clue_text,
                    semantic_tags=frozenset({"role:work_surface", "clue:key_relation", f"key:{reported_key}"}),
                    informative=True,
                    confidence=spec.report_confidence if spec.report_confidence is not None else 0.92,
                    metadata={"inspected": spec.work_surface_name, "candidate_key": reported_key},
                )]
            return [EnvironmentSignal(
                kind=EventKind.OBSERVATION,
                truth_status=TruthStatus.FACT,
                content=f"The {spec.work_surface_name} has tools but nothing identifying the lock key.",
                semantic_tags=frozenset({"role:work_surface", "observation:irrelevant"}),
                informative=False,
                metadata={"inspected": spec.work_surface_name},
            )]

        if action.kind == "inspect_container":
            self.container_seen = True
            return [EnvironmentSignal(
                kind=EventKind.OBSERVATION,
                truth_status=TruthStatus.FACT,
                content=f"The {spec.container_name} has an ordinary keyed lock with no visible damage.",
                semantic_tags=frozenset({"role:locked_container", "observation:lock"}),
                informative=False,
                metadata={"inspected": spec.container_name},
            )]

        if action.kind == "ask_informant":
            self.person_asked = True
            if spec.clue_source == "person":
                reported_key = spec.reported_key or spec.correct_key
                tags = {"role:informant", "clue:key_relation", f"key:{reported_key}"}
                metadata = {"asked_person": spec.person_name, "candidate_key": reported_key}
                if spec.changed_from_key:
                    tags.add(f"contradicts_key:{spec.changed_from_key}")
                    metadata["contradicts_key"] = spec.changed_from_key
                return [EnvironmentSignal(
                    kind=EventKind.OBSERVATION,
                    truth_status=TruthStatus.FACT,
                    person_id=spec.person_name,
                    content=spec.clue_text,
                    semantic_tags=frozenset(tags),
                    informative=True,
                    confidence=spec.report_confidence if spec.report_confidence is not None else 0.94,
                    metadata=metadata,
                )]
            return [EnvironmentSignal(
                kind=EventKind.OBSERVATION,
                truth_status=TruthStatus.FACT,
                person_id=spec.person_name,
                content=f"{spec.person_name} remembers there is a clue somewhere in the room but cannot recall where.",
                semantic_tags=frozenset({"role:informant", "observation:vague"}),
                informative=True,
                confidence=0.45,
                metadata={"asked_person": spec.person_name},
            )]

        if action.kind == "try_key":
            key = action.key_name or "unknown"
            success = key == spec.correct_key
            signals: list[EnvironmentSignal] = []
            for i in range(spec.delayed_feedback_events):
                signals.append(EnvironmentSignal(
                    kind=EventKind.OBSERVATION,
                    truth_status=TruthStatus.FACT,
                    content=f"A distracting workshop sound occurs while the lock response is pending ({i + 1}).",
                    semantic_tags=frozenset({"topic:distractor", "observation:delay"}),
                    informative=False,
                ))
            if success:
                self.opened = True
                signals.append(EnvironmentSignal(
                    kind=EventKind.OUTCOME,
                    truth_status=TruthStatus.OUTCOME,
                    outcome="success",
                    reward=1.0,
                    cause_event_id=cause_event_id,
                    content=f"The {key} key opens the {spec.container_name}; the object is retrieved.",
                    semantic_tags=frozenset({"outcome:success", "role:key", f"key:{key}", "goal:completed"}),
                    informative=True,
                ))
            else:
                signals.append(EnvironmentSignal(
                    kind=EventKind.OUTCOME,
                    truth_status=TruthStatus.OUTCOME,
                    outcome="failure",
                    reward=-0.35,
                    cause_event_id=cause_event_id,
                    content=f"The {key} key does not open the {spec.container_name}.",
                    semantic_tags=frozenset({"outcome:failure", "role:key", f"key:{key}"}),
                    informative=True,
                ))
            return signals

        raise ValueError(f"unknown action: {action.kind}")


def default_episode_specs() -> list[EpisodeSpec]:
    return [
        EpisodeSpec(
            episode_id="E1_initial",
            container_name="steel cabinet",
            storage_name="shallow drawer",
            work_surface_name="tool bench",
            person_name="Mira",
            keys=("brass", "blue", "black"),
            correct_key="brass",
            clue_source="storage",
            clue_text="A maintenance tag in the shallow drawer says the brass key matches the steel cabinet.",
        ),
        EpisodeSpec(
            episode_id="E2_similar",
            container_name="painted locker",
            storage_name="parts tray",
            work_surface_name="assembly table",
            person_name="Jon",
            keys=("copper", "white", "green"),
            correct_key="copper",
            clue_source="storage",
            clue_text="A parts card in the tray identifies the copper key as the locker key.",
        ),
        EpisodeSpec(
            episode_id="E3_temporal_interference",
            container_name="steel cabinet",
            storage_name="service drawer",
            work_surface_name="repair bench",
            person_name="Mira",
            keys=("brass", "steel", "black"),
            correct_key="steel",
            clue_source="storage",
            changed_from_key="brass",
            clue_text="A dated service notice says the lock was replaced: brass is obsolete and the steel key now fits.",
        ),
        EpisodeSpec(
            episode_id="E4_distractor_load",
            container_name="supply cabinet",
            storage_name="label drawer",
            work_surface_name="packing bench",
            person_name="Oren",
            keys=("yellow", "nickel", "red"),
            correct_key="nickel",
            clue_source="storage",
            clue_text="A current inventory label links the nickel key to the supply cabinet.",
        ),
        EpisodeSpec(
            episode_id="E5_delayed_feedback",
            container_name="test cabinet",
            storage_name="index drawer",
            work_surface_name="test bench",
            person_name="Aya",
            keys=("violet", "silver", "orange"),
            correct_key="silver",
            clue_source="storage",
            clue_text="The index card marks silver as the test-cabinet key.",
            delayed_feedback_events=3,
        ),
        EpisodeSpec(
            episode_id="E6_relational_transfer",
            container_name="portable toolbox",
            storage_name="document pouch",
            work_surface_name="rolling cart",
            person_name="Luis",
            keys=("bronze", "teal", "ivory"),
            correct_key="ivory",
            clue_source="person",
            clue_text="Luis explains that the ivory key is assigned to the portable toolbox.",
        ),
    ]


# Research-only procedural transfer environment; no production runtime imports it.
AXES = ("self", "salience", "time", "sequence", "person")
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

class UncertaintyWorld:
    """Four-choice, partially observed transfer world with a changing DAG.

    Truth remains inside the environment. Public reports are potentially wrong;
    only attempts and explicit, costly inspections establish current results.
    """

    def __init__(self, seed: int) -> None:
        rng = random.Random(seed * 101 + 29)
        selected = rng.sample(SYMBOLS, 4 * len(AXES))
        self.options = {
            axis: tuple(selected[4*i:4*i+4]) for i, axis in enumerate(AXES)
        }
        self.correct = {axis: rng.choice(self.options[axis]) for axis in AXES}
        self.order = rng.sample(list(AXES), len(AXES))
        self.active_order = list(self.order)
        self.original_time = self.correct["time"]
        self.revised = False

    def revise(self) -> None:
        if self.revised:
            return
        self.revised = True
        options = self.options["time"]
        self.correct["time"] = options[(options.index(self.correct["time"]) + 1) % len(options)]
        self.active_order[1], self.active_order[2] = self.active_order[2], self.active_order[1]

    def accepted(self, axis: str, token: str, *, historical: bool = False) -> bool:
        return token == (self.original_time if axis == "time" and historical
                         else self.correct[axis])

    def attempt(self, axis: str, token: str, completed: set[str]) -> tuple[str, str | None]:
        index = self.active_order.index(axis)
        if index and self.active_order[index - 1] not in completed:
            return "blocked", self.active_order[index - 1]
        return ("success" if self.accepted(axis, token) else "failure"), None

    def prerequisite(self, axis: str) -> str | None:
        index = self.active_order.index(axis)
        return self.active_order[index - 1] if index else None

    def scene(self, seed: int, round_no: int) -> dict:
        rng = random.Random(seed * 1009 + round_no * 83)
        return {"object": rng.choice(OBJECTS) + "-" + str(seed + round_no),
                "person": rng.choice(PEOPLE) + "-" + str(round_no),
                "context": "night" if round_no % 2 else "day"}

    def reports(self, seed: int, round_no: int, axis: str) -> list[dict]:
        """Generate context-dependent fallible testimony; not an answer key."""
        rng = random.Random(seed * 2903 + round_no * 107 + AXES.index(axis) * 17)
        choices = self.options[axis]
        context = "night" if round_no % 2 else "day"
        reports = []
        for source in ("Aster", "Beryl", "Cinder"):
            reliability = {"Aster": .84, "Beryl": .22,
                           "Cinder": .75 if context == "day" else .27}[source]
            actual = self.correct[axis]
            reported = (actual if rng.random() < reliability
                        else rng.choice([x for x in choices if x != actual]))
            reports.append({"source": source, "axis": axis, "token": reported,
                            "context": context, "issued_round": round_no})
        return reports


# Experiment 8 — independent contextual decision environment.
# World truth is private and is NEVER used as a policy feature or retrieval cue.
from itertools import product


@dataclass(frozen=True)
class ContextualTask:
    task_id: str
    device_name: str
    mode: str
    material: str
    phase: str
    actual_dependency: str
    reported_dependency: str
    source: str
    context: str
    consequence: float
    self_relevance: float
    revision_notice: bool
    profile: str
    protocol_epoch: str

    def public(self) -> dict:
        return {
            "task_id": self.task_id, "device_name": self.device_name,
            "mode": self.mode, "material": self.material,
            "phase": self.phase, "reported_dependency": self.reported_dependency,
            "source": self.source, "context": self.context,
            "consequence": self.consequence,
            "self_relevance": self.self_relevance,
            "revision_notice": self.revision_notice,
            "protocol_epoch": self.protocol_epoch,
        }


class ContextualWorld:
    """A four-choice conditional procedure, not five separate axis puzzles.

    Two independent causal interactions jointly determine the correct procedure.
    The pair relationships transfer across held-out four-way combinations.
    A fallible source reports the dependency, which can be inspected at a cost.
    """

    PROCEDURES = ("isolate", "purge", "recalibrate", "reseat")
    VALUES = ("stable", "unstable")
    SOURCES = ("Aster", "Beryl", "Cinder")
    FACTORS = tuple(product(range(2), repeat=4))
    # Each unseen combination has a different correct procedure, but its
    # causal sub-relations were individually encountered during training.
    HOLDOUT_KEYS = ((0, 0, 0, 0), (1, 0, 1, 0),
                    (0, 1, 1, 1), (1, 1, 0, 1))
    TRAIN_KEYS = tuple(x for x in FACTORS if x not in (
        (0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 1, 1), (1, 1, 0, 1)))

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def task(self, index: int, *, heldout: bool = False,
             complexity: int = 2) -> ContextualTask:
        if complexity not in (1, 2, 3):
            raise ValueError("complexity must be 1, 2, or 3")
        rng = random.Random(self.seed * 1000003 + index * 613 + (99991 if heldout else 0))
        pool = self.HOLDOUT_KEYS if heldout else self.TRAIN_KEYS
        bits = pool[index % len(pool)]
        mode, dependency, material, phase = bits
        context = ("day", "night")[index % 2]
        source = self.SOURCES[(index // 2) % 3]
        accuracy = {"Aster": .85, "Beryl": .22,
                    "Cinder": .75 if context == "day" else .3}[source]
        is_correct = rng.random() < (accuracy if complexity > 1 else 1.0)
        reported = dependency if is_correct else 1 - dependency
        return ContextualTask(
            task_id=f"{'transfer' if heldout else 'experience'}-{self.seed}-{index}",
            device_name=f"{rng.choice(('conveyor', 'press', 'pump', 'mixer'))}-{rng.randrange(100000)}",
            mode=("warm", "cold")[mode],
            material=("porous", "sealed")[material],
            phase=("initial", "recovery")[phase],
            actual_dependency=self.VALUES[dependency],
            reported_dependency=self.VALUES[reported],
            source=source, context=context,
            consequence=(.25, .7, 1.0)[complexity - 1],
            self_relevance=(.3, .65, .95)[complexity - 1],
            revision_notice=(complexity == 3 and index % 2 == 1),
            profile=("simple", "uncertain", "adversarial")[complexity - 1],
            protocol_epoch=("v2" if index >= 500 and not heldout
                            or (heldout and complexity == 3 and index % 2 == 1)
                            else "v1"),
        )

    @staticmethod
    def _bit(value: str, positive: str) -> int:
        return int(value == positive)

    def outcome(self, task: ContextualTask, procedure: str) -> bool:
        if procedure not in self.PROCEDURES:
            raise ValueError("unknown procedure")
        # The agent observes outcomes only; it never receives this equation.
        bit0 = self._bit(task.mode, "cold") ^ self._bit(task.actual_dependency, "unstable")
        if task.protocol_epoch == "v2" and task.mode == "cold":
            bit0 ^= 1  # Procedural reversal: earlier successes remain historically true.
        bit1 = self._bit(task.material, "sealed") ^ self._bit(task.phase, "recovery")
        return procedure == self.PROCEDURES[bit0 + 2 * bit1]

    def inspect_dependency(self, task: ContextualTask) -> dict:
        return {"dependency": task.actual_dependency, "verified": True,
                "inspection_cost": .25, "source": "instrument"}

    @staticmethod
    def relational_cues(public: dict, *, dependency: str | None = None) -> frozenset[str]:
        """Signals are conjunctions of contextual properties, not answer labels."""
        dep = dependency if dependency is not None else public["reported_dependency"]
        return frozenset({
            f"pair:mode={public['mode']}|dependency={dep}",
            f"pair:material={public['material']}|phase={public['phase']}",
            f"pair:context={public['context']}|source={public['source']}",
            f"pair:mode={public['mode']}|material={public['material']}",
            f"pair:protocol={public['protocol_epoch']}|mode={public['mode']}",
            f"pair:epoch={public['protocol_epoch']}|mode={public['mode']}|dependency={dep}",
            f"protocol:{public['protocol_epoch']}",
            f"pair:consequence={public['consequence']}|self={public['self_relevance']}",
            f"source:{public['source']}|context:{public['context']}",
            f"mode:{public['mode']}", f"dependency:{dep}",
            f"material:{public['material']}", f"phase:{public['phase']}",
        })


@dataclass(frozen=True)
class ReturnTimelineWorld:
    """Synthetic private truth for a time-dependent expectation.

    Only check_return yields instrument evidence. Neither pending-clock events
    nor source testimony grants the cognitive policy access to actual_arrival_at.
    """
    subject: str
    promised_duration: float
    actual_arrival_at: float | None
    observation_available: bool = True
    timestamp_available: bool = True

    def promise(self) -> EnvironmentSignal:
        return EnvironmentSignal(
            kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
            content=f"{self.subject} says they will return in {self.promised_duration:g} minutes",
            semantic_tags=frozenset({"signal:temporal", "temporal:promise"}),
            person_id=self.subject, confidence=.9, informative=True,
            metadata={"source_status": "reported_promise",
                      "promised_duration": self.promised_duration},
        )

    def testimony(self, *, reported_present: bool) -> EnvironmentSignal:
        return EnvironmentSignal(
            kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
            content="A bystander reports presence, without direct verification",
            semantic_tags=frozenset({"signal:temporal", "temporal:report"}),
            confidence=.35,
            metadata={"source_status": "unverified_testimony",
                      "reported_present": reported_present},
        )

    def check_return(self, now: float) -> EnvironmentSignal:
        if not self.observation_available:
            return EnvironmentSignal(
                kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
                content="The return sensor is unavailable",
                semantic_tags=frozenset({"signal:temporal", "temporal:unavailable"}),
                metadata={"source_status": "unavailable"},
            )
        arrived = self.actual_arrival_at is not None and self.actual_arrival_at <= now
        return EnvironmentSignal(
            kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
            content=f"Direct sensor check: {'returned' if arrived else 'absent'}",
            semantic_tags=frozenset({"signal:temporal",
                                     "temporal:returned" if arrived else "temporal:absent"}),
            informative=True, confidence=1.0,
            metadata={"source_status": "direct_observation",
                      "returned": arrived,
                      "arrival_at": (self.actual_arrival_at
                                     if arrived and self.timestamp_available else None)},
        )


@dataclass(frozen=True)
class TemporalDelivery:
    """Private arrival truth, exposed only through world observations/actions."""
    person: str
    context: str
    promised: float
    arrival_at: float | None
    deadline: float
    payoff: float
    premature_loss: float


class TemporalDecisionWorld:
    """E10 bounded synthetic decision environment, no oracle policy access.

    A decision-maker receives promises and public constraints. Arrival times and
    dependency changes stay private until experienced through a valid channel.
    """
    def __init__(self, deliveries: tuple[TemporalDelivery, ...], *,
                 inspection_cost: float = 0.12, wait_cost: float = 0.035,
                 switch_reward: float = 0.24, sensor_down: frozenset[str] = frozenset(),
                 reports: tuple[tuple[float, str, bool, str], ...] = (),
                 dependency_change: tuple[float, str, str] | None = None):
        if len(deliveries) < 2 or len(set(d.person for d in deliveries)) != len(deliveries):
            raise ValueError("at least two distinct pending expectations")
        self._deliveries = {d.person: d for d in deliveries}
        self._reports = tuple(reports)
        self._dependency_change = dependency_change
        self._sensor_down = sensor_down
        self.inspection_cost = inspection_cost
        self.wait_cost = wait_cost
        self.switch_reward = switch_reward
        self.now = 0.0
        self.reward = 0.0
        self.finished = False
        self._notices: set[str] = set()
        self.actions: list[str] = []

    def public_constraints(self) -> dict[str, dict]:
        return {d.person: {"context": d.context, "promised": d.promised,
                           "deadline": d.deadline, "payoff": d.payoff,
                           "premature_loss": d.premature_loss}
                for d in self._deliveries.values()}

    def promises(self) -> list[EnvironmentSignal]:
        return [EnvironmentSignal(
            kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
            content=f"{d.person} promises delivery in {d.promised:g} units",
            semantic_tags=frozenset({"signal:temporal", "temporal:promise"}),
            person_id=d.person, informative=True, confidence=.75,
            metadata={"source_status": "reported_promise",
                      "promised_duration": d.promised})
            for d in self._deliveries.values()]

    def advance(self, now: float) -> list[EnvironmentSignal]:
        if now < self.now:
            raise ValueError("time cannot move backwards")
        self.now = now
        signals = []
        for when, person, arrived, speaker in self._reports:
            key = f"report:{when}:{person}:{speaker}"
            if when <= now and key not in self._notices:
                self._notices.add(key)
                signals.append(EnvironmentSignal(
                    kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
                    content=f"{speaker} claims {person} {'arrived' if arrived else 'is absent'}",
                    semantic_tags=frozenset({"signal:temporal", "temporal:report"}),
                    person_id=speaker, confidence=.5,
                    metadata={"source_status": "unverified_testimony",
                              "about": person, "reported_present": arrived,
                              "reported_at": when}))
        if self._dependency_change is not None:
            when, person, dependency = self._dependency_change
            key = f"change:{when}:{person}"
            if when <= now and key not in self._notices:
                self._notices.add(key)
                signals.append(EnvironmentSignal(
                    kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
                    content=f"Task for {person} now depends on {dependency}",
                    semantic_tags=frozenset({"signal:temporal", "temporal:dependency"}),
                    informative=True,
                    metadata={"dependency_change": person, "required": dependency}))
        return signals

    def inspect(self, person: str) -> EnvironmentSignal:
        if person not in self._deliveries:
            raise ValueError("unknown person")
        if person in self._sensor_down:
            return EnvironmentSignal(
                kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
                content=f"Sensor for {person} unavailable",
                semantic_tags=frozenset({"signal:temporal", "temporal:unavailable"}),
                person_id=person, metadata={"source_status": "unavailable"})
        d = self._deliveries[person]
        present = d.arrival_at is not None and self.now >= d.arrival_at
        return EnvironmentSignal(
            kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
            content=f"Direct check of {person}: {'arrived' if present else 'absent'}",
            semantic_tags=frozenset({"signal:temporal", "temporal:direct"}),
            person_id=person, informative=True,
            metadata={"source_status": "direct_observation",
                      "returned": present, "arrival_at": d.arrival_at if present else None})

    def apply(self, action: Action, *, known_dependencies: dict[str, str]) -> float:
        """Only environment execution sees private completion truth."""
        if self.finished:
            raise ValueError("world already finished")
        self.actions.append(action.label)
        delta = 0.0
        if action.kind == "wait":
            delta = -self.wait_cost
        elif action.kind == "check":
            delta = -self.inspection_cost
        elif action.kind == "switch":
            delta = self.switch_reward
            self.finished = True
        elif action.kind == "stop":
            self.finished = True
        elif action.kind == "proceed":
            d = self._deliveries[action.target]
            done = d.arrival_at is not None and d.arrival_at <= self.now
            # Changed dependency truth matters only after its public signal.
            if self._dependency_change and self.now >= self._dependency_change[0]:
                subject, required = self._dependency_change[1:]
                if subject == action.target:
                    other = self._deliveries[required]
                    done = done and other.arrival_at is not None and other.arrival_at <= self.now
            if self.now > d.deadline:
                delta = -d.premature_loss
            else:
                delta = d.payoff if done else -d.premature_loss
            self.finished = True
        else:
            raise ValueError("unsupported action")
        self.reward += delta
        return delta


class IncrementalDispatchWorld:
    """E11 private environment. Public cues never expose its causal rule.

    Each world instance permits exactly one execution. Paired conditions use a
    deterministic potential outcome for each action (common environment draw),
    so different learned choices never shift later scenarios' random stream.
    """
    def __init__(self, *, family: str, factors: dict[str, str], epoch: int,
                 outcomes: dict[str, bool], truth: dict[str, bool] | None = None,
                 scripted: bool = False) -> None:
        if not scripted and family not in ("delivery", "repair", "coordination"):
            raise ValueError("unknown dispatch family")
        self.family, self.factors, self.epoch = family, dict(factors), epoch
        self._outcomes = dict(outcomes)
        # Noiseless rule; evaluation-only, never delivered to cognition.
        self._truth = dict(truth if truth is not None else outcomes)
        self.executed = False

    @staticmethod
    def causal_rule(family: str, factors: dict[str, str], epoch: int) -> dict[str, bool]:
        """The experiment's private noiseless causal rule for the E11/E12 families."""
        p, d, u, c, n = (factors[k] for k in
                          ("person", "delay", "urgency", "capacity", "channel"))
        if family == "delivery":
            standard = (u == "low" or d == "short") and not (
                n == "night" and p == "Jordan" and d == "long")
            express = c == "open" and (n == "day" or p == "Alex")
        elif family == "repair":
            standard = (c == "open" or d == "short") and not (
                u == "high" and n == "night")
            express = (p == "Alex" or n == "day") and (
                c == "open" or u == "low")
        else:
            standard = (u == "low" and d == "short") or (
                p == "Jordan" and n == "day")
            express = c == "open" and (u == "high" or n == "night")
            if epoch == 1:
                # A real but publicly signaled protocol change; historical
                # evidence is retained yet should not override the new epoch.
                express = c == "limited" and (u == "high" or n == "night")
        return {"standard": standard, "express": express}

    @staticmethod
    def _noisy(truths: dict[str, bool], key: str, noise: float) -> dict[str, bool]:
        import hashlib
        outcomes = dict(truths)
        for procedure in outcomes:
            digest = hashlib.sha256(f"{key}|{procedure}|noise".encode()).digest()
            draw = int.from_bytes(digest[:8], "big") / (2 ** 64)
            if draw < noise:
                outcomes[procedure] = not outcomes[procedure]
        return outcomes

    @classmethod
    def scenario(cls, *, seed: int, family: str, index: int,
                 factors: dict[str, str], epoch: int):
        """The experiment's only private causal ground truth, including noise."""
        if family not in ("delivery", "repair", "coordination"):
            raise ValueError("unknown dispatch family")
        truths = cls.causal_rule(family, factors, epoch)
        return cls(family=family, factors=factors, epoch=epoch,
                   outcomes=cls._noisy(truths, f"{seed}|{family}|{index}", .09), truth=truths)

    @classmethod
    def scripted(cls, *, seed: int, family: str, index: int, factors: dict[str, str],
                 truths: dict[str, bool], noise: float = .08, epoch: int = 0):
        """E12 controlled challenge world: truths come from the evaluation harness.

        The public epoch stays constant; any rule change is never announced.
        """
        if set(truths) != {"standard", "express"}:
            raise ValueError("scripted worlds define both procedures")
        if not 0.0 <= noise < .5:
            raise ValueError("noise must lie in [0, .5)")
        return cls(family=family, factors=factors, epoch=epoch, scripted=True,
                   outcomes=cls._noisy(truths, f"scripted|{seed}|{family}|{index}", noise),
                   truth=truths)

    def public_cues(self) -> frozenset[str]:
        return frozenset(
            {f"pair:domain:{self.family}", f"pair:epoch:{self.epoch}"} |
            {f"pair:{axis}:{value}" for axis, value in self.factors.items()})

    def execute(self, action: Action) -> EnvironmentSignal:
        if self.executed:
            raise RuntimeError("one unique experience may be executed only once")
        if action.kind != "route" or action.target not in self._outcomes:
            raise ValueError("unsupported dispatch action")
        self.executed = True
        succeeded = self._outcomes[action.target]
        cost = {"standard": .02, "express": .18}[action.target]
        return EnvironmentSignal(
            kind=EventKind.OUTCOME, truth_status=TruthStatus.OUTCOME,
            content="Observed dispatch " + ("success" if succeeded else "failure"),
            semantic_tags=self.public_cues() | frozenset({"action:route",
                               "procedure:" + action.target}),
            outcome="success" if succeeded else "failure",
            reward=(1.0 if succeeded else -1.0) - cost,
            confidence=1.0, informative=True,
            metadata={"procedure": action.target,
                      "context_cues": sorted(self.public_cues())})

    def counterfactual_truth(self, procedure: str) -> bool:
        """Offline evaluator only: the noiseless rule, never seen by cognition."""
        return self._truth[procedure]

    def counterfactual_reward(self, procedure: str) -> float:
        """Offline evaluator only; never delivered to the cognitive pipeline."""
        return (1.0 if self._outcomes[procedure] else -1.0) - {
            "standard": .02, "express": .18}[procedure]


class InvestigationWorld:
    """E12 investigation world: consequential variables may be hidden until inspected.

    Public (cognition may read): the visible factors, WHICH variables exist but
    are hidden, each one's possible values and inspection cost, every action's
    consequences (success gain, failure loss, cost), the safe deferral value
    and the authorization restrictions. Private: hidden values, the noiseless
    rule and the noisy potential outcomes. Outcomes report success or failure
    only; a hidden value is learned solely by inspecting it.

    Each instance permits one final action. Potential outcomes are common random
    numbers per (seed, family, index, procedure), so different decisions never
    shift another case's random draw.
    """
    ROUTES = ("standard", "express")

    def __init__(self, *, family: str, visible: dict[str, str], hidden: dict[str, str],
                 domains: dict[str, tuple[str, str]], truths: dict[str, bool],
                 outcomes: dict[str, bool], noise: float, inspect_costs: dict[str, float],
                 stakes: dict[str, dict[str, float]], prohibited: frozenset[str],
                 defer_value: float, epoch: int = 0, activity: dict | None = None) -> None:
        if set(truths) != set(self.ROUTES) or set(outcomes) != set(self.ROUTES):
            raise ValueError("investigation worlds define both routes")
        if set(hidden) != set(inspect_costs) or set(hidden) != set(domains):
            raise ValueError("every hidden variable needs a public domain and inspection cost")
        if any(hidden[a] not in domains[a] for a in hidden) or set(hidden) & set(visible):
            raise ValueError("hidden values must lie in their public domain")
        if set(stakes) != set(self.ROUTES) or not prohibited <= {"standard", "express"}:
            raise ValueError("stakes cover both routes; only routes can be prohibited")
        self.family, self.epoch = family, epoch
        self.visible, self._hidden = dict(visible), dict(hidden)
        self.domains = {a: tuple(v) for a, v in domains.items()}
        self._truth, self._outcomes, self.noise = dict(truths), dict(outcomes), noise
        self.inspect_costs = dict(inspect_costs)
        self.stakes = {p: dict(s) for p, s in stakes.items()}
        self.prohibited, self.defer_value = frozenset(prohibited), float(defer_value)
        self.revealed: dict[str, str] = {}
        self.inspection_cost = 0.0
        self.executed = False
        self.violations = 0  # evaluation: executed prohibited routes (must stay 0)
        # Private multi-step script. Only its individual factual observations can
        # enter cognition; scripts and eventual evaluator truth cannot.
        self._activity = dict(activity or {})
        self._steps = tuple(self._activity.get("steps", ()))
        self._checks = tuple(self._activity.get("verification", ()))
        self._step_index = 0
        self._check_index = 0
        self._started_action: Action | None = None

    @classmethod
    def scripted(cls, *, seed: int, family: str, index: int, visible: dict[str, str],
                 hidden: dict[str, str], domains: dict[str, tuple[str, str]],
                 truths: dict[str, bool], noise: float = .08,
                 inspect_costs: dict[str, float], stakes: dict[str, dict[str, float]],
                 prohibited=frozenset(), defer_value: float = -.3, epoch: int = 0,
                 activity: dict | None = None):
        if not 0.0 <= noise < .5:
            raise ValueError("noise must lie in [0, .5)")
        outcomes = IncrementalDispatchWorld._noisy(truths, f"investigation|{seed}|{family}|{index}", noise)
        return cls(family=family, visible=visible, hidden=hidden, domains=domains, truths=truths,
                   outcomes=outcomes, noise=noise, inspect_costs=inspect_costs, stakes=stakes,
                   prohibited=frozenset(prohibited), defer_value=defer_value, epoch=epoch,
                   activity=activity)

    # -- public interface --------------------------------------------------------
    def public_cues(self) -> frozenset[str]:
        hidden = {f"pair:{a}:{self.revealed.get(a, '?')}" for a in self._hidden}
        return frozenset({f"pair:domain:{self.family}", f"pair:epoch:{self.epoch}"} |
                         {f"pair:{a}:{v}" for a, v in self.visible.items()} | hidden)

    def public_constraints(self) -> dict:
        return {"hidden": {a: self.inspect_costs[a] for a in self._hidden if a not in self.revealed},
                "domains": dict(self.domains), "stakes": {p: dict(s) for p, s in self.stakes.items()},
                "prohibited": self.prohibited, "defer_value": self.defer_value}

    def inspect(self, axis: str) -> EnvironmentSignal:
        if self.executed:
            raise RuntimeError("inspection happens before the final action")
        if axis not in self._hidden or axis in self.revealed:
            raise ValueError("only an unrevealed hidden variable can be inspected")
        value = self._hidden[axis]
        self.revealed[axis] = value
        cost = self.inspect_costs[axis]
        self.inspection_cost += cost
        return EnvironmentSignal(
            kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
            content=f"Inspected {axis}: {value}",
            semantic_tags=frozenset({f"pair:{axis}:{value}", "action:inspect"}),
            reward=-cost, informative=True,
            metadata={"inspected": axis, "value": value, "cost": cost})

    def execute(self, action: Action) -> EnvironmentSignal:
        if self.executed:
            raise RuntimeError("one investigation case permits one final action")
        if action.kind != "route" or action.target not in ("standard", "express", "defer"):
            raise ValueError("unsupported final action")
        self.executed = True
        cues = sorted(self.public_cues())
        if action.target == "defer":
            return EnvironmentSignal(
                kind=EventKind.OUTCOME, truth_status=TruthStatus.OUTCOME,
                content="Request deferred safely", outcome="deferred",
                semantic_tags=frozenset(cues) | {"procedure:defer"}, reward=self.defer_value,
                metadata={"procedure": "defer", "context_cues": cues})
        if action.target in self.prohibited:
            self.violations += 1
        succeeded = self._outcomes[action.target]
        stake = self.stakes[action.target]
        return EnvironmentSignal(
            kind=EventKind.OUTCOME, truth_status=TruthStatus.OUTCOME,
            content="Observed dispatch " + ("success" if succeeded else "failure"),
            semantic_tags=frozenset(cues) | {"action:route", "procedure:" + action.target},
            outcome="success" if succeeded else "failure",
            reward=(stake["gain"] if succeeded else -stake["loss"]) - stake["cost"],
            confidence=1.0, informative=True,
            metadata={"procedure": action.target, "context_cues": cues})

    # -- opt-in unfolding activity (legacy single-step execute remains unchanged) ----
    def begin_activity(self, action: Action) -> EnvironmentSignal:
        if not self._activity or self.executed or self._started_action is not None:
            raise ValueError("activity must be configured and started only once")
        if action.kind != "route" or action.target not in (*self.ROUTES, "defer"):
            raise ValueError("invalid initial action")
        self._started_action = action
        return EnvironmentSignal(
            kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
            content="Dispatch activity started", semantic_tags=self.public_cues(),
            metadata={"activity_state": "active", "source": "dispatch_start"})

    def activity_step(self) -> EnvironmentSignal | None:
        if self._started_action is None or self.executed:
            raise RuntimeError("activity is not awaiting an intermediate observation")
        if self._step_index >= len(self._steps):
            return None
        step = self._steps[self._step_index]
        self._step_index += 1
        changes = dict(step.get("visible_updates", {}))
        self.visible.update(changes)
        # New public restrictions may tighten during execution but never lift.
        self.prohibited = self.prohibited | frozenset(step.get("prohibited", ()))
        return EnvironmentSignal(
            kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
            content=str(step.get("content", "Intermediate activity observation")),
            semantic_tags=self.public_cues(),
            metadata={"activity_state": "active", "source": str(step.get("source", "process")),
                      "elapsed": float(step.get("elapsed", 1.0)),
                      "visible_updates": changes, "prohibited": tuple(self.prohibited)})

    def goal_evidence_step(self) -> EnvironmentSignal | None:
        if not self.executed or not self._activity:
            raise RuntimeError("verification requires a finished activity")
        if self._check_index >= len(self._checks):
            return None
        check = self._checks[self._check_index]
        self._check_index += 1
        # A verified observation is independent of the mechanical action
        # outcome; no truth label or private rule reaches the learner.
        return EnvironmentSignal(
            kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
            content=str(check.get("content", "Independent goal observation")),
            semantic_tags=self.public_cues(),
            metadata={"goal_check": {
                "requirement": check["requirement"], "satisfied": check["satisfied"],
                "source": check["source"], "final": bool(check.get("final", False))},
                "elapsed": float(check.get("elapsed", 1.0))})

    # -- evaluation only: never delivered to cognition -----------------------------
    def counterfactual_truth(self, procedure: str) -> bool:
        return self._truth[procedure]

    def expected_utility(self, procedure: str) -> float:
        """Noiseless expected reward of a final action in the full private context."""
        if procedure == "defer":
            return self.defer_value
        p = 1 - self.noise if self._truth[procedure] else self.noise
        s = self.stakes[procedure]
        return p * s["gain"] - (1 - p) * s["loss"] - s["cost"]

    def permissible_routes(self) -> tuple[str, ...]:
        return tuple(p for p in ("standard", "express", "defer") if p not in self.prohibited)

    def hidden_truth(self) -> dict[str, str]:
        return dict(self._hidden)
