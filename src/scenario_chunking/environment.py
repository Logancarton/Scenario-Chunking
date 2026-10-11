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
