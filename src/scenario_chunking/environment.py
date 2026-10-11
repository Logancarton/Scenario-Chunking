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


