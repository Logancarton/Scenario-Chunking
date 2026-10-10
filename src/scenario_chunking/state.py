from __future__ import annotations

from dataclasses import dataclass, field

from .models import Action, EnvironmentSignal, EventKind, RetrievalQuery


@dataclass
class CognitiveState:
    episode_id: str
    goal_id: str = "retrieve_object_from_locked_container"
    current_time: float = 0.0
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
    beliefs: dict[str, object] = field(default_factory=dict)
    goal_complete: bool = False

    def apply_signal(self, signal: EnvironmentSignal) -> None:
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
        if "candidate_key" in meta:
            self.candidate_key = str(meta["candidate_key"])
            self.beliefs["candidate_key_confidence"] = float(signal.confidence)
        if "contradicts_key" in meta:
            self.contradicted_keys.add(str(meta["contradicts_key"]))
            if self.candidate_key == meta["contradicts_key"]:
                self.candidate_key = None
        if "inspected" in meta:
            self.inspected.add(str(meta["inspected"]))
        if "asked_person" in meta:
            self.asked_people.add(str(meta["asked_person"]))
        if signal.outcome == "success":
            self.goal_complete = True

    def record_action(self, action: Action) -> None:
        self.attempted_actions.append(action.label)

    def candidate_actions(self) -> list[Action]:
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
