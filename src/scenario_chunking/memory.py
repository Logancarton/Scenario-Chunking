from __future__ import annotations

from collections import defaultdict
from itertools import count

from .models import Event, EventKind, MemoryChunk, MemoryHeader, TruthStatus


class EpisodicMemory:
    """Stores immutable historical events plus compact, indexable episodic chunk headers."""

    def __init__(self) -> None:
        self.events: dict[str, Event] = {}
        self.chunks: dict[str, MemoryChunk] = {}
        self.chunk_order: list[str] = []
        self.goal_index: dict[str, set[str]] = defaultdict(set)
        self.tag_index: dict[str, set[str]] = defaultdict(set)
        self.person_index: dict[str, set[str]] = defaultdict(set)
        self.episode_index: dict[str, list[str]] = defaultdict(list)
        self._event_ids = count(1)
        self._chunk_ids = count(1)

    def next_event_id(self) -> str:
        return f"e{next(self._event_ids):06d}"

    def add_event(self, event: Event) -> None:
        if event.event_id in self.events:
            raise ValueError(f"duplicate event id {event.event_id}")
        self.events[event.event_id] = event

    def create_chunk(
        self,
        *,
        events: list[Event],
        transition_reason: str,
        self_relevance: float,
        salience: float,
        validity_tags: frozenset[str] = frozenset(),
    ) -> MemoryChunk:
        if not events:
            raise ValueError("a chunk requires at least one event")
        for event in events:
            if event.event_id not in self.events:
                self.add_event(event)

        first = events[0]
        last = events[-1]
        chunk_id = f"c{next(self._chunk_ids):06d}"
        previous = self.episode_index[first.episode_id][-1] if self.episode_index[first.episode_id] else None
        tags = frozenset().union(*(event.semantic_tags for event in events))
        people = frozenset(event.person_id for event in events if event.person_id)
        header = MemoryHeader(
            self_relevance=self_relevance,
            salience=salience,
            occurrence_time=first.occurrence_time,
            acquired_time=first.acquired_time,
            storage_time=last.storage_time,
            time_uncertainty=max(float(event.metadata.get("time_uncertainty", 0.0)) for event in events),
            sequence_index=len(self.episode_index[first.episode_id]),
            goal_id=first.goal_id,
            episode_id=first.episode_id,
            person_ids=people,
            previous_chunk_id=previous,
            activation_strength=1.0,
            activation_last_time=last.storage_time,
        )
        chunk = MemoryChunk(
            chunk_id=chunk_id,
            header=header,
            event_ids=[event.event_id for event in events],
            semantic_tags=tags,
            transition_reason=transition_reason,
            validity_tags=validity_tags,
        )
        self.chunks[chunk_id] = chunk
        self.chunk_order.append(chunk_id)
        self.episode_index[first.episode_id].append(chunk_id)
        self.goal_index[first.goal_id].add(chunk_id)
        for tag in tags:
            self.tag_index[tag].add(chunk_id)
        for person in people:
            self.person_index[person].add(chunk_id)

        if previous:
            prev = self.chunks[previous]
            prev.header.next_chunk_id = chunk_id
        return chunk

    def activate(self, chunk_id: str, now: float, boost: float = 0.35) -> None:
        header = self.chunks[chunk_id].header
        current = header.activation_at(now)
        header.activation_strength = min(3.0, current + boost)
        header.activation_last_time = now

    def chunk_events(self, chunk_id: str) -> list[Event]:
        return [self.events[eid] for eid in self.chunks[chunk_id].event_ids]

    def inject_distractors(self, count_: int, *, start_time: float, goal_id: str = "unrelated_goal") -> None:
        """Add realistic but irrelevant historical chunks without touching active episode state."""
        for i in range(count_):
            t = start_time + i * 0.001
            event = Event(
                event_id=self.next_event_id(),
                episode_id=f"distractor-{i // 25}",
                goal_id=goal_id,
                step=i,
                occurrence_time=t,
                acquired_time=t,
                storage_time=t,
                kind=EventKind.OBSERVATION,
                truth_status=TruthStatus.FACT,
                content=f"Irrelevant workshop memory {i}",
                semantic_tags=frozenset({f"topic:distractor-{i % 41}", f"object:irrelevant-{i % 97}", f"goal:{goal_id}"}),
                informative=False,
            )
            self.create_chunk(
                events=[event],
                transition_reason="distractor",
                self_relevance=0.05,
                salience=0.05,
            )

    @property
    def detailed_event_count(self) -> int:
        return len(self.events)
