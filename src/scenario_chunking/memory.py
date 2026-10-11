from __future__ import annotations

from collections import defaultdict
from collections import deque
import hashlib

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
        self.event_to_chunk: dict[str, str] = {}
        # Directed explanatory links, separate from chronological links.
        # Stored with provenance; a hypothesized edge is not an observed cause.
        self.causal_links: dict[str, dict[str, tuple[str, float]]] = defaultdict(dict)
        # Per-chunk learned interpretations reference original evidence; no event is copied.
        self.layer_meanings: dict[tuple[str, str, str], object] = {}
        # Bounded, incrementally maintained routing headers: no full-history scan.
        self.focus_examples: dict[str, list[str]] = defaultdict(list)
        self.focus_recent: dict[str, deque[str]] = {}
        # Plain integers (not itertools.count) so state pickles on Python 3.14+.
        self._event_ids = 0
        self._chunk_ids = 0

    def next_event_id(self) -> str:
        self._event_ids += 1
        return f"e{self._event_ids:06d}"

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
        self._chunk_ids += 1
        chunk_id = f"c{self._chunk_ids:06d}"
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
        for event in events:
            if event.cause_event_id:
                source_chunk = self.event_to_chunk.get(event.cause_event_id)
                if source_chunk and source_chunk != chunk_id:
                    self.link_causal(source_chunk, chunk_id, evidence="observed", confidence=event.confidence)
            self.event_to_chunk[event.event_id] = chunk_id
        self.chunk_order.append(chunk_id)
        self.episode_index[first.episode_id].append(chunk_id)
        self.goal_index[first.goal_id].add(chunk_id)
        for tag in tags:
            self.tag_index[tag].add(chunk_id)
            self._index_focus(tag, chunk_id)
        for person in people:
            self.person_index[person].add(chunk_id)

        if previous:
            prev = self.chunks[previous]
            prev.header.next_chunk_id = chunk_id
        return chunk

    def _index_focus(self, tag: str, chunk_id: str) -> None:
        """Retain high-value and distributed historical examples plus recent evidence.

        No environment answer, current goal, or action-selection authority enters
        this index. Historical events and complete postings remain untouched.
        """
        recent = self.focus_recent.setdefault(tag, deque(maxlen=32))
        recent.append(chunk_id)
        candidates = self.focus_examples[tag]
        candidates.append(chunk_id)
        if len(candidates) > 64:
            def priority(cid: str) -> tuple[float, str]:
                chunk = self.chunks[cid]
                quality = .4 * chunk.header.salience + .3 * chunk.header.self_relevance
                if "outcome:success" in chunk.semantic_tags:
                    quality += .25
                if "source:current_protocol" in chunk.semantic_tags:
                    quality += .3
                # Stable coverage means low-frequency counterexamples need not
                # disappear simply because thousands of duplicates arrive.
                digest = hashlib.blake2b((cid + tag).encode(), digest_size=8).digest()
                coverage = int.from_bytes(digest, "big") / (2 ** 64)
                return quality + .3 * coverage, cid
            candidates.sort(key=priority, reverse=True)
            del candidates[64:]

    def focused_postings(self, tag: str, count: int) -> list[str]:
        """Bounded compact headers, balanced between representative and fresh."""
        if count <= 0:
            return []
        keep = (count + 1) // 2
        stable = self.focus_examples.get(tag, ())[:keep]
        recent = list(reversed(self.focus_recent.get(tag, ())))[:count - keep]
        return list(dict.fromkeys([*stable, *recent]))[:count]

    def activate(self, chunk_id: str, now: float, boost: float = 0.35) -> None:
        header = self.chunks[chunk_id].header
        current = header.activation_at(now)
        header.activation_strength = min(3.0, current + boost)
        header.activation_last_time = now

    def link_causal(self, source_chunk_id: str, effect_chunk_id: str, *, evidence: str, confidence: float) -> None:
        """Connect a confirmed or proposed cause; only observed causes can earn strong spreading priority."""
        if source_chunk_id == effect_chunk_id:
            return
        if source_chunk_id not in self.chunks or effect_chunk_id not in self.chunks:
            raise KeyError("a causal link requires two stored chunks")
        if evidence not in ("observed", "hypothesis", "contradicted"):
            raise ValueError("unknown evidence provenance")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence outside [0, 1]")
        self.causal_links[source_chunk_id][effect_chunk_id] = (evidence, confidence)
        self.causal_links[effect_chunk_id][source_chunk_id] = (evidence, confidence)

    def neighbors(self, chunk_id: str) -> list[tuple[str, str, str, float]]:
        """Return only directly linked chunk headers, never full episode scans."""
        header = self.chunks[chunk_id].header
        nearby: list[tuple[str, str, str, float]] = []
        if header.previous_chunk_id:
            nearby.append((header.previous_chunk_id, "predecessor", "observed", 1.0))
        if header.next_chunk_id:
            nearby.append((header.next_chunk_id, "successor", "observed", 1.0))
        for neighbor, (evidence, confidence) in sorted(self.causal_links.get(chunk_id, {}).items()):
            nearby.append((neighbor, "causal", evidence, confidence))
        return nearby

    def store_layer_meaning(self, chunk_id: str, axis: str, action_label: str, meaning: object) -> None:
        if chunk_id not in self.chunks:
            raise KeyError(chunk_id)
        self.layer_meanings[(chunk_id, axis, action_label)] = meaning

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
