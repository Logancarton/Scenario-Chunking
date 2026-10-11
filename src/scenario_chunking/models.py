from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, FrozenSet, Iterable


class EventKind(str, Enum):
    OBSERVATION = "observation"
    PREDICTION = "prediction"
    ACTION = "action"
    OUTCOME = "outcome"


class TruthStatus(str, Enum):
    FACT = "fact"
    PREDICTION = "prediction"
    ATTEMPT = "attempt"
    OUTCOME = "outcome"


class RetrievalMode(str, Enum):
    FULL = "full"
    CHUNKED = "chunked"
    RECENCY = "recency"
    STRUCTURAL = "structural"
    SPREADING = "spreading"
    ACTIVATION = "activation"


@dataclass(frozen=True)
class Action:
    kind: str
    target: str | None = None
    key_name: str | None = None
    person: str | None = None

    @property
    def label(self) -> str:
        bits = [self.kind]
        if self.target:
            bits.append(self.target)
        if self.key_name:
            bits.append(self.key_name)
        if self.person:
            bits.append(self.person)
        return ":".join(bits)


@dataclass
class Event:
    event_id: str
    episode_id: str
    goal_id: str
    step: int
    occurrence_time: float
    acquired_time: float
    storage_time: float
    kind: EventKind
    truth_status: TruthStatus
    content: str
    semantic_tags: FrozenSet[str] = frozenset()
    person_id: str | None = None
    action: Action | None = None
    outcome: str | None = None
    reward: float = 0.0
    cause_event_id: str | None = None
    informative: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryHeader:
    self_relevance: float
    salience: float
    occurrence_time: float
    acquired_time: float
    storage_time: float
    time_uncertainty: float
    sequence_index: int
    goal_id: str
    episode_id: str
    person_ids: FrozenSet[str]
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    activation_strength: float = 1.0
    activation_last_time: float = 0.0

    def activation_at(self, now: float, decay_rate: float = 0.08) -> float:
        """Lazy closed-form working-memory activation; historical times are never rewritten."""
        elapsed = max(0.0, now - self.activation_last_time)
        return self.activation_strength * (2.718281828459045 ** (-decay_rate * elapsed))


@dataclass
class MemoryChunk:
    chunk_id: str
    header: MemoryHeader
    event_ids: list[str]
    semantic_tags: FrozenSet[str]
    transition_reason: str
    validity_tags: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class RetrievalQuery:
    goal_id: str
    semantic_tags: FrozenSet[str]
    person_ids: FrozenSet[str] = frozenset()
    current_time: float = 0.0
    contradicted_keys: FrozenSet[str] = frozenset()
    desired_event_kinds: FrozenSet[EventKind] = frozenset()


@dataclass
class RetrievalHit:
    chunk_id: str
    score: float
    reasons: list[str]


@dataclass
class RetrievalStats:
    total_chunks: int = 0
    candidate_chunks: int = 0
    header_evaluations: int = 0
    detail_loads: int = 0
    latency_ms: float = 0.0
    active_detail_events: int = 0
    unnecessary_activations: int = 0
    initial_hit_ids: list[str] = field(default_factory=list)
    spread_hit_ids: list[str] = field(default_factory=list)
    spread_justifications: dict[str, str] = field(default_factory=dict)
    spread_detailed_events: int = 0
    neighbor_headers_examined: int = 0
    budget_exhausted: bool = False
    reinstated_hit_ids: list[str] = field(default_factory=list)
    reinstatement_headers_examined: int = 0
    reinstatement_detail_events: int = 0

    def __iadd__(self, other: "RetrievalStats") -> "RetrievalStats":
        self.total_chunks += other.total_chunks
        self.candidate_chunks += other.candidate_chunks
        self.header_evaluations += other.header_evaluations
        self.detail_loads += other.detail_loads
        self.latency_ms += other.latency_ms
        self.active_detail_events += other.active_detail_events
        self.unnecessary_activations += other.unnecessary_activations
        self.spread_detailed_events += other.spread_detailed_events
        self.neighbor_headers_examined += other.neighbor_headers_examined
        self.reinstatement_headers_examined += other.reinstatement_headers_examined
        self.reinstatement_detail_events += other.reinstatement_detail_events
        self.reinstated_hit_ids.extend(other.reinstated_hit_ids)
        self.budget_exhausted = self.budget_exhausted or other.budget_exhausted
        return self


@dataclass
class SelectionDecision:
    action: Action
    score: float
    predicted_success: float
    reasons: list[str]
    retrieval_hits: list[RetrievalHit]
    meaning_frame: Any = None
    interpretation_event_evaluations: int = 0
    thought_trace: Any = None
    used_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class EnvironmentSignal:
    kind: EventKind
    content: str
    semantic_tags: FrozenSet[str]
    truth_status: TruthStatus
    person_id: str | None = None
    outcome: str | None = None
    reward: float = 0.0
    cause_event_id: str | None = None
    informative: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    episode_id: str
    success: bool
    actions: int
    failed_actions: int
    repeated_actions: int
    unnecessary_actions: int
    retrieval_stats: RetrievalStats
    trace: list[dict[str, Any]]
    outdated_retrievals: int = 0
    credit_accuracy: float | None = None


@dataclass
class ExperimentSummary:
    mode: RetrievalMode
    seed: int
    episodes: list[EpisodeResult]
    retrieval_recall_at_k: float
    temporal_order_accuracy: float
    delayed_credit_accuracy: float
    recency_negative_control_passed: bool | None = None
    recall_with_spreading: float = 0.0
    recovered_relevant_memories: int = 0
    spread_action_changes: int = 0
    spread_action_score_influences: int = 0

    @property
    def completion_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    @property
    def total_actions(self) -> int:
        return sum(e.actions for e in self.episodes)

    @property
    def failed_actions(self) -> int:
        return sum(e.failed_actions for e in self.episodes)

    @property
    def repeated_actions(self) -> int:
        return sum(e.repeated_actions for e in self.episodes)


def frozen(items: Iterable[str]) -> FrozenSet[str]:
    return frozenset(items)
