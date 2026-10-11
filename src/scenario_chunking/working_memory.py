"""Experimental working-thought accessibility; not a historical memory store.

This owner knows chunk references and routing headers, never copies or edits events.
An activation update is *not* a statement about evidence, validity, or confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp, log
from typing import Iterable


class ThoughtState(str, Enum):
    FOCUSED = "focused"
    PERIPHERAL = "peripheral"
    DORMANT = "dormant"


@dataclass
class ThoughtBundle:
    chunk_id: str
    goal_id: str
    tags: frozenset[str]
    strength: float
    last_time: float
    last_used_time: float | None = None
    meaningful_uses: int = 0
    reinstatements: int = 0
    reinstated_since_use: bool = False


class WorkingMemoryActivation:
    """Bounded multi-bundle accessibility with lazy elapsed-time decay.

    'now' must come from the experiment's GLOBAL, nondecreasing event clock.
    The episode-local CognitiveState.current_time is not suitable.
    """

    def __init__(
        self, *,
        capacity: int = 6,
        focused_capacity: int = 2,
        half_life: float = 7.0,
        dormant_threshold: float = 0.18,
        max_tracked: int = 96,
    ) -> None:
        if not 0 < focused_capacity <= capacity <= max_tracked:
            raise ValueError("invalid bundle capacities")
        if half_life <= 0 or not 0 < dormant_threshold < 1:
            raise ValueError("invalid decay parameters")
        self.capacity = capacity
        self.focused_capacity = focused_capacity
        self.half_life = half_life
        self.dormant_threshold = dormant_threshold
        self.max_tracked = max_tracked
        self.bundles: dict[str, ThoughtBundle] = {}
        self.latest_time = 0.0
        self.reinstatements = 0
        self.use_events = 0
        self.trajectories: list[dict[str, object]] = []

    def _clock(self, now: float) -> None:
        if now + 1e-9 < self.latest_time:
            raise ValueError("working-memory clock regressed; pass global event time")
        self.latest_time = max(self.latest_time, now)

    def activation(self, chunk_id: str, now: float, *, persistence_gain: float = 0.0) -> float:
        self._clock(now)
        bundle = self.bundles.get(chunk_id)
        if bundle is None:
            return 0.0
        # A bounded learned gain changes half-life, not truth or source confidence.
        half_life = self.half_life * exp(max(-1.1, min(1.1, persistence_gain)))
        elapsed = max(0.0, now - bundle.last_time)
        return bundle.strength * exp(-log(2.0) * elapsed / half_life)

    def _active_ranking(self, goal_id: str, now: float, persistence: dict[str, float]) -> list[tuple[str, float]]:
        active = []
        for cid, bundle in self.bundles.items():
            value = self.activation(cid, now, persistence_gain=persistence.get(cid, 0.0))
            if value >= self.dormant_threshold:
                # Goal relevance makes a thought competitive without halting its decay.
                affinity = 0.12 if bundle.goal_id == goal_id else -0.12
                active.append((cid, value + affinity))
        active.sort(key=lambda pair: (-pair[1], pair[0]))
        return active[:self.capacity]

    def state(self, chunk_id: str, goal_id: str, now: float, *,
              persistence: dict[str, float] | None = None) -> ThoughtState:
        gain = persistence or {}
        ranking = [cid for cid, _ in self._active_ranking(goal_id, now, gain)]
        if chunk_id not in ranking:
            return ThoughtState.DORMANT
        if chunk_id in ranking[:self.focused_capacity]:
            return ThoughtState.FOCUSED
        return ThoughtState.PERIPHERAL

    def bonus(self, chunk_id: str, goal_id: str, now: float, *,
              persistence: dict[str, float] | None = None) -> float:
        """Small routing-priority modifier, not a boost to evidence quality."""
        gain = persistence or {}
        bundle = self.bundles.get(chunk_id)
        if bundle is None or bundle.goal_id != goal_id:
            return 0.0
        strength = self.activation(chunk_id, now, persistence_gain=gain.get(chunk_id, 0.0))
        state = self.state(chunk_id, goal_id, now, persistence=gain)
        return (min(0.11, 0.055 * strength) if state == ThoughtState.FOCUSED else
                min(0.04, 0.02 * strength) if state == ThoughtState.PERIPHERAL else 0.0)

    def cue(self, chunk_id: str, goal_id: str, tags: frozenset[str],
            query_tags: frozenset[str], now: float, *,
            contradicted_keys: frozenset[str] = frozenset(),
            persistence: dict[str, float] | None = None) -> bool:
        """One bounded, evidence-compatible reinstatement; never chains to neighbors.

        Generic goal/role tags alone are insufficient cues. Repeated retrievals
        of an already-active chunk do not refresh its timestamp.
        """
        gain = persistence or {}
        self._clock(now)
        if any(f"key:{key}" in tags and "outcome:success" in tags for key in contradicted_keys):
            return False
        specific = lambda ts: {t for t in ts if not (t.startswith("goal:") or t.startswith("role:"))}
        overlap = specific(tags) & specific(query_tags)
        if not overlap or goal_id != self.bundles.get(chunk_id, ThoughtBundle(chunk_id, goal_id, tags, 0., now)).goal_id:
            return False
        if self.state(chunk_id, goal_id, now, persistence=gain) != ThoughtState.DORMANT:
            return False
        old = self.bundles.get(chunk_id)
        if old is not None and old.reinstated_since_use:
            # Repeated retrieval without intervening meaningful use must not
            # indefinitely resurrect the same unhelpful thought.
            return False
        self.bundles[chunk_id] = ThoughtBundle(
            chunk_id=chunk_id, goal_id=goal_id, tags=tags, strength=0.34,
            last_time=now, reinstated_since_use=True, reinstatements=self.bundles.get(
                chunk_id, ThoughtBundle(chunk_id, goal_id, tags, 0.0, now)
            ).reinstatements + 1,
        )
        self.reinstatements += 1
        self._enforce_capacity(goal_id, now, gain)
        return chunk_id in {cid for cid, _ in self._active_ranking(goal_id, now, gain)}

    def use(self, chunk_ids: Iterable[str], goal_id: str, now: float, *,
            known_headers: dict[str, frozenset[str]],
            persistence: dict[str, float] | None = None) -> None:
        """Only call for sources that contributed to live selected reasoning.

        Mere top-K retrieval MUST NOT call this method.
        """
        gain = persistence or {}
        self._clock(now)
        for cid in dict.fromkeys(chunk_ids):
            if cid not in known_headers:
                continue
            old = self.bundles.get(cid)
            value = self.activation(cid, now, persistence_gain=gain.get(cid, 0.0))
            self.bundles[cid] = ThoughtBundle(
                chunk_id=cid, goal_id=goal_id, tags=known_headers[cid],
                strength=min(1.5, value + 0.52), last_time=now,
                last_used_time=now,
                meaningful_uses=(old.meaningful_uses if old else 0) + 1,
                reinstatements=old.reinstatements if old else 0,
            )
            self.use_events += 1
        self._enforce_capacity(goal_id, now, gain)

    def _enforce_capacity(self, goal_id: str, now: float, gain: dict[str, float]) -> None:
        # Never erase historical memory; excess representations become dormant.
        keep = {cid for cid, _ in self._active_ranking(goal_id, now, gain)}
        for cid, bundle in self.bundles.items():
            if cid not in keep and self.activation(cid, now, persistence_gain=gain.get(cid, 0.0)) >= self.dormant_threshold:
                # Suppress excess activation; stored episodic evidence is untouched.
                bundle.strength = 0.0
                bundle.last_time = now
        if len(self.bundles) > self.max_tracked:
            # Dormant working-memory cache can be shed; episodic retrieval remains possible.
            removable = sorted(
                (b for b in self.bundles.values() if b.chunk_id not in keep),
                key=lambda b: (b.last_used_time if b.last_used_time is not None else -1.0, b.last_time),
            )
            for b in removable[:len(self.bundles) - self.max_tracked]:
                del self.bundles[b.chunk_id]

    def snapshot(self, goal_id: str, now: float, *,
                 persistence: dict[str, float] | None = None) -> dict[str, object]:
        gain = persistence or {}
        self._clock(now)
        focused, peripheral = [], []
        for cid, _ in self._active_ranking(goal_id, now, gain):
            entry = {"chunk_id": cid, "activation": round(
                self.activation(cid, now, persistence_gain=gain.get(cid, 0.0)), 5)}
            (focused if len(focused) < self.focused_capacity else peripheral).append(entry)
        snap = {
            "global_time": now, "focused": focused, "peripheral": peripheral,
            "dormant_tracked": len(self.bundles) - len(focused) - len(peripheral),
            "active_count": len(focused) + len(peripheral),
            "meaningful_uses": self.use_events,
            "reinstatements": self.reinstatements,
        }
        self.trajectories.append(snap)
        return snap
