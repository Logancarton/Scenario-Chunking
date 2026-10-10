from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ActionTrace:
    event_id: str
    features: frozenset[str]
    informative: bool = False
    resolved: bool = False


class OutcomeLearner:
    """Transparent linear policy learner with causal and eligibility-trace credit assignment."""

    def __init__(self, *, learning_rate: float = 0.32, gamma: float = 0.72, enabled: bool = True) -> None:
        self.learning_rate = learning_rate
        self.enabled = enabled
        self.gamma = gamma
        self.weights: dict[str, float] = defaultdict(float)
        self.episode_trace: list[ActionTrace] = []
        self.credit_log: list[dict[str, object]] = []

    def begin_episode(self) -> None:
        self.episode_trace = []

    def register_action(self, event_id: str, features: frozenset[str]) -> None:
        self.episode_trace.append(ActionTrace(event_id=event_id, features=features))

    def mark_informative(self, event_id: str) -> None:
        for item in reversed(self.episode_trace):
            if item.event_id == event_id:
                item.informative = True
                return

    def expected_utility(self, features: frozenset[str]) -> float:
        if not self.enabled or not features:
            return 0.0
        # Generalized features transfer across surface names; exact key identity receives no special prior.
        learnable = [f for f in features if not f.startswith("key:") and not f.startswith("goal:")]
        if not learnable:
            return 0.0
        return sum(self.weights[f] for f in learnable) / len(learnable)

    def apply_failure(self, cause_event_id: str, reward: float = -0.35) -> None:
        target = self._find(cause_event_id)
        if target is None:
            return
        self._update(target.features, reward)
        target.resolved = True
        self.credit_log.append({"cause_event_id": cause_event_id, "credited_event_id": cause_event_id, "credit": reward, "kind": "failure"})

    def apply_success(self, cause_event_id: str, reward: float = 1.0) -> None:
        cause_index = next((i for i, x in enumerate(self.episode_trace) if x.event_id == cause_event_id), None)
        if cause_index is None:
            return
        # The causal attempt gets full credit even if feedback arrives after unrelated observations.
        cause = self.episode_trace[cause_index]
        self._update(cause.features, reward)
        cause.resolved = True
        self.credit_log.append({"cause_event_id": cause_event_id, "credited_event_id": cause_event_id, "credit": reward, "kind": "causal_success"})

        # Earlier informative actions are eligible because they changed the belief state used by the success.
        distance = 1
        for item in reversed(self.episode_trace[:cause_index]):
            if item.informative:
                credit = reward * (self.gamma ** distance)
                self._update(item.features, credit)
                self.credit_log.append({"cause_event_id": cause_event_id, "credited_event_id": item.event_id, "credit": credit, "kind": "informative_predecessor"})
                distance += 1

    def _update(self, features: frozenset[str], reward: float) -> None:
        if not self.enabled:
            return
        learnable = [f for f in features if not f.startswith("key:") and not f.startswith("goal:")]
        if not learnable:
            return
        delta = self.learning_rate * reward / len(learnable)
        for feature in learnable:
            self.weights[feature] += delta

    def _find(self, event_id: str) -> ActionTrace | None:
        return next((item for item in self.episode_trace if item.event_id == event_id), None)

    def snapshot(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        return dict(sorted(self.weights.items()))
