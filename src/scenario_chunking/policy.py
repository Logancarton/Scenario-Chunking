from __future__ import annotations

import math
import random

from .learning import OutcomeLearner
from .memory import EpisodicMemory
from .models import Action, RetrievalHit, SelectionDecision
from .state import CognitiveState


class ActionSelector:
    def __init__(self, memory: EpisodicMemory, learner: OutcomeLearner, rng: random.Random) -> None:
        self.memory = memory
        self.learner = learner
        self.rng = rng

    def select(self, state: CognitiveState, actions: list[Action], hits: list[RetrievalHit]) -> SelectionDecision:
        if not actions:
            raise RuntimeError("no candidate actions")
        scored: list[tuple[float, Action, list[str]]] = []
        for action in actions:
            features = state.action_features(action)
            base = self._base_prior(action)
            learned = self.learner.expected_utility(features)
            memory_evidence = self._memory_evidence(action, state, hits)
            clue = 0.78 if "clue_supported" in features else 0.0
            contradiction = -1.15 if "contradicted_key" in features else 0.0
            repeat = -0.32 if action.label in state.attempted_actions else 0.0
            exploration = self.rng.uniform(-0.065, 0.065)
            score = base + learned + memory_evidence + clue + contradiction + repeat + exploration
            reasons = [
                f"base={base:.3f}",
                f"learned={learned:.3f}",
                f"memory={memory_evidence:.3f}",
                f"clue={clue:.3f}",
                f"contradiction={contradiction:.3f}",
                f"repeat={repeat:.3f}",
                f"explore={exploration:.3f}",
            ]
            scored.append((score, action, reasons))
        scored.sort(key=lambda item: (-item[0], item[1].label))
        score, action, reasons = scored[0]
        predicted_success = 1.0 / (1.0 + math.exp(-score))
        return SelectionDecision(
            action=action,
            score=score,
            predicted_success=predicted_success,
            reasons=reasons,
            retrieval_hits=hits,
        )

    @staticmethod
    def _base_prior(action: Action) -> float:
        return {
            "inspect_container": 0.24,
            "inspect_work_surface": 0.23,
            "ask_informant": 0.22,
            "inspect_storage": 0.21,
            "try_key": 0.18,
        }.get(action.kind, 0.0)

    def _memory_evidence(self, action: Action, state: CognitiveState, hits: list[RetrievalHit]) -> float:
        evidence = 0.0
        for hit in hits:
            events = self.memory.chunk_events(hit.chunk_id)
            matched_action = any(event.action and event.action.kind == action.kind for event in events)
            if not matched_action:
                continue
            has_success = any(event.outcome == "success" for event in events)
            has_failure = any(event.outcome == "failure" for event in events)
            if has_success:
                evidence += 0.16 * hit.score
            if has_failure:
                evidence -= 0.11 * hit.score
            if action.kind == "try_key" and action.key_name:
                if any(f"key:{action.key_name}" in event.semantic_tags and event.outcome == "success" for event in events):
                    evidence += 0.22 * hit.score
        return max(-0.45, min(0.45, evidence))
