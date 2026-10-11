from __future__ import annotations

import math
import random

from .learning import OutcomeLearner
from .meaning import MeaningInterpreter
from .memory import EpisodicMemory
from .models import Action, RetrievalHit, SelectionDecision
from .state import CognitiveState


class ActionSelector:
    def __init__(self, memory: EpisodicMemory, learner: OutcomeLearner, rng: random.Random,
                 *, meaning_mode: str | None = None, refiner=None) -> None:
        self.memory = memory
        self.learner = learner
        self.rng = rng
        if meaning_mode not in (None, "fixed", "learned", "adaptive", "gate_only"):
            raise ValueError(f"unknown meaning mode {meaning_mode}")
        self.meaning_mode = meaning_mode
        self.interpreter = MeaningInterpreter(memory, learner) if meaning_mode else None
        self.refiner = refiner

    def select(self, state: CognitiveState, actions: list[Action], hits: list[RetrievalHit], *,
               refine: bool = False, transfer_scores: dict[str, float] | None = None,
               transfer_context: dict | None = None) -> SelectionDecision:
        if not actions:
            raise RuntimeError("no candidate actions")
        scored: list[tuple[float, Action, list[str], object]] = []
        prepared = self.interpreter.prepare(hits) if self.interpreter else None
        interpretation_evaluations = prepared.examined_event_records if prepared else 0
        for action in actions:
            features = state.action_features(action)
            base = self._base_prior(action)
            # Transfer state can mark obsolete feature associations without
            # destroying historical outcome learning or changing the memory query.
            policy_features = (
                state.policy_features(action) if hasattr(state, "policy_features")
                else features
            )
            learned = self.learner.expected_utility(policy_features)
            frame = None
            if transfer_scores is not None:
                memory_evidence = max(-0.45, min(0.45, transfer_scores.get(action.label, 0.0)))
            elif self.interpreter is not None:
                frame = self.interpreter.evaluate(
                    state, action, hits,
                    learn_layers=self.meaning_mode in ("learned", "adaptive"),
                    adaptive_gate=self.meaning_mode in ("adaptive", "gate_only"),
                    prepared=prepared,
                )
                memory_evidence = max(-0.45, min(0.45, 0.52 * frame.integrated))
                interpretation_evaluations += frame.examined_event_records
            else:
                memory_evidence = self._memory_evidence(action, state, hits)
            clue = 0.78 if "clue_supported" in features else 0.0
            contradiction = -1.15 if "contradicted_key" in features else 0.0
            repeat = -0.32 if action.label in state.attempted_actions else 0.0
            exploration = self.rng.uniform(-0.065, 0.065)
            # Ordered tasks use observed prerequisite knowledge from the one
            # OutcomeLearner, not an independent planner or the world's graph.
            dependency_penalty = 0.0
            if (action.kind == "resolve" and action.key_name and
                    hasattr(state, "completed_axes") and
                    getattr(state, "excluded_axis", None) != "sequence"):
                dependent = action.key_name
                prerequisites = self.learner.prerequisites_for(dependent)
                if "sequence" not in getattr(state, "invalidated_since", {}):
                    dependency_penalty += sum(
                        0.85 * strength for before, strength in prerequisites.items()
                        if before not in state.completed_axes
                    )
                dependency_penalty += sum(
                    1.1 for before in state.current_prerequisites.get(dependent, set())
                    if before not in state.completed_axes
                )
            score = base + learned + memory_evidence + clue + contradiction + repeat + exploration - dependency_penalty
            reasons = [
                f"base={base:.3f}",
                f"learned={learned:.3f}",
                f"memory={memory_evidence:.3f}",
                f"clue={clue:.3f}",
                f"contradiction={contradiction:.3f}",
                f"repeat={repeat:.3f}",
                f"explore={exploration:.3f}",
            ]
            if frame is not None:
                reasons.append(f"interpreted={frame.integrated:.3f}")
                reasons.append(f"context={frame.context}")
            scored.append((score, action, reasons, frame))
        thought_trace = None
        if refine:
            if self.refiner is None:
                raise ValueError('no thought refiner configured')
            if transfer_scores is not None:
                scored, thought_trace = self.refiner.refine_transfer(
                    state, scored, hits, context=transfer_context or {},
                )
            else:
                scored, thought_trace = self.refiner.refine(
                    state, scored, hits, prepared=prepared,
                )
        scored.sort(key=lambda item: (-item[0], item[1].label))
        score, action, reasons, frame = scored[0]
        predicted_success = 1.0 / (1.0 + math.exp(-score))
        if thought_trace and thought_trace.get("prediction") is not None:
            predicted_success = float(thought_trace["prediction"])
        return SelectionDecision(
            action=action,
            score=score,
            predicted_success=predicted_success,
            reasons=reasons,
            retrieval_hits=hits,
            meaning_frame=frame,
            interpretation_event_evaluations=interpretation_evaluations,
            thought_trace=thought_trace,
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
