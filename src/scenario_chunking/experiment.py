from __future__ import annotations

from dataclasses import asdict
import copy
import json
from pathlib import Path
import random
import statistics
import tracemalloc
from time import perf_counter
from typing import Any

from .dashboard import write_dashboard
from .environment import LockedWorkshop, default_episode_specs
from .learning import OutcomeLearner
from .memory import EpisodicMemory
from .models import (
    Action,
    EnvironmentSignal,
    EpisodeResult,
    Event,
    EventKind,
    ExperimentSummary,
    RetrievalMode,
    RetrievalQuery,
    RetrievalStats,
    TruthStatus,
)
from .policy import ActionSelector
from .retrieval import MemoryRouter, retrieval_recall_at_k
from .state import CognitiveState


class ExperimentRun:
    def __init__(self, mode: RetrievalMode, seed: int, *, top_k: int = 6, distractor_count: int = 1000, learning_enabled: bool = True, meaning_mode: str | None = None) -> None:
        self.mode = mode
        self.meaning_mode = meaning_mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.memory = EpisodicMemory()
        self.learner = OutcomeLearner(enabled=learning_enabled)
        self.environment = LockedWorkshop(seed)
        self.router = MemoryRouter(self.memory, mode, top_k=top_k)
        self.selector = ActionSelector(self.memory, self.learner, self.rng, meaning_mode=meaning_mode)
        self.clock = 0.0
        self.distractor_count = distractor_count
        self.learning_enabled = learning_enabled
        self.success_chunks: list[tuple[str, str]] = []  # (chunk_id, key)
        self.preference_snapshots: list[dict[str, float]] = []
        self.recall_diagnostics: list[dict[str, Any]] = []
        self.same_snapshot_recovery_audit: list[dict[str, Any]] = []

    def run(self) -> ExperimentSummary:
        results: list[EpisodeResult] = []
        for spec in default_episode_specs():
            if spec.episode_id == "E4_distractor_load":
                self.memory.inject_distractors(self.distractor_count, start_time=self.clock + 0.001)
                self.clock += 1.0
            results.append(self._run_episode(spec))
            self.preference_snapshots.append(self.learner.snapshot())

        recall_queries = self._build_recall_queries()
        initial_recall = []
        post_recall = []
        recovered = 0
        for query, relevant in recall_queries:
            shadow_hits = []
            shadow_stats = None
            if self.mode == RetrievalMode.STRUCTURAL and self.router.top_k == 6:
                # Preserve this exact pre-retrieval memory state for a paired
                # top-6 vs spreading diagnostic. Shadow activations never feed
                # back into the structural control's state or future actions.
                shadow_memory = copy.deepcopy(self.memory)
                shadow_hits, shadow_stats = MemoryRouter(
                    shadow_memory, RetrievalMode.SPREADING, top_k=6
                ).retrieve(query)
            hits, stats = self.router.retrieve(query)
            original = set(stats.initial_hit_ids)
            all_hits = {h.chunk_id for h in hits}
            initial_recall.append(len(original & relevant) / max(1, len(relevant)))
            post_recall.append(len(all_hits & relevant) / max(1, len(relevant)))
            recovered += len((all_hits - original) & relevant)
            for cid in sorted(relevant):
                if shadow_stats is not None:
                    self.same_snapshot_recovery_audit.append({
                        "relevant_chunk_id": cid,
                        "query_tags": sorted(query.semantic_tags),
                        "baseline_initial_hits": list(stats.initial_hit_ids),
                        "shadow_initial_hits": list(shadow_stats.initial_hit_ids),
                        "shadow_spread_hits": list(shadow_stats.spread_hit_ids),
                        "baseline_missed": cid not in original,
                        "recovered_by_shadow_spread": cid not in original and cid in {
                            h.chunk_id for h in shadow_hits
                        },
                    })
                score, reasons = self.router._score_chunk(chunk_id=cid, query=query)
                self.recall_diagnostics.append({
                    "relevant_chunk_id": cid,
                    "query_tags": sorted(query.semantic_tags),
                    "initial_hits": list(stats.initial_hit_ids),
                    "spread_hits": list(stats.spread_hit_ids),
                    "initially_missed": cid not in original,
                    "recovered": cid not in original and cid in all_hits,
                    "still_missed": cid not in all_hits,
                    "relevant_score": score,
                    "scoring_reasons": reasons,
                    "direct_neighbors": [
                        {"chunk_id": nid, "relation": kind, "evidence": ev}
                        for nid, kind, ev, _ in self.memory.neighbors(cid)
                    ],
                })
        recall = sum(initial_recall) / len(initial_recall) if initial_recall else 0.0
        after_recall = sum(post_recall) / len(post_recall) if post_recall else 0.0
        temporal_accuracy = self._temporal_order_accuracy()
        delayed_credit = next((r.credit_accuracy for r in results if r.episode_id == "E5_delayed_feedback"), None) or 0.0
        return ExperimentSummary(
            mode=self.mode,
            seed=self.seed,
            episodes=results,
            retrieval_recall_at_k=recall,
            temporal_order_accuracy=temporal_accuracy,
            delayed_credit_accuracy=delayed_credit,
            recency_negative_control_passed=None,
            recall_with_spreading=after_recall,
            recovered_relevant_memories=recovered,
            spread_action_changes=sum(int(step.get("spread_changed_action", False)) for ep in results for step in ep.trace),
            spread_action_score_influences=sum(int(step.get("spread_changed_score", False)) for ep in results for step in ep.trace),
        )

    def _run_episode(self, spec) -> EpisodeResult:
        self.learner.begin_episode()
        state = CognitiveState(episode_id=spec.episode_id)
        initial = self.environment.reset(spec)
        init_events: list[Event] = []
        for signal in initial:
            event = self._event_from_signal(signal, state, step=0)
            init_events.append(event)
            state.apply_signal(signal)
        self.memory.create_chunk(
            events=init_events,
            transition_reason="goal_begins",
            self_relevance=1.0,
            salience=0.65,
        )

        aggregate = RetrievalStats()
        trace: list[dict[str, Any]] = []
        failed = repeated = unnecessary = outdated_retrievals = 0
        action_counts: dict[str, int] = {}
        max_actions = 14

        for step in range(1, max_actions + 1):
            query = state.retrieval_query()
            hits, stats = self.router.retrieve(query)
            aggregate += stats
            state.active_memory = [h.chunk_id for h in hits]
            for hit in hits:
                chunk = self.memory.chunks[hit.chunk_id]
                if any(f"key:{key}" in chunk.semantic_tags for key in state.contradicted_keys) and "outcome:success" in chunk.semantic_tags:
                    outdated_retrievals += 1

            actions = state.candidate_actions()
            if not actions:
                break
            original_decision = None
            rng_snapshot = self.rng.getstate()
            if self.mode == RetrievalMode.SPREADING:
                original_decision = self.selector.select(state, actions, hits[: self.router.top_k])
                self.rng.setstate(rng_snapshot)
            paired_direct_action = None
            if self.meaning_mode is not None:
                self.rng.setstate(rng_snapshot)
                paired_direct_action = ActionSelector(self.memory, self.learner, self.rng).select(
                    state, actions, hits).action.label
                self.rng.setstate(rng_snapshot)
            decision_started = perf_counter()
            decision = self.selector.select(state, actions, hits)
            selection_latency_ms = (perf_counter() - decision_started) * 1000.0
            rng_after_selection = self.rng.getstate()
            # Leave-one-out analysis for EACH additional activation. The policy
            # and random exploration state are exactly matched to the live choice.
            individual_spread_influence = {}
            for added in hits[self.router.top_k:]:
                self.rng.setstate(rng_snapshot)
                without = self.selector.select(
                    state, actions, [h for h in hits if h.chunk_id != added.chunk_id]
                )
                individual_spread_influence[added.chunk_id] = {
                    "selection_changed_without": without.action != decision.action,
                    "action_without": without.action.label,
                    "selected_score_delta": round(decision.score - without.score, 6),
                    "changed_selected_score": abs(decision.score - without.score) > 1e-9,
                }
            self.rng.setstate(rng_after_selection)
            spread_changed_action = bool(original_decision and original_decision.action != decision.action)
            spread_changed_score = bool(original_decision and abs(original_decision.score - decision.score) > 1e-9)
            action = decision.action
            before_features = state.action_features(action)
            repeated_now = action_counts.get(action.label, 0) > 0
            repeated += int(repeated_now)
            action_counts[action.label] = action_counts.get(action.label, 0) + 1
            state.record_action(action)

            prediction = self._prediction_event(state, step, decision)
            action_event = self._action_event(state, step, action, before_features)
            self.learner.register_action(action_event.event_id, before_features)
            signals = self.environment.step(action, cause_event_id=action_event.event_id)

            outcome_events: list[Event] = []
            action_informative = False
            action_failed = False
            action_succeeded = False
            for signal in signals:
                event = self._event_from_signal(signal, state, step)
                outcome_events.append(event)
                state.apply_signal(signal)
                if signal.informative and signal.kind == EventKind.OBSERVATION:
                    action_informative = True
                if signal.outcome == "failure":
                    failed += 1
                    action_failed = True
                    self.learner.apply_failure(signal.cause_event_id or action_event.event_id, signal.reward)
                if signal.outcome == "success":
                    action_succeeded = True
                    self.learner.apply_success(signal.cause_event_id or action_event.event_id, signal.reward)
            if action_informative:
                self.learner.mark_informative(action_event.event_id)
            if not action_informative and not action_succeeded:
                unnecessary += 1
            if self.meaning_mode and decision.meaning_frame is not None:
                # Only actual observed outcome or immediate information gain.
                # No reward from future hidden environment state.
                self.learner.learn_from_meaning(
                    decision.meaning_frame,
                    beneficial=action_succeeded or action_informative,
                    learn_layers=self.meaning_mode in ("learned", "adaptive"),
                    learn_gate=self.meaning_mode in ("adaptive", "gate_only"),
                )

            salience = 1.0 if action_succeeded else (0.82 if action_failed or action_informative else 0.35)
            reason = "action_succeeds" if action_succeeded else ("action_fails" if action_failed else ("state_changes" if action_informative else "action_cycle"))
            chunk = self.memory.create_chunk(
                events=[prediction, action_event, *outcome_events],
                transition_reason=reason,
                self_relevance=1.0,
                salience=salience,
                validity_tags=frozenset({f"contradicts:{k}" for k in state.contradicted_keys}),
            )
            if action_succeeded:
                for entry in self.learner.credit_log:
                    if entry.get("cause_event_id") != action_event.event_id or entry.get("kind") != "informative_predecessor":
                        continue
                    previous_id = self.memory.event_to_chunk.get(str(entry["credited_event_id"]))
                    if previous_id:
                        self.memory.link_causal(previous_id, chunk.chunk_id, evidence="observed", confidence=0.90)
            if action_succeeded and action.key_name:
                self.success_chunks.append((chunk.chunk_id, action.key_name))

            trace.append({
                "step": step,
                "time": state.current_time,
                "action": action.label,
                "score": round(decision.score, 5),
                "predicted_success": round(decision.predicted_success, 5),
                "selection_latency_ms": selection_latency_ms,
                "interpretation_event_evaluations": decision.interpretation_event_evaluations,
                "selection_reasons": decision.reasons,
                "retrieved_chunks": [{"id": h.chunk_id, "score": round(h.score, 5), "reasons": h.reasons} for h in hits],
                "initially_activated": list(stats.initial_hit_ids),
                "spread_activated": list(stats.spread_hit_ids),
                "spread_justifications": dict(stats.spread_justifications),
                "individual_spread_influence": individual_spread_influence,
                "spread_detailed_events": stats.spread_detailed_events,
                "total_detailed_events_this_retrieval": stats.active_detail_events,
                "neighbor_headers_examined": stats.neighbor_headers_examined,
                "spread_budget_exhausted": stats.budget_exhausted,
                "counterfactual_initial_action": original_decision.action.label if original_decision else None,
                "spread_changed_action": spread_changed_action,
                "spread_changed_score": spread_changed_score,
                "signals": [s.content for s in signals],
                "candidate_key": state.candidate_key,
                "contradicted_keys": sorted(state.contradicted_keys),
                "learner_weights": self.learner.snapshot(),
                "meaning": decision.meaning_frame.trace() if decision.meaning_frame else None,
                "meaning_learning": self.learner.meaning_snapshot() if self.meaning_mode else None,
                "paired_direct_action": paired_direct_action,
                "paired_direct_choice_changed": bool(paired_direct_action and paired_direct_action != action.label),
                "success": action_succeeded,
            })
            if state.goal_complete:
                break

        credit_accuracy = None
        if spec.episode_id == "E5_delayed_feedback":
            causal = [x for x in self.learner.credit_log if x.get("kind") == "causal_success"]
            credit_accuracy = 1.0 if causal and causal[-1]["cause_event_id"] == causal[-1]["credited_event_id"] else 0.0

        return EpisodeResult(
            episode_id=spec.episode_id,
            success=state.goal_complete,
            actions=len(trace),
            failed_actions=failed,
            repeated_actions=repeated,
            unnecessary_actions=unnecessary,
            retrieval_stats=aggregate,
            trace=trace,
            outdated_retrievals=outdated_retrievals,
            credit_accuracy=credit_accuracy,
        )

    def _prediction_event(self, state: CognitiveState, step: int, decision) -> Event:
        self.clock += 1.0
        return Event(
            event_id=self.memory.next_event_id(),
            episode_id=state.episode_id,
            goal_id=state.goal_id,
            step=step,
            occurrence_time=self.clock,
            acquired_time=self.clock,
            storage_time=self.clock,
            kind=EventKind.PREDICTION,
            truth_status=TruthStatus.PREDICTION,
            content=f"Predict {decision.action.label} has success probability {decision.predicted_success:.3f}.",
            semantic_tags=frozenset({f"goal:{state.goal_id}", f"action:{decision.action.kind}", "record:prediction"}),
            action=decision.action,
            confidence=decision.predicted_success,
        )

    def _action_event(self, state: CognitiveState, step: int, action: Action, features: frozenset[str]) -> Event:
        self.clock += 1.0
        return Event(
            event_id=self.memory.next_event_id(),
            episode_id=state.episode_id,
            goal_id=state.goal_id,
            step=step,
            occurrence_time=self.clock,
            acquired_time=self.clock,
            storage_time=self.clock,
            kind=EventKind.ACTION,
            truth_status=TruthStatus.ATTEMPT,
            content=f"Attempted {action.label}.",
            semantic_tags=features | frozenset({"record:attempt"}),
            person_id=action.person,
            action=action,
        )

    def _event_from_signal(self, signal: EnvironmentSignal, state: CognitiveState, step: int) -> Event:
        self.clock += 1.0
        return Event(
            event_id=self.memory.next_event_id(),
            episode_id=state.episode_id,
            goal_id=state.goal_id,
            step=step,
            occurrence_time=self.clock,
            acquired_time=self.clock,
            storage_time=self.clock,
            kind=signal.kind,
            truth_status=signal.truth_status,
            content=signal.content,
            semantic_tags=signal.semantic_tags | frozenset({f"goal:{state.goal_id}"}),
            person_id=signal.person_id,
            outcome=signal.outcome,
            reward=signal.reward,
            cause_event_id=signal.cause_event_id,
            informative=signal.informative,
            confidence=signal.confidence,
            metadata=dict(signal.metadata),
        )

    def _build_recall_queries(self) -> list[tuple[RetrievalQuery, set[str]]]:
        queries: list[tuple[RetrievalQuery, set[str]]] = []
        for chunk_id, key in self.success_chunks:
            chunk = self.memory.chunks[chunk_id]
            queries.append((
                RetrievalQuery(
                    goal_id=chunk.header.goal_id,
                    semantic_tags=frozenset({f"goal:{chunk.header.goal_id}", "role:key", f"key:{key}", "outcome:success"}),
                    current_time=self.clock + 1.0,
                ),
                {chunk_id},
            ))
        return queries

    def sparse_miss_cases(self) -> list[dict[str, object]]:
        if self.mode == RetrievalMode.FULL:
            return []
        cases: list[dict[str, object]] = []
        full_router = MemoryRouter(self.memory, RetrievalMode.FULL, top_k=self.router.top_k)
        for query, relevant in self._build_recall_queries():
            sparse_hits, _ = self.router.retrieve(query)
            full_hits, _ = full_router.retrieve(query)
            sparse_ids = {h.chunk_id for h in sparse_hits}
            full_ids = {h.chunk_id for h in full_hits}
            for chunk_id in sorted((relevant & full_ids) - sparse_ids):
                cases.append({
                    "relevant_chunk_id": chunk_id,
                    "query_tags": sorted(query.semantic_tags),
                    "sparse_top_k": [h.chunk_id for h in sparse_hits],
                    "full_top_k": [h.chunk_id for h in full_hits],
                })
        return cases

    def _temporal_order_accuracy(self) -> float:
        checks: list[bool] = []
        for chunks in self.memory.episode_index.values():
            for left, right in zip(chunks, chunks[1:]):
                a = self.memory.chunks[left]
                b = self.memory.chunks[right]
                checks.append(a.header.sequence_index < b.header.sequence_index)
                checks.append(a.header.occurrence_time <= b.header.occurrence_time)
                checks.append(a.header.next_chunk_id == right)
                checks.append(b.header.previous_chunk_id == left)
        return sum(checks) / len(checks) if checks else 1.0


def run_recency_negative_control() -> dict[str, Any]:
    memory = EpisodicMemory()
    goal = "retrieve_object_from_locked_container"
    old = Event(
        event_id=memory.next_event_id(), episode_id="NC_old", goal_id=goal, step=1,
        occurrence_time=1.0, acquired_time=1.0, storage_time=1.0,
        kind=EventKind.OUTCOME, truth_status=TruthStatus.OUTCOME,
        content="Older but causally confirmed: inspecting storage revealed the successful key relation.",
        semantic_tags=frozenset({f"goal:{goal}", "role:storage", "clue:key_relation", "outcome:success"}),
        outcome="success", reward=1.0, informative=True,
    )
    old_chunk = memory.create_chunk(events=[old], transition_reason="confirmed_success", self_relevance=1.0, salience=1.0)
    recent = Event(
        event_id=memory.next_event_id(), episode_id="NC_recent", goal_id=goal, step=1,
        occurrence_time=100.0, acquired_time=100.0, storage_time=100.0,
        kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
        content="Recent but noncausal storage observation: fresh labels were stacked nearby.",
        semantic_tags=frozenset({f"goal:{goal}", "role:storage", "observation:irrelevant"}),
        informative=False,
    )
    recent_chunk = memory.create_chunk(events=[recent], transition_reason="recent_distractor", self_relevance=0.2, salience=0.2)
    query = RetrievalQuery(
        goal_id=goal,
        semantic_tags=frozenset({f"goal:{goal}", "role:storage", "clue:key_relation"}),
        current_time=101.0,
    )
    fixed_hits, _ = MemoryRouter(memory, RetrievalMode.RECENCY, top_k=1).retrieve(query)
    structural_hits, _ = MemoryRouter(memory, RetrievalMode.STRUCTURAL, top_k=1).retrieve(query)
    return {
        "old_relevant_chunk": old_chunk.chunk_id,
        "recent_distractor_chunk": recent_chunk.chunk_id,
        "fixed_recency_top1": fixed_hits[0].chunk_id if fixed_hits else None,
        "structural_top1": structural_hits[0].chunk_id if structural_hits else None,
        "fixed_recency_worse": bool(fixed_hits and structural_hits and fixed_hits[0].chunk_id == recent_chunk.chunk_id and structural_hits[0].chunk_id == old_chunk.chunk_id),
    }


def _try_key_brier(raw: dict[str, Any]) -> float | None:
    observations = [
        (step["predicted_success"], int(step["success"]))
        for episode in raw["episodes"] for step in episode["trace"]
        if step["action"].startswith("try_key:")
    ]
    return (sum((p - y) ** 2 for p, y in observations) / len(observations)
            if observations else None)


def run_experiments(
    output_dir: str | Path,
    *,
    seeds: tuple[int, ...] = (7, 23, 61, 101, 149),
    modes: tuple[RetrievalMode, ...] = tuple(RetrievalMode),
    distractor_count: int = 1000,
    include_meaning: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    traces_path = output / "event_traces.jsonl"
    traces_path.write_text("", encoding="utf-8")

    recency_control = run_recency_negative_control()

    for mode in modes:
        for seed in seeds:
            tracemalloc.start()
            run = ExperimentRun(mode, seed, distractor_count=distractor_count)
            summary = run.run()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            summary.recency_negative_control_passed = recency_control["fixed_recency_worse"] if mode == RetrievalMode.RECENCY else None
            raw = asdict(summary)
            raw["mode"] = mode.value
            raw["try_key_brier"] = _try_key_brier(raw)
            raw["peak_memory_bytes"] = peak
            raw["learned_weights"] = run.learner.snapshot()
            raw["preference_snapshots"] = run.preference_snapshots
            raw["sparse_miss_cases"] = run.sparse_miss_cases()
            raw["recall_diagnostics"] = run.recall_diagnostics
            raw["same_snapshot_recovery_audit"] = run.same_snapshot_recovery_audit
            summaries.append(raw)
            with traces_path.open("a", encoding="utf-8") as fh:
                for episode in summary.episodes:
                    for entry in episode.trace:
                        fh.write(json.dumps({"mode": mode.value, "seed": seed, "episode": episode.episode_id, **entry}, sort_keys=True) + "\n")

    ablation_runs: list[dict[str, Any]] = []
    for seed in seeds:
        run = ExperimentRun(RetrievalMode.STRUCTURAL, seed, distractor_count=distractor_count, learning_enabled=False)
        summary = run.run()
        raw = asdict(summary)
        raw["mode"] = "structural_no_policy_learning"
        raw["try_key_brier"] = _try_key_brier(raw)
        raw["peak_memory_bytes"] = 0
        raw["learned_weights"] = run.learner.snapshot()
        raw["preference_snapshots"] = run.preference_snapshots
        raw["sparse_miss_cases"] = run.sparse_miss_cases()
        raw["recall_diagnostics"] = run.recall_diagnostics
        raw["same_snapshot_recovery_audit"] = run.same_snapshot_recovery_audit
        ablation_runs.append(raw)

    for seed in seeds:
        tracemalloc.start()
        run = ExperimentRun(RetrievalMode.STRUCTURAL, seed, top_k=8, distractor_count=distractor_count)
        summary = run.run()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        raw = asdict(summary)
        raw["mode"] = "structural_topk8"
        raw["try_key_brier"] = _try_key_brier(raw)
        raw["peak_memory_bytes"] = peak
        raw["learned_weights"] = run.learner.snapshot()
        raw["preference_snapshots"] = run.preference_snapshots
        raw["sparse_miss_cases"] = run.sparse_miss_cases()
        raw["recall_diagnostics"] = run.recall_diagnostics
        raw["same_snapshot_recovery_audit"] = run.same_snapshot_recovery_audit
        summaries.append(raw)
        with traces_path.open("a", encoding="utf-8") as fh:
            for episode in summary.episodes:
                for entry in episode.trace:
                    fh.write(json.dumps({"mode": "structural_topk8", "seed": seed, "episode": episode.episode_id, **entry}, sort_keys=True) + "\n")

    if include_meaning:
        # Four treatments isolate interpretation, axis learning, and gate learning.
        # Identical random seeds/candidate evidence within each start state; distinct
        # closed-loop action trajectories are explicitly NOT paired future states.
        for treatment in ("fixed", "learned", "gate_only", "adaptive"):
            for seed in seeds:
                tracemalloc.start()
                run = ExperimentRun(
                    RetrievalMode.STRUCTURAL, seed,
                    distractor_count=distractor_count, meaning_mode=treatment,
                )
                summary = run.run()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                raw = asdict(summary)
                raw["mode"] = "meaning_" + treatment
                raw["peak_memory_bytes"] = peak
                raw["learned_weights"] = run.learner.snapshot()
                raw["meaning_snapshot"] = run.learner.meaning_snapshot()
                raw["layer_meanings_stored"] = len(run.memory.layer_meanings)
                raw["paired_direct_action_changes"] = sum(
                    int(step.get("paired_direct_choice_changed", False))
                    for ep in summary.episodes for step in ep.trace
                )
                # Goal-outcome probability calibration uses try-key attempts only:
                # informative inspections do not count as completed goals.
                attempts = [(step["predicted_success"], int(step["success"]))
                            for ep in summary.episodes for step in ep.trace
                            if step["action"].startswith("try_key:")]
                raw["try_key_brier"] = (
                    sum((p - y) ** 2 for p, y in attempts) / len(attempts)
                    if attempts else None
                )
                summaries.append(raw)
                with traces_path.open("a", encoding="utf-8") as fh:
                    for episode in summary.episodes:
                        for entry in episode.trace:
                            fh.write(json.dumps({
                                "mode": raw["mode"], "seed": seed,
                                "episode": episode.episode_id, **entry,
                            }, sort_keys=True) + "\n")

    aggregate = _aggregate(summaries)
    ablation_aggregate = _aggregate(ablation_runs).get("structural_no_policy_learning", {})
    payload = {
        "seeds": list(seeds),
        "distractor_count": distractor_count,
        "recency_negative_control": recency_control,
        "neighbor_budget": {"max_additional_chunks": 2, "max_neighbor_headers": 18, "max_total_detail_events": 36, "max_hops": 1},
        "runs": summaries,
        "aggregate": aggregate,
        "meaning_experiment": {
            "enabled": include_meaning,
            "treatments": ["structural", "meaning_fixed", "meaning_learned",
                           "meaning_gate_only", "meaning_adaptive"] if include_meaning else [],
            "train_episodes": ["E1_initial", "E2_similar", "E3_temporal_interference",
                               "E4_distractor_load", "E5_delayed_feedback"],
            "held_out_episode": "E6_relational_transfer",
            "held_out_seeds": [x for x in seeds if x in (101, 149)],
            "interpretation": "Same initial seeds and retrieval algorithm. Different closed-loop "
                              "trajectories are not matched memory states; paired_direct_action "
                              "compares decisions on the SAME within-treatment evidence.",
        },
        "ablations": {
            "structural_no_policy_learning": {
                "aggregate": ablation_aggregate,
                "runs": ablation_runs,
                "interpretation": "Disables explicit outcome-driven policy-weight updates while leaving episodic storage and retrieval intact."
            }
        },
        "claims": {
            "transformer_flops_measured": False,
            "processing_cost_note": "Header evaluations and detail loads are architectural processing-count proxies; latency and peak memory are measured runtime values.",
            "consciousness_claim": False,
        },
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_dashboard(payload, output / "dashboard.html")
    return payload


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_mode.setdefault(run["mode"], []).append(run)
    result: dict[str, Any] = {}
    for mode, group in by_mode.items():
        all_eps = [ep for run in group for ep in run["episodes"]]
        result[mode] = {
            "completion_rate_mean": statistics.mean(sum(1 for ep in run["episodes"] if ep["success"]) / len(run["episodes"]) for run in group),
            "actions_mean": statistics.mean(sum(ep["actions"] for ep in run["episodes"]) for run in group),
            "failed_actions_mean": statistics.mean(sum(ep["failed_actions"] for ep in run["episodes"]) for run in group),
            "repeated_actions_mean": statistics.mean(sum(ep["repeated_actions"] for ep in run["episodes"]) for run in group),
            "episode2_actions_mean": statistics.mean(next(ep["actions"] for ep in run["episodes"] if ep["episode_id"] == "E2_similar") for run in group),
            "episode1_actions_mean": statistics.mean(next(ep["actions"] for ep in run["episodes"] if ep["episode_id"] == "E1_initial") for run in group),
            "improvement_e1_to_e2_mean": statistics.mean(
                next(ep["actions"] for ep in run["episodes"] if ep["episode_id"] == "E1_initial") - next(ep["actions"] for ep in run["episodes"] if ep["episode_id"] == "E2_similar")
                for run in group
            ),
            "retrieval_recall_at_k_mean": statistics.mean(run["retrieval_recall_at_k"] for run in group),
            "recall_after_spreading_mean": statistics.mean(run["recall_with_spreading"] for run in group),
            "recovered_relevant_memories_mean": statistics.mean(run["recovered_relevant_memories"] for run in group),
            "spread_action_changes_mean": statistics.mean(run["spread_action_changes"] for run in group),
            "spread_action_score_influences_mean": statistics.mean(run["spread_action_score_influences"] for run in group),
            "spread_details_mean": statistics.mean(sum(ep["retrieval_stats"]["spread_detailed_events"] for ep in run["episodes"]) for run in group),
            "neighbor_headers_mean": statistics.mean(sum(ep["retrieval_stats"]["neighbor_headers_examined"] for ep in run["episodes"]) for run in group),
            "temporal_order_accuracy_mean": statistics.mean(run["temporal_order_accuracy"] for run in group),
            "delayed_credit_accuracy_mean": statistics.mean(run["delayed_credit_accuracy"] for run in group),
            "outdated_retrievals_mean": statistics.mean(sum(ep["outdated_retrievals"] for ep in run["episodes"]) for run in group),
            "header_evaluations_mean": statistics.mean(sum(ep["retrieval_stats"]["header_evaluations"] for ep in run["episodes"]) for run in group),
            "detail_loads_mean": statistics.mean(sum(ep["retrieval_stats"]["detail_loads"] for ep in run["episodes"]) for run in group),
            "active_detail_events_mean": statistics.mean(sum(ep["retrieval_stats"]["active_detail_events"] for ep in run["episodes"]) for run in group),
            "unnecessary_memory_activations_mean": statistics.mean(sum(ep["retrieval_stats"]["unnecessary_activations"] for ep in run["episodes"]) for run in group),
            "retrieval_latency_ms_mean": statistics.mean(sum(ep["retrieval_stats"]["latency_ms"] for ep in run["episodes"]) for run in group),
            "peak_memory_bytes_mean": statistics.mean(run["peak_memory_bytes"] for run in group),
            "goal_completions": sum(1 for ep in all_eps if ep["success"]),
            "goal_attempts": len(all_eps),
            "selection_latency_ms_mean": statistics.mean(
                sum(step.get("selection_latency_ms", 0.0) for ep in run["episodes"]
                    for step in ep["trace"]) for run in group),
            "interpretation_event_evaluations_mean": statistics.mean(
                sum(step.get("interpretation_event_evaluations", 0) for ep in run["episodes"]
                    for step in ep["trace"]) for run in group),
            "total_decision_event_evaluations_mean": statistics.mean(
                sum(ep["retrieval_stats"]["active_detail_events"] for ep in run["episodes"])
                + sum(step.get("interpretation_event_evaluations", 0) for ep in run["episodes"]
                    for step in ep["trace"]) for run in group),
            "try_key_brier_mean": statistics.mean(
                r["try_key_brier"] for r in group if r.get("try_key_brier") is not None
            ) if any(r.get("try_key_brier") is not None for r in group) else None,
            "paired_direct_action_changes_mean": statistics.mean(
                r.get("paired_direct_action_changes", 0) for r in group
            ),
            "stored_meanings_mean": statistics.mean(
                r.get("layer_meanings_stored", 0) for r in group
            ),
            "actions_std": statistics.pstdev(sum(ep["actions"] for ep in run["episodes"]) for run in group),
            "improvement_e1_to_e2_std": statistics.pstdev(
                next(ep["actions"] for ep in run["episodes"] if ep["episode_id"] == "E1_initial") - next(ep["actions"] for ep in run["episodes"] if ep["episode_id"] == "E2_similar")
                for run in group
            ),
            "retrieval_recall_at_k_std": statistics.pstdev(run["retrieval_recall_at_k"] for run in group),
            "active_detail_events_std": statistics.pstdev(sum(ep["retrieval_stats"]["active_detail_events"] for ep in run["episodes"]) for run in group),
        }
    return result
