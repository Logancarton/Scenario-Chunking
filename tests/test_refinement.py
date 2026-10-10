"""Experiment 5 mechanism tests: continuity, bounded retrieval and live feedback."""
import random

from scenario_chunking.experiment import ExperimentRun
from scenario_chunking.learning import OutcomeLearner
from scenario_chunking.meaning import MeaningInterpreter
from scenario_chunking.memory import EpisodicMemory
from scenario_chunking.models import (
    Action, Event, EventKind, RetrievalHit, RetrievalMode, RetrievalQuery, TruthStatus,
)
from scenario_chunking.policy import ActionSelector
from scenario_chunking.refinement import ProgressiveRefiner
from scenario_chunking.retrieval import IndexedCueReinstater, MemoryRouter
from scenario_chunking.state import CognitiveState

GOAL = "retrieve_object_from_locked_container"


def outcome_chunk(memory, key, *, episode="historical", outcome="success"):
    action = Action("try_key", target="container", key_name=key)
    attempt = Event(
        event_id=memory.next_event_id(), episode_id=episode,
        goal_id=GOAL, step=1,
        occurrence_time=1, acquired_time=1, storage_time=1,
        kind=EventKind.ACTION, truth_status=TruthStatus.ATTEMPT,
        content="attempt", action=action, semantic_tags=frozenset({f"key:{key}"}),
    )
    result = Event(
        event_id=memory.next_event_id(), episode_id=episode, goal_id=GOAL, step=1,
        occurrence_time=2, acquired_time=2, storage_time=2,
        kind=EventKind.OUTCOME, truth_status=TruthStatus.OUTCOME,
        content="observed outcome", outcome=outcome, cause_event_id=attempt.event_id,
        semantic_tags=frozenset({f"key:{key}", f"outcome:{outcome}"}),
    )
    return memory.create_chunk(events=[attempt, result], transition_reason="outcome",
                               self_relevance=1.0, salience=0.8), attempt, result


def test_continuous_thought_revises_without_rewriting_observations():
    memory = EpisodicMemory()
    chunk, attempt, outcome = outcome_chunk(memory, "steel")
    learner = OutcomeLearner()
    refiner = ProgressiveRefiner(learner)
    selector = ActionSelector(memory, learner, random.Random(2),
                              meaning_mode="fixed", refiner=refiner)
    state = CognitiveState(episode_id="new")
    state.visible_keys = ["steel", "brass"]
    state.candidate_key = "steel"
    state.inspected.update([state.storage_name, state.work_surface_name, state.container_name])
    state.asked_people.add(state.person_name)
    hits, _ = MemoryRouter(memory, RetrievalMode.STRUCTURAL, top_k=2).retrieve(
        state.retrieval_query())
    first = selector.select(state, state.candidate_actions(), hits, refine=True)
    thread = refiner.thread(state)
    assert first.thought_trace["thought_id"] == thread.thought_id
    first_count = len(thread.versions)
    assert first_count >= 2  # provisional local belief, then historical evidence
    again = selector.select(state, state.candidate_actions(), hits, refine=True)
    assert len(thread.versions) == first_count
    assert not again.thought_trace["changed"]  # no spinning on identical evidence
    # New contradictory observation changes applicability, not historical truth.
    state.contradicted_keys.add("steel")
    state.candidate_key = None
    state.observed_evidence_ids.append(outcome.event_id)
    changed = selector.select(state, state.candidate_actions(), hits, refine=True)
    assert changed.thought_trace["changed"]
    assert thread.versions[-1].contradictions == ("steel",)
    assert thread.versions[-1].number > thread.versions[0].number
    assert memory.events[outcome.event_id].outcome == "success"
    assert memory.events[attempt.event_id].action.key_name == "steel"
    assert first.thought_trace["additional_detailed_reads"] == 0


def test_cue_reinstates_outside_top_k_without_scanning_detailed_history():
    memory = EpisodicMemory()
    forgotten, _, _ = outcome_chunk(memory, "steel", episode="older")
    recent, _, _ = outcome_chunk(memory, "brass", episode="newer")
    state = CognitiveState(episode_id="new")
    state.current_time = 200  # old activation has decayed lazily
    state.visible_keys = ["steel", "brass"]
    query = RetrievalQuery(goal_id=GOAL,
                           semantic_tags=frozenset({f"goal:{GOAL}", "key:brass"}),
                           current_time=200)
    router = MemoryRouter(memory, RetrievalMode.STRUCTURAL, top_k=1)
    hits, stats = router.retrieve(query)
    assert [h.chunk_id for h in hits] == [recent.chunk_id]
    assert forgotten.header.activation_at(200) < 0.01
    expanded = IndexedCueReinstater(memory).expand(
        query, state.candidate_actions(), hits, stats,
        rank_header=lambda cid, q: router._score_chunk(chunk_id=cid, query=q))
    assert [x.chunk_id for x in expanded] == [forgotten.chunk_id]
    assert stats.reinstated_hit_ids == [forgotten.chunk_id]
    assert stats.reinstatement_headers_examined <= 18
    assert stats.active_detail_events <= 36


def test_forged_outcome_tag_is_not_accepted_as_causal_evidence():
    memory = EpisodicMemory()
    known, _, _ = outcome_chunk(memory, "brass", episode="known")
    forged = Event(
        event_id=memory.next_event_id(), episode_id="misleading",
        goal_id=GOAL, step=1, occurrence_time=3, acquired_time=3, storage_time=3,
        kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT,
        content="unverified claim", semantic_tags=frozenset({"key:steel", "outcome:success"}),
    )
    memory.create_chunk(events=[forged], transition_reason="unverified_claim",
                        self_relevance=1.0, salience=1.0)
    query = RetrievalQuery(goal_id=GOAL, semantic_tags=frozenset({f"goal:{GOAL}", "key:brass"}),
                           current_time=100)
    router = MemoryRouter(memory, RetrievalMode.STRUCTURAL, top_k=1)
    hits, stats = router.retrieve(query)
    assert hits[0].chunk_id == known.chunk_id
    actions = [Action("try_key", target="container", key_name="steel")]
    extra = IndexedCueReinstater(memory).expand(
        query, actions, hits, stats,
        rank_header=lambda cid, q: router._score_chunk(chunk_id=cid, query=q))
    assert extra == []
    assert stats.reinstatement_detail_events == 1  # rejected reads are charged


def test_outcome_learning_is_calibration_not_thought_change_reward():
    learner = OutcomeLearner()
    before = learner.refinement_adjustment("exploration", "try_key")
    assert before == 0.0
    learner.learn_from_refinement("exploration", "try_key",
                                  predicted=0.8, outcome=False)
    assert learner.refinement_adjustment("exploration", "try_key") < 0.0
    assert len(learner.refinement_updates) == 1
    assert learner.refinement_adjustment("social", "ask_informant") == 0.0


def test_live_six_scenario_path_preserves_bounded_resources_and_global_clock():
    run = ExperimentRun(RetrievalMode.STRUCTURAL, 7, distractor_count=35,
                        refinement_mode="associate")
    summary = run.run()
    assert len(summary.episodes) == 6
    assert len(run.refiner.threads) == 6
    assert all(step["thought"] and step["thought"]["thought_id"]
               for ep in summary.episodes for step in ep.trace)
    assert all(step["reinstatement_headers_examined"] <= 18
               and step["total_detailed_events_this_retrieval"] <= 36
               for ep in summary.episodes for step in ep.trace)
    for ep in summary.episodes:
        clocks = [step["clock_at_decision"] for step in ep.trace]
        assert clocks == sorted(clocks)
        assert all(x > 0 for x in clocks)
    assert any(len(t.versions) >= 1 for t in run.refiner.threads.values())


def test_prepared_evidence_is_equivalent_and_processed_once():
    memory = EpisodicMemory()
    chunk, _, _ = outcome_chunk(memory, "brass")
    hits = [RetrievalHit(chunk.chunk_id, 0.9, ["fixture"])]
    learner = OutcomeLearner()
    interpreter = MeaningInterpreter(memory, learner)
    state = CognitiveState(episode_id="current")
    action = Action("try_key", target="container", key_name="brass")
    direct = interpreter.evaluate(state, action, hits, learn_layers=False,
                                  adaptive_gate=False)
    prepared = interpreter.prepare(hits)
    reused = interpreter.evaluate(state, action, hits, learn_layers=False,
                                  adaptive_gate=False, prepared=prepared)
    assert direct.meanings == reused.meanings
    assert direct.integrated == reused.integrated
    assert direct.examined_event_records == prepared.examined_event_records == 2
    assert reused.examined_event_records == 0
    state.visible_keys = ["brass", "steel"]
    choice = ActionSelector(memory, learner, random.Random(0),
                            meaning_mode="fixed").select(
                                state, state.candidate_actions(), hits)
    assert choice.interpretation_event_evaluations == 2


def test_refinement_threads_decay_and_resume_without_resetting_history():
    memory = EpisodicMemory()
    chunk, _, _ = outcome_chunk(memory, "brass")
    hits = [RetrievalHit(chunk.chunk_id, 0.9, ["fixture"])]
    learner = OutcomeLearner()
    refiner = ProgressiveRefiner(learner)
    selector = ActionSelector(memory, learner, random.Random(1),
                              meaning_mode="fixed", refiner=refiner)
    state = CognitiveState(episode_id="thread")
    state.visible_keys = ["brass", "steel"]
    selector.select(state, state.candidate_actions(), hits, refine=True)
    thread = refiner.thread(state)
    original = thread.thought_id
    versions = len(thread.versions)
    assert thread.status_at(state.current_time) == "focused"
    assert thread.status_at(500) == "dormant"
    state.current_time = 500
    selector.select(state, state.candidate_actions(), hits, refine=True)
    assert thread.status_at(500) == "focused"
    assert thread.thought_id == original
    assert len(thread.versions) == versions


def test_reinstated_grounded_outcome_can_change_snapshot_action():
    memory = EpisodicMemory()
    negative, _, _ = outcome_chunk(memory, "steel", episode="negative", outcome="failure")
    positive, _, _ = outcome_chunk(memory, "brass", episode="positive", outcome="success")
    state = CognitiveState(episode_id="now")
    state.visible_keys = ["steel", "brass"]
    original = [RetrievalHit(negative.chunk_id, 0.9, ["initial"])]
    additional = [RetrievalHit(positive.chunk_id, 0.7, ["reinstated"])]
    before = ActionSelector(memory, OutcomeLearner(), random.Random(3),
                            meaning_mode="fixed").select(
                                state, state.candidate_actions(), original)
    after = ActionSelector(memory, OutcomeLearner(), random.Random(3),
                           meaning_mode="fixed").select(
                               state, state.candidate_actions(), original + additional)
    assert before.action != after.action
    assert after.action.key_name == "brass"
    # This proves a grounded policy influence, NOT superior generalization
    # or an improved closed-loop outcome.


def test_focused_thought_capacity_is_strict_and_dormant_threads_reenter():
    memory = EpisodicMemory()
    chunk, _, _ = outcome_chunk(memory, "brass")
    hits = [RetrievalHit(chunk.chunk_id, 0.9, ["fixture"])]
    learner = OutcomeLearner()
    refiner = ProgressiveRefiner(learner, max_focused=1, max_active=2)
    selector = ActionSelector(memory, learner, random.Random(1),
                              meaning_mode="fixed", refiner=refiner)
    first = None
    initial_versions = None
    for i in range(5):
        state = CognitiveState(episode_id=f"thread-{i}")
        state.visible_keys = ["steel", "brass"]
        state.current_time = 10
        selector.select(state, state.candidate_actions(), hits, refine=True)
        assert len(refiner._active) <= 2
        assert sum(t.status_at(10) == "focused"
                   for t in refiner._active.values()) <= 1
        assert refiner.thread(state).status_at(10) == "focused"
        if i == 0:
            first = state
            initial_versions = len(refiner.thread(state).versions)
    assert len(refiner.threads) == 5  # archived continuity, not active cognition
    first.current_time = 500
    selector.select(first, first.candidate_actions(), hits, refine=True)
    assert refiner.thread(first).status_at(500) == "focused"
    assert len(refiner.thread(first).versions) == initial_versions
    assert len(refiner._active) <= 2
