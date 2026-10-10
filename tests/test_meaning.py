"""Mechanism and live-path tests for independently learned layer interpretations."""
import random

from scenario_chunking.experiment import ExperimentRun
from scenario_chunking.learning import OutcomeLearner
from scenario_chunking.memory import EpisodicMemory
from scenario_chunking.meaning import AXES, MeaningInterpreter
from scenario_chunking.models import (
    Action, Event, EventKind, RetrievalHit, RetrievalMode, TruthStatus,
)
from scenario_chunking.policy import ActionSelector
from scenario_chunking.state import CognitiveState

GOAL = "retrieve_object_from_locked_container"


def fixture(*, duplicate=False, informative_only=False, person=None):
    memory = EpisodicMemory()
    action = Action("ask_informant", person=person) if person else Action("try_key", target="container", key_name="steel")
    attempt = Event(
        event_id=memory.next_event_id(), episode_id="historical", goal_id=GOAL, step=1,
        occurrence_time=1.0, acquired_time=1.0, storage_time=1.0,
        kind=EventKind.ACTION, truth_status=TruthStatus.ATTEMPT,
        content="attempt", action=action, person_id=person,
        semantic_tags=frozenset({"role:key"}),
    )
    response = Event(
        event_id=memory.next_event_id(), episode_id="historical", goal_id=GOAL, step=1,
        occurrence_time=2.0, acquired_time=2.0, storage_time=2.0,
        kind=EventKind.OBSERVATION if informative_only else EventKind.OUTCOME,
        truth_status=TruthStatus.FACT if informative_only else TruthStatus.OUTCOME,
        content="information" if informative_only else "success",
        outcome=None if informative_only else "success",
        reward=0.0 if informative_only else 1.0,
        informative=informative_only,
        cause_event_id=None if informative_only else attempt.event_id,
        semantic_tags=frozenset({"role:key", "outcome:success"}),
        confidence=0.9,
        metadata={"source_event_id": "original-source"},
    )
    c1 = memory.create_chunk(events=[attempt, response], transition_reason="outcome",
                             self_relevance=1.0, salience=0.8)
    hits = [RetrievalHit(c1.chunk_id, 0.8, ["fixture"])]
    if duplicate:
        other_attempt = Event(
            event_id=memory.next_event_id(), episode_id="copy", goal_id=GOAL, step=1,
            occurrence_time=3.0, acquired_time=3.0, storage_time=3.0,
            kind=EventKind.ACTION, truth_status=TruthStatus.ATTEMPT,
            content="same source retold", action=action, person_id=person,
        )
        other_response = Event(
            event_id=memory.next_event_id(), episode_id="copy", goal_id=GOAL, step=1,
            occurrence_time=4.0, acquired_time=4.0, storage_time=4.0,
            kind=response.kind, truth_status=response.truth_status,
            content="duplicate report", outcome=response.outcome, informative=response.informative,
            cause_event_id=other_attempt.event_id if not informative_only else None,
            semantic_tags=response.semantic_tags, confidence=response.confidence,
            metadata={"source_event_id": "original-source"},
        )
        c2 = memory.create_chunk(events=[other_attempt, other_response], transition_reason="repeated_source",
                                 self_relevance=1.0, salience=0.8)
        hits.append(RetrievalHit(c2.chunk_id, 0.9, ["duplicate"]))
    return memory, action, hits


def test_all_five_layers_are_distinct_and_store_evidence_references():
    m, action, hits = fixture(person="Mira")
    state = CognitiveState(episode_id="now")
    state.person_name = "Mira"
    learner = OutcomeLearner()
    frame = MeaningInterpreter(m, learner).evaluate(
        state, action, hits, learn_layers=False, adaptive_gate=False)
    axes = {x.axis for x in frame.meanings}
    assert axes == {"self", "salience", "sequence", "person"}
    assert "time" not in axes  # historical time alone is not evidence of present validity
    assert frame.independent_sources == 1
    assert all(mg.source_event_ids == ("original-source",) for mg in frame.meanings)
    assert len(m.layer_meanings) == 4
    assert all(mg.provenance == "observed_outcome" for mg in frame.meanings)


def test_duplicate_chunk_does_not_multiply_evidence_or_layer_weight():
    m, action, hits = fixture(duplicate=True)
    state = CognitiveState(episode_id="now")
    interpreter = MeaningInterpreter(m, OutcomeLearner())
    single = interpreter.evaluate(state, action, hits[:1], learn_layers=False, adaptive_gate=False)
    repeated = interpreter.evaluate(state, action, hits, learn_layers=False, adaptive_gate=False)
    assert repeated.independent_sources == single.independent_sources == 1
    assert repeated.integrated == single.integrated


def test_temporal_invalidation_is_separate_from_positive_outcome_and_ground_truth():
    m, action, hits = fixture()
    state = CognitiveState(episode_id="new")
    state.contradicted_keys.add("steel")
    frame = MeaningInterpreter(m, OutcomeLearner()).evaluate(
        state, action, hits, learn_layers=False, adaptive_gate=False)
    time = next(x for x in frame.meanings if x.axis == "time")
    assert time.value < 0
    assert not time.temporal_valid
    assert m.events[next(iter(m.events))].action.key_name == "steel"  # historical record unchanged


def test_similar_key_does_not_count_as_evidence_for_a_different_key():
    m, _, hits = fixture()
    state = CognitiveState(episode_id="now")
    frame = MeaningInterpreter(m, OutcomeLearner()).evaluate(
        state, Action("try_key", target="container", key_name="brass"),
        hits, learn_layers=False, adaptive_gate=False)
    assert frame.integrated == 0.0
    assert frame.independent_sources == 0


def test_informative_temporal_adjacency_is_not_claimed_causal():
    m, action, hits = fixture(person="Mira", informative_only=True)
    state = CognitiveState(episode_id="now")
    frame = MeaningInterpreter(m, OutcomeLearner()).evaluate(
        state, action, hits, learn_layers=False, adaptive_gate=False)
    assert "sequence" not in {x.axis for x in frame.meanings}
    assert all("informative" in x.provenance for x in frame.meanings)


def test_separate_feedback_learning_and_integration_gating():
    m, action, hits = fixture(person="Mira")
    learner = OutcomeLearner()
    interpreter = MeaningInterpreter(m, learner)
    state = CognitiveState(episode_id="now")
    before = interpreter.evaluate(state, action, hits, learn_layers=True, adaptive_gate=True)
    for _ in range(8):
        learner.learn_from_meaning(before, beneficial=True, learn_layers=True, learn_gate=True)
        before = interpreter.evaluate(state, action, hits, learn_layers=True, adaptive_gate=True)
    snap = learner.meaning_snapshot()
    assert snap["feedback_updates"] == 8
    assert any(abs(v) > 0 for v in snap["layer_associations"].values())
    assert any(abs(v) > 0 for v in snap["integration_logits"].values())
    assert any(abs(v - 0.25) > 0.00001 for v in before.weights.values())

    frozen = OutcomeLearner()
    frozen_frame = MeaningInterpreter(m, frozen).evaluate(
        state, action, hits, learn_layers=False, adaptive_gate=False)
    frozen.learn_from_meaning(frozen_frame, beneficial=True, learn_layers=False, learn_gate=False)
    assert not frozen.meaning_snapshot()["layer_associations"]
    assert not frozen.meaning_snapshot()["integration_logits"]


def test_paired_action_comparison_uses_identical_available_evidence_and_rng():
    m, action, hits = fixture()
    state = CognitiveState(episode_id="new")
    state.visible_keys = ["brass", "steel"]
    state.inspected = {state.storage_name, state.work_surface_name, state.container_name}
    state.asked_people = {state.person_name}
    actions = state.candidate_actions()
    before = ActionSelector(m, OutcomeLearner(), random.Random(3)).select(state, actions, hits)
    after = ActionSelector(m, OutcomeLearner(), random.Random(3), meaning_mode="fixed").select(
        state, actions, hits)
    assert before.retrieval_hits == after.retrieval_hits
    assert after.meaning_frame is not None
    assert after.meaning_frame.independent_sources == 1


def test_six_scenario_live_path_updates_meaning_and_gate_without_new_retrieval():
    run = ExperimentRun(RetrievalMode.STRUCTURAL, 7, distractor_count=35, meaning_mode="adaptive")
    summary = run.run()
    assert len(summary.episodes) == 6
    assert sum(ep.actions for ep in summary.episodes) > 0
    assert run.router.top_k == 6
    assert any(step["meaning"] and step["meaning"]["layers"]
               for ep in summary.episodes for step in ep.trace)
    assert run.memory.layer_meanings
    assert run.learner.meaning_updates
    assert run.learner.meaning_snapshot()["layer_associations"]
    assert run.learner.meaning_snapshot()["integration_logits"]
    assert all(
        step["meaning"]["independent_sources"] <= len(step["retrieved_chunks"]) * 5
        for ep in summary.episodes for step in ep.trace if step["meaning"]
    )
