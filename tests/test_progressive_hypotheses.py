"""Falsifiable Experiment 5 tests: hypothesis revision, direct outcome and credit.

The answers live only in these synthetic environments; the policy receives cues
and causal observations, never the environment's correct_key field.
"""
import random

from scenario_chunking.environment import EpisodeSpec, LockedWorkshop
from scenario_chunking.learning import OutcomeLearner
from scenario_chunking.memory import EpisodicMemory
from scenario_chunking.models import (
    Action, EnvironmentSignal, Event, EventKind, RetrievalHit,
    TruthStatus,
)
from scenario_chunking.policy import ActionSelector
from scenario_chunking.refinement import ProgressiveRefiner
from scenario_chunking.state import CognitiveState


GOAL = "retrieve_object_from_locked_container"


def _causal_chunk(memory, *, target, key, result="success", episode="training"):
    attempted = Action("try_key", target=target, key_name=key)
    attempt = Event(
        memory.next_event_id(), episode, GOAL, 1, 1, 1, 1,
        EventKind.ACTION, TruthStatus.ATTEMPT, "attempt",
        semantic_tags=frozenset({f"key:{key}"}), action=attempted,
    )
    outcome = Event(
        memory.next_event_id(), episode, GOAL, 1, 2, 2, 2,
        EventKind.OUTCOME, TruthStatus.OUTCOME, "outcome",
        semantic_tags=frozenset({f"key:{key}", f"outcome:{result}"}),
        outcome=result, cause_event_id=attempt.event_id,
    )
    chunk = memory.create_chunk(events=[attempt, outcome], transition_reason="observed",
                                self_relevance=1.0, salience=1.0)
    return chunk


def _state(target, keys):
    state = CognitiveState(episode_id="unseen_probe")
    state.container_name = target
    state.visible_keys = list(keys)
    state.inspected.update([state.storage_name, state.work_surface_name, state.container_name])
    state.asked_people.add(state.person_name)
    return state


def test_heldout_low_reliability_clue_is_revised_using_matching_causal_evidence():
    """Matched evidence/RNG; revision changes action AND actual environmental outcome."""
    memory = EpisodicMemory()
    learned = _causal_chunk(memory, target="unfamiliar-lock", key="silver")
    hits = [RetrievalHit(learned.chunk_id, 0.85, ["observed_historical_outcome"])]
    state = _state("unfamiliar-lock", ("copper", "silver"))
    observation = EnvironmentSignal(
        EventKind.OBSERVATION, "A doubtful note points to copper.",
        frozenset({"key:copper", "clue:key_relation"}), TruthStatus.FACT,
        informative=True, confidence=0.22, metadata={"candidate_key": "copper"},
    )
    state.apply_signal(observation, event_id="observed-note")
    learner = OutcomeLearner()
    original = ActionSelector(memory, learner, random.Random(3),
                              meaning_mode="fixed").select(
                                  state, state.candidate_actions(), hits)
    refiner = ProgressiveRefiner(learner)
    selector = ActionSelector(memory, learner, random.Random(3),
                              meaning_mode="fixed", refiner=refiner)
    improved = selector.select(state, state.candidate_actions(), hits, refine=True)
    assert original.action.key_name == "copper"
    assert improved.action.key_name == "silver"
    assert improved.thought_trace["meaningfully_changed_action"]
    assert improved.thought_trace["decision_basis"] == "evidence_supported_prediction"
    assert len(refiner.thread(state).versions) >= 2
    assert improved.thought_trace["hypotheses"]
    silver = next(h for h in improved.thought_trace["hypotheses"] if h["action"] == improved.action.label)
    assert silver["supports"] and silver["confidence"] > 0.78
    assert improved.predicted_success == silver["confidence"]

    task = EpisodeSpec(
        episode_id="heldout-contradiction",
        container_name="unfamiliar-lock", storage_name="drawer",
        work_surface_name="bench", person_name="Alex",
        keys=("copper", "silver"), correct_key="silver",
        clue_source="storage", clue_text="uncertain note",
    )
    old_env = LockedWorkshop(19)
    new_env = LockedWorkshop(19)
    old_env.reset(task)
    new_env.reset(task)
    assert old_env.step(original.action, cause_event_id="baseline")[-1].outcome == "failure"
    assert new_env.step(improved.action, cause_event_id="refined")[-1].outcome == "success"
    # Reusing unchanged evidence does not create more versions.
    prior = len(refiner.thread(state).versions)
    again = selector.select(state, state.candidate_actions(), hits, refine=True)
    assert len(refiner.thread(state).versions) == prior
    assert not again.thought_trace["changed"]


def test_histories_for_different_objects_do_not_prove_current_key():
    memory = EpisodicMemory()
    unrelated = _causal_chunk(memory, target="different-lock", key="silver")
    hits = [RetrievalHit(unrelated.chunk_id, 0.9, ["ambiguous_tag_match"])]
    state = _state("unfamiliar-lock", ("copper", "silver"))
    refiner = ProgressiveRefiner(OutcomeLearner())
    choice = ActionSelector(memory, refiner.learner, random.Random(3),
                            meaning_mode="fixed", refiner=refiner).select(
                                state, state.candidate_actions(), hits, refine=True)
    assert not any(h["sources"] for h in choice.thought_trace["hypotheses"])
    assert choice.thought_trace["decision_basis"] == "one_pass_adequate"
    assert not choice.thought_trace["meaningfully_changed_action"]


def test_failed_attempt_becomes_current_contradiction_with_source_provenance():
    state = _state("lock", ("blue", "green"))
    state.candidate_key = "blue"
    state.track_outcome_contradictions = True
    failure = EnvironmentSignal(
        EventKind.OUTCOME, "Blue failed.", frozenset({"key:blue", "outcome:failure"}),
        TruthStatus.OUTCOME, outcome="failure", cause_event_id="attempt-1",
    )
    state.apply_signal(failure, event_id="failure-1")
    assert state.candidate_key is None
    assert "blue" in state.contradicted_keys
    assert ("failure-1", False, 1.0, "causal_attempt_outcome") in state.key_observations["blue"]
    # Unlinked claims are not allowed to invalidate a candidate.
    state.candidate_key = "green"
    untrusted = EnvironmentSignal(
        EventKind.OBSERVATION, "Green allegedly failed.", frozenset({"key:green"}),
        TruthStatus.FACT, outcome="failure", cause_event_id="alleged",
    )
    state.apply_signal(untrusted, event_id="claim-1")
    assert state.candidate_key == "green"
    assert "green" not in state.contradicted_keys


def test_strict_credit_excludes_irrelevant_informative_predecessors():
    learner = OutcomeLearner(strict_credit=True)
    learner.begin_episode()
    learner.register_action("inspection-used", frozenset({"action:inspect_storage"}))
    learner.mark_informative("inspection-used", produced_evidence_ids=["evidence-A"])
    learner.register_action("inspection-noise", frozenset({"action:ask_informant"}))
    learner.mark_informative("inspection-noise", produced_evidence_ids=["unrelated"])
    learner.register_action("successful-action", frozenset({"action:try_key"}),
                            used_evidence_ids=["evidence-A"])
    learner.apply_success("successful-action")
    credited = [x["credited_event_id"] for x in learner.credit_log]
    assert "inspection-used" in credited
    assert "inspection-noise" not in credited
    supported = [x for x in learner.credit_log
                 if x["kind"] == "evidence_supported_predecessor"]
    assert len(supported) == 1
    assert supported[0]["source_evidence_ids"] == ["evidence-A"]

    learner.begin_episode()
    learner.credit_log.clear()
    learner.register_action("unrelated-inspect", frozenset({"action:inspect_storage"}))
    learner.mark_informative("unrelated-inspect", produced_evidence_ids=["clue"])
    learner.register_action("lucky-action", frozenset({"action:try_key"}))
    learner.apply_success("lucky-action")
    assert not any(x["kind"] == "evidence_supported_predecessor" for x in learner.credit_log)


def test_live_experiment_path_uses_reported_evidence_not_hidden_correct_key():
    """Unfamiliar clue reliability reaches the real runtime and hypothesis owner."""
    from scenario_chunking.experiment import ExperimentRun
    from scenario_chunking.models import RetrievalMode

    spec = EpisodeSpec(
        episode_id="heldout_live",
        container_name="unfamiliar-lock", storage_name="parts drawer",
        work_surface_name="shelf", person_name="Dana",
        keys=("copper", "silver"), correct_key="silver",
        clue_source="storage", clue_text="Faint pencil mark suggests copper.",
        reported_key="copper", report_confidence=0.22,
    )
    outcomes = {}
    for mode in (None, "associate"):
        run = ExperimentRun(RetrievalMode.STRUCTURAL, 7, distractor_count=0,
                            refinement_mode=mode)
        _causal_chunk(run.memory, target="unfamiliar-lock", key="silver")
        episode = run._run_episode(spec)
        outcomes[mode or "structural"] = episode
        assert episode.trace
        assert all(step["interpretation_event_evaluations"]
                   <= step["total_detailed_events_this_retrieval"]
                   for step in episode.trace)
        if mode:
            assert any(step["thought"] and step["thought"]["hypotheses"]
                       for step in episode.trace)
            assert all(step["total_detailed_events_this_retrieval"] <= 36
                       for step in episode.trace)
    assert outcomes["associate"].success
    assert outcomes["structural"].success
    # Measure actual action efficiency rather than assuming that a changed
    # thought improved it: no hard assertion that refinement must win.
    assert outcomes["associate"].actions <= 14


def test_irrelevant_historical_success_triggers_information_seeking():
    """Do not reuse a key outcome from a different object as current proof."""
    memory = EpisodicMemory()
    unrelated = _causal_chunk(memory, target="different-lock", key="silver")
    hits = [RetrievalHit(unrelated.chunk_id, 0.9, ["same_key_different_target"])]
    state = CognitiveState(episode_id="unfamiliar")
    state.container_name = "unfamiliar-lock"
    state.visible_keys = ["silver", "copper"]
    learner = OutcomeLearner()
    baseline = ActionSelector(memory, learner, random.Random(3),
                              meaning_mode="fixed").select(
                                  state, state.candidate_actions(), hits)
    refiner = ProgressiveRefiner(learner)
    new = ActionSelector(memory, learner, random.Random(3),
                         meaning_mode="fixed", refiner=refiner).select(
                             state, state.candidate_actions(), hits, refine=True)
    assert baseline.action.kind == "try_key"
    assert new.action.kind != "try_key"
    assert new.thought_trace["decision_basis"] == "reject_unverified_transfer"
    assert new.thought_trace["meaningfully_changed_action"]


def test_unrelated_success_cannot_reward_unsupported_refinement():
    memory = EpisodicMemory()
    history = _causal_chunk(memory, target="different-lock", key="silver")
    state = _state("unfamiliar-lock", ("silver",))
    learner = OutcomeLearner()
    refiner = ProgressiveRefiner(learner, learning=True)
    decision = ActionSelector(
        memory, learner, random.Random(3), meaning_mode="fixed", refiner=refiner,
    ).select(state, state.candidate_actions(),
             [RetrievalHit(history.chunk_id, 0.9, ["tag-only"])], refine=True)
    assert decision.meaning_frame.independent_sources
    assert not decision.thought_trace["contributing_evidence_ids"]
    refiner.feedback(state, decision, outcome="success", outcome_event_ids=("outcome-1",))
    assert not learner.refinement_updates
    assert refiner.thread(state).outcomes[-1]["outcome"] == "success"
