import random

from scenario_chunking.learning import OutcomeLearner
from scenario_chunking.memory import EpisodicMemory
from scenario_chunking.models import (
    Action, Event, EventKind, RetrievalHit, RetrievalMode, RetrievalQuery, RetrievalStats, TruthStatus,
)
from scenario_chunking.policy import ActionSelector
from scenario_chunking.retrieval import MemoryRouter
from scenario_chunking.spreading import BoundedNeighborActivator, SpreadBudget
from scenario_chunking.state import CognitiveState


GOAL = "retrieve_object_from_locked_container"


def chunk(mem, idx, tags, *, goal=GOAL, episode="fixture", salience=0.4, outcome=None):
    evt = Event(
        event_id=mem.next_event_id(), episode_id=episode, goal_id=goal, step=idx,
        occurrence_time=float(idx), acquired_time=float(idx), storage_time=float(idx),
        kind=EventKind.OUTCOME if outcome else EventKind.OBSERVATION,
        truth_status=TruthStatus.OUTCOME if outcome else TruthStatus.FACT,
        content=f"fixture {idx}", semantic_tags=frozenset({f"goal:{goal}", *tags}),
        outcome=outcome, reward=1.0 if outcome == "success" else 0.0,
        action=Action("try_key", key_name=next((tag[4:] for tag in tags if tag.startswith("key:")), None)) if outcome else None,
    )
    return mem.create_chunk(events=[evt], transition_reason="fixture", self_relevance=1.0, salience=salience)


def query(*tags, contradicted=frozenset()):
    return RetrievalQuery(goal_id=GOAL, semantic_tags=frozenset({f"goal:{GOAL}", *tags}),
                          current_time=50.0, contradicted_keys=contradicted)


def direct_expand(mem, seed, q, budget=None):
    stats = RetrievalStats(active_detail_events=len(seed.event_ids), detail_loads=1)
    base = MemoryRouter(mem, RetrievalMode.STRUCTURAL)
    spread = BoundedNeighborActivator(mem, budget)
    hits = spread.expand(q, [RetrievalHit(seed.chunk_id, 0.9, ["selected"])], stats,
                         rank_header=lambda cid, qry: base._score_chunk(chunk_id=cid, query=qry))
    return hits, stats


def test_direct_successor_recovered_without_widening_initial_k():
    m = EpisodicMemory()
    selected = chunk(m, 1, {"role:key", "clue:key_relation"})
    missing = chunk(m, 2, {"role:key", "key:silver", "outcome:success"}, outcome="success")
    hits, stats = direct_expand(m, selected, query("role:key", "key:silver", "outcome:success"))
    assert [h.chunk_id for h in hits] == [missing.chunk_id]
    assert stats.initial_hit_ids == [selected.chunk_id]
    assert stats.spread_hit_ids == [missing.chunk_id]
    assert stats.spread_detailed_events == 1
    assert stats.neighbor_headers_examined <= 18


def test_misleading_causal_link_does_not_override_provenance():
    m = EpisodicMemory()
    seed = chunk(m, 1, {"role:key"}, episode="one")
    misleading = chunk(m, 2, {"role:key", "key:silver", "outcome:success"},
                        episode="two", outcome="success")
    m.link_causal(seed.chunk_id, misleading.chunk_id, evidence="hypothesis", confidence=1.0)
    hits, _ = direct_expand(m, seed, query("role:key", "key:silver", "outcome:success"))
    assert misleading.chunk_id not in [h.chunk_id for h in hits]


def test_irrelevant_goal_neighbors_are_blocked():
    m = EpisodicMemory()
    seed = chunk(m, 1, {"role:key"}, goal=GOAL)
    unrelated = chunk(m, 2, {"role:key", "key:silver", "outcome:success"},
                      goal="unrelated_goal", outcome="success")
    hits, _ = direct_expand(m, seed, query("role:key", "key:silver", "outcome:success"))
    assert unrelated.chunk_id not in [h.chunk_id for h in hits]


def test_outdated_success_neighbor_is_blocked_by_current_contradiction():
    m = EpisodicMemory()
    seed = chunk(m, 1, {"role:key"})
    old = chunk(m, 2, {"role:key", "key:brass", "outcome:success"}, outcome="success")
    hits, _ = direct_expand(m, seed, query("role:key", "outcome:success",
                                          contradicted=frozenset({"brass"})))
    assert old.chunk_id not in [h.chunk_id for h in hits]


def test_long_distracting_chain_and_high_degree_node_cannot_cascade():
    m = EpisodicMemory()
    seed = chunk(m, 1, {"role:key"}, episode="seed")
    for i in range(2, 200):
        c = chunk(m, i, {"role:key", "key:silver"}, episode=f"other{i}")
        m.link_causal(seed.chunk_id, c.chunk_id, evidence="observed", confidence=0.9)
    hits, stats = direct_expand(
        m, seed, query("role:key", "key:silver"),
        budget=SpreadBudget(max_additional_chunks=2, max_neighbor_headers=18,
                            max_total_detail_events=3, max_hops=1))
    assert len(hits) <= 2
    assert stats.active_detail_events <= 3
    assert stats.neighbor_headers_examined <= 18
    assert stats.budget_exhausted
    assert all(h.chunk_id != seed.chunk_id for h in hits)


def test_tight_detail_budget_blocks_activation():
    m = EpisodicMemory()
    seed = chunk(m, 1, {"role:key"})
    missing = chunk(m, 2, {"role:key", "key:silver"})
    hits, stats = direct_expand(
        m, seed, query("role:key", "key:silver"),
        budget=SpreadBudget(max_additional_chunks=2, max_neighbor_headers=18,
                            max_total_detail_events=1, max_hops=1))
    assert hits == []
    assert stats.spread_detailed_events == 0
    assert stats.budget_exhausted


def test_recovered_causal_evidence_can_change_action_selection():
    m = EpisodicMemory()
    seed = chunk(m, 1, {"role:key"})
    outcome = chunk(m, 2, {"role:key", "key:steel", "outcome:success"},
                    episode="previous_success", outcome="success")
    m.link_causal(seed.chunk_id, outcome.chunk_id, evidence="observed", confidence=0.95)
    q = query("role:key", "key:steel", "outcome:success")
    extra, _ = direct_expand(m, seed, q)
    assert [hit.chunk_id for hit in extra] == [outcome.chunk_id]

    state = CognitiveState(episode_id="new")
    state.visible_keys = ["brass", "steel"]
    state.inspected = {state.storage_name, state.work_surface_name, state.container_name}
    state.asked_people = {state.person_name}
    actions = state.candidate_actions()
    chooser_without = ActionSelector(m, OutcomeLearner(), random.Random(0))
    chooser_with = ActionSelector(m, OutcomeLearner(), random.Random(0))
    initial = [RetrievalHit(seed.chunk_id, 0.5, ["initial"])]
    without = chooser_without.select(state, actions, initial)
    with_spread = chooser_with.select(state, actions, initial + extra)
    assert without.action.key_name == "brass"
    assert with_spread.action.key_name == "steel"
