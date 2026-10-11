from __future__ import annotations

from scenario_chunking.working_memory import WorkingMemoryActivation, ThoughtState


def test_thought_persists_during_reasoning_then_decays_without_use():
    wm = WorkingMemoryActivation(capacity=4, half_life=4)
    tags = {"a": frozenset({"goal:g", "clue:key_relation", "key:a"})}
    wm.use(["a"], "g", 10, known_headers=tags)
    assert wm.state("a", "g", 11) == ThoughtState.FOCUSED
    wm.use(["a"], "g", 13, known_headers=tags)
    assert wm.state("a", "g", 16) == ThoughtState.FOCUSED
    assert wm.state("a", "g", 80) == ThoughtState.DORMANT
    assert "a" in wm.bundles  # no historical deletion


def test_retrieval_without_use_does_not_maintain_activation():
    wm = WorkingMemoryActivation(half_life=2)
    tags = frozenset({"goal:g", "clue:key_relation", "key:a"})
    wm.use(["a"], "g", 1, known_headers={"a": tags})
    previous = wm.bundles["a"].last_time
    for t in (2, 3, 4, 5):
        wm.bonus("a", "g", t)
        wm.state("a", "g", t)
    assert wm.bundles["a"].last_time == previous
    assert wm.state("a", "g", 25) == ThoughtState.DORMANT


def test_specific_cue_reinstates_dormant_but_generic_goal_does_not():
    wm = WorkingMemoryActivation(half_life=2)
    tags = frozenset({"goal:g", "role:key", "key:bronze"})
    wm.use(["a"], "g", 1, known_headers={"a": tags})
    assert not wm.cue("a", "g", tags, frozenset({"goal:g", "role:key"}), 40)
    assert wm.cue("a", "g", tags, frozenset({"goal:g", "key:bronze"}), 41)
    assert wm.state("a", "g", 41) == ThoughtState.FOCUSED
    assert not wm.cue("a", "g", tags, frozenset({"goal:g", "key:bronze"}), 42)


def test_competition_and_goal_relevance_without_permanent_pinning():
    wm = WorkingMemoryActivation(capacity=3, focused_capacity=1, half_life=4)
    headers = {str(i): frozenset({"goal:g", f"key:{i}"}) for i in range(9)}
    for i in range(9):
        wm.use([str(i)], "g" if i < 6 else "other", 10 + i, known_headers={str(i): headers[str(i)]})
    s = wm.snapshot("g", 19)
    assert s["active_count"] <= 3
    assert len(s["focused"]) <= 1
    assert wm.snapshot("g", 300)["active_count"] == 0


def test_invalid_time_cannot_reanimate_older_episode():
    wm = WorkingMemoryActivation()
    wm.use(["a"], "g", 900, known_headers={"a": frozenset({"goal:g"})})
    try:
        wm.snapshot("g", 1)
    except ValueError as e:
        assert "global" in str(e) or "regressed" in str(e)
    else:
        raise AssertionError("episode-local time must be rejected")


def test_activation_is_not_confidence_or_truth():
    wm = WorkingMemoryActivation()
    tags = frozenset({"goal:g", "outcome:success", "key:old"})
    wm.use(["old"], "g", 1, known_headers={"old": tags})
    assert not wm.cue("old", "g", tags, frozenset({"key:old"}), 100,
                      contradicted_keys=frozenset({"old"}))
    assert "old" in wm.bundles
    assert wm.state("old", "g", 100) == ThoughtState.DORMANT


def test_adapted_lifetime_bounded_and_not_permanent():
    wm = WorkingMemoryActivation(half_life=4)
    tags = frozenset({"goal:g", "clue:c"})
    wm.use(["a"], "g", 1, known_headers={"a": tags})
    assert wm.activation("a", 8, persistence_gain=1.0) > wm.activation("a", 8, persistence_gain=-1.0)
    assert wm.state("a", "g", 1000, persistence={"a": 100}) == ThoughtState.DORMANT


def test_peripheral_state_is_distinct_from_focus():
    wm = WorkingMemoryActivation(capacity=4, focused_capacity=1, half_life=30)
    headers = {"a": frozenset({"goal:g", "key:a"}),
               "b": frozenset({"goal:g", "key:b"})}
    wm.use(["a"], "g", 1, known_headers=headers)
    wm.use(["b"], "g", 2, known_headers=headers)
    states = {wm.state(cid, "g", 2) for cid in headers}
    assert ThoughtState.FOCUSED in states
    assert ThoughtState.PERIPHERAL in states


def test_irrelevant_and_outdated_cues_do_not_resurrect_thoughts():
    wm = WorkingMemoryActivation(half_life=1)
    tags = frozenset({"goal:g", "key:old", "outcome:success"})
    wm.use(["a"], "g", 1, known_headers={"a": tags})
    assert not wm.cue("a", "g", tags, frozenset({"goal:other", "key:irrelevant"}), 100)
    assert not wm.cue("a", "g", tags, frozenset({"goal:g", "key:old"}), 101,
                      contradicted_keys=frozenset({"old"}))
    assert wm.state("a", "g", 101) == ThoughtState.DORMANT
