from scenario_chunking.memory import EpisodicMemory
from scenario_chunking.models import Event, EventKind, TruthStatus


def test_activation_decay_does_not_rewrite_historical_time():
    memory = EpisodicMemory()
    event = Event(
        event_id=memory.next_event_id(), episode_id="e", goal_id="g", step=0,
        occurrence_time=10.0, acquired_time=11.0, storage_time=12.0,
        kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT, content="fact",
        semantic_tags=frozenset({"topic:test"}),
    )
    chunk = memory.create_chunk(events=[event], transition_reason="test", self_relevance=1.0, salience=1.0)
    before = (chunk.header.occurrence_time, chunk.header.acquired_time, chunk.header.storage_time)
    a1 = chunk.header.activation_at(20.0)
    a2 = chunk.header.activation_at(40.0)
    assert a2 < a1
    assert before == (chunk.header.occurrence_time, chunk.header.acquired_time, chunk.header.storage_time)


def test_chunk_sequence_links_are_bidirectional():
    memory = EpisodicMemory()
    chunks = []
    for i in range(3):
        event = Event(
            event_id=memory.next_event_id(), episode_id="e", goal_id="g", step=i,
            occurrence_time=float(i), acquired_time=float(i), storage_time=float(i),
            kind=EventKind.OBSERVATION, truth_status=TruthStatus.FACT, content=str(i),
            semantic_tags=frozenset({"topic:test"}),
        )
        chunks.append(memory.create_chunk(events=[event], transition_reason="test", self_relevance=1.0, salience=1.0))
    assert chunks[0].header.next_chunk_id == chunks[1].chunk_id
    assert chunks[1].header.previous_chunk_id == chunks[0].chunk_id
    assert chunks[1].header.next_chunk_id == chunks[2].chunk_id
