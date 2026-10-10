# Scenario-Chunking

A standalone, executable cognitive-architecture experiment testing whether structurally organized episodic memory can improve decisions through sequential experience while avoiding full historical retrieval.

This is **not Sonara production code**, does not use an LLM, and does not claim consciousness or human-level cognition. It is deliberately transparent Python so the mechanisms and failures can be inspected.

## What it tests

The agent works through six versions of a locked-workshop problem. It must discover useful information, select actions, receive real environment feedback, preserve historical facts separately from current activation, assign delayed credit to the causal attempt, and reuse experience on later tasks. The environment knows the correct key; the policy does not.

Experiments compare four retrieval systems under the same task rules and deterministic seeds:

- `full`: scan all historical chunks and process all detailed events.
- `chunked`: use sparse structural indexes without a recency assumption.
- `recency`: sparse chunking plus a fixed exponential recency weight.
- `structural`: sparse chunking plus factual time, sequence links, contradiction handling, and activation that decays without rewriting history.

All systems use the same outcome learner. This avoids making the baselines artificially incapable of learning; the comparison focuses on memory routing and temporal structure.

## Cognitive signal path

```text
Environment observation
        ↓
CognitiveState (current belief / goal)
        ↓
MemoryRouter (five-axis sparse routing)
        ↓
EpisodicMemory chunks + selected detailed events
        ↓
ActionSelector (prediction + action choice)
        ↓
LockedWorkshop environment action
        ↓
Observed outcome
        ↓
OutcomeLearner causal / eligibility credit
        ↓
Memory + policy update
        ↓
Next cognitive state
```

## Five-axis organization

Each episodic chunk has a compact routing header. The axes are intentionally not forced into one identical equation:

- **Self:** `self_relevance` in the chunk header.
- **Salience:** consequence/information-sensitive salience.
- **Time:** occurrence, acquisition, storage time, uncertainty, and lazy working-memory activation.
- **Sequence:** episode sequence index plus previous/next chunk links.
- **Person:** indexed person/entity relationship IDs.

Semantic content is stored separately on events/chunks. Historical timestamps do not decay. Only activation decays using a closed-form function when queried.

## Learning mechanism

`OutcomeLearner` is a small linear feature learner. Failed actions receive negative causal credit. On success, the actual causal action receives full credit using its event ID, while earlier actions are eligible only when they changed the belief state (for example, revealing a clue). This prevents unrelated events immediately before success from receiving reward merely because they were recent.

Observed facts, predictions, attempts, outcomes, and learned feature weights are stored as separate record types. A prediction never becomes a historical fact.

## Scenarios

1. **Initial attempt:** discover a workable strategy from feedback.
2. **Similar problem:** different surface names, same important causal structure; test whether learned behavior reduces wasted actions.
3. **Temporal interference:** an old successful key becomes obsolete after a lock replacement.
4. **Distractor load:** 1,000 unrelated chunks are inserted before solving the next task.
5. **Delayed feedback:** unrelated observations are inserted between a key attempt and its outcome; causal credit must still reach the initiating action.
6. **Relational transfer:** a different container and person test transfer of role-level knowledge rather than exact names.

A separate negative control creates a case where fixed recency prefers a newer noncausal memory over an older causally confirmed one.

## Run

Python 3.11+ is sufficient; the experiment has no runtime dependencies.

```bash
python -m pip install -e .
scenario-chunking --output results
```

Or without installing the console script:

```bash
PYTHONPATH=src python -m scenario_chunking.cli --output results
```

For a faster development run:

```bash
PYTHONPATH=src python -m scenario_chunking.cli --output results --seeds 7 --distractors 250
```

Run tests:

```bash
pytest -q
python -m compileall -q src tests
```

## Outputs

`results/results.json` contains per-seed/per-mode quantitative metrics and aggregate comparisons. `results/event_traces.jsonl` records each decision, retrieved chunks, action scores, feedback, and learner state. `results/dashboard.html` is a dependency-free local dashboard showing trajectories, chunk activation, preference changes, historical-vs-current temporal behavior, learning across episodes, and retrieval/computation costs.

The experiment reports goal completion, actions, failed/repeated actions, E1→E2 improvement, recall@K, outdated evidence retrieval, temporal ordering, delayed credit accuracy, unnecessary memory activations, header evaluations, detailed-memory loads, retrieval latency, peak Python memory, and active detailed-event volume. Means and deterministic-seed variability are reported. A structural no-policy-learning ablation isolates the contribution of explicit feedback-driven weight updates while leaving episodic recall intact.

## Interpreting cost

Header evaluations, detail loads, and active detailed-event volume are **architectural processing-count proxies**. Retrieval latency and peak Python memory are measured runtime values. The experiment does **not** implement transformer attention and therefore does not claim transformer FLOP savings.

## What a positive result means

A useful result would show that feedback changes future action preferences, later episodes require fewer or better-targeted actions, structural time avoids blindly treating historical success as current truth, delayed outcomes credit the causal attempt, and sparse routing keeps detailed-memory activation far below full retrieval under distractor load.

It would **not** establish consciousness, general intelligence, or biological equivalence. It would only support or falsify specific architectural hypotheses worth carrying into Sonara.
