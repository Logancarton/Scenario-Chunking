# Scenario-Chunking

**Research history:** [COGNITIVE_RESEARCH_JOURNAL.md](docs/COGNITIVE_RESEARCH_JOURNAL.md) — three experiments, preserved measurements, failures and hypotheses.


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

To run the independent layer-meaning and contextual integration ablations in addition to the historical retrieval baselines:

```bash
scenario-chunking --output results --meaning
```

The five comparison conditions are existing structural direct scoring; fixed per-layer interpretations; learned per-layer associations with fixed integration; learned meanings with adaptive contextual integration; and fixed meanings with adaptive integration. The last four modes are named `meaning_fixed`, `meaning_learned`, `meaning_adaptive`, and `meaning_gate_only` in output.

Layer meaning traces preserve source-event IDs, confidence and temporal validity. Context gates update after real feedback. Source-deduplicated evidence is integrated only across axes that actually have supporting information. This is a small, interpretable numerical hypothesis, not a claim to replicate biological cognition. The research journal documents the original results and failures without substituting newer runs.

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


## Bounded one-hop causal-neighbor experiment

The 2026-10-10 archived baseline reports **two sparse-retrieval misses**: seed 7 missed
the E5 success chunk `c001020` while selecting its predecessor `c001019`;
seed 149 missed E5 success chunk `c001016` while selecting its predecessor
`c001015`. These are **ranking/top-K displacement**, not candidate-index misses:
all chunks share the task goal and were eligible for candidate routing.
The structural score awards 0.18 to a chunk with a successful successor and a
working-memory activation bonus; consequently a predecessor can outrank the
relevant terminal success chunk. Full retrieval ranked each relevant success
at the top, while structural top-6 omitted it. The new mechanism must be
evaluated against these exact documented failures but contains **no seed,
key, chunk ID, or correct-trajectory special cases**.

### Mechanism

The unchanged sparse top-K runs first. Only the resulting chunk IDs are seeds
for a **one-hop** inspection of directly linked predecessor, successor, or
observed-causal neighbors. An observed cross-chunk causal edge can be recorded
after the environment confirms causal feedback or the learner credits an
informative predecessor. Hypotheses and contradicted edges are retained as
historical claims but do **not** qualify as strong causal evidence.

The independent spreading owner ranks neighbors using goal match, semantic
overlap, positive evidence consistency, provenance, and link type. It does not
apply a universal recency preference. Its default per-retrieval caps are
**two additional chunks, 18 neighbor headers, 36 detailed events in total
(including the original top-K), and one hop**. Neighbor chunks never become
new seeds, preventing chain reaction through long sequences. Historical
timestamps are unchanged by activation.

This experiment compares:
- Original structural top-6 (no spreading).
- Structural top-8 (larger retrieval-window control).
- Structural top-6 plus bounded one-hop neighbor activation.

All use the same six environment episodes, random seeds, feedback learner,
and action-selection rules. The initial top-K recall and recall after spreading
are separate metrics. Every trace records initial activations, additional
activations and evidence justification, budget use, the actual selected action,
and whether that action differs from a same-random-state top-K-only
counterfactual. This is a **selection influence** check, not itself evidence
of behavioral *improvement*. The benchmark also compares completed goals,
action counts, repeated/failed actions, measured latency, and detailed-event
processing proxies.

### Adversarial tests

Focused tests include false causal links with only hypothetical provenance,
cross-goal neighbors, contradicted/outdated successes, long high-degree
distractor chains, a detail-budget exhaustion case, and an action-selection
counterfactual with distinct keys. Test and benchmark failures must remain
visible rather than being hidden or automatically tuned away.

The GitHub research gate runs focused tests, full tests, compilation,
`git diff --check`, and five-seed/1,000-distractor benchmark runs, and
uploads the generated HTML, JSON and trace artifacts. Recorded outcomes,
including negative results, are added below only after actual execution.

### Result interpretation

Memory recovery alone is **not** cognitive improvement. A valid claim
requires improvement in completed goals or decisions/action efficiency on
matched seeds while retaining most of the detailed-memory saving.
An unchanged completion rate with fewer retrieved misses but no better
actions supports a *retrieval improvement only*. The prior baseline
experiment achieved 30/30 successful tasks across the five seeds, making
**action efficiency** a more sensitive behavioral metric than raw completion.


### Measured bounded-spreading findings (2026-10-10)

The five-seed/1,000-distractor research gate completed all six original
scenarios for each retrieval variant. The following are observed
per-seed **means**, not hypothetical outcomes:

| Mode | Initial recall@K | After-spread recall | Goal completion | Actions | Detail-event processing | Retrieval latency* |
|---|---:|---:|---:|---:|---:|---:|
| Full historical retrieval | 1.000 | 1.000 | 100% | 16.8 | 8,294 | — |
| Structural top-6 | 0.933 | 0.933 | 100% | 16.4 | 214.8 | 11.77 ms |
| Structural top-8 | 0.933 | 0.933 | 100% | 15.6 | 241.6 | 10.98 ms |
| Structural top-6 + one-hop spreading | 0.633 | 0.967 | 100% | 15.6 | 260.4 | 16.44 ms |

*Retrieval latency is measured on the GitHub-hosted Python runner and
includes all retrievals per seed-run; it is not an LLM or GPU inference
benchmark. Detailed-event counts are processing proxies, not measured FLOPs.

Spreading recovered an average of **2 relevant chunks per seed-run** beyond
its initial top-6, affected the selected action in **2.6 decisions per run**
and altered the chosen score in **7 decisions per run**. The final recall of
0.967 versus initial 0.633 shows successful neighbor recovery, but the lower
initial recall versus the unmodified structural baseline (0.933) indicates
the altered policy/activation trajectory has changed later retrieval state.
These are not paired retrieval states and should not be mistaken for
recovery of the two exact archived failures under identical memory snapshots.

**Negative behavioral result:** Spreading did NOT outperform a simple
top-8 control on total actions (both 15.6). It had **more failed actions**
(5.8 vs 5.0 per run), **more detailed-event processing** (260.4 vs 241.6),
and **higher retrieval latency** (16.44 vs 10.98 ms). All modes already
achieved 100% goal completion. The evidence supports retrieval recovery,
not demonstrated net cognitive improvement over top-8. Spreading still
uses about **96.9% fewer** detailed-event processing operations than full
history retrieval, preserving most of the overall sparse-memory advantage.

The new trace includes a same-RNG top-K-only counterfactual and, for each
additional activated memory, a leave-one-out counterfactual with action
change, selected-score delta, routing justification, and budget use. The
dashboard exposes these records. It is possible for an activation to change
a score without changing action, and for an action change not to be beneficial.

**Known failure modes and methodological limits:**
- A false causal relationship *asserted* with the trusted `observed`
  provenance passes the routing gate. A dedicated negative control preserves
  this vulnerability. Provenance checks are not independent causal proof.
- Top-6 retrieval after spreading has lower baseline recall because its live
  decisions and activation history can diverge from structural-only mode.
  A separate paired-snapshot replay (below) isolates recovery independently of
  these trajectory changes.
- The current action learner has a simple linear policy and the benchmark
  has ceiling-level completion; outcomes may not generalize to harder tasks.
- Measured detailed-memory operations remain a proxy, not proof of
  reduced compute on transformer or other neural hardware.

**Decision:** Retain the new mode as an experimental option, not Sonara's
default, until it beats a stronger large-top-K control on behavioral quality
at similar or lower memory cost. Do not promote these findings as a net
cognitive improvement.


### Paired-state audit of the original two misses

A subsequent verification replay cloned the *entire pre-retrieval memory
state* just before each baseline structural recall query, so the experimental
one-hop router could run without changing the baseline state or action history.
All other inputs and original top-6 rankings were preserved. The two archived
misses were reproduced and **both recovered**:

| Seed | Baseline miss | Present in unchanged top-6? | Found through one-hop spreading? |
|---|---|---|---|
| 7 | `c001020` | No | **Yes** (directly linked to retrieved `c001019`) |
| 149 | `c001016` | No | **Yes** (directly linked to retrieved `c001015`) |

The per-seed `same_snapshot_recovery_audit` records in `results.json`
contain exact initial chunk IDs and extra activated IDs. Regression tests
replay these documented failures with the historical IDs **only in tests**;
the retrieval code has no special handling for their IDs, seeds or keys.

This resolves the narrow question of whether one-hop spreading can recover
these two missing episodes. It does **not** change the negative behavioral
conclusion above: the spreading mode matched top-8 in actions while processing
more detailed events and incurring greater latency. Its additional memory
activations are not yet justified as the best action-selection policy.


## Experiment 7 — Adaptive Reasoning Under Uncertainty (research branch)

Implementation is limited to the independent Scenario-Chunking research system; Sonara production is unchanged. Run:

```bash
scenario-chunking --uncertainty --output results-uncertainty --seeds 7,23,61,101,149 --rounds 4 --action-budget 9
```

The four-choice `UncertaintyWorld` does not construct a correct proposed plan for the agent. The selector chooses each action from publicly visible options and receives explicit, costly causal feedback. Two mid-series changes invalidate a token relationship and swap prerequisite ordering. Source testimony is fallible; learned person-by-context reliability is calibrated only after subsequent verified outcomes. Blocked dependencies teach ordering without incorrectly punishing the proposed token.

The active pathway reuses `MemoryRouter`, `EpisodicMemory`, `MeaningInterpreter`, `ActionSelector`, `ProgressiveRefiner` and `OutcomeLearner`. The refiner may request a costly instrument check, but does not gain invisible access to ground truth. The benchmark reports all seven main reasoning conditions plus larger retrieval, corrupted-testimony, category ablations and per-seed outcomes. See `results-uncertainty/uncertainty_summary.json` and `uncertainty_trials.json`.

**Important limitations:** the underlying role tokens remain shared between training and transfer; these are not genuinely held-out causal *combinations*. Blocked-predecessor feedback explicitly names a prerequisite. The strong one-pass baseline is measured, but not strictly compute-matched with iterative reflection. Serialization restoration is verified in-process by the runner and independently in a focused subprocess test. A positive cognitive benefit must come from observed benchmark metrics, not from test passage or architecture presence.

## Experiment 8 — corrected contextual scaling research

Research branch: \`experiment/cognitive-scaling-e8-corrected\`. No Sonara production changes.

The earlier Experiment 8 implementation is superseded for causal inference:
it separated five contextual dimensions into independent symbol puzzles, tested
different problems at successive experience stages, and restricted retrieval
with fixed heuristics. Those outcomes must NOT be interpreted as tests of
cognitive scaling.

The corrected experiment tests single decisions conditioned jointly on
self-relevance, consequence/salience, time/protocol epoch, procedural dependencies,
object properties, source identity and fallible testimony. All four held-out
procedural outcomes are represented. Training experiences arise from actions and
observed feedback; learned conditional relations and source reliability remain
revisable. A bounded cue index retrieves evidence through the existing
EpisodicMemory and MemoryRouter; MeaningInterpreter, OutcomeLearner,
ActionSelector, ProgressiveRefiner and the live environment feedback path remain
the cognitive owners. Reflection can make targeted inspection choices or stop
when evidence is sufficient. Task-state never receives the private answer.

Most importantly, a factorial evaluation holds **the exact held-out task bank
fixed** across all experience checkpoints and independently varies added
distractor-memory load and task complexity. Each treatment sees the same training
interventions and observations (except the untrained control). Decision outcomes,
causal recall/precision, missed memories, computation, attention and evidence
provenance are reported separately. The evaluation state is forked and never
teaches the continuing training state.

Verified smoke command:

\`\`\`bash
scenario-chunking --scaling --output results-scaling-corrected \
  --seeds 7,23 --rounds 4 --training-stages 20,100,500 \
  --memory-stages 50,100 --complexities 1,3 \
  --scaling-conditions untrained_structural,strong_one_pass,adaptive_focus,bounded_without_learned_focus \
  --scaling-action-budget 3 --candidate-budget 32
\`\`\`

Outputs: \`scaling_trials.json\`, \`scaling_summary.json\`.

A full 30-seed, 10,000-experience, 20,000-distractor study is available through
the CLI but has **not yet been verified** and may require splitting memory
conditions into separate controlled jobs. The smoke does not establish a
reliable cognitive advantage, strictly equal computational budgets, or
general reasoning beyond the synthetic causal world. See the research journal.
