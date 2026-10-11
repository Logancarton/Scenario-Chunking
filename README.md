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

Research branch: `experiment/cognitive-scaling-e8-corrected`. No Sonara production changes.

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

```bash
scenario-chunking --scaling --output results-scaling-corrected \
  --seeds 7,23 --rounds 4 --training-stages 20,100,500 \
  --memory-stages 50,100 --complexities 1,3 \
  --scaling-conditions untrained_structural,strong_one_pass,adaptive_focus,bounded_without_learned_focus \
  --scaling-action-budget 3 --candidate-budget 32
```

Outputs: `scaling_trials.json`, `scaling_summary.json`.

A full 30-seed, 10,000-experience, 20,000-distractor study is available through
the CLI but has **not yet been verified** and may require splitting memory
conditions into separate controlled jobs. The smoke does not establish a
reliable cognitive advantage, strictly equal computational budgets, or
general reasoning beyond the synthetic causal world. See the research journal.


## Experiment 9 — time-triggered temporal prediction (isolated research)

Research branch: `experiment/temporal-prediction-e9`. This experiment tests a
**functional** sense of elapsed time, not subjective time or consciousness. It
builds directly on the corrected Experiment 8 branch; Sonara production is
unchanged.

`CognitiveState` now separates externally advanced, monotonic `world_time`
from processing/episode ticks. A heap of sourced expectations wakes only those
whose deadlines are due. "Due but unverified" is a provisional belief, **not**
an observed failure. Direct observations may establish witnessed absence or
return; a sensor may be unavailable or know presence without knowing the
precise arrival time. Only authenticated-by-the-test-environment, timestamped
feedback changes the existing `OutcomeLearner`'s estimated durations.
Historical source/prediction/observation events remain in `EpisodicMemory`.

The existing `ProgressiveRefiner` updates thought versions only when the
evidence/status changes and considers inspection cost. The single
`ActionSelector` still selects `wait`, `check_return`, `follow_up` or
`acknowledge_return`. `temporal_experiment.py` is orchestration and
measurement, not a parallel cognitive owner or runtime.

Run focused tests and the small paired benchmark:

```bash
python -m pytest -q tests/test_temporal.py
python - <<'PY'
from scenario_chunking.temporal_experiment import write_temporal_benchmark
print(write_temporal_benchmark()["summary"])
PY
```

Controls include memory-only, clock-only, learned-information one-pass and
temporal refinement. All share available actions and the synthetic world's
private outcomes. Positive results require improved **observed decisions**,
not merely a due-event log or more thought versions. This finite set of timed
return cases does not establish cross-domain time understanding or superior
refinement over the matched one-pass control.


## Experiment 10 — predictive temporal decision value (research-only)

Research branch: `experiment/predictive-decision-value-e10`. Draft PR:
[Experiment 10](https://github.com/Logancarton/Scenario-Chunking/pull/4).
This continues E9 **without changing Sonara production**. The private-truth
multi-person world tests concurrent deliveries, misleading testimony,
deadline expiration, temporary observation outages and a changing dependency.
Public promises, observed signals, execution feedback and clock passage are
the only inputs to the cognition pathway.

The existing `OutcomeLearner` maintains context-conditioned temporal
survival estimates using direct arrivals and right-censored observations.
The `MeaningInterpreter` estimates sourced probability and expected option
values. The `ProgressiveRefiner` compares expected decisions with thinking
cost, preserves version provenance and prevents repeated unchanged thoughts.
The original `ActionSelector` retains final selection authority. Execution
feedback enters `EpisodicMemory` and the existing `OutcomeLearner`.

The corrected 30-independent-seed replication uses **identical held-out worlds
and training experiences** across eight treatments, including a strong
compute-matched one-pass control allowed the *same final value-based action
ranking*. Each seed evaluates six held-out environments at 0, 20 and 100
training experiences. The reported normal-approximation intervals cluster
paired reward differences by seed, not by individual scenario.

| Experiences | Full reward | Compute-matched one-pass reward | Paired full-minus-one-pass reward (approx. 95% interval) | Full / one-pass completion |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.3547 | 0.3593 | −0.0046 [−0.0293, +0.0202] | 17.2% / 17.8% |
| 20 | 0.4712 | 0.4667 | +0.0045 [−0.0039, +0.0129] | 57.2% / 57.8% |
| 100 | 0.5339 | 0.5239 | +0.0099 [+0.0044, +0.0155] | 62.2% / 62.8% |

After adding 500 irrelevant chunks, the small **five-seed** replications had
paired reward intervals including zero: at 20 experiences +0.0015
[−0.0067, +0.0097] and at 100 experiences +0.0177
[−0.0041, +0.0394]. All main treatments recorded zero fabricated
`verified_absent` failures. Forecast mean absolute error for the shared
learned model declined from 2.375 time units at zero experience to 2.002
at 100. More refinement versions by themselves are **not** a benefit.

**Verdict:** The 100-experience synthetic reward comparison is weak positive
evidence of decision-value gains, **not** proof that thinking longer improves
goal completion or that gains generalize. Completion slightly trails the
compute-matched one-pass control; the distractor replication is inconclusive.
The one-pass computation control matches available decision values and
per-decision repeated interpretation, but the measured operation totals are
not exactly equal because trajectories differ. The earlier exploratory run
that deprived one-pass reasoning of equivalent final scoring authority is
invalid for cognitive-superiority claims and is not used here.

To independently reproduce the results:

```bash
python -m pytest -q tests/test_decision_value.py tests/test_temporal.py
python - <<'PY'
from scenario_chunking.decision_value_experiment import write_decision_value_benchmark
report = write_decision_value_benchmark(
    output="results-e10", seeds=tuple(range(30)),
    training_stages=(0, 20, 100), cases=tuple(range(6)))
print(report["paired_comparisons"])
PY
```

An independent GitHub Actions job additionally runs the 500-distractor
challenge and uploads detailed per-trial records. See the append-only
[cognitive research journal](docs/COGNITIVE_RESEARCH_JOURNAL.md) for
limitations and the next falsifiable research question. The mechanism is
**Built, Integrated and Enabled within the research runner**; cross-process
restart of learned state, expectation identities and thought versions is
tested. This does not imply biological temporal awareness, consciousness, or
general-purpose predictive intelligence.


## Experiment 11 — unique-experience incremental conditional refinement (isolated research)

Branch: `experiment/incremental-refinement-e11`, [draft PR #5](https://github.com/Logancarton/Scenario-Chunking/pull/5), based on E10. Production Sonara remains untouched.

This is a **prequential** test: each distinct public scenario is predicted and acted upon **before** the feedback from that scenario enters the learner. A Gray-code walk varies one substantive binary factor at a time over 32 unique combinations per domain (delivery, repair, coordination), with a deliberately irrelevant marker, nine-percent potential-outcome noise, and an observed change of protocol in the final domain. Each individual policy/seed run sees each unique experience once. Five independently seeded replicas are for measurement, not repeated training passes.

Existing `OutcomeLearner` contextual associations now support bounded, shrinkage-based conditional predictions and per-outcome incremental belief updates; `MeaningInterpreter` translates them into action values, `MemoryRouter` uses bounded causal-neighbor expansion, `ProgressiveRefiner` can reconsider risk, and the **single** `ActionSelector` makes the decision. Outcome events and provenance remain in `EpisodicMemory`. The experiment controller only sequences inputs and measures private, post-action counterfactual regret.

The five controls are frozen learning, episodic-only (no generalized belief update), strong one-pass (same learned conditionals and expected values), full incremental risk-sensitive refinement, and full refinement with two-variable interactions ablated. Each has the same action set, matched scenario order, environment randomness and bounded retrieval budgets.

**Initial corrected five-seed measured result (120 held-out, pre-outcome-scored choices per treatment):**

| Treatment | Held-out mean reward | Success | Chosen-action Brier |
| --- | ---: | ---: | ---: |
| Frozen | −0.1367 | 44.17% | 0.2500 |
| Episodic only | −0.1220 | 45.83% | 0.2419 |
| Strong one-pass | −0.1653 | 43.33% | 0.2500 |
| Full refinement | −0.1727 | 42.50% | 0.2362 |
| No two-variable interactions | −0.1367 | 44.17% | 0.2503 |

The full system's Brier score improved relative to one-pass in this particular replication, but its actual **reward and completion were worse**, including versus episodic memory alone. Relation-interaction ablation **did not establish causal benefit**; the ablation has higher reward. Full refinement accounts two nominal cognition operations per decision versus one for one-pass, with similar memory work (held-out mean 60 inspected headers and 16 detailed events). Those operation counts are harness accounting rather than precisely matched CPU FLOPs. Early-to-late changes are confounded by case ordering and changed difficulty, so they do **not** show monotonic improvement.

**Scientific conclusion:** E11 demonstrates integrated single-experience updates, sourced episodic persistence, contradiction-tolerant conditional revisions, minimally varying scenarios, and subprocess restart. It **does not demonstrate improved cognitive decisions**. The key unresolved gap is reliably distinguishing causal differences from many correlated cue combinations under bandit feedback (only the selected action's outcome is observed). Minimal scenario variation alone cannot resolve that attribution problem; future work must use observed counterfactual contrasts, value-of-information, and independently held-out transfer to test whether conditional refinement becomes useful. Do not transplant this experimental mechanism into Sonara on these data.

Reproduce with:

```bash
python -m pytest -q tests/test_incremental.py
python - <<'PY'
from scenario_chunking.incremental_experiment import write_incremental_benchmark
p = write_incremental_benchmark(seeds=(7,23,61,101,149))
print(p["summary"])
print(p["paired_seed_deltas"])
PY
```

GitHub Actions runs this experiment with the full E6–E10 historical research gate. Review the [research journal](docs/COGNITIVE_RESEARCH_JOURNAL.md) and [CI runs](https://github.com/Logancarton/Scenario-Chunking/actions) for current verification; code tests and passing CI do not establish a cognitive advantage.
