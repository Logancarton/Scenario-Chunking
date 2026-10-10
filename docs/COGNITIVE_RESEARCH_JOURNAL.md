# Sonara cognitive research journal

> Append-only evidence record. This repository is an isolated research sandbox, not Sonara production. Dates, observations, hypotheses, and implementation stages are kept separate. Do not interpret these experiments as biological equivalence, general intelligence, or proof of lower transformer FLOPs.

## Original hypothesis

A memory system might reduce computation by routing to a few relevant episodic chunks instead of rescanning every prior experience. We began with five complementary, **not** independent agents: **Self** (relevance to goals), **Salience** (consequence and informativeness), **Time** (historical validity), **Sequence** (event ordering and possible causal links), and **Person** (identity and relationships). These routing axes are distinct from semantic tags, and historical event records are distinct from temporary activation.

The system should use environmental feedback to revise what it expects, not merely replay a fixed rule for a locked cabinet. The initial six workshops were controlled learning environments, not real-world reasoning benchmarks.

## Experiment 1 — Sparse structural episodic chunking

**Built and tested:** Immutable event records with explicit observation/prediction/action/outcome types; compact chunk headers and indexes; top-K retrieval; historical timestamps and lazy activation decay; sequential links; current-state contradictions; an action selector; and a separate outcome-driven linear learner.

**Historical five-seed, six-scenario findings (locked original measurement):**

| Measure | Recorded finding |
|---|---:|
| Structural goal completion | 30/30 |
| Structural mean actions per seed | 16.4 |
| No explicit policy learning, mean actions | approximately 21.2 |
| First vs similar second episode, mean actions | 4.0 vs 2.4 |
| Sparse detailed-event processing | approximately 215 |
| Full-history detailed-event processing | approximately 8,294 |
| Structural retrieval recall | approximately 93.3% |
| Relevant memories displaced from top-6 | 2 |

**Supported:** A small action-feature learner altered preferences following actual feedback. Sparse routing greatly reduced detailed event processing while finishing these six deterministic tasks. Fixed recency could prefer newer irrelevant material over an older relevant result.

**Not supported:** This benchmark does not establish general-purpose reasoning, reduced neural inference FLOPs, or that sparse retrieval is sufficient for robust decisions.

## Experiment 2 — Bounded one-hop spreading

**Built, integrated and tested:** A selected chunk can propose immediately linked predecessor, successor, or observed-causal neighbors. Hard caps are two added chunks, 18 neighbor headers, 36 detailed events total, and one hop. Newly activated chunks never become seeds. The original sparse top-6 remains unchanged before spreading.

**Why:** Seed 7's E5 success chunk `c001020` and seed 149's E5 success chunk `c001016` were eligible for sparse retrieval but ranked behind other evidence. Both were recoverable from the preceding chunk using identical pre-retrieval memory snapshots. The failure was top-K displacement, not missing indexes.

**Historical five-seed/1,000-distractor measurements (preserved, not rerun-and-substituted):**

| Mode | Recall after retrieval | Mean actions | Detailed events |
|---|---:|---:|---:|
| Structural top-6 | 93.3% | 16.4 | 214.8 |
| Structural top-8 | 93.3% | 15.6 | 241.6 |
| Bounded spreading | 96.7% | 15.6 | 260.4 |
| Full retrieval | 100% | 16.8 | 8,294 |

Spreading recovered both archived misses on matched snapshots. On independent trajectories, initial recall was lower because actions and memory states diverged. It changed action selection, **but did not beat top-8** for mean action count. It used more detailed-event processing, incurred greater retrieval latency, and produced more failed actions than top-8. These are meaningful **negative findings**, not a success claim.

**Unresolved risks:** An asserted causal edge labeled "observed" could be false; the system did not independently establish causality. The learner sometimes credited an informative predecessor without proving it caused the eventual success. Neighboring memories could overlap as evidence.

## What we learned

A retrieval hit is not the same as an accurate belief. Evidence can be retrieved and activated yet remain outdated, redundant, contradictory, speculative, or irrelevant to a particular action. Finding a success chunk does not by itself justify applying that success to the current situation.

**Confirmed:** Sparse memory processing and explicit outcome learning are useful within the measured toy environment. Some missing results can be recovered from nearby links.

**Plausible but not demonstrated:** Independently interpreted representations could improve decision quality, especially where axes disagree.

**Unresolved:** Can a self-trained interpretation distinguish trusted but wrong information from independently observed outcomes? Can it infer genuinely useful causality without hand-authored scaffolding? Can it generalize beyond repeated workshop roles?

## Why we changed direction

The previous experiments optimized which memories became active. The limiting factor now appears to be **how active evidence is interpreted**.

- **Activation:** Which stored records receive limited processing.
- **Meaning:** What each record implies about a candidate action, goal, source, temporal validity or sequence.
- **Evidence:** Which observations support or challenge that meaning; one event can support several layers without becoming several independent sources.
- **Prediction:** The expected utility or likelihood of the chosen action.
- **Learning:** How actual feedback changes future interpretations and the context-dependent mixture of those interpretations.

More activated records can increase ambiguity and cost. The next test must show **better decisions** rather than better recall alone.

## Experiment 3 — Independent meanings and contextual integration

**Hypothesis:** Five facets of an experience can construct their own bounded numerical interpretations, referencing their original source events. Their relative weights can then change with task context and feedback.

A compact representation for an axis includes a signed value, confidence, source-event IDs, temporal validity, goal association, evidence provenance and the learned association responsible for the value. No language model is required.

The computational form is:

`m_i = clip(raw_i(evidence) * (1 + association_i(context, action)), -1, 1)`

`w_i = softmax(gate_logit_i(context))` **over axes with evidence**, not invented neutral axes.

`integrated = Σ_i w_i * m_i`

This **proposes relevance**, not truth. Contradictions and source deduplication happen before integration. The selector combines the bounded interpreted score with the pre-existing prior, explicit clue/contradiction constraints, repeat penalty and action-feature learner. The formula does not grant causal status to temporal order.

### Four controlled treatments and baseline

- **A — structural direct scoring:** Original memory-score/linear learner.
- **B — meaning_fixed:** Independent structured layer interpretation, no meaning learning, equal weights.
- **C — meaning_learned:** Per-axis learned gains, equal weights.
- **D — meaning_adaptive:** Per-axis learned gains plus context-gated learned integration.
- **E — meaning_gate_only:** Fixed layer values plus learned integration logits.

Every treatment shares the initial seeds, candidate action rules, structural retrieval top-K, and environment. Closed-loop action choices can change subsequent states, **so later events are not matched**. Same-state, same-RNG direct-scoring counterfactuals are logged at each treatment's decision. They identify choice influence, not causal improvements to eventual performance.

The five observable contexts in this first implementation are actually **four**: exploration, supported clue, contradicted candidate and social assistance. The classifier uses only current agent state and selected action, never hidden correct-key state. This context vocabulary is hand-engineered and a generalization limitation.

Feedback for try-key actions comes from their observed outcome. For information gathering, immediate informativeness is a learning target, **not** proof that the step caused later goal success. Separate associations and gates can be frozen for ablations. Interpretation gains learned from past outcomes do **not** rewrite past events.

### Source consistency safeguards and limits

A verified outcome contribution requires an explicit causal-action event ID present in the same retrieved chunk. A merely adjacent informative observation has lower confidence and cannot claim causal success. The same source ID referenced by multiple chunks contributes only once. Five interpretations of that source are not five independent confirmations.

The time axis contributes explicit detected invalidation, not automatic recency decay or automatic endorsement of old success. Contradicted keys retain their historical successes but are discounted in the current selector. Missing axis evidence remains missing; equal weights are normalized only across available evidence.

**Unresolved vulnerability:** Claimed provenance and source IDs can be dishonest. The environment currently provides signed outcomes; this experiment does not authenticate external testimony or independently verify all causal links.

## Architecture changes

The original ownership remains:
`Environment → CognitiveState → MemoryRouter → EpisodicMemory → ActionSelector → Environment → OutcomeLearner → memory/parameters`

**Integrated branch implementation:**
- `meaning.py`: One meaning interpretation owner; constructs self/salience/time/sequence/person outputs from **retrieved events only**, deduplicates evidence, computes available-axis integration.
- `memory.py`: Stores lightweight layer meaning records by original chunk, axis and candidate action. No copied event streams.
- `learning.py`: Same existing learning owner now also updates per-axis association gains and context-specific gate logits after observed immediate feedback; old linear policy remains.
- `policy.py`: Live action selector can choose existing scoring or one of four meaning treatments; no parallel planner.
- `models.py`: Optional selected-action meaning trace for inspectability.
- `experiment.py`: Runs ablations through the **real** six-episode path; logs frames, weights, direct-choice counterfactuals, learned parameters, Brier on actual try-key outcomes, latency, retrieval cost and peak memory.
- `cli.py`: `--meaning` enables the separate comparison; the baseline CLI default remains unchanged.
- `tests/test_meaning.py`: Focused provenance, duplicate-source, temporal contradiction, causal inference, learning ablation and live-path tests.
- `.github/workflows/research-gate.yml`: Runs both historical and meaning tests and five-seed benchmarks.

**Stage labels:** Code written = **Built**. Live path and integration tests passing = **Integrated / Tested** only after verified gate output. Result support requires comparative data.

## Experimental results

**Pending verification.** Results must be transcribed from an executed five-seed research gate, not inferred from the intended algorithm. Preserve the earlier two experiments' numbers above.

Record for each treatment: completion, total/failed/repeated actions, E6 transfer performance, try-key Brier, retrieval recall, detailed-event processing, retrieval and interpretation timing, peak memory, per-axis association changes, context gate changes and action counterfactuals. Distinguish tested behavior from stronger claims about generalization.

If adaptive meanings match or lose to fixed meanings or direct scoring, retain the result as **not supported**, even when the weights visibly change.

## What happens next

After the gated comparison, the most consequential unresolved question is whether layer meanings generalize to *new causal structures* rather than improving workshop task priors. A discriminating next experiment should hold out different clue reliability patterns and changing rules, train on disjoint scenarios, require calibrated predictions, and compare compute/parameter-matched fixed versus adaptive controls.

Before Sonara integration, demand reproducible improvement over a strong simple baseline, reliable contradiction handling, source provenance robustness and measured learning transfer.
