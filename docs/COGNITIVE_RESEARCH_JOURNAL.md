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

The four observable contexts in this first implementation are exploration, supported clue, contradicted candidate and social assistance. The classifier uses only current agent state and selected action, never hidden correct-key state. This context vocabulary is hand-engineered and a generalization limitation.

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

## Experimental results — 2026-10-10

**Tested:** The GitHub Actions research gates for Experiment 3 passed 18 focused tests, 25 total tests, compilation, `git diff --check`, clean-worktree checks, and the full six-scenario/five-seed/1,000-distractor benchmark. Runs #22 and #24 reproduced the outcome differences; run #33 added direct-baseline calibration and interpretation-cost accounting. The archived Experiment 1 and Experiment 2 values above were not overwritten.

**Five fixed seeds: 7, 23, 61, 101, 149. These figures are per-seed means across the original six episodes.**

| Architecture | Goal completion | Actions | Failed actions | Repeated actions | Recall | Try-key Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Original direct structural | 100% | 16.4 | 4.8 | 0.0 | 93.3% | 0.240 |
| Independent meanings, fixed | 100% | 20.0 | 5.0 | 1.6 | 90.0% | 0.229 |
| Learned meanings, fixed integration | 100% | 19.4 | 3.8 | 0.8 | 96.7% | 0.216 |
| Fixed meanings, learned gate | 100% | 20.0 | 5.0 | 1.6 | 90.0% | 0.229 |
| Learned meanings and learned gate | 100% | 19.4 | 3.8 | 0.8 | 96.7% | 0.216 |

Lower Brier indicates more accurate estimated probabilities for observed *try-key success/failure* decisions only. It does not measure confidence calibration for informational steps, and lower Brier on five deterministic seeds is not a held-out generalization claim.

**Computation must include the additional interpretation passes over memory**, not just the unchanged router's sparse detail loads:

| Architecture | Retrieval detailed events | Extra event interpretations | Total event examinations | Total selector latency* |
|---|---:|---:|---:|---:|
| Original direct structural | 214.8 | 0 | 214.8 | 7.16 ms |
| Independent meanings, fixed | 273.4 | 1,664.4 | 1,937.8 | 16.50 ms |
| Learned meanings, fixed integration | 264.6 | 1,578.2 | 1,842.8 | 15.65 ms |
| Learned meanings + learned gate | 264.6 | 1,578.2 | 1,842.8 | 15.91 ms |

*Measured cumulative selector latency per six-episode seed-run on a GitHub-hosted runner; runs execute different numbers of selected actions. Retrieval adds separate measured latency. Counts include repeated interpreter event examinations for different candidate actions and are processing **proxies**, not CPU FLOPs.*

**Observed learning:** Each of the five adaptive runs updated at least one axis association, and each updated its gate logits. The gate-only condition likewise changed parameters. Fixed conditions changed neither. Approximately 29 chunk/axis/action meaning records were retained per seed-run. **Axis-level audit (gate run #35):** The Self, Salience, Sequence, and Person layers produced evidence-grounded records and nonzero learned association gains. **Time produced no live meaning record and no learned association** in these six episodes. Integration logits changed for Self, Salience and Person; **not** for Sequence or Time. Fixed-meaning treatments learned no layer associations, as intended. Hence the five-layer *interface* is built and connected, but evidence does **not** support the stronger claim that all five layers learned through the live benchmark. Time's contradiction logic was tested with direct synthetic evidence; it was not exercised by actual selected-action meaning frames in these trajectories. This is an architectural/evaluation coverage gap, not a reason to retroactively force a positive result.

**What improved:** Learned meanings reduced failed actions relative to fixed meanings (3.8 vs 5.0), reduced repeated actions (0.8 vs 1.6), improved recall and try-key Brier, and used 0.6 fewer total actions. Thus the layer-learning switch had an observed effect within the new architecture.

**What did not improve:** Against the original structural direct scorer, the complete adaptive architecture took 3.0 additional actions per run (19.4 vs 16.4), and consumed about **8.6×** the detailed event examinations (1,842.8 vs 214.8). Adding gate learning to already learned meanings produced **identical measured actions, failures, retrieval volume and Brier**, despite learned logits. More learning parameters are not sufficient evidence of useful integration.

The last original episode, `E6_relational_transfer`, required means of 2.8 actions under direct structural and 3.4 under each of fixed, learned, and adaptive meanings. This is only one pre-existing transfer scenario, not a genuinely disjoint evaluation suite. Seeds 101 and 149 were not used to tune parameters, but all seeds still ran the same six episode definitions.

**Verdict: Supported in part, otherwise not supported.** Distinct evidence-grounded numerical interpretations can be *built*, integrated, and updated from feedback. Learning their gains changes behavior and improves one limited prediction metric. **The stronger hypothesis—contextual integration produces better overall predictions and actions with acceptable computational cost—was not demonstrated.** There is currently no evidence to promote this architecture into Sonara production.

**Methodological limitations and documented failures:**
- The five interpretation rules and four context labels are partly hand-engineered. This is not evidence of spontaneous concept formation.
- Evidence deduplication relies on claimed source IDs; outcome provenance relies on the event recording process. Forged claims can still pass.
- Existing informative-predecessor policy credit can assign causal influence without proving contribution to later success.
- Layer-specific gains can update only when that axis already contributed evidence; **Time remained dormant in all measured live decisions** despite a passing direct contradiction test. A new benchmark must test whether it can learn useful temporal validity from observations without requiring the agent to select a known-obsolete action.
- Adaptive versus fixed variants have different state trajectories after their first divergent action. Same-RNG local direct-score counterfactuals are available, but later closed-loop states are not paired snapshots.
- Relative parameter-count/compute-matched training controls, genuinely held-out causal structures, and calibrated confidence for misleading information are still missing.
- A passing gate establishes execution and measurement integrity; it does not certify learning generalization.

**Artifacts:** CI uploads `independent-meaning-research` containing `results.json`, `event_traces.jsonl` (including axis values, sources, integration weights and updates per decision), and `dashboard.html`. The log prints each experimental mode's measured aggregate. See [GitHub Actions](https://github.com/Logancarton/Scenario-Chunking/actions/workflows/research-gate.yml).

## What happens next

After the gated comparison, the most consequential unresolved question is whether layer meanings generalize to *new causal structures* rather than improving workshop task priors. A discriminating next experiment should hold out different clue reliability patterns and changing rules, train on disjoint scenarios, require calibrated predictions, and compare compute/parameter-matched fixed versus adaptive controls.

Before Sonara integration, demand reproducible improvement over a strong simple baseline, reliable contradiction handling, source provenance robustness and measured learning transfer.


## Experiment 5 — Progressive thought refinement (experimental implementation, October 10, 2026)

**Hypothesis:** A persistent thought identity, versioned evidence-grounded interpretations, and bounded cue-based reinstatement might improve decisions beyond one-pass interpretation or longer activation.

**Built in isolated branch (verification pending):** A thought registry in the meaning/refinement owner, contextual provisional interpretations followed by reuse of the existing five-axis evidence frames, immutable version lineage, explicit evidence and contradiction references, uncertainty as an uncalibrated decision-margin heuristic, and a stop condition that avoids a new revision from repeating the same evidence signature. This is *not* an autonomous general reasoning engine; the maximum two stages are contextual hypothesis and evidence integration. It does not independently derive arbitrary new causal hypotheses.

**Retrieval experiment:** A bounded indexed cue reinstater can discover key-associated chunks outside the initial top-K from existing inverted indexes. It examines at most 18 additional headers, restores at most two chunks, and caps total retrieval detail events at 36. It verifies causal attempt/outcome linkage before accepting an association. Both rejected reads and accepted reads count toward the budget. This is not unrestricted associative spreading and does not scan full detailed history.

**Feedback path:** The existing learner calibrates the influence of evidence-supported action interpretation only following observed success/failure of the selected key attempt. The system never rewards a revision merely for changing or treats information gain as successful key retrieval.

**Temporal correction:** The cognitive state's decision clock now uses the global event clock after observation and feedback boundaries. Original historical occurrence, acquisition, and storage timestamps remain unchanged. Historical archived baseline measurements remain locked.

**Comparison switches:** `--refinement` adds context refinement alone, refinement plus cue reinstatement, and refinement with outcome-grounded calibration, alongside the unchanged structural controls. Trace fields report thought/version lineage, retrieved and reinstated chunk IDs, grounded evidence sources, changed paired actions, clocks, and detail costs.

**Initial CI result (run #38093453592, commit `5f3d2e1`):** 23 focused and 30 total tests passed, compilation and integrity passed, and the original five-seed baseline benchmark completed. A limited **one-seed, 35-distractor** smoke comparison gave 17 actions and 100% completion for both corrected-clock structural and all three refinement variants; refinement caused **zero same-snapshot action changes**. Structural detailed-event examinations were 228 versus 1,768 (refinement alone) or 1,876 (with reinstatement). All refinement variants produced the same observed try-key Brier of approximately 0.204 versus structural 0.239, but this is not independent decision improvement. Retrieved memories added cost without reducing actions.

**Architectural correction following the negative smoke test (verification pending):** The inherited five-axis interpreter repeatedly scanned identical retrieved events for every candidate action. Prepare and index those evidence records once per decision, then reuse their validated source references across candidates and refinement stages. Also add lazy thought focus/peripheral/dormant transitions without rewriting historical timestamps. These changes aim to remove redundant examinations and ensure inactive reasoning consumes no repeated processing.

Do not treat a passing unit test, increased version count, or a changed decision as improvement. The existing structural and top-8 baselines remain the primary skeptics. Further held-out test generation, stronger causal-inference checks, budget-matched evaluation across all modes, and independent cross-domain reasoning evidence remain outstanding before any Sonara promotion.
