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


### Experiment 5 — Second gate (run #38093694728, commit `a7ad6f3`)

**Verified:** 26 focused and 33 full-suite tests passed, compilation and diff checks passed, and both the standard full baseline benchmark and refinement smoke comparison succeeded.

The evidence reuse change reduced the **single-seed refinement + reinstatement** cost from 1,876 to **514 measured unique detailed-event examinations** (~73% reduction), while preserving 17 actions and 100% completion. Refinement alone required 484 examinations. The corrected-clock structural control used 228 examinations. Thus the refinement path remains approximately 2.25× the structural event-examination proxy even after caching.

**Negative result remains:** No measured action-count improvement and zero within-treatment same-snapshot action changes attributable to the refinement stage on the smoke seed. The small difference in try-key Brier does not establish better decisions. The outcome calibration treatment did not improve these measured actions or Brier compared with fixed refinement. The improvement is computational efficiency **relative to the initial experimental implementation**, not evidence of superior cognition.

**Follow-up correction pending verification:** Rework evidence preparation into a one-pass event index with causal links, cap active-thought bookkeeping without scanning dormant threads, and accurately count the second cognitive stage only when grounded evidence actually exists. Keep the draft experiment isolated. Do not promote to Sonara or claim the Experiment 5 behavioral hypothesis succeeded.


### Experiment 5 — Evidence-based revision and held-out skepticism gate (October 10, 2026)

**Isolation:** All changes remain on `experiment/progressive-thought-refinement` in Scenario-Chunking. Sonara production is untouched.

**Architectural finding and implementation:** The previous refiner mostly changed already-computed action scores; it did not explicitly represent competing predictions. The revised meaning/refinement owner now builds candidate *key–object relationship* hypotheses from current observation provenance and previously indexed causal attempt/outcome pairs. It distinguishes support from contradiction, records alternatives and predicted outcomes, and chooses either a supported action or an information-seeking action when historical applicability remains unresolved. It reuses the meaning owner's one-pass prepared evidence index, makes no new detail reads within the refinement stage, and preserves thread identity, lazy activation/dormancy and version history. This is constrained relational inference, **not general causal reasoning** or independent concept formation.

**Wiring and learning:** The state owner records observation confidence, source IDs and witnessed key-failure contradictions; the experiment runner passes actual observed event IDs into state and action-selection feedback. The existing learner has an opt-in strict-credit comparison for refinement treatments. Informative-predecessor credit requires that the producing observation was used in the successful action's interpretation. No new observed causal memory edges are manufactured from mere temporal precedence. Refinement prediction calibration requires evidence actually contributing to the selected hypothesis. Historical structural-control behavior is preserved separately to avoid quietly changing the comparison.

**Behavioral tests:** Same-state/RNG tests prove a weak, incorrect clue can be displaced by verified matching historical outcome, changing both predicted probability and actual tested key success. Object-mismatched historical outcomes do not count as target-matched evidence. Tests cover misleading history, current observed key failure, irrelevant informative predecessors, unrelated feedback, stable-thought termination, dormancy and the real `ExperimentRun._run_episode` entrypoint.

**Original six-scenario smoke, seed 7, 35 distractors:** A verified intermediate gate after adding a historical-applicability check (commit `bc18726e`) completed **32 focused and 39 total tests**, full baseline research benchmark, compilation and integrity checks. The corrected-clock structural control completed 6/6 scenarios in **17 actions** with **228 event examinations**. Refinement completed 6/6 in **14 actions** with **380 examinations**, and registered **1 same-snapshot action difference** versus its paired one-pass interpretation. Brier for observed try-key actions was **0.0219** for refinement versus **0.2393** for structural. These numbers are from one small, previously known task family, not evidence of broad generalization. An earlier revision with aggressive unverified historical action selection took 19 actions and 552–588 examinations; the observed regressions motivated applicability-aware information seeking.

**Independent held-out synthetic challenge:** GitHub Actions run for commit `850e1d4f` passed all CI stages and generated a five-seed comparison across five separately generated challenge conditions and five treatments (125 independent trials). Training memory contains, at most, one synthetic historical causal outcome preceding each held-out test; each condition uses the same seed/task/evidence across treatments. There is **no cross-trial learner training or evidence of transfer to genuinely different domains**. Comparison is closed-loop, so trajectories after divergent actions are not matched snapshots.

Five-seed means for the verified `refinement_associate` treatment versus structural:

| Challenge | Structural actions / failed attempts | Refinement actions / failed attempts |
|---|---:|---:|
| Uncertain misleading current report | 1.0 / 0.0 | 4.8 / 0.0 |
| Replaced lock with obsolete historical relationship | 4.8 / 1.0 | 4.8 / 0.0 |
| Irrelevant historical object relationship | 4.8 / 1.0 | 4.8 / 0.0 |
| Clear current evidence | 4.8 / 0.6 | 4.8 / 0.6 |
| Delayed feedback under stable historical relation | 1.0 / 0.0 | 4.8 / 0.0 |

All 125 independent trials completed the task. Refinement protected against failed attempts on the changed/irrelevant-history controls, but it did **not** save actions there, and it was materially less efficient in stable-history controls. Additional inspections also increase memory/selection processing cost. The refinement and reinstatement treatments had equal means in these simple cases; the added reinstatement or learned calibration has **not** proved separate benefits.

**Falsification and next scientific question:** The agent has no initial observation distinguishing a stable reused lock from a replaced lock. Without information or a learned hazard model, it cannot always choose optimally between directly trying a historical key and inspecting for changes. The current conservative confirmation rule can avoid errors but over-checks stable situations. The next falsifiable improvement belongs in the existing interpretation/learning and action policy owners: estimate the *expected value of additional evidence* from uncertainty, outcome loss, inspection cost, and observed transfer reliability, then terminate or investigate accordingly. Compare against these frozen current results with new held-out episodes and matched cognitive budgets. Do not tune against the same five cases to force a win.

**What is not yet demonstrated:** independent generation of genuinely new causal explanations, contradiction-driven targeted retrieval beyond current bounded indexed reinstatement, robust calibration across task families, learned volatility-dependent reasoning termination, proven cross-domain transfer, restart-safe thought identity, or acceptable cost-benefit over a compute-matched baseline. Working-memory continuity, bounded hypothesis reasoning and the live experimental path are integrated and enabled; **the full cognitive research question remains open**. No Sonara promotion is justified.


## Experiment 6 — Five-category learning followed by stepwise transfer (October 10, 2026)

**Research branch:** `experiment/category-transfer-e6`, based on `experiment/progressive-thought-refinement`. This experiment is separate from Sonara production and does not change the original six workshop scenarios or archived measurements.

**Question:** Do extra bounded interpretation cycles produce reusable knowledge that measurably improves subsequent novel decisions enough to compensate for computation? This is a hypothesis, not a conclusion.

**Implemented mechanism:** A reproducible procedural environment randomly associates two neutral cues with each of Self, Salience, Time, Sequence and Person. The hidden association is accessible only to the environment and its outcome-feedback generator. Balanced Phase-1 interactions produce attempted actions and causally linked success/failure events. The existing `EpisodicMemory`, `OutcomeLearner`, `MemoryRouter`, `MeaningInterpreter`, `ActionSelector` and `ProgressiveRefiner` are the receiving owners; no second action selector, learning store or production runtime has been introduced. Predictions, unverified reports, and repeated exposure are not upgraded to verified causal outcomes.

**Transfer:** The trained memory and learner state are serialized and reconstructed before Phase 2; episode-specific working state is reset. New equipment names, people and problem orderings are procedurally generated. Each round offers a joint proposal that depends on all five categories, followed by five feedback-bearing decisions. A finite action budget means a brute-force solution is not automatically counted as task completion. Later novel rounds keep earlier outcome learning. Phase-1 knowledge is not regenerated from hidden truth for the policy; policy gets candidate cue features and retrieved historical outcomes only. The research gate includes a separate-process memory retrieval test.

**Comparisons:** Seven matched seeded conditions: untrained structural, trained structural with no thought-review override, trained/no reflection, limited reflection, extended bounded reflection, strong one-pass with wider historical evidence, and relationship-shuffle negative control. Each of five axes is ablated separately. A selected-action counterfactual is computed on the same evidence and random seed to detect changes attributable to additional evidence review. Separate closed-loop trajectories are not falsely described as counterfactually identical.

**Reflection:** The existing meaning owner reviews only causally paired observations from retrieved chunks. No-reflection evaluation uses the available evidence together; limited and extended reviews use progressively narrower, chronological evidence windows to reconsider stale or contradicted associations. The existing refiner stores thought identity, evidence provenance and revised versions; extra review does not itself trigger positive learning credit. New causal action outcomes can update the existing learner and memory. The latency and event examination costs of interpretation and the shadow counterfactual are recorded.

**What to inspect:** `scenario-chunking --transfer --output results-transfer --seeds 7,23,61,101,149 --rounds 4` writes `transfer_trials.json` with source-linked decisions and `transfer_summary.json` with phase-1 learning, category recall, stepwise completion, initial-plan accuracy, errors, action efficiency, prediction Brier, restart state, retrieval cost, reflection processing, across-round completion, per-axis ablation and controls. The original benchmark runner remains unchanged without `--transfer`.

**Limits that prevent overstating cognition:** These are synthetic binary associations with a shared task grammar, not spontaneously discovered human concepts or unseen causal structures. Task order is presented by the environment; this benchmark does not yet prove that the agent learned an arbitrary dependency graph. The plan affordances are generated by the environment, not hallucinated or inferred. Phase-1 examples are balanced exposure trials rather than free exploration. The time axis has an obsolete early association; outcome validation and chronological review are tested, but sophisticated calendar reasoning is not. Reflection here is evidence reweighting, not unlimited thought. Category ablation can demonstrate dependence on learned cues but does not establish open-domain transfer. A longer thought trace or an altered numerical preference is not itself an intelligence improvement.

**Verification status:** The initial focused Experiment 6 tests, existing focused tests, full suite, compilation, and integrity checks have passed on the research branch. Full five-seed benchmark outcome and the newer separate-process persistence test remain pending at the time of this journal entry. Preserve positive and negative results when the gate completes; do not claim a behavioral victory, cost justification, or production integration without completed comparative data.


### Experiment 6 — Verified five-seed adversarial transfer result (GitHub gate, October 10, 2026)

**Gate:** Commit `895cabbb`; 34 earlier focused tests, 8 Experiment-6-focused tests, **49 full-suite tests**, compilation, diff integrity, historical six-scenario benchmark, held-out refinement gate and five-seed Experiment-6 research benchmark all passed. A separate Python process retrieved correct evidence from a serialized memory and learner state. The research environment and its outcomes remain isolated from production.

**Protocol:** Five seeds (7, 23, 61, 101, 149) x four novel rounds per treatment = 20 evaluated tasks per treatment. Phase 1 has balanced causal outcome experiences for two neutral symbolic cues in each of five axes. Phase 1B does zero, limited or extended offline consolidation without new factual evidence or learning credit. After half of the Phase-2 rounds, the environment changes the formerly effective Time relation. A visible notice announces a change but not the new correct cue. The six-action deadline penalizes incorrect initial decisions. Controls include zero prior training, a strong wider-retrieval one-pass solver, scrambled relationships and one-axis-at-a-time ablations.

**Measured:**

| Treatment | Completion | Initial joint-plan accuracy | Actions/task | Interpretation operations/run | Detail events/run |
|---|---:|---:|---:|---:|---:|
| Untrained structural | 10% | 55% | 7.85 | 0 | 619 |
| Trained structural | 50% | 90% | 6.60 | 0 | 938 |
| Trained one-pass interpretation | 50% | 90% | 6.60 | 1,772 | 938 |
| Limited reflection | 50% | 90% | 6.60 | 2,152 + 360 offline | 1,238 |
| Extended reflection | 50% | 90% | 6.60 | 2,472 + 390 offline | 1,238 |
| Strong one-pass (top-32) | 50% | 90% | 6.60 | 3,548 | 1,875 |
| Shuffled relationships | 0% | 0% | 11.50 | 3,070 + 390 offline | 1,718 |

The paired shadow one-pass diagnostic adds **1,772** interpretation operations to the limited/extended research runs, but it is not charged as live cognitive processing in the table. Wall-clock decision+retrieval time was approximately 11.5 ms/run for ordinary one-pass interpretation and 16.0 ms/run for extended reflection on this CI host; do not generalize these host-specific latencies. Extended reflection Brier was 0.1266 versus no-reflection 0.1272, an extremely small numerical difference without behavioral benefit.

**Category findings:** Training preference accuracy and Phase-1 category recall were 100% for Self, Salience, Time, Sequence, Person in the trained conditions. Disabling each category reduced overall goal completion from 50%: Person 20%, Salience 20%, Self 25%, Sequence 30%, Time 35%. This supports causal sensitivity to each trained cue in this *toy* environment, not domain-general understanding of those categories. Scrambling the cue-to-outcome relationship eliminated goal completions.

**Decisive negative result:** Both no-reflection and extended-reflection completion by round were **[100%, 100%, 0%, 0%]**. All trained treatments failed to recover successful end-to-end behavior after the temporal relation reversed. Extended reflection changed **zero same-snapshot actions**, saved **zero actions**, and required approximately **1,090 additional live interpretation operations/run** over no-reflection (including its offline consolidation). Its extra computation was therefore **not justified by measured outcomes**. Repeated experience after the change did not produce compounding completion gains within the four-round horizon.

**Answers to the seven research questions:** (1) The agent learned five independently trained synthetic cue/outcome associations; this is narrower than learning rich conceptual categories. (2) It recombined those associations for previously unseen object/person surfaces and joint plans, but failed the temporal reversal. (3) Each category measurably affected choices or deadline success when ablated. (4) More reflection did not improve selection or completion. (5) No advantage grew across rounds; performance collapsed after a rule change. (6) The added cost was not worthwhile. (7) Temporal belief invalidation, evidence conflict handling and recovery failed; a future finding of no advantage against genuinely conditional, evolving causal tasks would further falsify the broad claim that repeated thinking pays for itself.

**Interpretation and next experiment:** The system has demonstrated limited associative transfer and causal sensitivity to memory cues, **not flexible belief revision**. A longer reflection pass that merely reweights retrieved outcomes cannot repair uncorrected temporal validity, insufficient evidence reinstatement, or a learned prior that persists after new outcomes. Next, design a challenger where an observed contradiction can *invalidate an obsolete relationship*, selectively retrieve its causal alternatives, and cause a changed live decision—then compare against the same strong one-pass baseline under matched budgets. Do not tune for an artificially positive result or merge this branch into Sonara production.


### Experiment 6 — Repair of disconnected reflection and revision pathway (October 10, 2026)

**Why the first experiment was invalid as a test of progressive thinking:** The transfer runner computed different historical evidence windows but never called `ProgressiveRefiner.refine()` during its decisions; post-hoc `record_transfer()` merely logged the chosen action. Phase-1B consolidated interpretations were only consulted when all normal evidence was absent. Old outcomes remained treated as current despite a recorded rule-change notice. The learner reset the eligibility trace for every sub-action and the joint-plan outcome was not stored as causally linked decision feedback. The deadline of six actions made correction of a mistaken action appear as goal failure even when later sub-actions succeeded. These problems invalidated a strong claim that reflection itself had been fairly evaluated.

**Corrected mechanism:** The existing `ActionSelector` now invokes the `ProgressiveRefiner` *before* transfer action selection. The refiner retains thought identity, records changed evidence and contradictions, and can choose an available information-seeking action when a named relationship is known to be obsolete but no applicable replacement has been observed. `MeaningInterpreter` refuses to transfer obsolete causal outcomes across an observed validity boundary; a current posted-protocol observation is eligible as an explicitly identified report rather than invented confirmed causal feedback. Learned action features after the change use a separate observed-epoch namespace so outdated policy priors are not silently carried over. Every treatment sees the same optional inspection action, and only environment-generated observations reveal what it reports. Full-per-episode eligibility traces and causally linked integrated plan outcomes are retained; an unknown-cause joint failure no longer assigns negative blame to every category. Recovery and deadline success are reported separately. Research world, task state and decision orchestration are placed in the existing environment, state and experimental-run owners, respectively; Sonara production remains unchanged.

**Verification (initial repaired five-seed full gate, commit `583f3b7`):** 34 existing focused, 13 Experiment 6 focused, and **54 total tests** passed; compilation, diff integrity and all historical benchmarks passed. A separate interpreter-process restart test remains included. The refiner generated **five same-snapshot action differences** in extended reflection, replacing an unverified joint proposal with current-protocol inspection after temporal invalidation. After those observations, the agent resumed choosing joint plans using evidence with explicit provenance. The old zero-change result was therefore due partly to missing live wiring.

**Measured behavior after repair:** All trained treatments, including ordinary trained structural, no-reflection one-pass, limited reflection, extended reflection and strong wider-retrieval one-pass, completed **20/20** novel tasks across five seeds and four rounds. Untrained structural completed 45%; scrambled cue/outcome relationships completed 0%. Extended reflection produced **100% initial joint-plan accuracy**, zero wrong decisions and 6.4 actions/task; no-reflection one-pass produced **95% initial joint-plan accuracy**, 0.1 wrong decisions/task and **6.2 actions/task**. Extended processing was more expensive (mean 2,651 live interpretation operations plus 390 offline/run versus 1,829 live and zero offline for one-pass; paired counterfactual diagnostics add a further 1,978 operations to the research run only). Reflection did not beat the strong one-pass baseline on completion or action efficiency. Host-specific decision+retrieval latency also increased, from approximately 18.7 to 27.8 ms/run in the first repaired CI run. Treat these as synthetic small-sample results, not statistically established improvements.

**Verdict:** The repair is an **integration improvement with a real, measurable selection consequence**, not evidence that longer reflection compounds intelligence. The previous failure under the six-action deadline was partly a scoring artifact; after separating recovery from efficiency and representing rule invalidation, the simpler trained controls recovered just as reliably. More thinking *can now change decisions* when a real evidence gap is present, but it has not yet been shown to be worth its cost. No Sonara production promotion is warranted.

**Remaining skeptical gaps:** The five domains are synthetic cue categories rather than true self-valuation, person reliability, temporal knowledge, salience or dependency-graph learning. Current-protocol observations are reliable by construction; misleading or incomplete protocol reports, ambiguous revision notices, longer intervals, conditional dependencies, incremental confidence calibration and genuinely unknown causal structures remain untested. The current refiner is a bounded two-stage evidence check, not free-form recursive inference. Consolidated Phase-1B knowledge is still mostly backup evidence rather than a source of decisions when full direct causal history is present; this is a research limitation, not proof that unlimited offline thought would help. Avoid overfitting the next challenge to inspection-triggered success.


## Experiment 7: independent choices under uncertain testimony — implementation record

**Scope:** Standalone research branch, not Sonara production. **Status:** mechanism wired into the live research pathway; cognitive superiority is *not established* by implementation.

**Motivation:** Experiment 6 presented a candidate set constructed around the hidden correct plan and relied on binary choices. Therefore 100% task recovery did not demonstrate independent planning or a reason to spend more computation. Experiment 7 removes the prefabricated correct joint plan, introduces four actions per category, fallible source claims, dependency gates, bounded information acquisition and a mid-series protocol change.

**Ownership:** `environment.py` owns hidden procedural world truth and physical action outcomes. `state.py` tracks task-local observed completion, prior actions, and current observed dependencies. `learning.py` stores verified source reliability and prerequisite associations, distinct from ground-truth environment data. `meaning.py` scores only retrieved causal evidence and explicitly demoted testimony with event provenance. `policy.py` applies observed prerequisite penalties during live action selection. `refinement.py` maintains a thought identity and can trigger costly inspection when a changed relationship lacks positive applicable evidence. `uncertainty.py` only sequences existing owners and collects controls/metrics.

**Controls:** Untrained structural, trained structural, trained interpreted no-reflection, limited reflection, extended reflection, larger retrieval, stronger one-pass with measured (not identical) processing budget, shuffled evidence, corrupted testimony, and individual category ablations. Identical seeds establish comparable private worlds. Metrics report goal completion, first-action accuracy, observed prerequisite blocks, mistakes, inspections, latency, active detail events, processing evaluations, calibration and seed-level results.

**Critical constraints / possible failures:** Sharing role tokens across train and held-out scenes is not compositional novelty. The environment directly reports the blocker after a failed prerequisite action; it does not require independently inferring latent causes. Baseline access to costly inspection is the same, but one-pass and iterative compute budgets are not strictly matched. Correct-first-plan accuracy is not yet established: this world measures independently selected actions rather than complete submitted plans. Outcomes must be measured separately; do not claim Experiment 7 proves progressive thought helps unless matched results support it. A negative result should be retained without reducing task difficulty.


### Experiment 7 verified five-seed result (revision after training-leakage correction)

Run: https://github.com/Logancarton/Scenario-Chunking/actions/runs/38100108827 (62 complete-suite tests, 8 E7-focused tests, compilation, existing research scenarios, and diff-integrity gate passed).

**Five seeds x four novel tasks = 20 tasks/condition:**

| Condition | Completion | Mean actions | First action |
| --- | ---: | ---: | ---: |
| Untrained structural | 15% | 8.75 | 50% |
| Trained structural | 75% | 6.75 | 60% |
| Trained interpretation, no reflection | 80% | 6.70 | 70% |
| Strong one-pass | 80% | 6.70 | 70% |
| Larger retrieval | 80% | 6.70 | 65% |
| Limited reflection | 85% | 6.25 | 95% |
| Extended reflection | 85% | 6.25 | 95% |
| Shuffled evidence control | 0% | 9.00 | 0% |
| Corrupted testimony control | 80% | 6.70 | 70% |

The extended-reflection condition made 15 counterfactual action changes. Paired seed completion deltas versus strong one-pass were `[0, 0, 0, 0, +0.25]`; the 2,000-draw, seed-resampled 95% bootstrap interval was `[0, +0.15]`. Therefore **no robust or statistically established improvement** is demonstrated. Limited and extended reflection were identical in completion and mean actions; deeper reflection is not justified by these results. Extended reflection executed 32,692 interpretation evaluations versus 49,869 for the larger-retrieval one-pass comparator, but this is not a matched compute allocation or matched algorithmic search budget. Do not infer superior cognition per compute from that number alone.

Mean held-out source-prediction Brier score was approximately 0.170 for trained conditions, 0.250 for untrained, and 0.333 for corrupted-testimony control. This shows prediction calibration sensitivity but not a causal behavioral benefit from testimony: the corrupted-testimony control still finished 80% of tasks. Remaining source calibration and action-selection influence need stricter matched source ablations.

Signal-cue ablation completion with extended reflection: Self 25%, Salience 0%, Time 20%, Sequence 5%, Person 0%. **These remove the category's signal-token cue, not every axis-specific retrieval or interpretation mechanism.** They cannot establish unique biological-style responsibilities for all five categories.

**Conclusion:** This environment is harder than Experiment 6 and is executed through the shared live cognitive pathway. Reflection can cause information-seeking and sometimes help after changes, but the one-task completion improvement is too small to justify a cognitive advantage claim. The strongest next challenge is held-out causal combinations and a genuinely compute-matched one-pass policy with matched information acquisition, followed by independent replication on substantially more seeds. Sonara production remains untouched.

## Experiment 8 — Cognitive scaling and adaptive focus (implementation stage)

**Branch:** `experiment/cognitive-scaling-e8`. **Production:** untouched. The bounded-focus mechanism is built, and the research CLI, incremental-training harness and behavioral gates are connected. Positive reasoning results must **not** be asserted until CI and the full multi-seed study produce measured evidence.

**Signal ownership:** Experiment 7's `UncertaintyWorld` remains the private source of procedural truth. `EpisodicMemory` incrementally maintains a compact index of historically salient/representative and recent cue-linked chunks; `MemoryRouter` optionally uses it to bound candidate header evaluations. `OutcomeLearner.focus_axes` derives category priority only from observed prerequisite outcomes, preserving explicit invalidation of out-of-date sequencing. `UncertaintyRun` still invokes the same interpreter, selector and progressive thought refiner. `scaling.py` sequences these existing owners; it is not a second cognitive runtime or selector.

**Design:** Each treatment observes a seed-matched stream of action attempts and observed success/failure, accumulating 20, 100, 500, 2,000, then 10,000 experience steps in the **same learner and memory**. Test-only forked states are confronted with new private environment mappings and novel procedural scenes at every stage. Distractor memory sizes are controlled independently (100, 1,000, 5,000, 20,000, 20,000 minimum chunks), including low-trust same-goal misleading reports. The untrained control receives no observed training. Stage test outcomes never leak back into the continuing training stream. The experiment records stage/seed decision accuracy, budgeted goal completion, action/inspection cost, active detail reads, header scans, latency, processing evaluations and paired bootstrap intervals.

**Current limitations / open research gates:**
- Adaptive **category focus** is integrated; an **outcome-learned reflection stopping rule is not yet implemented**. Existing limited/extended refiner modes still use fixed review depths. Thus this experiment must not report that adaptive reflection itself has been proved superior.
- The held-out world changes equipment, identity strings and symbol assignments, but preserves the same five categories and procedural family. This tests recombination and interference, not unconstrained abstraction or broad general intelligence.
- The representative/recent focus index is a new attention heuristic, not yet a demonstrated optimal abstraction. Its priority must be compared against random, recent-only and enlarged retrieval under matched processing budgets.
- Full study (30 seeds, all conditions, 10,000 experiences, up to 20,000 memories), independent reproducibility and quantified uncertainty are not satisfied by the smoke gate.
- A separate-process continuation test is implemented. An in-process evaluation fork is not independent process restart validation of a full multi-stage deployed run.
- Certain training and memory-size factors remain correlated at the largest stages; a factorial subset is still needed to isolate their effects.
- The original Experiment 7 tests and controls remain essential. Negative findings should be retained rather than adjusting task difficulty to improve the score.

**Research decision rule:** Look for paired gains in goal completion *and* lower irrelevant processing per correct decision, not higher episodic recall alone. Do not claim cognitive maturity from smoke compilation or passing infrastructure tests.

### Experiment 8 — first verified smoke (October 10, 2026, Phoenix)

Run: https://github.com/Logancarton/Scenario-Chunking/actions/runs/38102511995

**Verification:** Seven focused behavioral tests passed (including separate-process continuation); two-stage executable scaling gate, payload assertions, compilation and diff check passed. This is a **one-seed, two-round-per-stage smoke**, *not* the 30-seed research verdict. Legacy full regression gate is separate.

| Experience stage | Untrained completion | Strong one-pass | Adaptive learned focus | Bounded retrieval without learned focus |
| --- | ---: | ---: | ---: | ---: |
| 20 | 0% | 0% | 0% | 0% |
| 100 | 0% | 0% | 0% | 50% |

At 100 experience steps, adaptive learned focus averaged 32 evaluated headers/decision versus 232 for the larger-access one-pass comparator. Both capped treatments were limited to 32 candidate headers. However learned focus finished **0/2** tasks; the otherwise bounded condition without category selection finished **1/2**. Thus **the candidate-budget mechanism reduced header processing, but learned category prioritization did not show better reasoning** in the smoke test and may be discarding important alternatives. No inference of statistical significance is warranted.

**Priority for next research iteration:** separate category-focus quality from purely reducing the candidate-header budget. Add paired exact information/compute controls and harder novel causal combinations, then evaluate learning curves over 30 seeds if computationally practical. Do not increase reflection merely to improve completion.

## Experiment 8 correction — held-out contextual causal scaling

**Branch:** `experiment/cognitive-scaling-e8-corrected`. This entry supersedes the
initial E8 architecture as a test of causal scaling; the earlier implementation
record and negative result are retained for provenance.

**Why corrected:** E8 initially treated `self`, `salience`, `time`,
`sequence`, and `person` as five independent procedural symbol choices.
Its learned focus prioritized only two predetermined categories using
prerequisite heuristics, and the hidden evaluation world changed with each
checkpoint. It could neither cleanly measure experience effects nor demonstrate
genuinely learned, compositional causal understanding.

**New live research path:** `ContextualWorld` owns the private procedure
equation and instrument outcomes. Every problem has four plausible intervention
choices and one joint scene carrying time/protocol, source, consequence,
self-relevance, condition and dependencies. `EpisodicMemory` keeps immutable
events; `OutcomeLearner` accumulates outcome-conditioned relationship
statistics and attribution/attention utility and retains source verification.
`MemoryRouter` provides bounded indexed retrieval; `MeaningInterpreter`
combines provenance-bearing episodic evidence with revisable learned contextual
relationships and appropriately weighted testimony. `ProgressiveRefiner`
preserves thought identity, revises evidence-backed preferences and can choose
inspection or stop; `ActionSelector` remains the one action authority.
`scaling.py` sequences owners but contains no duplicate selector.

**Experiment validity:** Frozen held-out combinations cover all four possible
procedure outcomes. Their sub-relations appear in training but the joint
combinations do not. Each treatment has identical public signals and the
same exploratory training interventions; outcomes come from actual attempts,
not a generated answer plan. Training checkpoints maintain one original
persistent learner/memory. Test episodes operate on forks. Added distractors
and difficulty vary factorially and independently from training volume.
Statistics report by-seed completion, first procedural action, action and
inspection cost, active event reads, header evaluations, causal retrieval
precision/recall/misses, latency, reflection versions and paired seed deltas.
Time/protocol changes are observed as epoch-bearing situations and new
outcomes can revise conditional knowledge without rewriting old events.

**Verified two-seed smoke (four distinct held-out procedures, two complexities,
two independent memory loads, four conditions):** 13 focused tests and the
end-to-end smoke passed in GitHub Actions. Completion by 20 / 100 / 500
training experiences: untrained 38% / 38% / 38%;
strong one-pass 28% / 56% / 44%;
adaptive learned focus 38% / 56% / 53%;
bounded without learned focus 28% / 56% / 44%.
These values pool correlated cells from only two independent seeds, so they
do **not** establish a statistically reliable improvement. At the highest
smoke checkpoint adaptive focus was 53% versus 44% for one-pass, but its
retrieval and reflection are not precisely compute-matched. No robust
generalization claim follows.

**Remaining limitations:** This is a small, transparent synthetic conditional
causal world; there is no open-ended symbolic concept invention, no natural
language or multimodal input, and no claim of broad human-like cognition.
Category selection is learned through predictive feedback, but a rigorously
outcome-optimized reflection stopping policy is not proven; learned inspection
value is an observed information-gain proxy. Source confidence does not yet
model full counterfactual testimony. The research still needs 30 independent
seeds, more diverse held-out causal transformations, strict information/compute
matching, longer-term interference controls and a complete 10k/20k run before
any superiority or mature learning curve can be asserted.

### Experiment 8 corrective replication — 30 independent seeds

**Why this gate matters:** An earlier 30-seed replication of the initial
corrected mechanism showed deterioration with experience: at 500 experiences,
untrained completed 35% of tasks, strong one-pass 27%, and learned focus 18%.
Reflection provided exactly zero goal-completion benefit. This negative result
is retained; its cause prompted corrections rather than weakening the baseline.

**Mechanism repair:** OutcomeLearner now evaluates each learned contextual
relationship against the action's empirical base success rate, avoiding
"learning" from common failures alone. MeaningInterpreter marginalizes
uncertain source testimony across both publicly possible dependency values.
ProgressiveRefiner uses verified source reliability, counterfactual hypothesis
disagreement, consequence and an observed information-value estimate to
investigate selectively. Same-task random exploration is aligned across
treatments and frozen across stages. An identical-learned-focus one-pass
control isolates attention from reflection.

**30-seed corrected research run:** [GitHub Actions run
38104001062](https://github.com/Logancarton/Scenario-Chunking/actions/runs/38104001062).
Thirty independent seeds, four balanced held-out actions per seed, 100
additional distractor memories, fixed complexity 2, checkpoints at 20, 100
and 500 experiences. The same held-out task bank and action budget apply
to all conditions at each checkpoint.

| Training experiences | Untrained | Strong one-pass | Learned focus one-pass | Learned focus + reflection |
| --- | ---: | ---: | ---: | ---: |
| 20 | 41% | 52% | 34% | 34% |
| 100 | 41% | 9% | 48% | 44% |
| 500 | 41% | 26% | 56% | 54% |

Seed-bootstrap 95% intervals for adaptive focus+reflection minus strong
one-pass: 20 experiences **−17.5 percentage points**
[−25.8, −8.3]; 100 experiences **+35.0 points**
[+24.2, +45.0]; 500 experiences **+28.3 points**
[+22.5, +34.2]. Thus focus is worse early and better after more
observed experience **in this fixed synthetic task family**.

The *same focus and initial retrieval evidence* without reflection finishes
56% versus 54% with reflection at 500 experiences. The paired reflection
increment is −1.7 points, 95% interval [−4.2, 0]. No positive reflection
benefit is demonstrated. This is a meaningful **negative** finding about
the current reflection policy, not grounds to count extra thought as cognition.

**Interpretation:** One causal learning and attention mechanism now exhibits
a measurable, experience-dependent advantage against the tested strong
one-pass policy at later checkpoints. However that comparator evaluates a
broader cue set, not strictly the same computational operations; the
focus-matched one-pass ablation better isolates reflection. The sampled
tasks still share a small underlying procedural grammar. Selection and
calibration improvements must not be extrapolated to human-style open-world
reasoning or to untested 2,000/10,000 experience stages. Per-task compute,
distractor interactions and causal memory recall are reported by the
corrected research harness; further factorial and domain-transfer
replication remains necessary.

**Additional replication checks (same final 30-seed run):** Comparing
*learned-focus one-pass* on the **same held-out tasks** at 500 versus
20 experiences yielded **+21.7 percentage points** (seed-bootstrap
95% interval [+15.0, +27.5]). Against the untrained baseline at 500
experiences, learned-focus one-pass gained **+15.0 points**
([+6.7, +22.5]). Adaptive focus with reflection gained **+13.3**
([+5.0, +20.8]). In contrast, strong one-pass was **−15.0**
([−23.3, −6.7]) compared with untrained at that stage.

The result is evidence for experience-dependent, bounded focus on this
procedural family, *not* evidence that reflection added utility. The
one-pass-with-identical-focus comparison keeps initial retrieval cues,
retrieval budget and source observations matched. Reflection adds
investigations and alters trajectories, but no positive net completion
benefit was measured. Report this negative conclusion alongside the
focus advantage.

**Verification:** 15 focused behavioral/durability tests, frozen
factorial smoke and independent 30-seed replication passed in GitHub
Actions. Full historical regression and Research Gates 6/7 remain the
separate cumulative gate. The configured 2,000/10,000-experience and
20,000-distractor stages are implemented but have not been executed as
a verified large-scale study.


## Experiment 9 — event-driven temporal memory and prediction (implementation record)

**Source:** Branch `experiment/temporal-prediction-e9`, based on the corrected
Experiment 8 research branch. This remains independent of Sonara production.

**Mechanism:** Time is no longer just a recency feature or a protocol label.
`CognitiveState` maintains a monotonic elapsed-world clock independent of
cognitive processing ticks, retains sourced `TimedExpectation` records, and
uses a bounded due-event queue to wake relevant expectations. Deadline
passage produces `due_unverified`, not a fabricated absence or failure.
The environment alone owns private arrival times and direct sensor feedback.
EpisodicMemory stores the immutable source promise, prediction and
subsequent observations. The existing ActionSelector invokes the existing
ProgressiveRefiner when a due-event or verified observation justifies revision;
refinement may choose information acquisition based on learned punctuality,
consequence, and inspection cost. New versions cannot be created by repeated
reasoning on unchanged evidence. Only OutcomeLearner learns duration error and
on-time calibration, using direct event-timestamp evidence; a witnessed
presence without an arrival timestamp never becomes an exact duration label.

**Experimental constraints:** This is a simple, falsifiable "return in ten
minutes" research environment. Controls use the same synthetic test worlds and
retrieval/action pathway: clock-only, memory-only, learned-information one-pass,
and temporal refinement. Two prior verified observations supply equal
calibration to the learned treatments. Test cases include early arrival, late
arrival, verified nonarrival, an unavailable sensor and an unverified
arrival timestamp, plus misleading bystander testimony. Focused tests cover
no-input waking, bounded selective retrieval, clock regression, duplicate
evidence, stable thought termination, outcome updates and independent-process
serialization.

**Scientific status:** Built and wired into the research path, with CI
verification pending at implementation time. Neither "due" nor a new thought
version proves improved reasoning. The current synthetic cases cannot test
complex social semantics, repeated appointments, causal time inference, or
biological subjective experience. Keep all previous negative Experiment 8
reflection findings intact and **do not promote Experiment 9** until live
tests pass and a held-out, information/compute-matched comparison demonstrates
benefits over strong simpler policies.


## Experiment 10 — predictive temporal reasoning and decision value (2026-10-10)

**Scientific question:** Can learned temporal expectations and decision-value
refinement improve subsequent actions beyond equally informed, compute-matched
one-pass interpretation? This entry adds to the Experiment 1–9 record without
editing older observations or negative findings.

**Built and integrated in the isolated research branch:**
`experiment/predictive-decision-value-e10`, based on
`experiment/temporal-prediction-e9`; production Sonara is untouched.
`OutcomeLearner` now records observed arrival durations and right-censored
intervals by expectation identity, computes a smoothed survival distribution
conditional on context and promise length, and weights verified source
reliability by age. A censored direct absence is evidence of survival past
an observed time, never an invented exact arrival or generic failure.
EpisodicMemory retains source promises, direct observations and causally
linked attempted-action outcomes as separate historical events.

`MeaningInterpreter` interprets public information, probabilities, reports,
dependency notices, deadlines and costs into option values. `ProgressiveRefiner`
can revise an option only if its expected gain beats thought cost and records
distinct sourced versions; unchanged evidence/probability bands stop version
growth. `ActionSelector` alone executes the final ranking. The synthetic
world hides arrival and dependency truth until legally exposed by public
notices, inspections or actual execution feedback. Deterministic fake clocks
and a separate interpreter process test episodic/expectation/thought
continuity after pickle-based restart. The research runner sequences owners
and does not implement an alternative planner or memory store.

**Environment and comparisons:** Three overlapping delivery expectations,
several contextual duration distributions, misleading reports with evolving
reliability, deadline losses, dependency changes, sensor failures, absent
deliveries, waiting, checking, switching, proceeding and stopping. Seeded
six-case held-out worlds combine factors not supplied as the same training
trajectories. Eight controls: untrained, fixed clock, learned prediction
without refinement, learned-focus one-pass, time-triggered refinement,
predictive-value refinement, compute-matched one-pass and repeated
interpretation without useful new evidence. Per-decision memory candidate
inspection is bounded to 32 headers. After discovering that an early
implementation gave refinement a stronger final scoring override than the
one-pass comparator, the latter was upgraded to the **same
value-ranking authority**; the earlier exploratory 30-seed advantage is
invalidated and not included in the inferential result.

**Corrected replication, 30 independent seeds:** Each seed contains six
held-out scenarios per treatment at each training stage (180 paired scenarios
per stage; 4,320 treatment/scenario rows in total). In the following pairs,
the first number is predictive-value refinement and the second is the
compute-matched one-pass comparator. Paired reward intervals are
normal-approximation 95% intervals from 30 seed-level differences and are
not preregistered or a proof of generalization.

| Training experiences | Mean reward full / one-pass | Mean completion full / one-pass | Full − one-pass reward [approx. 95% interval] |
| --- | --- | --- | --- |
| 0 | 0.3547 / 0.3593 | 0.1722 / 0.1778 | −0.00456 [−0.02929, +0.02018] |
| 20 | 0.4712 / 0.4667 | 0.5722 / 0.5778 | +0.00450 [−0.00393, +0.01293] |
| 100 | 0.5339 / 0.5239 | 0.6222 / 0.6278 | +0.00994 [+0.00441, +0.01547] |

At 100 training experiences the full mechanism made 977 live refinement
action changes over the 180 held-out scenarios and averaged 11.97
accounted cognitive operations, compared with 13.46 for compute-matched
one-pass. These totals measure the harness's counted interpretation cycles
rather than calibrated neural/CPU FLOPs; paths differ as actions cause
different subsequent observations. Mean inspected headers were 209.24
versus 215.29 across trajectories, not per retrieval, and detailed events
39.23 versus 40.37. Mean unsuccessful/unavailable inspections per trial
were 0.0111 in both conditions. At 100 experiences the mean presence
Brier score was 0.0687 versus 0.0716. Both shared a temporal duration
forecast MAE of 2.002 (versus 2.375 at zero experience). Neither policy
fabricated `verified_absent` status in the measured runs. These are
descriptive metrics; no independent attribution analysis of action changes
was completed.

**Distraction replication:** The independent five-seed, 500-distractor
challenge yielded full-minus-one-pass rewards +0.00150
[−0.00671, +0.00971] after 20 training experiences and +0.01767
[−0.00408, +0.03942] after 100, both including zero. This is too small
a replication to establish resistance to greater memory load. Increasing
training experiences helped both methods, so learning and the marginal
effect of refinement must be kept conceptually separate.

**Interpretation and negative findings:** 100-experience cumulative reward
modestly favored value-refinement on this particular benchmark, with a
seed-cluster interval excluding zero. However **goal completion was slightly
worse at all three checkpoints** than the strong one-pass control, and
benefits failed the small distractor replication. Therefore E10 does not
demonstrate a general or robust superiority of additional thought. The
strong one-pass condition uses the same computed option values and can
choose directly without producing thought revisions. The compute-match is
approximate because actual trajectories and operation counts differ,
and the world has a deliberately small grammar. No human cognition,
consciousness, generic temporal understanding, or production readiness
may be inferred.

**Next scientifically meaningful test:** Freeze learned forecasts and
retrieved evidence per decision, randomize whether the additional
value-refinement pass executes, and pair the exact same decision state with
common-random-number environment continuations. Measure causal reward
delta for *individual decisions changed by refinement*, subtract measured
CPU/evidence-processing cost, and introduce an affordable observation with
a delayed but decisive outcome. This discriminates whether actual revisable
inference, versus merely improved learned forecasts or using the same
decision values in one pass, generates net future benefit.

**Verification:** E10 focused and legacy E9 temporal gates and the
independent 30-seed replication passed in GitHub Actions at the tested
implementation. Independent-process restart, immutable source evidence,
censored direct checks and source provenance are covered by E10 tests.
Cumulative E6–E9 and full-suite completion is tracked by the GitHub
research workflow for the branch; passing code tests must not be
conflated with scientific support for refinement superiority.


## Experiment 11 — incremental cognition from minimally changing, once-seen experiences (2026-10-10)

**Question:** Does a persistent cognitive system learn context-dependent consequences from every distinct observation, without training again on identical scenarios, and does this produce better *new* decisions than equally informed one-pass inference or episodic case memory?

**Implementation ownership:** Research branch `experiment/incremental-refinement-e11` and [draft PR #5](https://github.com/Logancarton/Scenario-Chunking/pull/5) continue E10's verified owners. `IncrementalDispatchWorld` contains the only private causal outcomes. Each world executes once. `CognitiveState` holds the currently observed context, the E10 `EpisodicMemory` stores every immutable observation/attempt/outcome and causal predecessor link, `MemoryRouter` retrieves bounded relevant chunks with one-hop causal spreading, `OutcomeLearner` updates existing contextual outcome counts and confidence from *observed* action feedback, `MeaningInterpreter` produces calibrated, provenance-bearing conditional action values, `ProgressiveRefiner` records evidence-grounded thought revisions, and `ActionSelector` remains final authority. No parallel memory, selector or learning system was introduced. Production Sonara is unchanged.

**Experience design:** Three independent domain families, each with a 32-step Gray-code sequence of public person, delay, urgency, capacity and channel variables. The adjacent sequence changes exactly one of those five substantive factors. A marker changes without affecting private causal truth. A documented protocol epoch changes in one family, permitting old evidence to remain historically valid but need new applicability. The environment flips some binary potential outcomes with a deterministic nine-percent probability. Each chosen action has a deterministic paired potential outcome across comparator runs; the agent sees only the outcome for its *executed* action. Twenty-four experiences per family precede eight prequential held-out choices (each held-out choice is scored before its own feedback and may inform later distinct choices). Five random seeds are independent replications, not repeated training epochs within an agent.

**Controls and causal ablation:** Frozen/no generalized learning; episodic nearest memories without generalized relationships; strongly matched one-pass conditioned on the *same learned aggregates and values*; full value-sensitive risk reconsideration; and the same full mechanism with interaction cues disabled. The ablation removes two-variable conditionals while preserving basic marginals and the episodic record, distinguishing relationship inference from mere history accumulation. All report reward, binary success, Brier, regret, bounded retrieval costs, thought-accounting operations, and context-local updates. Restart serialization is checked in a separate Python process. Explicit tests exercise observed-world uniqueness, noise/irrelevant marker invariance, signaled reversal, evidence provenance and unrelated belief nonmutation.

**Five-seed research replication, held-out prequential window, 120 choices per treatment:**

| Control | Mean reward | Success | Chosen-action Brier | Mean observed detail events | Mean header evaluations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen | −0.1367 | 44.17% | 0.2500 | 16 | 60 |
| Episodic memory only | −0.1220 | 45.83% | 0.2419 | 16 | 60 |
| One-pass conditional | −0.1653 | 43.33% | 0.2500 | 16 | 60 |
| Full conditional refinement | −0.1727 | 42.50% | 0.2362 | 16 | 60 |
| No two-variable interactions | −0.1367 | 44.17% | 0.2503 | 16 | 60 |

These are observed results from the GitHub E11 replication job following the initial retrieval correction. A first pass incorrectly deprived the episodic baseline of causal outcome retrieval; after adding one-hop bounded spreading and better procedure-focused routing, episodic held-out mean reward improved from −0.1367 to −0.1220, without giving the generalized learner different world truth. This error and repair are part of the research record. New negative controls were added afterward and remain subject to the final CI revision.

**Verdict and falsifiable weakness:** Persistent incremental updates and supported beliefs work mechanically. This is not evidence that new contexts cause better decisions. The full treatment underperforms stronger simpler controls on actual held-out reward and completion, while selected-action calibration improves modestly versus one-pass. Interaction ablation produces at least as much reward, so improved decisions **cannot** be attributed to learned two-variable relationships. A single observed outcome updates 36 overlapping cues: feature co-occurrence is not yet causal discrimination. The observational bandit policy can select which outcome it sees, generating biased training evidence. A prequential held-out window also differs from a strictly frozen model transfer test. The most valuable next test is to add conservative one-variable contrast attribution from adjacent *observed* action outcomes, with explicit residual uncertainty, and independently evaluate untrained combinations with stable action-specific counterfactual scoring. Do not patch environment answers into the learned model.

**Verification and maturity:** Built, Integrated and Enabled only in the research runner. Target CI: E11 focused tests and independent-process restart, the E6–E10 cumulative suite, independent five-seed replication, compilation and diff integrity. Completed gate status is determined by the linked GitHub Actions run; a unit or restart pass alone is not a full research gate. No general-human-cognition claim, no evidence for promoting this mechanism to Sonara production.

## Experiment 12 — persistent, revisable explanations predicting never-experienced combinations (2026-10-11)

**Question:** Can Sonara experience something, form an explanation, experience something different, revise *that same* explanation, and correctly predict a combination it has never experienced — producing better decisions on genuinely new situations than episodic case memory?

**Protocol changes from E11:** Before any outcome, eight of each family's 32 factor combinations are withheld (`curriculum`). Every held-out decision is made from a deep copy of the *same frozen* post-training state with learning disabled, so held-out labels neither teach nor reactivate memory (E11's final-eight window kept learning). The marker is randomized and balanced within each split and epoch. Coordination's two epochs each span the full factorial space, so epoch cannot proxy channel. Five design seeds (7, 23, 61, 101, 149); 160 held-out decisions per condition.

### Historical treatment 1 — contrast fragments (negative; formerly labelled E12a)

Codex added a persistent per-family `ThoughtThread`, a bounded same-action/same-scope contrast diagnosis (`ProgressiveRefiner.diagnose_causal`), and one- and two-literal `CausalHypothesis` fragments with retained support and contradiction provenance. Result: held-out reward −0.0375 vs episodic-only −0.0205 (Δ −0.017, interval [−0.120, +0.086]), and the continuity ablation produced *identical* decisions. Diagnosis of the trace: every condition chose `standard` on 90–95% of training cases, so almost no evidence about `express` existed. Every learner converged on "nearly always standard" (held-out −0.0575). Fragments also never form a complete explanation (e.g. `open AND (day OR Alex)`), so there is nothing to revise.

**Headroom check.** An idealized Bayesian learner that enumerates ~5,000 small DNF rules (ε = 0.09) reaches +0.116 held-out reward from uniformly explored *bandit* data (+0.144 with full feedback); the oracle is +0.345. The benchmark can show the effect; the E12a learner could not.

### Historical treatment 2 — one revisable explanation per action, with selector-owned exploration (formerly labelled E12b)

* **Exploration (plan item 5, done first because it is a precondition).** `ActionSelector._optimistic` adds a count-based UCB bonus κ·√(ln(1+N)/(1+nₐ)) per task scope, during training only. It is learner-agnostic, so every condition explores under the same rule; predictions stay unbiased. Held-out choices remain greedy. κ = 0.5 was fixed before measurement; κ ∈ {0.25, 1.0} were also run on E12a and did not rescue it.
* **Explanation (plan item 2).** `OutcomeLearner.explanation_feedback` keeps an `ExplanationBank` per (domain, epoch, action): a beam of ≤48 complete rules (DNF, ≤3 terms of ≤2 literals), each scored against ≤64 retained observations under an explicit noise model (ε = 0.1) with a complexity price. Each experience *edits the current best eight rules* (add term, drop term, specialize, generalize) using literals from the new case; contrasts (plan items 1 and 3) only seed edit material. Every rule carries its operator, originating evidence id, and support and contradiction ids. `explanation_prediction` is read-only. The `ThoughtThread` records each revised explanation as a new version.
* **Ablations (evidence-equal):** `no_continuity` rebuilds the explanation from scratch on every experience while keeping all evidence, which directly tests "revise the same explanation". `no_contrast` disables contrast seeding. `contrast_fragments` is the E12a learner.

**Primary result (design seeds, κ = 0.5, 160 frozen held-out decisions per condition):**

| Condition | Held-out reward | Success | All-action Brier | Training reward |
| --- | ---: | ---: | ---: | ---: |
| episodic_only | −0.0415 | 49.38% | 0.2364 | +0.0877 |
| strong_one_pass | −0.0510 | 48.75% | 0.2401 | +0.0488 |
| contrast_fragments (E12a) | −0.0510 | 51.25% | 0.2525 | +0.0613 |
| **explanation_full** | **+0.0530** | 56.25% | **0.2091** | +0.1207 |
| no_continuity | −0.0260 | 52.50% | 0.2538 | +0.0922 |
| no_contrast | +0.0990 | 58.75% | 0.2070 | +0.1315 |

**Fresh-seed replication (frozen design, 10 seeds never used during design, 320 decisions per condition):**

| explanation_full minus | Mean Δ reward | ≈95% interval | Seeds positive |
| --- | ---: | --- | ---: |
| episodic_only | +0.1155 | [+0.047, +0.184] | 9/10 |
| strong_one_pass | +0.0965 | [+0.010, +0.183] | 8/10 |
| contrast_fragments | +0.1550 | [+0.089, +0.221] | 10/10 |
| no_continuity | +0.1037 | [+0.012, +0.196] | 8/10 |
| no_contrast | +0.0282 | [+0.005, +0.052] | 6/10 |

Replication held-out reward: explanation_full +0.1725, episodic_only +0.0570, no_continuity +0.0688. All-action Brier: 0.2115 vs 0.2383 and 0.2716.

**Without exploration (κ = 0, design seeds):** explanation_full −0.0075 vs episodic −0.0205 (Δ +0.013, interval [−0.056, +0.082]). The explanation learner explores somewhat on its own (24% express), but the decision advantage needs deliberate evidence gathering. The continuity effect survives (Δ +0.047, [+0.023, +0.071], 5/5 seeds).

**Verdict:** First positive E-series result on genuinely unseen combinations. A persistent explanation that is *revised*, rather than re-derived or fragmented, predicts withheld contexts better (Brier 0.21 vs 0.24–0.27) and yields better frozen decisions than episodic memory, one-pass conditional values, and E12a. The effect replicates on fresh seeds. Removing continuity (same evidence, explanation rebuilt each time) removes most of the gain, which supports the "revise the same explanation" hypothesis specifically. **Contrast seeding is not supported:** its sign differs between design and replication seeds (−0.046, then +0.028), so plan items 1 and 3 add no demonstrated value beyond the revisable explanation.

**Limitations and threats:** (1) The beam design was prototyped on the design seeds, which is a forking-paths risk; the fresh-seed replication is the evidence to rely on. (2) Synthetic world with a known small-DNF rule class; the explanation language matches it, so this does not show discovery of structure outside that language. (3) Intervals are normal approximations over 5 and 10 seeds. (4) Training reward is not uniformly better (replication: +0.067 vs episodic +0.074) because exploration has a price. (5) Held-out reward is still well below the oracle (+0.345) and near the ideal-Bayes ceiling (+0.116 on design seeds). (6) Five-axis integration (plan item 4) has not been attempted; per the E12 objective it should follow, not precede, this result.

**Verification:** `tests/test_causal.py` covers forming, revising in place and predicting an unseen context; read-only prediction and replay idempotence; bounded beam and evidence; an evidence-equal continuity ablation; selector-owned, training-only exploration; a frozen held-out fork; potential-outcome isolation; and subprocess restart. Full suite: 125 passed locally (Python 3.14). `EpisodicMemory` id counters became plain integers because `itertools.count` no longer pickles on 3.14. Reproduce with `python -m scenario_chunking.causal_experiment`, which writes `results-e12/`, `results-e12/replication/` and `results-e12/no-exploration/`. Research runner only; Sonara production unchanged.

### Unified persistent cognitive learning — internal controls, matched evidence, adaptation (2026-10-11)

Branch `experiment/e12-unified-cognitive-learning`, started from `c69a51b` (identical on `main` and the E12 branch at start). This continues Experiment 12 inside the same framework; every comparison below is an internal E12 treatment, not a new experiment.

**Question.** Can one continuing learner build an explanation, revise it from individual experiences, keep still-valid knowledge, tell a contextual exception from a changed world, and apply what it learned to new situations? Which parts of the historical E12 gain are attributable to explanation persistence, thought identity, retrieval anchoring, search effort or contrast?

**Baseline reproduced before any change** (Python 3.11.17, macOS, seeds as published). Replication: episodic +0.0570 / 54.38%, one-pass +0.0760 / 55.00%, no_continuity +0.0688 / 57.19%, no_contrast +0.1443 / 60.94%, explanation_full +0.1725 / 62.50%. This is exact to four decimals. Full suite: 125 passed. After the refactor, all three historical result sets (primary, replication, no-exploration) are content-identical: same predictions, actions, rewards and update contents. The order of the diagnostic `proposed` list already varied with Python's string-hash seed. Golden digests of the original treatments are now tested (`test_original_e12_treatments_are_frozen_byte_for_byte`).

**Receiving owners (traced, caller to callee).** `CausalRun.act/deliver` → `CognitiveState` (context, time) → `ProgressiveRefiner.thread` (identity, anchors, attention) → `MemoryRouter.retrieve` (bounded, anchor-aware) → `MeaningInterpreter.incremental_action_values` → `OutcomeLearner.explanation_prediction` (`ExplanationBank`) → `ActionSelector.select` (exploration, refinement) → world execution → outcome event in `EpisodicMemory` → `ProgressiveRefiner.diagnose_causal` (contrast) → `OutcomeLearner.explanation_feedback` (the only owner that revises explanations) → `ProgressiveRefiner.record_causal_revision`.

Tracing showed the following before any measurement. In original E12 the `ThoughtThread` never influences a prediction: predictions come from the learner's bank keyed by (domain, epoch, action). The thread's only path to behaviour is anchors → retrieval → availability of a contrast. `refine_incremental` with `risk_price = 0` picks exactly the one-pass value argmax. The historical `no_continuity` control removed three things at once: thread identity, anchors and the persistent rule beam.

**Donors inspected (read-only):**
* Sonara `test_stage_12_7c_hypothesis_revision.py`, `cognition.py`, `state.py`, `contracts.py`. Adopted: supported/contradicted/superseded history retained, "one event never converges", "repetition is not evidence", owner-validated scoped revision. Rejected: the contract/ledger machinery.
* Sonara `test_active_thought_{continuity,residency}.py`, `test_thought_chain_identity.py`. Adopted: "decay is attention fading, not forgetting", analytic decay on elapsed time, and suspend/resume without deleting knowledge.
* sonara-clean `outcome_learning.py`, `da_rpe.py`, `credit_assignment.py`. Adopted: surprise = error × confidence. Rejected: neuromodulator EMA telemetry.
* Sonara-2 `neural/representation.py`, `population.py`. Adopted: deterministic k-winner competition with stable tie-break and local adaptation of winners. Rejected: LIF dynamics and the NumPy dependency, since no dependency was added.

**Architecture changes (all in existing owners; every new switch defaults off, which is exactly original E12):**
* `ExplanationBank` (learning.py):
  * **Durable evidence:** per-context sufficient statistics (bounded at 256 contexts, eviction counted) separate durable knowledge from the 64-item active provenance window. Duplicate ids are rejected beyond the window.
  * **Append-only explanation lineage**, keyed to the explanation held *before* the evidence. The historical `explanation_before`/`revised` fields are computed after the evidence is added and therefore miss re-scoring-only changes; they are kept unchanged and this limitation is documented here.
  * **Regime detection.** The trigger is prequential misses (wrong predictions made *before* each outcome) ≥ 5 of the last 12. The test is then a segmented likelihood: the best old rule plus the best recent rule must beat the joint rule by 8 nats, using a bounded 2-round refit on the recent window. The superseded regime is archived as the rule that best explained *its* evidence, never relabelled false, bounded to 4, and reactivated if it again explains recent evidence. A first version triggered on post-update fit; it failed because ordinary revision contorts a 3-term rule to fit both regimes. That failure is why the trigger is prequential.
  * **Directed review:** a confident wrong prediction recruits error-directed edits (exclude or include that context) and opens a review.
  * **Sparse recruitment:** k = 3 winners among the context literals, frozen or locally adapted toward marginal association.
  * Explicit `rules_scored` / `scoring_work` accounting. Malformed evidence (non-boolean outcome, empty id) raises before any mutation.
* `ProgressiveRefiner`/`ThoughtThread` (refinement.py): open reviews carried by identity (bounded at 2, patience of 8 experiences, resolved/abandoned/displaced history), chronology (`last_outcome`), a read-only `attention_status`, and simulated acquisition time instead of processing ticks.
* `MeaningInterpreter.contextual_literals`: sequence (previous procedure and outcome of the continuing thought) and elapsed-time literals. They are offered as candidate evidence and enter an explanation only if the evidence-scored rules adopt them.
* `Timeline` (state.py): an injectable, monotonic simulated clock, separate from experience count. `CognitiveState.apply_signal` still advances a processing tick for older experiments; E12 no longer uses that tick as time.
* `IncrementalDispatchWorld`: private noiseless rule separated (`causal_rule`, `counterfactual_truth`, evaluation-only), plus `scripted` challenge worlds. Existing outcomes are unchanged.
* `CausalRun`: one path, with `act` and `deliver` split (delayed feedback, event time separate from acquisition time). Matched-evidence `behavior`. `Treatment` switches.
* Harness: `causal_challenges.py` contains streams and probes only.

**Internal controls (one factor at a time from `explanation_full` unless named):** `beam_reset` (structure rebuilt each experience, identity and anchors kept), `fresh_search` (rebuilt with 3 search rounds, more compute), `no_identity` (no continuing thread, hence no anchors; beam persists), `no_anchor` (identity kept, anchors not used), `no_contrast`, `durable`, `regime`, `directed`, `meaning`, `decay`, `sparse_frozen`, `sparse_adaptive`. Unified variants: `unified`, `unified_no_identity`, `unified_no_anchor`, `unified_plus_{directed,sparse,meaning}`. The six original conditions are kept frozen.

**Protocols and seed separation.**
* *Matched evidence:* a fixed hash-uniform behavior policy acts during training, so all learners receive the identical public context, executed action, outcome, evidence id and chronology. Unexecuted outcomes never enter learning (tested by flipping them).
* *Online:* each agent acts with the shared UCB exploration (κ = 0.5).
* *Challenges:* matched-evidence streams with private truths, 16–17 treatments each.

Development used design seeds 7–149 and challenge seeds 11–29 only. The unified configuration was chosen on those, then frozen. **Final seeds 1301–2213 (10) were run once.** Intervals are seed-paired percentile bootstraps (deterministic, 2000–4000 resamples). Two harness bugs were fixed before the final run: the `long_gap` truth was computed after the family tag had been removed, and the `sequence` probe scored the wrong step. A `meaning` crash (outcome stamped one processing tick ahead of the clock) was also fixed before the final run.

**Mechanism selection on development data.**
* Regime detection was supported: abrupt +0.25 and gradual +0.33 final accuracy, 0 false flags, 0 flags on exceptions.
* Durable evidence helped abrupt change but *hurt gradual change when used alone* (−0.19). It is enabled only together with regime detection.
* Directed review, meaning, decay and sparse recruitment were not supported.

**Unified = explanation_full + durable evidence + regime detection.**

#### Final results — E12 benchmark (10 fresh seeds, 320 frozen held-out decisions per condition)

| Condition | Matched reward | Matched all-action Brier | Online reward | Online success | Online all-action Brier | Rules scored / experience |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| episodic_only | +0.0727 | 0.2388 | +0.0992 | 56.56% | 0.2379 | 0 |
| strong_one_pass | +0.0597 | 0.2388 | +0.0780 | 55.62% | 0.2460 | 0 |
| explanation_full (original E12) | +0.0295 | **0.2356** | +0.0838 | 57.81% | **0.2093** | 328 / 323 |
| no_continuity (historical bundle) | +0.0760 | 0.2645 | +0.0483 | 55.94% | 0.2698 | 47 |
| beam_reset | +0.0760 | 0.2645 | +0.0483 | 55.94% | 0.2698 | 47 |
| fresh_search (≈1.6× compute) | +0.0268 | 0.2578 | +0.0685 | 56.88% | 0.2564 | 518 / 538 |
| no_identity / no_anchor | identical to explanation_full | — | identical | — | — | 328 |
| unified (and both continuity variants) | identical to explanation_full | — | identical | — | — | 328 |
| unified_plus_directed | +0.0260 | 0.2308 | +0.1165 | 59.38% | 0.2032 | 329 |
| unified_plus_sparse | +0.0070 | 0.2604 | +0.0845 | 58.13% | 0.2229 | 215 |
| unified_plus_meaning | +0.0348 | 0.2517 | +0.0388 | 55.94% | 0.2343 | 449 |

Paired (explanation_full minus X), 10 seeds:
* **Online reward vs episodic:** −0.016 [−0.059, +0.031] (5/5). The historical +0.116 reward advantage **does not replicate** on fresh seeds.
* **Online all-action Brier vs episodic:** −0.029 [−0.046, −0.011]. Better predictions do replicate.
* **Persistence (vs beam_reset):** Brier −0.029 [−0.048, −0.008] matched and −0.061 [−0.092, −0.032] online. Explanation truth accuracy 0.757 vs 0.670 (matched).
* **Persistence and reward:** matched reward −0.047 [−0.115, +0.022] and online +0.036 [−0.017, +0.094]. Not significant either way.
* **Thought identity and anchoring:** exactly 0.000 on every metric in both protocols.
* **Contrast:** matched reward −0.009 [−0.024, 0.000] and online +0.020 [−0.018, +0.066]. Unsupported, as historically.
* **Directed review (online):** +0.033 [−0.004, +0.077], 5 positive / 1 negative seeds. The direction matches development (+0.059) but the interval includes zero.

Historical gate re-run (`write_all`) reproduces the published E12 numbers exactly.

#### Final results — controlled challenges (10 fresh seeds; explanation-family probes are read-only)

| Challenge (metric) | one-pass | beam_reset | explanation_full | unified | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Contextual exception: exception-context accuracy / retention of original context | 0.975 / 0.675 | 0.931 / 0.744 | **1.000 / 1.000** | **1.000 / 1.000** | 0 regime flags; adapts in 11 steps |
| Same with feedback delayed by 6 experiences | 0.975 / 0.675 | 0.931 / 0.744 | 1.000 / 1.000 | 1.000 / 1.000 | adapts in 17 steps |
| Abrupt change A→B→A (final accuracy) | 0.709 | 0.594 | 0.534 | **0.922** | unified − full +0.388 [+0.300, +0.484], 10/10 |
| Steps to ≥ 80% after change 1 / change 2 | 549 / 519 | 536 / 506 | never in window | **31 / 34** | historical regime accuracy 0.934 |
| Gradual change (final accuracy) | 0.412 | 0.494 | 0.459 | **0.847** | +0.388 [+0.297, +0.469], 10/10; durable alone 0.384 |
| No-change control (false regime flags; explanation changes) | — | 0 ; 124 | 0 ; 11.2 | **0 ; 10.8** | |
| Random unseen structures, held-out combinations | 0.669 | 0.631 | 0.963 | 0.963 | structures drawn per seed |
| Representable 2-term DNF / unrepresentable 3-parity | 0.766 / 0.487 | 0.703 / 0.500 | 0.988 / 0.738 | 0.994 / 0.713 | parity is a representation limit |
| Sequence-dependent rule (final; prequential agreement) | 0.731 ; 0.843 | 0.681 ; 0.758 | 0.756 ; 0.848 | 0.688 ; 0.849 | meaning: **1.000 ; 0.978** (+0.244, 9/0) |
| Salient random distractor (accuracy; unsupported literals) | 0.786 | 0.925 ; 0.3 | 0.988 ; 0.5 | 0.988 ; 0.5 | |
| Long gaps (years between blocks): accuracy α/β | 0.709 / 0.778 | 0.844 / 0.747 | 0.931 / 0.978 | 0.931 / 0.978 | decay changes anchors 158→144, headers unchanged |

Mechanism-level results on the challenges:
* **Durable alone:** abrupt +0.237 [+0.147, +0.328] but gradual −0.075 [−0.191, +0.016]. This is the negative interaction seen in development.
* **Unified costs:** about 12% more rule scoring than explanation_full (46.6k vs 42.2k on the exception stream). Parity −0.025 [−0.072, +0.003] and sequence −0.069 [−0.175, +0.019] are inconclusive.
* **Meaning:** useful only where order matters (sequence +0.244, 9/0). It costs elsewhere (unified_plus_meaning parity −0.119 [−0.178, −0.066]; matched benchmark Brier +0.016) and adds about 35% scoring work.
* **Sparse adaptive:** 28–38% less scoring work. Accuracy is within ±0.03 on most streams, but parity −0.056 with unified and benchmark Brier is worse.
* **Directed review:** opens about 12 reviews per stream and carries them across experiences, with no measurable effect on any challenge.

**Answers.**
1. *Form and evolve a continuing explanation:* **yes**, and lineage shows revision in place. The **persistent rule beam is the active ingredient**: removing it is the worst explanation treatment on every challenge and on Brier in both protocols. Fresh search with ~1.6× compute does not substitute for it.
2. *Correct without destroying valid knowledge:* **yes** for contextual exceptions (100% retention vs 67–74% for one-pass and beam reset). The durable-evidence test shows the original window learner forgets after 64 observations while durable evidence keeps the relationship.
3. *Thought identity independently:* **no measurable effect** in any protocol or challenge (identical to 9 decimals). The historical "continuity" gain belongs to beam persistence. Identity carries reviews and chronology, but neither produced a benefit.
4. *Prediction-error-directed refinement:* as a regime trigger (prequential surprise), **yes**: it is what makes change detection work. As directed per-rule review, **no demonstrated benefit beyond the existing likelihood update** (online +0.033, inconclusive).
5. *Exception vs changed world:* **yes, on these streams.** 0 regime flags on exceptions and on 180-step no-change noise; 2/2 changes flagged on abrupt; about 31 steps to adapt instead of never within the window. History is retained (93% accuracy of the archived regime). One reactivation occurred on development seeds and is proven in a test; archived rules are often contorted, so reactivation is not reliable.
6. *Time-sensitive attention:* attention and knowledge are separated (one minute vs one year apart gives identical learned state; dormant threads reactivate on relevant cues; unrelated experiences change nothing). **No accuracy or efficiency gain**, because anchors compete inside the same header budget.
7. *Sparse recruitment:* **efficiency, not generalization**: 28–38% fewer rule evaluations at similar accuracy on most streams, worse calibration on the benchmark. Not enabled.
8. *Integrated learner vs original on unseen conditions:* **identical on the E12 benchmark**, so no regression and no gain. **Large gains under nonstationarity** (+0.39 on abrupt and gradual change), **equal** on exceptions, unseen structures, distractors and long gaps, and slightly worse (inconclusive) on parity and sequence.
9. *Failed to improve:* thought identity, retrieval anchoring, time-decay attention, directed review, contrast seeding, meaning in general (helps only on sequence), sparse recruitment (accuracy), and durable evidence without change detection.
10. *Strongest limitation:* the decision-reward advantage of explanation learning over episodic memory does not replicate on fresh seeds even though prediction quality does. Second limitation: the hypothesis language (≤3 terms of ≤2 literals; parity at 0.71–0.74).

**Cognitive costs.** Explanation learners score about 320–330 rules per experience (beam_reset 47, fresh_search about 520) and store about 160 KB (matched) to 300 KB (online) of pickled learner+thought state per run, vs 2 KB for episodic counters. Retrieval headers and detail events are identical across explanation treatments; anchors do not reduce retrieval work. Unified adds no scoring work on the benchmark (327.7 = 327.7 rules per experience) and about 10–12% on challenge streams, from regime tests and refits.

**Completion status.**
* *Built + Integrated + Enabled:* durable evidence, regime detection with archive and reactivation, explanation lineage, injectable timeline, timeline-based thread chronology.
* *Built + Integrated (controls, not enabled):* directed review, meaning literals, decay-gated anchors, sparse recruitment, the identity/anchor/beam/search controls, matched protocol, delayed delivery.
* *Restart verified:* the unified run restores in a fresh process and learns the next experience with the same thought identity.
* *Full gate:* local (145 tests; historical gates). CI status is recorded with the commit.

**Limitations.** Synthetic worlds whose truths are mostly within the DNF language; 10 final seeds; bootstrap intervals over seeds. Regime parameters (window 12, 8 nats) were set on development seeds. Detection requires about 12 contradicting observations of the *same action*. The archived rule is a beam member and can be contorted. The `regime_reactivated` path fired once in development and was not measured as a separate outcome. Meaning covers only sequence and elapsed time: person is already a public literal, salience is not causal evidence, and source reliability does not exist in this world. The benchmark's labelled epoch keeps regime detection inert there.

**Next research question.** Why does better-calibrated explanatory prediction (Brier −0.03, truth agreement +0.12) fail to yield better held-out decisions than episodic recall on fresh seeds? Is the bottleneck the conversion of explanation uncertainty into action value and exploration, or noise in a 320-decision reward estimate?

**Reproduce.**
* Historical gate: `python -m scenario_chunking.causal_experiment` writes `results-e12/`.
* Unified benchmark: `python -c "from scenario_chunking.causal_experiment import write_unified; write_unified()"` writes `results-e12-unified/{matched,online}`.
* Challenges: `python -m scenario_chunking.causal_challenges` writes `results-e12-unified/challenges`.
* Tests: `python -m pytest tests/test_causal.py tests/test_causal_unified.py`.
* CI reproduces the seed-1301 unified reference in `docs/e12_unified_reference.json` to 9 decimals on Linux.

Research runner only; the Sonara production repositories were read, not modified.

### Prediction-to-decision diagnostic — why better prediction did not reliably raise reward (2026-10-11)

Branch `experiment/e12-prediction-decision-diagnostic`, from `main` at `483be0c`. This is an evaluation harness inside Experiment 12, not a new experiment. No learning, valuation, selection or refinement code changed. Draft PR #8 was not touched.

**Question.** Unified E12 reported online all-action Brier 0.2093 vs 0.2379 for episodic memory, but held-out reward +0.0838 vs +0.0992 (paired −0.016 [−0.059, +0.031], 10 seeds). Is this a prediction-to-decision bottleneck, and if so, where?

**Signal-flow and decision-equation audit (from the code, before measuring).**
* `CausalRun.act` → `CognitiveState` → `MemoryRouter.retrieve` → `MeaningInterpreter.incremental_action_values` (explanation/one-pass) or `incremental_episodic_values` (episodic) → `ActionSelector.select` → `ProgressiveRefiner.refine_incremental` (explanation only) → world → `deliver` → `OutcomeLearner`.
* Valuation (`meaning.py`): V(a) = p − (1 − p) − cost(a), with costs 0.02/0.18. This is exactly the world's expected reward (±1 − cost). Express wins iff p_E − p_S > 0.08; for the explanation learner (p = 0.9q + 0.1(1 − q)) the posterior mass must differ by 0.10. Ties go to express (label order).
* Selector, both paths: the incremental context replaces the heuristic score with the value argmax (`one_pass_conditional_value` / `incremental_value_refinement`). The refiner computes V − risk_price·u·(1 − p) with `risk_price = 0`, so it chooses exactly the direct EV argmax. Its `meaningfully_changed_action` flag compares against the noisy clipped heuristic order, not the value ranking (it fired in about 5% of decisions without ever changing the EV choice).
* Inactive under E12: predictive uncertainty (multiplied by zero; UCB exploration is count-based per scope, not per context), thread identity/anchors (previously shown identical), regime detection (labelled epoch), durable evidence (≤ 24 observations per bank, window 64). Unified is therefore identical to explanation_full on this benchmark (verified: 0 differing predictions).
* Benchmark structure: express is optimal only when standard fails and express works, which is 24 of 128 contexts. In 66 of 128 contexts both actions share a truth value and the decision is worth only 0.16. Brier weights every action in every context equally.

**Design (`causal_diagnostic.py`).** All learners train in lockstep on one evidence stream. At each held-out checkpoint every decision is `CausalRun.act(case, learn=False)` on a deep copy of the frozen learner, exactly the historical procedure. Interventions replace only the interpreter's prediction source (`SubstitutedBeliefs`) on a frozen explanation host (refining path) and a one-pass host (direct path). Sources: each learner's own predictions; a causal oracle (P* = 0.91/0.09 from the private rule); a hindsight oracle (realized outcomes); partial oracles that correct only one action. Oracles are built after acting and never enter learner state. Decisions are scored by noiseless expected utility EU(a) = 2P*(a) − 1 − cost(a), alongside the historical realized reward. Streams: matched (uniform behavior policy), online (own UCB choices), and swap (each learner retrained on another learner's exact online action stream).

**Integrity controls (all passed).**
* Frozen-state digest unchanged at all 960 final checkpoints (+600 historical/dev).
* Replay reproduces the committed seed-1301 reference and the historical held-out choices exactly.
* 0 of 99,840 audited decisions departed from the value argmax; 0 host-invariance violations; 0 misattributed outcomes.
* A focused test proves that substituted predictions flip the action exactly at p_E − p_S = 0.08 on both selector paths.
* Oracle evaluation leaves continued learning identical to an untouched twin.
* Deterministic rerun; independent-process restart; replaying a learner's own online stream reproduces its state.

**Protocol and seeds.** Protocol, metrics and analysis rules were committed in `c220623` before any historical or final run. Dev: the 15 E12 design and replication seeds. Historical reanalysis: unified final seeds 1301–2213 (where the discrepancy was observed; not independent). Final: 40 never-used seeds 4001 + 37k, run once. Intervals are seed-paired percentile bootstraps.

#### Results — 40 independent final seeds (1,280 held-out decisions per learner and stream)

| | Matched episodic | Matched explanation | Online episodic | Online explanation |
| --- | ---: | ---: | ---: | ---: |
| Expected reward (oracle 0.292; always-standard 0.024) | 0.037 | **0.139** | 0.049 | **0.113** |
| Realized reward | 0.042 | 0.149 | 0.061 | 0.128 |
| Best action chosen | **79.5%** | 73.2% | **80.2%** | 71.5% |
| Express chosen (optimal 18.1%) | 7.0% | 36.2% | 6.7% | 36.2% |
| False express / missed express | 4.7% / 15.9% | 22.4% / 4.4% | 4.2% / 15.6% | 23.3% / 5.2% |
| All-action Brier | 0.2375 | **0.2068** | 0.2384 | **0.2051** |
| Noiseless squared error | 0.156 | 0.130 | 0.157 | 0.129 |
| AUC standard / express | 0.71 / 0.67 | 0.80 / 0.81 | 0.71 / 0.66 | 0.83 / 0.76 |
| Advantage error \|Â − A\| | 0.787 | 0.763 | 0.792 | 0.774 |

Paired explanation_full − episodic_only (40 seeds):
* **Expected reward:** matched **+0.102 [+0.068, +0.135]** (34/6 seeds); online **+0.064 [+0.037, +0.093]** (31/9).
* **Realized reward:** matched +0.106 [+0.056, +0.156]; online +0.067 [+0.025, +0.106].
* **Brier:** −0.031 and −0.033. Advantage error −0.024 [−0.058, +0.013] and −0.018 [−0.051, +0.015], neither significant.
* **Best-action accuracy:** −0.063 and −0.087. The learner with better reward chooses the best action *less* often.
* Same against strong_one_pass: +0.093 and +0.077.

Where the difference comes from (online; the matched stream is similar):
* By stakes: high-stakes contexts +0.089, low-stakes contexts −0.025.
* By family: coordination 0 and 1 +0.031 / +0.035, delivery +0.003, repair −0.004. Delivery improves prediction most (SE 0.123 vs 0.154) yet gains almost no reward. Its always-standard baseline is already 0.226, and only 10% of its contexts reward express.

Decision-change taxonomy (online):
* 68% of decisions with improved predictions leave the action unchanged.
* Improved and changed: 143 beneficial (+0.149) vs 124 harmful (−0.016).
* Worsened and changed: 13 beneficial (+0.010) vs 143 harmful (−0.079).
* Changed decisions are 43% high-stakes; they carry nearly all of the value.

Margin and calibration:
* Explanation errors are confident: best-action accuracy is only 0.79–0.81 at \|V_E − V_S\| ≥ 0.6, and 0.41–0.55 near the boundary.
* Reliability: a predicted 0.87 corresponds to P* ≈ 0.77–0.82 for standard and ≈ 0.68–0.71 for express; 0.13–0.14 corresponds to ≈ 0.15–0.23.
* Episodic predictions sit at about 0.5: 89–95% fall in the 0.4–0.6 bin. Its decisions are mostly the cheap default.

Interventions on the frozen explanation host (expected reward, paired vs its own predictions):
* Causal oracle: +0.153 (matched) and +0.179 (online), 40/40.
* Correcting only express: +0.112 / +0.143. Correcting only standard: +0.120 / +0.125.
* Episodic predictions in the same host: −0.102 / −0.064.
* Both actions' errors matter; neither alone is the bottleneck.

Experience (online swap):
* Learner effect with the evidence stream fixed: +0.090 [+0.063, +0.118] on the episodic stream and +0.073 [+0.043, +0.103] on the explanation stream.
* Stream effect with the learner fixed: −0.026 [−0.055, +0.002] for explanation (it does slightly *worse* on its own exploration) and +0.009 [−0.007, +0.025] for episodic.
* Online actions first diverge after a median of 4 experiences (range 2–29); 31% of training actions differ.
* Exploration history does not explain the discrepancy.

Cost: explanation 6.4 ms CPU per training experience vs 2.0 (episodic); 3.5 vs 2.2 ms per decision; 326 rules scored per experience; 232 KB vs 2 KB learned state.

Measurement:
* The per-seed SD of the paired expected-reward difference is 0.09–0.11, with outcome noise *already removed*; realized reward raises it to 0.13–0.16.
* Minimum detectable effect at 80% power: about 0.04 for expected reward with 40 seeds, vs about 0.08–0.10 with 10 seeds and about 0.14 for realized reward with 10 seeds.
* Seed-level correlation between prediction improvement and reward improvement is moderate: ΔSE vs ΔEU −0.34 (matched) and −0.56 (online).

**Historical reanalysis (seeds 1301–2213; not independent).**
* The harness reproduces the journal's figures exactly: online realized 0.0992 vs 0.0838; Brier 0.2379 vs 0.2093; matched 0.0727 vs 0.0295.
* Under expected reward, online explanation − episodic is **+0.016 [−0.024, +0.051]** (realized: −0.016). Outcome noise alone flipped the sign.
* Matched: −0.014 [−0.074, +0.054]. On these seeds the matched explanation learner barely improved prediction (SE −0.006), and its false-express rate was 28.7%.

**Post-hoc exploratory check (added after the final run; not pre-registered).** A per-action reliability map was fitted on dev seeds using evaluation-only truth and applied to the final seeds. It keeps discrimination and recalibrates probabilities. Result: decisions do not improve (matched +0.003 [−0.010, +0.016], online −0.006 [−0.021, +0.008]). Calibration is therefore **not** the decision bottleneck. On the historical matched seeds the same map helped (+0.048, 8/2), which is consistent with those seeds being atypically overconfident.

**Diagnosis.**
1. *Measurement problem (primary, supported).* The reported non-replication came from 10 seeds scored by one noisy realized outcome per decision. That benchmark could not detect effects below about 0.1. With noise-free expected reward and 40 fresh seeds, the explanation learner's decision advantage is clear in both protocols and also holds in realized reward.
2. *No valuation or selection defect (supported).* The value equation equals the world's expected reward. The selector and the refiner follow the value argmax in 100% of decisions, and substituted predictions move the decision exactly at the analytic boundary.
3. *The conversion is lossy (supported), for structural reasons.*
   * Brier improves on every context, but only about 18% of contexts reward deviating from the cheap default, and 52% are low-stakes.
   * About two-thirds of prediction improvements change nothing. Changed decisions split nearly evenly by count into beneficial and harmful, and value comes from the few high-stakes flips.
   * The explanation learner pays a consistent tax of false-express choices (−0.025 from low-stakes contexts plus about −0.08 from confidently wrong flips).
4. *Not calibration, not exploration (falsified as primary causes).* Post-hoc recalibration does not help, and the swap design shows the evidence stream is not responsible.
5. *Remaining gap: discrimination on unseen contexts.* Explanation_full still leaves 0.15–0.18 expected reward below the causal oracle. The partial oracles show that errors in *both* per-action explanations generalising to withheld combinations cause it (AUC about 0.8; the advantage estimate is no better than episodic's). Which representation change would fix that is a hypothesis, not a finding.

**Not established.** That any particular explanation-language or beam change would reduce the remaining false-express rate; generality beyond this synthetic two-action world.

**Reproduce.**
* `python -m scenario_chunking.causal_diagnostic {dev|historical|final}` writes `results-e12-diagnostic/<set>/` (summary, decisions, training, raw results).
* `exploratory_calibration(...)` regenerates the post-hoc check.
* Committed summary: `docs/e12_diagnostic_summary.json`.
* Tests: `tests/test_causal_diagnostic.py`.

## Working-memory activation — experimental candidate, unverified integration

*Written 2026-10-10 against the Experiment 3 code base; it was left uncommitted and merged onto the post-E12 `main` on 2026-10-11. Its hypothesis has not been tested since.*

This optional `activation` treatment preserves the original experiments and
five-axis meaning modes. Episodic event times and evidence remain unmodified.
Its bounded current-thought owner decays accessibility lazily using a global
clock, exposes focused/peripheral/dormant states, and competes for six active
bundle slots (two focused). Only evidence used by the actual chosen action
receives maintenance; specific matching cues can reinstate dormant chunks.
The existing outcome learner adapts persistence after actual observed feedback.
Activation is never a probability of truth, source confidence, or memory validity.

**Hypothesis, not demonstrated benefit:** Such selective maintenance might
improve goal completion/action efficiency relative to the structural fixed-decay
control at the same sparse top-K and distractor budget. Run the identical-seed
six-scenario benchmark plus adversarial controls; publish measured detailed
memory operations, latency, active counts, reactivations, action choices and
negative results. Do not promote to Sonara without live behavioral evidence.
