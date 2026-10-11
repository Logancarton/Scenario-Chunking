"""E12 orchestration: persistent explanations, frozen unseen combinations.

The existing owners retain memory, learning, diagnosis, valuation and choice.
This module only constructs public curricula, executes worlds and measures.

Every E12 treatment is a configuration of the SAME CausalRun path: one memory,
one explanation learner (OutcomeLearner.ExplanationBank), one thought owner
(ProgressiveRefiner), one interpreter and one selector. Switches exist only to
compare mechanisms fairly; they never select a different cognitive owner.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import statistics

from .environment import IncrementalDispatchWorld
from .incremental_experiment import IncrementalRun, AXES, VALUES, FAMILIES, COSTS
from .models import EventKind, TruthStatus
from .state import CognitiveState, Timeline
from .working_memory import WorkingMemoryActivation


@dataclass(frozen=True)
class Treatment:
    """Internal E12 control switches; all False/off is not a learner by itself."""
    learner: str  # episodic | one_pass | fragments | explanation
    persist: bool = True      # explanation structure (rule beam) persists
    identity: bool = True     # one continuing ThoughtThread per task family
    anchoring: bool = True    # thread anchors enter bounded retrieval
    contrast: bool = True     # contrast pairs seed edit material
    durable: bool = False     # durable evidence statistics beyond the window
    change: bool = False      # regime-change detection with archived regimes
    directed: bool = False    # prediction-error-directed review
    meaning: bool = False     # sequence/elapsed-time meaning literals
    decay: bool = False       # dormant (time-faded) threads do not anchor retrieval
    activation: bool = False  # bounded WM prioritizes genuinely used evidence
    recruit: str = "off"      # sparse literal recruitment: off | frozen | adaptive
    search_rounds: int = 1

    @property
    def causal(self) -> bool:
        return self.learner in ("fragments", "explanation")

    def explanation_options(self) -> dict:
        return {"durable": self.durable, "change_detection": self.change,
                "directed": self.directed, "recruit": self.recruit,
                "search_rounds": self.search_rounds}


_EXPLANATION = Treatment("explanation")
TREATMENTS: dict[str, Treatment] = {
    # Original E12 conditions (frozen; exactly the historical behavior).
    "episodic_only": Treatment("episodic", identity=False, anchoring=False, contrast=False),
    "strong_one_pass": Treatment("one_pass", identity=False, anchoring=False, contrast=False),
    "contrast_fragments": Treatment("fragments"),
    "explanation_full": _EXPLANATION,
    "no_continuity": replace(_EXPLANATION, persist=False, identity=False, anchoring=False),
    "no_contrast": replace(_EXPLANATION, contrast=False),
    # One-factor isolation of the original "continuity" bundle.
    "beam_reset": replace(_EXPLANATION, persist=False),
    "fresh_search": replace(_EXPLANATION, persist=False, search_rounds=3),
    "no_identity": replace(_EXPLANATION, identity=False, anchoring=False),
    "no_anchor": replace(_EXPLANATION, anchoring=False),
    # New mechanisms, each added alone to the original learner.
    "durable": replace(_EXPLANATION, durable=True),
    "regime": replace(_EXPLANATION, change=True),
    "directed": replace(_EXPLANATION, directed=True),
    "meaning": replace(_EXPLANATION, meaning=True),
    "decay": replace(_EXPLANATION, decay=True),
    "sparse_frozen": replace(_EXPLANATION, recruit="frozen"),
    "sparse_adaptive": replace(_EXPLANATION, recruit="adaptive"),
}
# Unified configuration: mechanisms selected on development evidence only
# (design seeds 7-149 and challenge seeds 11-29), frozen before FINAL_SEEDS.
# Durable evidence + regime detection were supported; durable alone slowed
# adaptation to gradual change, so it is enabled only with regime detection.
# Directed review, meaning literals, decay and sparse recruitment were not
# supported on development data and are measured as add-on controls.
UNIFIED = replace(_EXPLANATION, durable=True, change=True)
TREATMENTS.update({
    "unified": UNIFIED,
    "unified_activation": replace(UNIFIED, activation=True),
    "unified_no_identity": replace(UNIFIED, identity=False, anchoring=False),
    "unified_no_anchor": replace(UNIFIED, anchoring=False),
    "unified_plus_directed": replace(UNIFIED, directed=True),
    "unified_plus_sparse": replace(UNIFIED, recruit="adaptive"),
    "unified_plus_meaning": replace(UNIFIED, meaning=True),
})

# contrast_fragments is the historical fragment learner. explanation_* use one
# persistent, revisable rule explanation per action. CONDITIONS is the frozen
# original E12 gate; ALL_CONDITIONS adds the internal unified-E12 controls.
CONDITIONS = ("episodic_only", "strong_one_pass", "contrast_fragments",
              "explanation_full", "no_continuity", "no_contrast")
ALL_CONDITIONS = tuple(TREATMENTS)
REFERENCE = "explanation_full"
EXPLANATION = tuple(k for k, t in TREATMENTS.items() if t.learner == "explanation")
PRIMARY_EXPLORATION = 0.5


def curriculum(seed: int, family: str, epoch: int) -> tuple[list[dict], list[dict]]:
    """Reserve exact substantive combinations BEFORE outcomes or fitting.

    Every epoch spans the full factorial space: epoch cannot be a surrogate
    for channel. Marker assignment is randomized and balanced within each
    training/evaluation split and epoch, independently of private outcomes.
    """
    rng = random.Random(f"e12-split:{seed}:{family}")
    indices = list(range(32))
    rng.shuffle(indices)
    heldout = set(indices[:8])
    groups = ([i for i in range(32) if i not in heldout], sorted(heldout))
    result = []
    for stage, group in zip(("train", "heldout"), groups):
        markers = ["blue", "orange"] * (len(group) // 2)
        random.Random(f"e12-marker:{seed}:{family}:{epoch}:{stage}").shuffle(markers)
        cases = []
        for i, marker in zip(group, markers):
            gray = i ^ (i >> 1)
            factors = {axis: values[(gray >> j) & 1]
                       for j, (axis, values) in enumerate(zip(AXES, VALUES))}
            factors["marker"] = marker
            cases.append({"family": family, "epoch": epoch, "index": i,
                          "factors": factors, "stage": stage})
        result.append(cases)
    return result[0], result[1]


def behavior_action(seed: int, family: str, epoch: int, index: int) -> str:
    """Matched-evidence data-collection policy: fixed, learner-independent, uniform."""
    digest = hashlib.sha256(f"e12-behavior:{seed}:{family}:{epoch}:{index}".encode()).digest()
    return "express" if digest[0] & 1 else "standard"


class CausalRun(IncrementalRun):
    def __init__(self, seed: int, mode: str, *, exploration: float = 0.0,
                 treatment: Treatment | None = None, gap_threshold: float = 30.0):
        if treatment is None and mode not in TREATMENTS:
            raise ValueError("unknown E12 treatment")
        self.treatment = treatment or TREATMENTS[mode]
        super().__init__(seed, "episodic_only" if self.treatment.learner == "episodic"
                         else "strong_one_pass")
        self.mode = mode
        if self.treatment.learner == "explanation":
            self.learner.explanation_options = self.treatment.explanation_options()
        # Same selector-owned exploration rule for every learner, training only.
        self.exploration = float(exploration)
        self.clock = Timeline()
        self.gap_threshold = gap_threshold
        # Compose with the SAME bounded SPREADING router; no parallel retrieval
        # authority or change to historical E12 unless explicitly enabled.
        self.working_memory = (WorkingMemoryActivation(capacity=6, focused_capacity=2)
                               if self.treatment.activation else None)
        if self.working_memory is not None:
            self.router.working_memory = self.working_memory
            self.router.activation_persistence = self.learner.activation_persistence

    def learner_cues(self, primitive: frozenset[str], thread, now: float) -> frozenset[str]:
        """Public cues plus interpreted chronology (only when meaning is enabled)."""
        if not self.treatment.meaning:
            return primitive
        last = thread.last_outcome if thread is not None else None
        return primitive | self.interpreter.contextual_literals(
            last, now, gap_threshold=self.gap_threshold)

    def act(self, case: dict, *, learn: bool = True, behavior: str | None = None) -> dict:
        """Perceive, retrieve, predict and act once. Feedback is a separate delivery."""
        t = self.treatment
        family, epoch, index = case["family"], case["epoch"], case["index"]
        identity = (family, epoch, index)
        if identity in self.seen:
            raise ValueError("cannot repeat a scoped context")
        self.seen.add(identity)
        if "truths" in case:
            # Controlled challenge world; truths stay inside the environment.
            world = IncrementalDispatchWorld.scripted(
                seed=self.seed, family=family, index=epoch * 100000 + index,
                factors=case["factors"], truths=case["truths"],
                noise=case.get("noise", .08), epoch=epoch)
        else:
            world = IncrementalDispatchWorld.scenario(
                seed=self.seed, family=family, epoch=epoch,
                index=10000 + epoch * 32 + index, factors=case["factors"])
        primitive = world.public_cues()
        # Preserve E11's actual predictor for the baseline, including cue
        # interactions; all conditions receive identical public information.
        cues = primitive | frozenset(a + "&" + b for a in primitive for b in primitive if a < b)
        state = CognitiveState(f"e12:{self.seed}:{family}:{epoch}:{index}",
                               goal_id="response_dispatch")
        if t.causal and t.identity:
            state.thought_key = "dispatch:" + family
        now = self.clock.advance(elapsed=case.get("elapsed", 1.0))
        state.incremental_context(primitive, now)
        attention = self.refiner.attention_status(state) if t.causal else "absent"
        self.learner.begin_episode()
        self._event(state, kind=EventKind.OBSERVATION, status=TruthStatus.FACT,
                    content="Received public dispatch context", tags=cues)
        thread = self.refiner.thread(state) if t.causal else None
        learner_cues = self.learner_cues(primitive, thread, now)
        stale = t.decay and attention == "dormant"
        anchors = (tuple(thread.causal_anchors.values())
                   if thread is not None and t.anchoring and not stale else ())
        actions = state.candidate_actions()
        hits, stats = self.router.retrieve(state.retrieval_query(), anchor_event_ids=anchors)
        # Materialize only already-routed records. Diagnosis does not rescan history.
        activated = [event for hit in hits for event in self.memory.chunk_events(hit.chunk_id)]
        if t.learner == "episodic":
            interpreted = self.interpreter.incremental_episodic_values(hits, cues, actions, COSTS)
        else:
            interpreted = self.interpreter.incremental_action_values(
                learner_cues if t.causal else cues, actions, costs=COSTS, causal=t.causal,
                explanation=t.learner == "explanation")
        values, predictions = interpreted["values"], interpreted["predictions"]
        if behavior is not None:
            # Matched evidence: the fixed data-collection policy acts, not cognition.
            chosen = next(a for a in actions if a.target == behavior)
            decision = None
        else:
            context = {"cognitive_mode": "incremental", "candidate_values": values,
                       "predictions": predictions, "risk_price": 0.0,
                       "reason": "revisable_conditional_hypotheses"}
            if learn and self.exploration > 0:
                context["exploration"] = {"bonus": self.exploration,
                                          "scope": "|".join(self.learner.causal_scope(primitive))}
            decision = self.selector.select(
                state, actions, hits, refine=t.causal,
                transfer_scores={k: v / 2 for k, v in values.items()},
                transfer_context=context)
            chosen = decision.action
        # Only actively selected reasoning sources count as meaningful use.
        # A memory merely appearing in top-K does NOT refresh activation.
        used_provenance = []
        if self.working_memory is not None and decision is not None:
            supporting_ids = set(predictions[chosen.label].get("evidence_ids", ()))
            used_provenance = [
                hit.chunk_id for hit in hits
                if supporting_ids.intersection(self.memory.chunks[hit.chunk_id].event_ids)
            ][:2]
            if used_provenance:
                self.working_memory.use(
                    used_provenance, state.goal_id, now,
                    known_headers={cid: self.memory.chunks[cid].semantic_tags
                                   for cid in used_provenance},
                    persistence=self.learner.activation_persistence)
        attempt = self._event(state, kind=EventKind.ACTION, status=TruthStatus.ATTEMPT,
                              content="Dispatch " + chosen.label, tags=cues, action=chosen)
        observed = world.execute(chosen)
        return {"case": case, "world": world, "state": state, "cues": cues,
                "primitive": primitive, "learner_cues": learner_cues, "actions": actions,
                "hits": hits, "stats": stats, "activated": activated, "anchors": anchors,
                "attention": attention, "predictions": predictions, "decision": decision,
                "chosen": chosen, "attempt": attempt, "observed": observed,
                "acted_at": now, "learn": learn, "used_provenance": used_provenance}

    def deliver(self, pending: dict) -> dict:
        """Observe the attributable outcome once, then learn exactly once from it."""
        t = self.treatment
        state, chosen, observed = pending["state"], pending["chosen"], pending["observed"]
        world, actions, predictions = pending["world"], pending["actions"], pending["predictions"]
        learn = pending["learn"]
        acquired = self.clock.now  # simulated time the outcome becomes known
        delayed = acquired > pending["acted_at"]
        if delayed:
            state.current_time = acquired
        forecast = predictions[chosen.label]
        outcome = self._event(state, kind=EventKind.OUTCOME, status=TruthStatus.OUTCOME,
                              content=observed.content, tags=observed.semantic_tags,
                              outcome=observed.outcome, cause=pending["attempt"].event_id,
                              reward=observed.reward, metadata=observed.metadata,
                              occurred=pending["acted_at"] if delayed else None)
        state.apply_signal(observed, event_id=outcome.event_id)
        state.record_action(chosen)
        success = observed.outcome == "success"
        diagnosis, revision, update = None, None, {"updated": []}
        if learn:
            if t.causal:
                diagnosis = self.refiner.diagnose_causal(
                    state, outcome, pending["activated"], probability=forecast["probability"])
                if t.learner == "explanation":
                    reviews = (self.refiner.pending_reviews(
                        state, scope=self.learner.causal_scope(pending["learner_cues"]),
                        procedure=chosen.target) if t.directed else ())
                    update = self.learner.explanation_feedback(
                        pending["learner_cues"], chosen.target, success=success,
                        evidence_id=outcome.event_id, contrast=diagnosis["contrast"],
                        allow_contrast=t.contrast, persist=t.persist, reviews=reviews)
                else:
                    update = self.learner.causal_feedback(
                        pending["primitive"], chosen.target, success=success,
                        evidence_id=outcome.event_id, contrast=diagnosis["contrast"],
                        allow_contrast=t.contrast)
                revision = self.refiner.record_causal_revision(state, update, outcome, now=acquired)
                # Retrieval only receives credit when the learner actually uses
                # the retrieved contrast to seed a rule revision.
                if self.working_memory is not None and update.get("contrast_used"):
                    contrast = (diagnosis or {}).get("contrast") or {}
                    cid = self.memory.event_to_chunk.get(contrast.get("evidence_id"))
                    if cid is not None:
                        self.working_memory.use(
                            [cid], state.goal_id, acquired,
                            known_headers={cid: self.memory.chunks[cid].semantic_tags},
                            persistence=self.learner.activation_persistence)
            else:
                update = self.learner.incremental_feedback(
                    pending["cues"], chosen.target, success=success,
                    evidence_id=outcome.event_id)
        # PRIVATE counterfactuals are read only after the single real execution,
        # for evaluation. No oracle action is fed back to any cognitive owner.
        best = max(world.counterfactual_reward(a.target) for a in actions)
        decision, stats = pending["decision"], pending["stats"]
        case = {k: v for k, v in pending["case"].items() if k != "truths"}
        row = {"seed": self.seed, "mode": self.mode, "exploration": self.exploration, **case,
               "action": chosen.label, "prediction": forecast["probability"],
               "selection_prediction": (decision.predicted_success if decision
                                        else forecast["probability"]),
               "predictions": predictions, "outcome": observed.outcome,
               "reward": observed.reward, "regret": best-observed.reward,
               "brier": (forecast["probability"]-float(success))**2,
               # Evaluation-only: realized potential outcomes of every action.
               "all_action_brier": statistics.mean(
                   (predictions[a.label]["probability"] - float(world.counterfactual_reward(a.target) > 0))**2
                   for a in actions),
               # Evaluation-only: agreement with the noiseless private rule.
               "truth_agreement": statistics.mean(
                   float((predictions[a.label]["probability"] > .5) == world.counterfactual_truth(a.target))
                   for a in actions),
               "thought": decision.thought_trace if decision else None,
               "diagnosis": diagnosis, "revision": revision,
               "update": update, "outcome_event_id": outcome.event_id,
               "time": pending["acted_at"], "acquired_time": acquired,
               "attention": pending["attention"],
               "learner_literals": sorted(pending["learner_cues"]),
               "retrieved_chunks": [h.chunk_id for h in pending["hits"]],
               "anchor_event_ids": list(pending["anchors"]), "detail_events": stats.active_detail_events,
               "header_evaluations": stats.header_evaluations + stats.neighbor_headers_examined,
               "hypotheses_examined": sum(p.get("hypotheses_examined", 0) for p in predictions.values()),
               "rules_scored": update.get("rules_scored", 0),
               "scoring_work": update.get("scoring_work", 0)}
        if self.working_memory is not None:
            state_snapshot = self.working_memory.snapshot(
                state.goal_id, acquired,
                persistence=self.learner.activation_persistence)
            row["working_memory"] = state_snapshot
            row["used_provenance"] = list(pending["used_provenance"])
        self.trace.append(row)
        return row

    def step_case(self, case: dict, *, learn: bool = True, behavior: str | None = None) -> dict:
        return self.deliver(self.act(case, learn=learn, behavior=behavior))

    def predict(self, factors: dict, *, family: str, epoch: int = 0) -> dict[str, float]:
        """Read-only probe of current learned expectations (evaluation harness).

        Builds the same public literals a world would expose; touches no memory,
        thought or learner state. Episodic baselines have no read-only probe.
        """
        t = self.treatment
        primitive = frozenset({f"pair:domain:{family}", f"pair:epoch:{epoch}"} |
                              {f"pair:{a}:{v}" for a, v in factors.items()})
        if t.learner == "one_pass":
            cues = primitive | frozenset(a + "&" + b for a in primitive for b in primitive if a < b)
            return {p: self.learner.incremental_prediction(cues, p)["probability"]
                    for p in ("standard", "express")}
        if t.learner != "explanation":
            raise ValueError("no read-only probe for this learner")
        key = (("dispatch:" + family) if t.identity else None, "response_dispatch")
        thread = self.refiner.threads.get(key) if key[0] else None
        cues = self.learner_cues(primitive, thread, self.clock.now)
        return {p: self.learner.explanation_prediction(cues, p)["probability"]
                for p in ("standard", "express")}


# ---------------------------------------------------------------------------
# Measurement helpers (evaluation only)
# ---------------------------------------------------------------------------

def paired_interval(differences: list[float], *, resamples: int = 4000, label: str = "") -> dict:
    """Seed-paired mean with a deterministic percentile bootstrap interval."""
    mean = statistics.mean(differences)
    rng = random.Random("e12-bootstrap:" + label)
    n = len(differences)
    boots = sorted(statistics.mean(rng.choice(differences) for _ in range(n))
                   for _ in range(resamples))
    return {"mean": mean, "bootstrap_95_interval": [boots[int(.025 * resamples)],
                                                     boots[int(.975 * resamples) - 1]],
            "positive_seeds": sum(d > 0 for d in differences),
            "negative_seeds": sum(d < 0 for d in differences), "seeds": n}


def _explanation_quality(run: CausalRun, family: str, epoch: int) -> dict | None:
    """Evaluation-only agreement of the learned explanation with the private rule."""
    if run.treatment.learner != "explanation":
        return None
    correct, unsupported, total = 0, 0, 0
    relevant = {p: set() for p in ("standard", "express")}
    combos = []
    for i in range(32):
        gray = i ^ (i >> 1)
        combos.append({axis: values[(gray >> j) & 1] for j, (axis, values) in enumerate(zip(AXES, VALUES))})
    for factors in combos:
        truth = IncrementalDispatchWorld.causal_rule(family, factors, epoch)
        for axis, values in zip(AXES, VALUES):
            flipped = dict(factors, **{axis: values[1 - values.index(factors[axis])]})
            other = IncrementalDispatchWorld.causal_rule(family, flipped, epoch)
            for p in truth:
                if truth[p] != other[p]:
                    relevant[p].add(axis)
    for factors in combos:
        truth = IncrementalDispatchWorld.causal_rule(family, factors, epoch)
        for marker in ("blue", "orange"):
            probe = run.predict(dict(factors, marker=marker), family=family, epoch=epoch)
            for p, q in probe.items():
                total += 1
                correct += (q > .5) == truth[p]
    for p in ("standard", "express"):
        primitive = frozenset({f"pair:domain:{family}", f"pair:epoch:{epoch}"})
        bank = run.learner.explanations.get((run.learner.causal_scope(primitive), p))
        if bank is None:
            continue
        best = bank.posterior()[0][0]
        unsupported += sum(1 for term in best for l in term
                           if l.split(":", 2)[1] not in relevant[p])
    return {"truth_accuracy": correct / total, "unsupported_literals": unsupported}


def _execute(job: tuple) -> list[dict]:
    seed, mode, protocol, exploration, families = job
    run = CausalRun(seed, mode, exploration=exploration)
    trials = []
    for family in families:
        for epoch in range(2 if family == "coordination" else 1):
            train, heldout = curriculum(seed, family, epoch)
            for case in train:
                behavior = (behavior_action(seed, family, epoch, case["index"])
                            if protocol == "matched" else None)
                trials.append(run.step_case(case, behavior=behavior))
            quality = _explanation_quality(run, family, epoch)
            for case in heldout:
                # Every held-out query starts from the SAME frozen
                # training state. Neither labels nor activation leak.
                row = deepcopy(run).step_case(case, learn=False)
                row["explanation_quality"] = quality
                trials.append(row)
    state_bytes = len(pickle.dumps(run.learner)) + len(pickle.dumps(run.refiner))
    if run.working_memory is not None:
        state_bytes += len(pickle.dumps(run.working_memory))
    for row in trials:
        row["protocol"] = protocol
        row["learned_state_bytes"] = state_bytes
    return trials


def _map_jobs(jobs: list[tuple], workers: int | None) -> list[list[dict]]:
    if workers == 1 or len(jobs) == 1:
        return [_execute(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 1)) as pool:
        return list(pool.map(_execute, jobs))


def _summarize(trials: list[dict], seeds, conditions, reference: str) -> tuple[dict, dict]:
    summary = {}
    for mode in conditions:
        rows = [r for r in trials if r["mode"] == mode and r["stage"] == "heldout"]
        train = [r for r in trials if r["mode"] == mode and r["stage"] == "train"]
        summary[mode] = {"n": len(rows),
                         "success_rate": statistics.mean(r["outcome"] == "success" for r in rows),
                         **{key: statistics.mean(r[key] for r in rows) for key in
                            ("reward", "regret", "brier", "all_action_brier", "truth_agreement",
                             "detail_events", "header_evaluations", "hypotheses_examined")}}
        summary[mode]["train_reward"] = statistics.mean(r["reward"] for r in train)
        summary[mode]["train_brier"] = statistics.mean(r["brier"] for r in train)
        summary[mode]["train_express_share"] = statistics.mean(
            r["action"] == "route:express" for r in train)
        summary[mode]["heldout_express_share"] = statistics.mean(
            r["action"] == "route:express" for r in rows)
        summary[mode]["rules_scored_per_experience"] = statistics.mean(r["rules_scored"] for r in train)
        summary[mode]["scoring_work_per_experience"] = statistics.mean(r["scoring_work"] for r in train)
        summary[mode]["learned_state_bytes"] = statistics.mean(r["learned_state_bytes"] for r in rows)
        qualities = [r["explanation_quality"] for r in rows if r.get("explanation_quality")]
        if qualities:
            summary[mode]["explanation_truth_accuracy"] = statistics.mean(
                q["truth_accuracy"] for q in qualities)
            summary[mode]["unsupported_literals"] = statistics.mean(
                q["unsupported_literals"] for q in qualities)
    paired = {}
    for other in conditions:
        if other == reference:
            continue
        stats = {}
        for metric, sign in (("reward", 1), ("all_action_brier", -1), ("truth_agreement", 1)):
            def avg(seed, mode):
                return statistics.mean(r[metric] for r in trials
                    if r["seed"] == seed and r["mode"] == mode and r["stage"] == "heldout")
            differences = [avg(seed, reference) - avg(seed, other) for seed in seeds]
            stats[metric] = {"by_seed": dict(zip(seeds, differences)),
                             **paired_interval(differences, label=f"{reference}-{other}-{metric}")}
        # Kept for continuity with the historical E12 report.
        rewards = list(stats["reward"]["by_seed"].values())
        half = (1.96 * statistics.stdev(rewards) / len(rewards) ** .5) if len(rewards) > 1 else None
        stats["reward"]["approximate_95_interval"] = (None if half is None else
            [stats["reward"]["mean"] - half, stats["reward"]["mean"] + half])
        paired[other] = {**stats["reward"], "metrics": stats}
    return summary, paired


def run_causal_benchmark(*, seeds=(7, 23, 61, 101, 149), families=FAMILIES,
                         exploration: float = PRIMARY_EXPLORATION, conditions=CONDITIONS,
                         protocol: str = "online", reference: str = REFERENCE,
                         workers: int | None = 1) -> dict:
    """protocol 'online': each agent acts and learns from its own outcomes.
    protocol 'matched': every learner receives the identical experience stream
    produced by a fixed behavior policy; held-out decisions are still greedy.
    """
    if protocol not in ("online", "matched"):
        raise ValueError("unknown E12 protocol")
    jobs = [(seed, mode, protocol, exploration, tuple(families))
            for seed in seeds for mode in conditions]
    trials = [row for batch in _map_jobs(jobs, workers) for row in batch]
    summary, paired = _summarize(trials, list(seeds), conditions, reference)
    return {"experiment": 12, "protocol": protocol, "seeds": list(seeds),
            "conditions": list(conditions), "exploration": exploration,
            "treatments": {m: asdict(TREATMENTS[m]) for m in conditions},
            "reference": reference, "summary": summary, "paired_reward_deltas": paired,
            "trials": trials,
            "limitations": ["Synthetic bandit feedback; observational contrasts are not causal proof.",
                            "Explanations are DNFs of at most 3 terms of at most 2 literals.",
                            "no_continuity rebuilds the explanation AND drops thread identity and anchors.",
                            "Exploration is count-based UCB shared by all learners, training only.",
                            "Matched protocol: a fixed uniform behavior policy collects evidence.",
                            "Small seed sample; intervals are paired percentile bootstraps over seeds."]}


# Seeds never used while designing E12's explanation learner (historical replication).
REPLICATION_SEEDS = (211, 307, 401, 503, 601, 701, 809, 907, 1009, 1103)
# Seeds reserved for the unified-E12 final evaluation; never used during design
# or mechanism selection. Run once after the architecture was frozen.
FINAL_SEEDS = (1301, 1409, 1511, 1607, 1709, 1811, 1913, 2017, 2113, 2213)


def _write(report: dict, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary", {k: v for k, v in report.items() if k != "trials"}),
                       ("trials", report["trials"])):
        (path / (name + ".json")).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_causal_benchmark(output="results-e12", **kwargs) -> dict:
    report = run_causal_benchmark(**kwargs)
    _write(report, Path(output))
    return report


def write_all(output="results-e12", **kwargs) -> dict:
    """Original E12 gate: primary (design seeds), replication, and greedy (no exploration)."""
    return {"primary": write_causal_benchmark(output, **kwargs),
            "replication": write_causal_benchmark(output + "/replication", seeds=REPLICATION_SEEDS, **kwargs),
            "no_exploration": write_causal_benchmark(output + "/no-exploration", exploration=0.0, **kwargs)}


def write_unified(output="results-e12-unified", *, seeds=FINAL_SEEDS, workers=None) -> dict:
    """Unified E12 internal comparison on one seed set: matched evidence and online."""
    return {"matched": write_causal_benchmark(output + "/matched", seeds=seeds, protocol="matched",
                                              conditions=ALL_CONDITIONS, workers=workers),
            "online": write_causal_benchmark(output + "/online", seeds=seeds, protocol="online",
                                             conditions=ALL_CONDITIONS, workers=workers)}


if __name__ == "__main__":
    print(json.dumps({k: {"summary": r["summary"], "paired_reward_deltas": {
        m: {x: d[x] for x in ("mean", "bootstrap_95_interval")} for m, d in r["paired_reward_deltas"].items()}}
        for k, r in write_all().items()}, indent=2))
