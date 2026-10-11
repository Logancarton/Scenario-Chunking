"""E12 orchestration: persistent explanations, frozen unseen combinations.

The existing owners retain memory, learning, diagnosis, valuation and choice.
This module only constructs public curricula, executes worlds and measures.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import statistics

from .environment import IncrementalDispatchWorld
from .incremental_experiment import IncrementalRun, AXES, VALUES, FAMILIES, COSTS
from .models import EventKind, TruthStatus
from .state import CognitiveState

# contrast_fragments is the E12a learner (independent 1-2 literal fragments).
# explanation_* use one persistent, revisable rule explanation per action.
CONDITIONS = ("episodic_only", "strong_one_pass", "contrast_fragments",
              "explanation_full", "no_continuity", "no_contrast")
REFERENCE = "explanation_full"
EXPLANATION = ("explanation_full", "no_continuity", "no_contrast")
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


class CausalRun(IncrementalRun):
    def __init__(self, seed: int, mode: str, *, exploration: float = 0.0):
        if mode not in CONDITIONS:
            raise ValueError("unknown E12 treatment")
        super().__init__(seed, "episodic_only" if mode == "episodic_only" else "strong_one_pass")
        self.mode = mode
        # Same selector-owned exploration rule for every learner, training only.
        self.exploration = float(exploration)

    def step_case(self, case: dict, *, learn: bool = True) -> dict:
        family, epoch, index = case["family"], case["epoch"], case["index"]
        identity = (family, epoch, index)
        if identity in self.seen:
            raise ValueError("cannot repeat a scoped context")
        self.seen.add(identity)
        world = IncrementalDispatchWorld.scenario(
            seed=self.seed, family=family, epoch=epoch,
            index=10000 + epoch * 32 + index, factors=case["factors"])
        primitive = world.public_cues()
        # Preserve E11's actual predictor for the baseline, including cue
        # interactions; all conditions receive identical public information.
        cues = primitive | frozenset(a + "&" + b for a in primitive for b in primitive if a < b)
        state = CognitiveState(f"e12:{self.seed}:{family}:{epoch}:{index}",
                               goal_id="response_dispatch")
        causal = self.mode not in ("episodic_only", "strong_one_pass")
        if causal and self.mode != "no_continuity":
            state.thought_key = "dispatch:" + family
        state.incremental_context(primitive, float(len(self.seen)))
        self.learner.begin_episode()
        self._event(state, kind=EventKind.OBSERVATION, status=TruthStatus.FACT,
                    content="Received public dispatch context", tags=cues)
        thread = self.refiner.thread(state) if causal else None
        anchors = tuple(thread.causal_anchors.values()) if thread else ()
        actions = state.candidate_actions()
        hits, stats = self.router.retrieve(state.retrieval_query(), anchor_event_ids=anchors)
        # Materialize only already-routed records. Diagnosis does not rescan history.
        activated = [event for hit in hits for event in self.memory.chunk_events(hit.chunk_id)]
        if self.mode == "episodic_only":
            interpreted = self.interpreter.incremental_episodic_values(hits, cues, actions, COSTS)
        else:
            interpreted = self.interpreter.incremental_action_values(
                primitive if causal else cues, actions, costs=COSTS, causal=causal,
                explanation=self.mode in EXPLANATION)
        values, predictions = interpreted["values"], interpreted["predictions"]
        context = {"cognitive_mode": "incremental", "candidate_values": values,
                   "predictions": predictions, "risk_price": 0.0,
                   "reason": "revisable_conditional_hypotheses"}
        if learn and self.exploration > 0:
            context["exploration"] = {"bonus": self.exploration,
                                      "scope": "|".join(self.learner.causal_scope(primitive))}
        decision = self.selector.select(
            state, actions, hits, refine=causal,
            transfer_scores={k: v / 2 for k, v in values.items()},
            transfer_context=context)
        chosen = decision.action
        forecast = predictions[chosen.label]
        attempt = self._event(state, kind=EventKind.ACTION, status=TruthStatus.ATTEMPT,
                              content="Dispatch " + chosen.label, tags=cues, action=chosen)
        observed = world.execute(chosen)
        outcome = self._event(state, kind=EventKind.OUTCOME, status=TruthStatus.OUTCOME,
                              content=observed.content, tags=observed.semantic_tags,
                              outcome=observed.outcome, cause=attempt.event_id,
                              reward=observed.reward, metadata=observed.metadata)
        state.apply_signal(observed, event_id=outcome.event_id)
        state.record_action(chosen)
        diagnosis, revision, update = None, None, {"updated": []}
        if learn:
            if causal:
                diagnosis = self.refiner.diagnose_causal(
                    state, outcome, activated, probability=forecast["probability"])
                feedback = (self.learner.explanation_feedback if self.mode in EXPLANATION
                            else self.learner.causal_feedback)
                extra = {"persist": self.mode != "no_continuity"} if self.mode in EXPLANATION else {}
                update = feedback(
                    primitive, chosen.target, success=observed.outcome == "success",
                    evidence_id=outcome.event_id, contrast=diagnosis["contrast"],
                    allow_contrast=self.mode != "no_contrast", **extra)
                revision = self.refiner.record_causal_revision(state, update, outcome)
            else:
                update = self.learner.incremental_feedback(
                    cues, chosen.target, success=observed.outcome == "success",
                    evidence_id=outcome.event_id)
        # PRIVATE counterfactuals are read only after the single real execution,
        # for evaluation. No oracle action is fed back to any cognitive owner.
        best = max(world.counterfactual_reward(a.target) for a in actions)
        row = {"seed": self.seed, "mode": self.mode, "exploration": self.exploration, **case,
               "action": chosen.label, "prediction": forecast["probability"],
               "selection_prediction": decision.predicted_success,
               "predictions": predictions, "outcome": observed.outcome,
               "reward": observed.reward, "regret": best-observed.reward,
               "brier": (forecast["probability"]-float(observed.outcome == "success"))**2,
               "all_action_brier": statistics.mean(
                   (predictions[a.label]["probability"] - float(world.counterfactual_reward(a.target) > 0))**2
                   for a in actions),
               "thought": decision.thought_trace, "diagnosis": diagnosis, "revision": revision,
               "update": update, "outcome_event_id": outcome.event_id,
               "retrieved_chunks": [h.chunk_id for h in hits],
               "anchor_event_ids": list(anchors), "detail_events": stats.active_detail_events,
               "header_evaluations": stats.header_evaluations + stats.neighbor_headers_examined,
               "hypotheses_examined": sum(p.get("hypotheses_examined", 0) for p in predictions.values())}
        self.trace.append(row)
        return row


def run_causal_benchmark(*, seeds=(7, 23, 61, 101, 149), families=FAMILIES,
                         exploration: float = PRIMARY_EXPLORATION) -> dict:
    trials = []
    for seed in seeds:
        for mode in CONDITIONS:
            run = CausalRun(seed, mode, exploration=exploration)
            for family in families:
                for epoch in range(2 if family == "coordination" else 1):
                    train, heldout = curriculum(seed, family, epoch)
                    for case in train:
                        trials.append(run.step_case(case))
                    for case in heldout:
                        # Every held-out query starts from the SAME frozen
                        # training state. Neither labels nor activation leak.
                        trials.append(deepcopy(run).step_case(case, learn=False))
    summary = {}
    for mode in CONDITIONS:
        rows = [r for r in trials if r["mode"] == mode and r["stage"] == "heldout"]
        summary[mode] = {"n": len(rows),
                         "success_rate": statistics.mean(r["outcome"] == "success" for r in rows),
                         **{key: statistics.mean(r[key] for r in rows) for key in
                            ("reward", "regret", "brier", "all_action_brier", "detail_events",
                             "header_evaluations", "hypotheses_examined")}}
        train = [r for r in trials if r["mode"] == mode and r["stage"] == "train"]
        summary[mode]["train_reward"] = statistics.mean(r["reward"] for r in train)
        summary[mode]["train_express_share"] = statistics.mean(
            r["action"] == "route:express" for r in train)
        summary[mode]["heldout_express_share"] = statistics.mean(
            r["action"] == "route:express" for r in rows)
    paired = {}
    for other in CONDITIONS:
        if other == REFERENCE:
            continue
        differences = []
        for seed in seeds:
            def avg(mode):
                return statistics.mean(r["reward"] for r in trials
                    if r["seed"] == seed and r["mode"] == mode and r["stage"] == "heldout")
            differences.append(avg(REFERENCE)-avg(other))
        mean = statistics.mean(differences)
        half = 1.96 * statistics.stdev(differences) / len(differences)**.5 if len(differences) > 1 else None
        paired[other] = {"by_seed": dict(zip(seeds, differences)), "mean": mean,
                         "approximate_95_interval": None if half is None else [mean-half, mean+half]}
    return {"experiment": 12, "seeds": list(seeds), "conditions": list(CONDITIONS),
            "exploration": exploration,
            "reference": REFERENCE, "summary": summary, "paired_reward_deltas": paired,
            "trials": trials,
            "limitations": ["Synthetic bandit feedback; observational contrasts are not causal proof.",
                            "Explanations are DNFs of at most 3 terms of at most 2 literals.",
                            "Continuity ablation rebuilds the explanation each experience; evidence is kept.",
                            "Five-axis integration is not implemented in E12.",
                            "Exploration is count-based UCB shared by all learners, training only.",
                            "Small seed sample; intervals are descriptive normal approximations."]}


# Seeds never used while designing E12b; the frozen design is replicated here.
REPLICATION_SEEDS = (211, 307, 401, 503, 601, 701, 809, 907, 1009, 1103)


def write_causal_benchmark(output="results-e12", **kwargs) -> dict:
    report = run_causal_benchmark(**kwargs)
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary", {k:v for k,v in report.items() if k != "trials"}),
                       ("trials", report["trials"])):
        (path / (name + ".json")).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return report


def write_all(output="results-e12") -> dict:
    """Primary (design seeds), fresh-seed replication, and greedy (no exploration)."""
    return {"primary": write_causal_benchmark(output),
            "replication": write_causal_benchmark(output + "/replication", seeds=REPLICATION_SEEDS),
            "no_exploration": write_causal_benchmark(output + "/no-exploration", exploration=0.0)}


if __name__ == "__main__":
    print(json.dumps({k: {"summary": r["summary"], "paired_reward_deltas": r["paired_reward_deltas"]}
                      for k, r in write_all().items()}, indent=2))
