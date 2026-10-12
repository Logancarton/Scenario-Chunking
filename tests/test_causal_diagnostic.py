"""E12 prediction-to-decision diagnostic: integrity of the replay and interventions.

The diagnostic must measure the EXISTING live decision path. These tests prove
that its replay reproduces the historical held-out procedure, that substituted
predictions really drive the existing selector across the decision boundary,
and that no evaluation-only information can reach learner state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from scenario_chunking.causal_diagnostic import (
    COSTS, PROCEDURES, SubstitutedBeliefs, WORLD_NOISE, _taxonomy, action_value, decide,
    expected_utility, oracle_vectors, run_stream, score, state_digest)
from scenario_chunking.causal_experiment import CausalRun, _execute, behavior_action, curriculum
from scenario_chunking.environment import IncrementalDispatchWorld
from scenario_chunking.meaning import MeaningInterpreter
from scenario_chunking.state import CognitiveState

ROOT = Path(__file__).resolve().parents[1]


def frozen(mode: str, seed: int = 7, family: str = "delivery"):
    run = CausalRun(seed, mode, exploration=.5)
    train, heldout = curriculum(seed, family, 0)
    for case in train:
        run.step_case(case, behavior=behavior_action(seed, family, 0, case["index"]))
    return run, heldout


def reference_rows(stream):
    return [d for d in stream["decisions"] if not d["substituted"]]


def test_replay_reproduces_committed_reference_and_historical_choices():
    reference = json.loads((ROOT / "docs/e12_unified_reference.json").read_text())
    modes = ("episodic_only", "explanation_full", "unified")
    for protocol in ("matched", "online"):
        stream = run_stream(1301, protocol, modes, interventions=False)
        for mode in modes:
            mine = [d for d in reference_rows(stream) if d["host"] == mode]
            historical = [r for r in _execute((1301, mode, protocol, .5,
                                               ("delivery", "repair", "coordination")))
                          if r["stage"] == "heldout"]
            assert [d["chosen"] for d in mine] == [r["action"].split(":")[1] for r in historical]
            reward = sum(d["realized_reward"] for d in mine) / len(mine)
            brier = sum(d["all_action_brier"] for d in mine) / len(mine)
            assert round(reward, 9) == reference[protocol][mode]["reward"]
            assert round(brier, 9) == reference[protocol][mode]["all_action_brier"]
        assert all(i["frozen_state_unchanged"] for i in stream["integrity"])


@pytest.mark.parametrize("host", ["explanation_full", "strong_one_pass"])
def test_substituted_predictions_cross_the_actual_decision_boundary(host):
    """V = 2p - 1 - cost, so express wins exactly when p_E - p_S > .18 - .02 = .08."""
    run, heldout = frozen(host)
    before = state_digest(run)
    case = heldout[0]
    below = decide(run, case, beliefs={"standard": .50, "express": .57})
    above = decide(run, case, beliefs={"standard": .50, "express": .59})
    assert below["chosen"] == "standard" and above["chosen"] == "express"
    assert below["refined"] == (host == "explanation_full")
    # Shifting both probabilities equally never changes the decision.
    assert decide(run, case, beliefs={"standard": .10, "express": .19})["chosen"] == "express"
    assert decide(run, case, beliefs={"standard": .80, "express": .87})["chosen"] == "standard"
    assert state_digest(run) == before


def test_selector_and_refiner_follow_the_value_argmax_exactly():
    explanation, heldout = frozen("explanation_full")
    one_pass, _ = frozen("strong_one_pass")
    grid = [i / 10 for i in range(11)]
    for ps in grid:
        for pe in grid:
            beliefs = {"standard": ps, "express": pe}
            values = {a: action_value(beliefs[a], a) for a in PROCEDURES}
            argmax = min(PROCEDURES, key=lambda a: (-values[a], "route:" + a))
            refined = decide(explanation, heldout[1], beliefs=beliefs)
            direct = decide(one_pass, heldout[1], beliefs=beliefs)
            assert refined["chosen"] == direct["chosen"] == argmax, beliefs
            assert score(refined, refined["world"])["follows_values"]


def test_value_equation_is_the_world_expected_reward():
    interpreter = MeaningInterpreter(None, SubstitutedBeliefs({"standard": .3, "express": .7}))
    state = CognitiveState("x", goal_id="response_dispatch")
    values = interpreter.incremental_action_values(
        frozenset(), state.candidate_actions(), costs=COSTS, causal=True, explanation=True)["values"]
    assert values == {"route:standard": action_value(.3, "standard"),
                      "route:express": action_value(.7, "express")}
    flips, total = 0, 0
    factors = {"person": "Alex", "delay": "short", "urgency": "low", "capacity": "open",
               "channel": "day", "marker": "blue"}
    for index in range(4000):
        world = IncrementalDispatchWorld.scenario(seed=1, family="delivery", index=index,
                                                  factors=factors, epoch=0)
        for a in PROCEDURES:
            success = world.counterfactual_reward(a) > 0
            assert world.counterfactual_reward(a) == (1.0 if success else -1.0) - COSTS[a]
            flips += success != world.counterfactual_truth(a)
            total += 1
    assert abs(flips / total - WORLD_NOISE) < .01
    assert expected_utility(True, "standard") - expected_utility(True, "express") == pytest.approx(.16)


def test_oracles_bound_regret_and_never_enter_learner_state():
    run, heldout = frozen("explanation_full")
    twin = pickle.loads(pickle.dumps(run))
    # Digests compare one object over time; a pickle round trip can reorder sets.
    before = state_digest(run)
    for case in heldout:
        own = decide(run, case)
        for name, vector in oracle_vectors(own["world"]).items():
            d = decide(run, case, beliefs=vector)
            s = score(d, d["world"])
            key = "expected_regret" if name == "oracle_causal" else "realized_regret"
            assert s[key] == 0.0
    assert state_digest(run) == before
    # Continue learning both: the evaluated learner and an untouched twin stay identical.
    train, _ = curriculum(7, "repair", 0)
    for case in train[:6]:
        a = run.step_case(case, behavior=behavior_action(7, "repair", 0, case["index"]))
        b = twin.step_case(case, behavior=behavior_action(7, "repair", 0, case["index"]))
        assert a["predictions"] == b["predictions"] and a["update"] == b["update"]
    for case in heldout:
        assert decide(run, case)["p"] == decide(twin, case)["p"]


def test_replaying_a_learners_own_online_actions_reproduces_its_state():
    """The swap design reuses exact evidence: (context, executed action, outcome)."""
    online = run_stream(23, "online", ("explanation_full",), interventions=False,
                        families=("delivery",))
    replay = run_stream(23, online["actions"]["explanation_full"], ("explanation_full",),
                        interventions=False, stream_name="swap:self", families=("delivery",))
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("cpu", "stream")} for r in rows]
    assert strip(replay["training"]) == strip(online["training"])
    assert strip(replay["decisions"]) == strip(online["decisions"])


def test_deterministic_rerun_and_attribution():
    a = run_stream(61, "matched", ("episodic_only", "explanation_full"), families=("repair",))
    b = run_stream(61, "matched", ("episodic_only", "explanation_full"), families=("repair",))
    strip = lambda rows: [{k: v for k, v in r.items() if k != "cpu"} for r in rows]
    assert strip(a["decisions"]) == strip(b["decisions"])
    assert all(d["attributed"] and d["follows_values"] for d in a["decisions"])
    # Every substituted vector produces the same choice on both hosts.
    chosen = {}
    for d in a["decisions"]:
        if d["substituted"]:
            chosen.setdefault((d["index"], d["source"]), set()).add(d["chosen"])
    assert chosen and all(len(v) == 1 for v in chosen.values())


def test_taxonomy_contributions_sum_to_expected_reward_difference():
    stream = run_stream(101, "matched", ("episodic_only", "explanation_full"),
                        interventions=False, families=("delivery", "repair"))
    rows = reference_rows(stream)
    a = [d for d in rows if d["host"] == "explanation_full"]
    b = [d for d in rows if d["host"] == "episodic_only"]
    t = _taxonomy(a, b)
    difference = (sum(d["expected_reward"] for d in a) - sum(d["expected_reward"] for d in b)) / len(a)
    assert sum(t["expected_reward_contribution"].values()) == pytest.approx(difference)
    assert sum(t["counts"].values()) == len(a)
    for key in ("improve_same", "worsen_same", "equal_prediction"):
        assert t["expected_reward_contribution"][key] == 0.0


def test_frozen_decision_survives_independent_process_restart(tmp_path):
    run, heldout = frozen("explanation_full", seed=149)
    expected = decide(run, heldout[2])
    snapshot = tmp_path / "frozen.pkl"
    snapshot.write_bytes(pickle.dumps(run))
    script = (
        "import json, pickle, sys\n"
        "from scenario_chunking.causal_diagnostic import decide\n"
        "run = pickle.loads(open(sys.argv[1], 'rb').read())\n"
        "case = json.loads(sys.argv[2])\n"
        "d = decide(run, case)\n"
        "print(json.dumps({'chosen': d['chosen'], 'p': d['p']}))\n")
    output = subprocess.check_output([sys.executable, "-c", script, str(snapshot),
                                      json.dumps(heldout[2])],
                                     text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    restored = json.loads(output)
    assert restored == {"chosen": expected["chosen"], "p": expected["p"]}


def test_episodic_host_rejects_substitution():
    run, heldout = frozen("episodic_only")
    with pytest.raises(ValueError):
        decide(run, heldout[0], beliefs={"standard": .5, "express": .5})
    with pytest.raises(ValueError):
        SubstitutedBeliefs({"standard": .5})
