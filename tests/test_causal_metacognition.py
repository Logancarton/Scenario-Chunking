"""E12 investigation, constraints and process learning through the real live path.

These tests prove mechanisms, isolation and integrity. They make no
reward-superiority claims: comparative results live in the measured reports.
"""
from copy import deepcopy
import hashlib
import inspect
import os
import pickle
import subprocess
import sys

import pytest

from scenario_chunking import causal_metacognition as cm
from scenario_chunking.environment import InvestigationWorld
from scenario_chunking.cognitive_control import CognitiveControl
from scenario_chunking.learning import ExplanationBank, OutcomeLearner, describe_rule
from scenario_chunking.meaning import MeaningInterpreter
from scenario_chunking.memory import EpisodicMemory
from scenario_chunking.models import Action
from scenario_chunking.policy import ActionSelector
from scenario_chunking.refinement import ProgressiveRefiner
from scenario_chunking.state import CognitiveState
import random

SCOPE = ("pair:domain:t", "pair:epoch:0")


def cues(**factors):
    return frozenset(set(SCOPE) | {f"pair:{a}:{v}" for a, v in factors.items()})


def teach(learner, rows, procedure="express", prefix="e"):
    """rows: (factors, success). Revealed hidden values are recorded as observations."""
    for i, (factors, ok) in enumerate(rows):
        c = cues(**factors)
        for a, v in factors.items():
            if a == "capacity" and v != "?":
                learner.observe_variable(c, f"pair:{a}:{v}")
        learner.explanation_feedback(c, procedure, success=ok, evidence_id=f"{prefix}{i}",
                                     allow_contrast=False)


def partial_learner():
    learner = OutcomeLearner()
    learner.explanation_options = {"durable": True, "change_detection": True, "partial": True}
    return learner


def capacity_rows(n=40, inspect_every=1):
    rows = []
    for i in range(n):
        cap = ("open", "limited")[i % 2]
        shown = cap if (i // 2) % inspect_every == 0 else "?"  # revelation independent of value
        rows.append(({"capacity": shown, "urgency": ("low", "high")[(i // 4) % 2]}, cap == "open"))
    return rows


# -- partial observability and self-diagnosis (knowledge owner) ----------------------

def test_association_evidence_is_model_free_and_occam_penalized():
    learner = partial_learner()
    teach(learner, capacity_rows(40))
    relevant = learner.variable_association(cues(), "capacity", ("express",))
    assert relevant["log_bayes_factor"] > 5 and relevant["revealed"] == 40
    noise = partial_learner()
    rng = random.Random(3)
    teach(noise, [({"capacity": rng.choice(("open", "limited")), "urgency": "low"}, rng.random() < .5)
                  for _ in range(40)])
    assert noise.variable_association(cues(), "capacity", ("express",))["log_bayes_factor"] < 0


def test_uninspected_variable_is_marginalized_and_never_becomes_edit_material():
    learner = partial_learner()
    teach(learner, capacity_rows(48, inspect_every=2))
    bank = learner.explanations[(SCOPE, "express")]
    assert describe_rule(bank.posterior()[0][0]) == "pair:capacity:open"
    assert not any(l.endswith(":?") for rule in bank.beam for term in rule for l in term)
    p_open = learner.explanation_prediction(cues(capacity="open", urgency="low"), "express")["probability"]
    p_unknown = learner.explanation_prediction(cues(capacity="?", urgency="low"), "express")["probability"]
    p_limited = learner.explanation_prediction(cues(capacity="limited", urgency="low"), "express")["probability"]
    assert p_limited < p_unknown < p_open
    assert abs(p_unknown - .5) < .1  # capacity was revealed half open, half limited


def test_partial_option_off_is_exactly_the_original_bank():
    rows = [(frozenset({"pair:capacity:open", "pair:urgency:low"}), True, "a"),
            (frozenset({"pair:capacity:limited", "pair:urgency:low"}), False, "b")]
    plain, partial = ExplanationBank(), ExplanationBank(partial=True)
    for bank in (plain, partial):
        bank.evidence = list(rows)
        bank.beam[frozenset({frozenset({"pair:capacity:open"})})] = {"operator": "t", "origin": None, "revision": 0}
    assert [plain.score(r) for r in plain.beam] == [partial.score(r) for r in partial.beam]
    assert plain.adequacy()["signals"] == partial.adequacy()["signals"]


@pytest.mark.parametrize("kind,status", [
    ("simple", "adequate"), ("hidden", "unexplained_variability"),
    ("parity", "representation_or_search_limit"), ("few", "insufficient_evidence"),
    ("narrow", "limited_coverage")])
def test_adequacy_self_diagnosis_separates_causes(kind, status):
    learner = OutcomeLearner()
    learner.explanation_options = {"durable": True}
    rng = random.Random(5)
    n = 10 if kind == "few" else 160
    for i in range(n):
        f = {a: rng.choice(v) for a, v in (("capacity", ("open", "limited")), ("channel", ("day", "night")),
                                           ("urgency", ("low", "high")))}
        if kind == "narrow":  # only two of the eight combinations are ever experienced
            f = rng.choice(({"capacity": "open", "channel": "day", "urgency": "low"},
                            {"capacity": "limited", "channel": "night", "urgency": "high"}))
        truth = {"simple": f["capacity"] == "open", "few": f["capacity"] == "open",
                 "narrow": f["capacity"] == "open",
                 "hidden": f["capacity"] == "open",
                 "parity": (f["capacity"] == "open") ^ (f["channel"] == "day") ^ (f["urgency"] == "high")}[kind]
        ok = truth if rng.random() > .05 else not truth
        visible = {k: v for k, v in f.items() if not (kind == "hidden" and k == "capacity")}
        learner.explanation_feedback(cues(**visible), "express", success=ok, evidence_id=f"x{i}",
                                     allow_contrast=False)
    assert learner.explanations[(SCOPE, "express")].adequacy()["status"] == status


def test_world_reveals_only_inspected_values_charges_costs_and_counts_violations():
    world = InvestigationWorld.scripted(
        seed=1, family="t", index=0, visible={"urgency": "low"}, hidden={"capacity": "open"},
        domains={"capacity": ("open", "limited")}, truths={"standard": True, "express": True},
        inspect_costs={"capacity": .06}, stakes=cm.stakes("high"), prohibited={"express"})
    assert "pair:capacity:?" in world.public_cues()
    assert world.public_constraints()["prohibited"] == {"express"}
    signal = world.inspect("capacity")
    assert signal.reward == -.06 and "pair:capacity:open" in world.public_cues()
    with pytest.raises(ValueError):
        world.inspect("capacity")  # each variable is examined at most once
    world.execute(Action("route", target="express"))
    assert world.violations == 1  # the world records it; cognition must never cause it
    with pytest.raises(RuntimeError):
        world.execute(Action("route", target="standard"))


# -- valuation and value of information (interpreter) --------------------------------

def constraints(prohibited=frozenset(), hazard="low", cost=.06):
    return {"hidden": {"capacity": cost}, "domains": {"capacity": ("open", "limited")},
            "stakes": cm.stakes(hazard), "defer_value": cm.DEFER, "prohibited": prohibited}


def test_value_of_information_is_zero_for_an_unknown_variable_and_positive_once_learned():
    learner = partial_learner()
    interpreter = MeaningInterpreter(EpisodicMemory(), learner)
    c = cues(capacity="?", urgency="low")
    teach(learner, [({"capacity": "?", "urgency": u}, i % 2 == 0)
                    for i, u in enumerate(["low", "high"] * 10)])
    blind = interpreter.decision_analysis(c, constraints=constraints(), permitted=frozenset({"standard", "express"}))
    assert blind["voi"]["capacity"] == 0.0  # a model that never saw capacity cannot value it
    teach(learner, capacity_rows(40), prefix="k")
    informed = interpreter.decision_analysis(c, constraints=constraints(), permitted=frozenset({"standard", "express"}))
    assert informed["voi"]["capacity"] > .2


def test_information_that_cannot_change_a_permitted_decision_is_worthless():
    learner = partial_learner()
    teach(learner, capacity_rows(40))
    interpreter = MeaningInterpreter(EpisodicMemory(), learner)
    c = cues(capacity="?", urgency="low")
    open_ = interpreter.decision_analysis(c, constraints=constraints(), permitted=frozenset({"standard", "express"}))
    # With express prohibited, knowing capacity cannot change what may be done.
    closed = interpreter.decision_analysis(c, constraints=constraints(), permitted=frozenset({"standard"}))
    assert open_["voi"]["capacity"] > 0 and closed["voi"]["capacity"] == 0.0


def test_severity_enters_valuation():
    learner = partial_learner()
    teach(learner, capacity_rows(40))
    interpreter = MeaningInterpreter(EpisodicMemory(), learner)
    c = cues(capacity="?", urgency="low")
    low = interpreter.decision_analysis(c, constraints=constraints(hazard="low"), permitted=frozenset({"standard", "express"}))
    high = interpreter.decision_analysis(c, constraints=constraints(hazard="high"), permitted=frozenset({"standard", "express"}))
    assert high["values"]["route:express"] < low["values"]["route:express"] - .5
    assert high["values"]["route:standard"] == low["values"]["route:standard"]


# -- hard constraints (selector authority) --------------------------------------------

class Certain:
    """Adversarial prediction source: the prohibited route always succeeds."""
    def __init__(self, p):
        self.p = p

    def explanation_beliefs(self, queries, procedure):
        return [{"probability": self.p[procedure], "uncertainty": 0.0, "familiarity": 9,
                 "evidence_ids": [], "hypotheses": []} for _ in queries]

    def variable_distribution(self, cues, axis, domain):
        return {v: 1 / len(domain) for v in domain}


def adversarial_decision(prohibited, *, bonus=0.0, standard=0.0):
    learner = OutcomeLearner()
    memory = EpisodicMemory()
    state = CognitiveState("adv", goal_id="investigative_dispatch")
    state.investigation_context({**constraints(prohibited=frozenset(prohibited), hazard="low"), "budget": 0})
    state.incremental_context(cues(capacity="?", urgency="high"), 1.0)
    actions = state.candidate_actions()
    interpreter = MeaningInterpreter(memory, Certain({"standard": standard, "express": 1.0}))
    permitted = frozenset(p for p in ("standard", "express") if p not in state.prohibited)
    analysis = interpreter.investigation_values(state.context_cues, actions,
                                                constraints=state.investigation, permitted=permitted)
    values = dict(analysis["values"])
    values["route:express"] += 1000.0  # a reward estimate no evidence could justify
    selector = ActionSelector(memory, learner, random.Random(1), refiner=ProgressiveRefiner(learner))
    context = {"cognitive_mode": "incremental", "candidate_values": values,
               "predictions": analysis["predictions"], "risk_price": 0.0}
    if bonus:
        context["exploration"] = {"bonus": bonus, "scope": "s", "labels": ("route:express",)}
    return selector.select(state, actions, [], refine=True,
                           transfer_scores={k: v / 2 for k, v in values.items()}, transfer_context=context)


@pytest.mark.parametrize("bonus", [0.0, 50.0])
def test_reward_and_exploration_cannot_override_a_hard_constraint(bonus):
    assert adversarial_decision(set(), bonus=bonus).action.target == "express"
    # Express prohibited; standard is believed certain to fail, so deferring is best.
    assert adversarial_decision({"express"}, bonus=bonus).action.target == "defer"
    assert adversarial_decision({"express"}, bonus=bonus, standard=1.0).action.target == "standard"
    assert adversarial_decision({"standard", "express"}, bonus=bonus).action.target == "defer"


def test_selector_refuses_when_nothing_is_permitted():
    state = CognitiveState("x", goal_id="investigative_dispatch")
    state.authorize({"standard", "express"})
    learner = OutcomeLearner()
    selector = ActionSelector(EpisodicMemory(), learner, random.Random(1), refiner=ProgressiveRefiner(learner))
    with pytest.raises(PermissionError):
        selector.select(state, [Action("route", target="express")], [])


def test_authorization_only_tightens_and_learning_owners_cannot_reach_it():
    state = CognitiveState("x", goal_id="investigative_dispatch")
    state.authorize({"express"})
    state.authorize(set())
    assert state.prohibited == {"express"}
    import scenario_chunking.learning as learning_module
    import scenario_chunking.meaning as meaning_module
    import scenario_chunking.cognitive_control as control_module
    for module in (learning_module, meaning_module, control_module):
        source = inspect.getsource(module)
        assert ".prohibited" not in source and "authorize(" not in source and "permissible(" not in source


def test_online_runs_never_execute_a_prohibited_action_even_when_it_would_pay():
    for mode in ("metacognitive", "voi_ucb", "always_investigate"):
        result = cm.run_condition(41, mode, families=("restricted",), per_family=80, checkpoints=(80,))
        rows = result["rows"]
        assert sum(r["violations"] for r in rows) == 0
        restricted = [r for r in rows if r["prohibited"]]
        assert restricted and all(r["action"] not in r["prohibited"] for r in restricted)
        assert any(r["prohibited"] == ["express", "standard"] and r["action"] == "defer" for r in rows)
    current = cm.run_condition(41, cm.CURRENT, families=("restricted",), per_family=80, checkpoints=(80,))
    assert sum(r["violations"] for r in current["rows"]) > 0  # the unmodified path has no permission concept


# -- process appraisal and process learning -------------------------------------------

def test_errors_are_attributed_to_missing_information_only_when_an_examination_was_warranted():
    refiner = ProgressiveRefiner(OutcomeLearner())
    state = CognitiveState("a", goal_id="investigative_dispatch")
    state.thought_key = "t"
    record = {"scope": SCOPE, "inspected": {}, "followed_values": True, "permissible": True,
              "effort": {}, "route": {"procedure": "express", "prediction": .9, "familiarity": 0, "success": False}}
    warranted = {"capacity": {"voi": .4, "cost": .06, "competence": .8}}
    unwarranted = {"capacity": {"voi": .01, "cost": .06, "competence": .8}}
    a = refiner.appraise_process(state, record=record, hindsight=warranted, unexplained=0.0, now=1.0)
    b = refiner.appraise_process(state, record=record, hindsight=unwarranted, unexplained=0.0, now=2.0)
    assert a["error"] == "missing_information" and a["missed"] == ["capacity"]
    assert b["error"] == "knowledge_or_noise" and b["missed"] == []
    assert len(refiner.thread(state).process_appraisals) == 2


def test_process_learning_never_touches_explanations_and_one_outcome_cannot_swing_it():
    learner = partial_learner()
    teach(learner, capacity_rows(30))

    def appraise(evidence_id, success, prediction=.95):
        return {"scope": list(SCOPE), "evidence_id": evidence_id, "error": None, "unexplained": 0.0,
                "route": {"procedure": "express", "prediction": prediction, "familiarity": 0,
                          "success": success},
                "hindsight": {}, "association": {
                    "capacity": learner.variable_association(cues(), "capacity", ("express",))}}
    learner.process_feedback(appraise("p0", True))
    before = pickle.dumps(learner.explanations)
    belief = learner.control.relevance_belief(SCOPE, "capacity")
    trust = learner.control.calibrate(SCOPE, "express", 0, .95)
    assert belief > .9
    learner.process_feedback(appraise("p1", True))
    assert pickle.dumps(learner.explanations) == before  # process learning never edits knowledge
    # One contradicting experience reaches knowledge, then process; the change is bounded.
    teach(learner, [({"capacity": "limited", "urgency": "low"}, True)], prefix="late")
    learner.process_feedback(appraise("p2", False))
    assert abs(belief - learner.control.relevance_belief(SCOPE, "capacity")) < .15
    assert 0 < trust - learner.control.calibrate(SCOPE, "express", 0, .95) < .15


def test_exploration_value_fades_with_competence_and_with_evidence_of_independence():
    control = CognitiveControl()
    fresh = control.exploration_value(SCOPE, "color", 0.0)
    assert control.exploration_value(SCOPE, "color", .9) < fresh / 5
    control.record_relevance(SCOPE, "color", -2.5)  # outcomes clearly do not depend on it
    assert control.exploration_value(SCOPE, "color", 0.0) < min(fresh / 2, .06)  # below the cost of looking
    control.record_relevance(SCOPE, "capacity", 6.0)  # outcomes clearly depend on it
    assert control.exploration_value(SCOPE, "capacity", 0.0) > fresh
    # Pooled evidence transfers to an unfamiliar scope, scaled by its own lack of evidence.
    other = ("pair:domain:u", "pair:epoch:0")
    assert control.exploration_value(other, "color", 0.0) < fresh
    control.drive[other] = .9  # self-diagnosed missing information raises the prior
    assert control.exploration_value(other, "color", 0.0) > fresh


# -- live path integration, isolation, frozen evaluation, restart ----------------------

def test_live_run_reaches_every_investigation_and_process_mechanism():
    run = cm.InvestigativeRun(43, "metacognitive")
    rows = [run.step_case(c) for c in cm.training_stream(43, ("delivery", "coordination"), 60)]
    assert sum(len(r["inspections"]) for r in rows) > 0           # inspection actions chosen
    assert run.learner.variable_counts                              # variable frequencies learned
    assert any("relevance:" in u for r in rows for u in r["process_updated"])
    assert any("drive" in r["process_updated"] for r in rows)
    assert run.learner.control.relevance and not run.learner.control.trust  # trust is not selected
    plus = cm.InvestigativeRun(43, "meta_plus_trust")
    for c in cm.training_stream(43, ("delivery",), 20):
        plus.step_case(c)
    assert plus.learner.control.trust
    thread = run.refiner.threads[("dispatch:delivery", "investigative_dispatch")]
    assert thread.process_appraisals and thread.versions
    assert any(b.partial and b.axis_counts for b in run.learner.explanations.values())


def test_uninspected_hidden_values_never_reach_cognition():
    case = cm.training_stream(47, ("coordination",), 1)[0]
    flipped = deepcopy(case)
    flipped["hidden"] = {a: v[1] if case["hidden"][a] == v[0] else v[0] for a, v in cm.HIDDEN.items()}
    flipped["truths"] = case["truths"]  # same private outcome; only unseen values differ
    a, b = cm.InvestigativeRun(47, "voi_fixed"), cm.InvestigativeRun(47, "voi_fixed")
    ra, rb = a.step_case(case), b.step_case(flipped)
    assert not ra["inspections"] and not rb["inspections"]
    assert canonical(a.learner) == canonical(b.learner)


def canonical(learner, keys=None):
    """Order-independent content of explanation knowledge (sets pickle in hash order)."""
    out = []
    for key in sorted(keys or learner.explanations):
        bank = learner.explanations[key]
        out.append((key, [(sorted(c), y, e) for c, y, e in bank.evidence],
                    sorted((sorted(c), v) for c, v in bank.contexts.items()),
                    sorted((describe_rule(r), bank.score(r)) for r in bank.beam),
                    sorted((a, sorted(v.items())) for a, v in bank.axis_counts.items())))
    return out


def test_scopes_isolate_knowledge_from_unrelated_experience():
    run = cm.InvestigativeRun(53, "metacognitive")
    for c in cm.training_stream(53, ("delivery",), 40):
        run.step_case(c)
    keys = list(run.learner.explanations)
    delivery = canonical(run.learner, keys)
    for c in cm.training_stream(53, ("repair",), 40, start=500):
        run.step_case(c)
    assert canonical(run.learner, keys) == delivery  # knowledge is scoped
    repair = ("pair:domain:repair", "pair:epoch:0")
    # Process evidence about weather in repair exists, despite delivery having
    # taught that weather is irrelevant THERE (no negative transfer lock-out).
    assert repair in run.learner.control.relevance.get("weather", {})


def _digest(run):
    return hashlib.sha256(pickle.dumps((run.learner, run.refiner, run.selector.exploration_counts,
                                        run.clock.now, sorted(run.seen)))).hexdigest()


def test_frozen_evaluation_never_teaches_and_one_copy_equals_per_case_copies():
    run = cm.InvestigativeRun(59, "metacognitive")
    for c in cm.training_stream(59, ("delivery", "remote"), 40):
        run.step_case(c)
    before = _digest(run)
    shared = cm.evaluate(run, ("delivery", "remote"), 80)
    fresh = cm.evaluate(run, ("delivery", "remote"), 80, per_case_copies=True)
    assert _digest(run) == before
    keys = ("action", "inspections", "expected_reward", "immediate_action")
    assert [tuple(str(r[k]) for k in keys) for r in shared] == [tuple(str(r[k]) for k in keys) for r in fresh]
    assert not any(r["process_updated"] for r in shared)


def test_process_reset_changes_only_process_state():
    run = cm.InvestigativeRun(61, "metacognitive")
    for c in cm.training_stream(61, ("delivery",), 40):
        run.step_case(c)
    rows = cm.evaluate(run, ("delivery",), 40, reset_process=True)
    assert rows and all(r["mode"] == "metacognitive@process_reset" for r in rows)
    assert run.learner.control.updates > 0  # the trained run itself is untouched


def test_investigative_state_restores_in_a_fresh_process_and_continues_identically(tmp_path):
    stream = cm.training_stream(67, ("delivery", "repair"), 30)
    run = cm.InvestigativeRun(67, "metacognitive")
    for c in stream[:30]:
        run.step_case(c)
    path = tmp_path / "run.pkl"
    path.write_bytes(pickle.dumps(run))
    rows = [run.step_case(c) for c in stream[30:]]
    expected = [(r["action"], tuple(r["inspections"]), round(r["expected_reward"], 9)) for r in rows]
    code = ("import pickle,sys; from scenario_chunking import causal_metacognition as cm\n"
            "run = pickle.loads(open(sys.argv[1],'rb').read())\n"
            "stream = cm.training_stream(67, ('delivery','repair'), 30)\n"
            "print([(r['action'], tuple(r['inspections']), round(r['expected_reward'], 9)) "
            "for r in (run.step_case(c) for c in stream[30:])])")
    env = {**os.environ, "PYTHONPATH": "src"}
    out = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True,
                         env=env, check=True).stdout.strip()
    assert out == str(expected)


def test_option_pricing_reserves_a_scarce_action_for_larger_future_advantages():
    refiner = ProgressiveRefiner(OutcomeLearner())
    state = CognitiveState("o", goal_id="investigative_dispatch")
    state.thought_key = "fleet"
    values = {"route:express": 1.5, "route:standard": -.86, "route:defer": -.3}
    for _ in range(12):
        refiner.option_costs(state, values=values, scarce="route:express",
                             alternatives=["route:standard", "route:defer"], horizon=3)
    small = {"route:express": .2, "route:standard": -.86, "route:defer": -.3}
    cost = refiner.option_costs(state, values=small, scarce="route:express",
                                alternatives=["route:standard", "route:defer"], horizon=3)
    assert cost["route:express"] > .5 + .3  # worth more than the small advantage available now
    assert refiner.option_costs(state, values=small, scarce="route:express",
                                alternatives=["route:standard", "route:defer"], horizon=0) == {}
