"""Bounded, inspectable interpretations of retrieved evidence, not a second planner.

A single historical event may inform several axes, but it counts as ONE source.
Chronology is not upgraded to causality, and missing evidence stays missing.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

from .models import Action, EventKind, TruthStatus
from .memory import EpisodicMemory
from .state import CognitiveState

AXES = ("self", "salience", "time", "sequence", "person")


@dataclass(frozen=True)
class LayerMeaning:
    axis: str
    value: float
    confidence: float
    source_event_ids: tuple[str, ...]
    temporal_valid: bool
    goal_related: bool
    provenance: str
    association: float

    def trace(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MeaningFrame:
    context: str
    action_kind: str
    meanings: tuple[LayerMeaning, ...]
    weights: dict[str, float]
    integrated: float
    independent_sources: int
    examined_event_records: int

    def trace(self) -> dict:
        return {
            "context": self.context,
            "action_kind": self.action_kind,
            "layers": {m.axis: m.trace() for m in self.meanings},
            "weights": dict(self.weights),
            "integrated": round(self.integrated, 6),
            "independent_sources": self.independent_sources,
            "examined_event_records": self.examined_event_records,
        }


@dataclass(frozen=True)
class PreparedEvidence:
    """One pass over retrieved events; indexed by candidate action identity."""
    pairs: dict[tuple[str, str | None], tuple[tuple[str, object, object], ...]]
    examined_event_records: int

    def matches(self, action: Action):
        key = (action.kind, action.key_name if action.kind == "try_key" else None)
        return self.pairs.get(key, ())


class MeaningInterpreter:
    """Interpret already retrieved events; never read unretrieved memory or environment truth."""

    def __init__(self, memory: EpisodicMemory, learner) -> None:
        self.memory = memory
        self.learner = learner

    @staticmethod
    def contextual_literals(last_outcome: dict | None, now: float, *,
                            gap_threshold: float = 30.0) -> frozenset[str]:
        """E12: sequence and elapsed-time meaning as candidate explanatory evidence.

        Interprets the continuing thought's own chronology (the previous executed
        procedure and its outcome, and the elapsed time since). These literals
        are only offered to the explanation learner; they matter only if its
        evidence-scored rules adopt them. No axis receives a fixed importance.
        Without a continuing investigation there is no sequence to interpret.
        """
        if not last_outcome or last_outcome.get("procedure") is None:
            return frozenset({"pair:sequence:none"})
        elapsed = now - float(last_outcome["time"])
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("chronology cannot run backwards")
        return frozenset({
            f"pair:sequence:{last_outcome['procedure']}_{last_outcome['outcome']}",
            "pair:gap:" + ("long" if elapsed > gap_threshold else "short")})

    def incremental_action_values(self, cues: frozenset[str], actions: list[Action],
                                  *, success_reward: float = 1.0,
                                  failure_loss: float = 1.0,
                                  costs: dict[str, float] | None = None,
                                  causal: bool = False, explanation: bool = False) -> dict:
        """Translate the one learner's conditional beliefs to action values."""
        costs = costs or {}
        predictions = {}
        values = {}
        provenance = {}
        for action in actions:
            procedure = action.target or action.kind
            belief = (self.learner.explanation_prediction(cues, procedure) if explanation
                      else self.learner.causal_prediction(cues, procedure) if causal
                      else self.learner.incremental_prediction(cues, procedure))
            p = belief["probability"]
            predictions[action.label] = belief
            values[action.label] = (
                p * success_reward - (1.0 - p) * failure_loss
                - costs.get(procedure, 0.0))
            provenance[action.label] = belief["evidence_ids"]
        return {"values": values, "predictions": predictions,
                "provenance": provenance}

    ROUTES = ("standard", "express")

    def decision_analysis(self, cues: frozenset[str], *, constraints: dict,
                          permitted: frozenset[str], calibrate=None,
                          active_axes: frozenset[str] | None = None) -> dict:
        """E12 investigation valuation from the one learner's CURRENT beliefs only.

        V(route) = p * gain - (1 - p) * loss - cost with public stakes, so the
        severity of failure enters every comparison; deferral has its public
        safe value. For each uninspected variable X the myopic value of
        information is E_x[max_a V(a | x)] - max_a E_x[V(a | x)], over the final
        actions that are permitted (what could actually be done after looking).
        ``calibrate(procedure, familiarity, p)`` optionally replaces the model's
        claimed probability by the organism's learned track record (trust).
        No hidden value, private rule or counterfactual outcome is read here.
        """
        stakes, defer = constraints["stakes"], float(constraints["defer_value"])
        unknown = sorted(l.split(":", 2)[1] for l in cues if l.startswith("pair:") and l.endswith(":?"))
        # When all routes are prohibited, inspecting cannot improve the sole
        # permitted final choice (defer). Keep live route predictions but avoid
        # computing counterfactual contexts that cannot change the decision.
        eligible = ([axis for axis in unknown if active_axes is None or axis in active_axes]
                    if permitted else [])
        queries, index = [cues], {}
        for axis in eligible:
            for value in constraints["domains"][axis]:
                index[(axis, value)] = len(queries)
                queries.append((cues - {f"pair:{axis}:?"}) | {f"pair:{axis}:{value}"})
        beliefs = {p: self.learner.explanation_beliefs(queries, p) for p in self.ROUTES}
        def value(p, belief):
            q = belief["probability"]
            if calibrate is not None:
                q = calibrate(p, belief["familiarity"], q)
            s = stakes[p]
            return q, q * s["gain"] - (1 - q) * s["loss"] - s["cost"]
        table = {p: [value(p, b) for b in beliefs[p]] for p in self.ROUTES}
        finals = [p for p in self.ROUTES if p in permitted]
        def best(i):
            return max([table[p][i][1] for p in finals] + [defer])
        voi = {}
        for axis in eligible:
            dist = self.learner.variable_distribution(cues, axis, constraints["domains"][axis])
            informed = sum(dist[v] * best(index[(axis, v)]) for v in dist)
            now = max([sum(dist[v] * table[p][index[(axis, v)]][1] for v in dist) for p in finals] + [defer])
            voi[axis] = round(max(0.0, informed - now), 9)
        voi.update({axis: 0.0 for axis in unknown if axis not in voi})
        values = {"route:" + p: table[p][0][1] for p in self.ROUTES}
        values["route:defer"] = defer
        predictions = {}
        for p in self.ROUTES:
            record = dict(beliefs[p][0])
            record["model_probability"], record["probability"] = record["probability"], table[p][0][0]
            predictions["route:" + p] = record
        predictions["route:defer"] = {"probability": 1.0, "uncertainty": 0.0, "evidence_ids": [],
                                      "hypotheses": []}
        return {"values": values, "predictions": predictions, "voi": voi,
                "now": max([values["route:" + p] for p in finals] + [defer]),
                "unknown": unknown, "belief_queries": len(queries) * len(self.ROUTES)}

    def investigation_values(self, cues: frozenset[str], actions: list[Action], *,
                             constraints: dict, permitted: frozenset[str], calibrate=None,
                          active_axes: frozenset[str] | None = None) -> dict:
        """Translate beliefs into values for every candidate, including inspection.

        Inspecting X is valued as deciding now plus the information value of X
        minus its public cost; the selector compares it with acting directly.
        """
        analysis = self.decision_analysis(cues, constraints=constraints, permitted=permitted,
                                          calibrate=calibrate, active_axes=active_axes)
        values, predictions = dict(analysis["values"]), dict(analysis["predictions"])
        for action in actions:
            if action.kind == "inspect":
                axis = action.target
                gain = analysis["voi"][axis] - constraints["hidden"][axis]
                values[action.label] = analysis["now"] + gain
                predictions[action.label] = {"probability": 1.0, "uncertainty": 0.0,
                                             "evidence_ids": [], "hypotheses": [],
                                             "information_value": analysis["voi"][axis]}
        return {**analysis, "values": values, "predictions": predictions}

    def incremental_episodic_values(self, hits, cues: frozenset[str],
                                    actions: list[Action], costs: dict[str, float]) -> dict:
        """A strong pure-case-memory baseline with no generalized relationships."""
        values, predictions = {}, {}
        for action in actions:
            similar = []
            for hit in hits:
                for event in self.memory.chunk_events(hit.chunk_id):
                    if (event.kind != EventKind.OUTCOME or
                            event.truth_status != TruthStatus.OUTCOME or
                            event.outcome not in ("success", "failure") or
                            event.metadata.get("procedure") != action.target):
                        continue
                    tags = frozenset(event.metadata.get("context_cues", ()))
                    overlap = len(cues & tags) / max(1, len(cues | tags))
                    similar.append((overlap * hit.score, event.event_id,
                                    event.outcome == "success"))
            similar.sort(reverse=True)
            top = similar[:5]
            total = sum(weight for weight, _, _ in top)
            p = ((1.0 + sum(weight * int(ok) for weight, _, ok in top)) /
                 (2.0 + total))
            predictions[action.label] = {
                "probability": p, "uncertainty": 1 / (1 + total),
                "evidence_ids": [eid for _, eid, _ in top]}
            values[action.label] = 2 * p - 1 - costs.get(action.target or action.kind, 0.0)
        return {"values": values, "predictions": predictions}

    def temporal_action_values(self, state: CognitiveState, public: dict,
                               *, inspection_cost: float, wait_cost: float,
                               switch_reward: float, forecast_step: float = 1.0) -> dict:
        """E10: interpret public evidence into estimated consequences.

        This owner estimates option values, never chooses the final action.
        Reports are Bayesian cues weighted by historically VERIFIED reliability;
        neither source claims nor hidden world data are treated as ground truth.
        """
        now = state.world_time
        beliefs = {}
        soon = {}
        provenance = {}
        values = {"switch": float(switch_reward), "stop": 0.0}
        reports = state.beliefs.get("temporal_reports", {})
        deps = state.beliefs.get("temporal_dependencies", {})
        for person, props in public.items():
            exp = state.timed_expectations["delivery:" + person]
            promised = float(props["promised"])
            elapsed = max(0.0, now - exp.started_at)
            last_absent = (None if exp.last_verified_absent_at is None
                           else exp.last_verified_absent_at - exp.started_at)
            if exp.status == "fulfilled":
                probability = 1.0
                future = 1.0
            else:
                survivor = self.learner.temporal_survival(
                    exp.context, promised, elapsed)
                if last_absent is not None:
                    before = self.learner.temporal_survival(
                        exp.context, promised, last_absent)
                    probability = 1.0 - survivor / max(1e-9, before)
                else:
                    probability = 1.0 - survivor
                future = probability + (1 - probability) * self.learner.temporal_arrival_probability(
                    exp.context, promised, elapsed, elapsed + forecast_step)
                if person in reports and last_absent is None:
                    reported, speaker, evidence_id, _ = reports[person]
                    reliability = self.learner.temporal_source_reliability(
                        speaker, exp.context, now)
                    likelihood = reliability if reported else (1 - reliability)
                    inverse = (1 - reliability) if reported else reliability
                    probability = (probability * likelihood /
                                   max(1e-9, probability * likelihood
                                       + (1.0 - probability) * inverse))
            beliefs[person] = max(0.0, min(1.0, probability))
            soon[person] = max(beliefs[person], min(1.0, future))
            evidence_ids = [exp.source_event_id] + list(exp.observation_ids)
            if person in reports:
                evidence_ids.append(reports[person][2])
            if person in deps:
                evidence_ids.append(deps[person][1])
            provenance[person] = tuple(dict.fromkeys(evidence_ids))

        for person, props in public.items():
            prob = beliefs[person]
            future = soon[person]
            required = deps.get(person)
            if required:
                dep = required[0]
                if dep in beliefs:
                    prob *= beliefs[dep]
                    future *= soon[dep]
            payoff = float(props["payoff"])
            loss = float(props["premature_loss"])
            deadline = float(props["deadline"])
            proceed = (prob * payoff - (1.0 - prob) * loss
                       if now <= deadline else -loss)
            # Inspection consumes one unit of time. It can reveal presence
            # without forcing a premature proceed, but is not free.
            inspect = (future * payoff + (1 - future) * max(0.0, switch_reward)
                       - inspection_cost - wait_cost
                       if now + forecast_step <= deadline else -inspection_cost)
            values[f"proceed:{person}"] = proceed
            values[f"check:{person}"] = inspect

        # Waiting preserves the option to act near a future arrival. Compare
        # expected deadline reward against cumulative carrying costs; do not
        # confuse "not arrived yet" with "will never arrive".
        waiting_values = [0.0, switch_reward]
        for person, props in public.items():
            deadline = float(props["deadline"])
            if now + forecast_step > deadline:
                continue
            exp = state.timed_expectations["delivery:" + person]
            elapsed = max(0.0, now - exp.started_at)
            elapsed_deadline = max(elapsed, deadline - exp.started_at)
            probability = (1.0 if exp.status == "fulfilled" else
                1.0 - self.learner.temporal_survival(
                    exp.context, float(props["promised"]), elapsed_deadline))
            required = deps.get(person)
            if required and required[0] in public:
                dep = required[0]
                dep_props = public[dep]
                probability *= 1.0 - self.learner.temporal_survival(
                    dep_props["context"], float(dep_props["promised"]),
                    elapsed_deadline)
            expected_wait = min(max(0.0, float(props["promised"]) - now),
                                max(0.0, deadline - now))
            carrying_cost = wait_cost * max(forecast_step, expected_wait)
            waiting_values.append(
                probability * float(props["payoff"])
                - (1.0 - probability) * float(props["premature_loss"])
                - carrying_cost)
        values["wait"] = max(waiting_values) - wait_cost
        return {"values": values, "presence": beliefs, "future": soon,
                "provenance": provenance, "report_count": len(reports)}

    def prepare(self, hits) -> PreparedEvidence:
        """Build one index of grounded responses without duplicating any event."""
        pairs = {}
        examined = 0
        for hit in hits:
            events = self.memory.chunk_events(hit.chunk_id)
            examined += len(events)
            attempts = []
            outcomes_by_cause = {}
            observations_by_step = {}
            positions = {}
            for index, event in enumerate(events):
                positions[event.event_id] = index
                if event.kind == EventKind.ACTION and event.action:
                    attempts.append(event)
                elif (event.kind == EventKind.OUTCOME and
                      event.truth_status == TruthStatus.OUTCOME and
                      event.outcome in ("success", "failure") and event.cause_event_id):
                    outcomes_by_cause.setdefault(event.cause_event_id, []).append(event)
                elif (event.kind == EventKind.OBSERVATION and
                      event.truth_status == TruthStatus.FACT and event.informative):
                    observations_by_step.setdefault(event.step, []).append(event)
            for attempt in attempts:
                responses = list(outcomes_by_cause.get(attempt.event_id, ()))
                if attempt.action.kind != "try_key":
                    responses.extend(observations_by_step.get(attempt.step, ()))
                responses.sort(key=lambda event: positions[event.event_id])
                key = (attempt.action.kind,
                       attempt.action.key_name if attempt.action.kind == "try_key" else None)
                pairs.setdefault(key, []).extend(
                    (hit.chunk_id, attempt, response) for response in responses
                )
        return PreparedEvidence(
            pairs={key: tuple(value) for key, value in pairs.items()},
            examined_event_records=examined,
        )

    @staticmethod
    def context(state: CognitiveState, action: Action) -> str:
        if action.kind == "ask_informant":
            return "social"
        if action.kind == "try_key" and action.key_name in state.contradicted_keys:
            return "contradicted"
        if action.kind == "try_key" and action.key_name == state.candidate_key:
            return "supported_clue"
        return "exploration"

    @staticmethod
    def _matches(recorded: Action, candidate: Action) -> bool:
        if recorded.kind != candidate.kind:
            return False
        # Do not transfer success of an arbitrary historical key to another key.
        return recorded.kind != "try_key" or recorded.key_name == candidate.key_name

    def transfer_evidence(self, hits, features: frozenset[str], *,
                          review_depth: int = 0,
                          invalidated_since: dict[str, float] | None = None):
        """Evaluate provenance-bearing causal outcomes against observed validity changes.

        An observed change notice invalidates old applicability, not the historical
        facts. A documented current protocol can inform a provisional prediction
        but is not recast as an observed action outcome. Rumors are never promoted.
        """
        from collections import defaultdict
        invalidated_since = invalidated_since or {}
        records = defaultdict(dict)
        evaluations = 0
        invalidated = 0
        for hit in hits:
            events = self.memory.chunk_events(hit.chunk_id)
            attempts = {e.event_id: e for e in events if e.kind == EventKind.ACTION}
            evaluations += len(events)
            for event in events:
                causal = (
                    event.kind == EventKind.OUTCOME and
                    event.truth_status == TruthStatus.OUTCOME and
                    event.outcome in ("success", "failure") and
                    event.cause_event_id in attempts
                )
                protocol = (
                    event.kind == EventKind.OBSERVATION and
                    event.truth_status == TruthStatus.FACT and
                    event.metadata.get("source_status") == "current_protocol" and
                    isinstance(event.metadata.get("reported_axis"), str) and
                    isinstance(event.metadata.get("reported_token"), str)
                )
                if causal:
                    attempt = attempts[event.cause_event_id]
                    axis = attempt.action.key_name if attempt.action else None
                    token = attempt.action.target if attempt.action else None
                    value = 1.0 if event.outcome == "success" else -1.0
                    provenance = "causal_outcome"
                elif protocol:
                    axis = event.metadata["reported_axis"]
                    token = event.metadata["reported_token"]
                    value = 0.9
                    provenance = "protocol_observation"
                else:
                    continue
                cue = f"signal:{axis}:{token}"
                if cue not in features:
                    continue
                if event.occurrence_time < invalidated_since.get(axis, float("-inf")):
                    invalidated += 1
                    continue
                sid = str(event.metadata.get("source_event_id") or event.event_id)
                records[cue][sid] = (
                    event.occurrence_time, value,
                    max(0.0, min(1.0, event.confidence)),
                    event.event_id, provenance,
                )
        category_values = {}
        provenance = {}
        source_types = {}
        for cue in sorted(f for f in features if f.startswith("signal:")):
            history = sorted(records.get(cue, {}).values(), key=lambda r: (r[0], r[3]))
            if review_depth == 1:
                chosen = history[-6:]
            elif review_depth >= 3:
                chosen = history[-3:]
            else:
                chosen = history
            evaluations += len(chosen) * max(0, review_depth)
            total = sum(r[2] for r in chosen)
            category_values[cue] = (
                sum(r[1] * r[2] for r in chosen) / total if total else 0.0
            )
            provenance[cue] = [r[3] for r in chosen]
            source_types[cue] = sorted({r[4] for r in chosen})
        return {
            "score": sum(category_values.values()) / max(1, len(category_values)),
            "categories": category_values,
            "evidence_ids": sorted({eid for ids in provenance.values() for eid in ids}),
            "provenance": provenance,
            "source_types": source_types,
            "invalidated_evidence": invalidated,
            "evaluations": evaluations,
        }

    def evaluate(self, state: CognitiveState, action: Action, hits, *,
                 learn_layers: bool, adaptive_gate: bool,
                 prepared: PreparedEvidence | None = None) -> MeaningFrame:
        context = self.context(state, action)
        # Only the initial preparation reads events. Later candidate evaluations
        # examine already indexed evidence pairs without rescanning all events.
        owned_preparation = prepared is None
        if prepared is None:
            prepared = self.prepare(hits)
        examined_event_records = prepared.examined_event_records if owned_preparation else 0
        sources: dict[str, tuple[dict[str, float], float, bool, str, str]] = {}
        for chunk_id, attempt, response in prepared.matches(action):
            source_id = str(response.metadata.get("source_event_id") or response.event_id)
            if source_id in sources:
                continue
            chunk = self.memory.chunks[chunk_id]
            causal = response.kind == EventKind.OUTCOME
            signed = (1.0 if response.outcome == "success" else -1.0) if causal else 0.35
            contradicted = bool(action.key_name and action.key_name in state.contradicted_keys)
            goal_related = chunk.header.goal_id == state.goal_id
            person_matches = bool(action.person and attempt.person_id == action.person)
            values = {
                "self": signed if goal_related else 0.0,
                "salience": signed * chunk.header.salience,
                "time": -1.0 if contradicted and signed > 0 else 0.0,
                "sequence": signed if causal else 0.0,
                "person": signed if person_matches else 0.0,
            }
            conf = min(1.0, max(0.0, response.confidence))
            if not causal:
                conf *= 0.65
            sources[source_id] = (
                values, conf, not contradicted,
                "observed_outcome" if causal else "informative_observation",
                chunk_id,
            )

        meanings: list[LayerMeaning] = []
        for axis in AXES:
            matches = [(sid, record) for sid, record in sources.items() if record[0][axis] != 0.0]
            if not matches:
                continue  # No fabricated neutral "evidence" for a missing axis.
            raw = sum(record[0][axis] * record[1] for _, record in matches) / len(matches)
            association = self.learner.layer_association(axis, context, action.kind) if learn_layers else 0.0
            value = max(-1.0, min(1.0, raw * (1.0 + association)))
            confidence = sum(record[1] for _, record in matches) / len(matches)
            meaning = LayerMeaning(
                axis=axis, value=value, confidence=confidence,
                source_event_ids=tuple(sorted(sid for sid, _ in matches)),
                temporal_valid=all(record[2] for _, record in matches),
                goal_related=any(self.memory.chunks[record[4]].header.goal_id == state.goal_id
                                 for _, record in matches),
                provenance="observed_outcome" if all(record[3] == "observed_outcome" for _, record in matches)
                else "includes_informative_observation",
                association=association,
            )
            meanings.append(meaning)
            for _, record in matches:
                self.memory.store_layer_meaning(record[4], axis, action.label, meaning)

        if not meanings:
            return MeaningFrame(context, action.kind, (), {}, 0.0, 0, examined_event_records)
        logits = {m.axis: (self.learner.integration_logit(context, m.axis)
                           if adaptive_gate else 0.0) for m in meanings}
        pivot = max(logits.values())
        normalizer = sum(math.exp(v - pivot) for v in logits.values())
        weights = {axis: math.exp(logit - pivot) / normalizer for axis, logit in logits.items()}
        integrated = sum(weights[m.axis] * m.value for m in meanings)
        return MeaningFrame(context, action.kind, tuple(meanings), weights, integrated, len(sources), examined_event_records)

    def contextual_evidence(self, hits, features: frozenset[str], *,
                            procedure: str, source: str, source_context: str,
                            verified: bool, attention_cues: frozenset[str],
                            alternative_attention_cues: frozenset[str] | None = None) -> dict:
        """Evaluate combined context, observed testimony and causal abstractions.

        A correct procedure is never supplied here. Historical episodic evidence
        remains accessible; the learner stores revisable aggregates of observed
        outcomes with evidence provenance, not an answer table.
        """
        active = frozenset(c for c in attention_cues if c.startswith("pair:"))
        if not active:
            active = frozenset(c for c in features if c.startswith("pair:"))
        abstraction = self.learner.contextual_support(active, procedure)
        retrieved = []
        evaluations = 0
        for hit in hits:
            events = self.memory.chunk_events(hit.chunk_id)
            evaluations += len(events)
            attempts = {e.event_id: e for e in events if e.kind == EventKind.ACTION}
            for event in events:
                if (event.kind != EventKind.OUTCOME or
                        event.outcome not in ("success", "failure") or
                        event.cause_event_id not in attempts):
                    continue
                attempt = attempts[event.cause_event_id]
                if not attempt.action or attempt.action.target != procedure:
                    continue
                overlap = len(active & attempt.semantic_tags)
                if overlap:
                    retrieved.append((event.event_id,
                                      1.0 if event.outcome == "success" else -1.0,
                                      overlap / len(active)))
        retrieved.sort(key=lambda x: x[0])
        # Retrieved episodic outcomes have bounded influence and cannot
        # overwhelm current observations simply by being repetitive.
        history = retrieved[-8:]
        episodic = (sum(value * overlap for _, value, overlap in history) /
                    max(1, len(history))) if history else 0.0
        reliability = (1.0 if verified else
                       self.learner.source_reliability(source, source_context))
        # Unknown reports are *probabilistic claims*, not facts. Marginalize
        # over publicly possible alternatives rather than reinforcing a wrong
        # dependency merely because it was reported confidently.
        alternative = (self.learner.contextual_support(
            alternative_attention_cues, procedure)
            if alternative_attention_cues is not None and not verified else None)
        if alternative is None:
            causal = abstraction["score"]
        else:
            causal = (reliability * abstraction["score"] +
                      (1.0 - reliability) * alternative["score"])
        score = 0.82 * causal + 0.18 * episodic
        ids = sorted(set(abstraction["evidence_ids"]) |
                     set(alternative["evidence_ids"] if alternative else ()) |
                     {event_id for event_id, _, _ in history})
        return {"score": max(-1.0, min(1.0, score)),
                "evidence_ids": ids,
                "causal_evidence_ids": ids,
                "provenance": {"contextual_outcomes": ids},
                "covered_relations": abstraction["covered_relations"],
                "source_reliability": reliability,
                "alternative_causal_score": (
                    alternative["score"] if alternative is not None else None),
                "reported_causal_score": abstraction["score"],
                "evaluations": evaluations + len(history),
                "hypothesis": procedure,
                "verified": verified}

    def uncertainty_evidence(self, hits, features: frozenset[str], *,
                             context: str, invalidated_since: dict[str, float] | None = None,
                             review_depth: int = 0) -> dict:
        """Ground fallible reports against source calibration and causal history.

        A testimony is an observation of a claim, never an observed success.
        Invalidated causal associations remain historically true but cease to
        govern the current epoch. Processing is restricted to retrieved chunks.
        """
        base = self.transfer_evidence(
            hits, features, review_depth=review_depth,
            invalidated_since=invalidated_since)
        invalidated_since = invalidated_since or {}
        reports = {}
        checked = 0
        conflicts = set()
        report_axis_tokens = {}
        for hit in hits:
            for event in self.memory.chunk_events(hit.chunk_id):
                checked += 1
                if event.kind != EventKind.OBSERVATION:
                    continue
                meta = event.metadata
                if meta.get("source_status") != "testimony" or meta.get("context") != context:
                    continue
                axis, token, source = (meta.get("reported_axis"),
                                       meta.get("reported_token"), meta.get("source_name"))
                if not all(isinstance(x, str) for x in (axis, token, source)):
                    continue
                if event.occurrence_time < invalidated_since.get(axis, float("-inf")):
                    continue
                cue = f"signal:{axis}:{token}"
                report_axis_tokens.setdefault(axis, set()).add(token)
                reliability = self.learner.source_reliability(source, context)
                # Values at or below chance can count as contradictory, not
                # corroborating; the confidence comes from verified history.
                weight = max(-0.5, min(0.5, reliability - 0.5)) * 2.0
                if cue in features:
                    reports[event.event_id] = (axis, weight, source, reliability)
        for axis, tokens in report_axis_tokens.items():
            if len(tokens) > 1:
                conflicts.add(axis)
        report_score = sum(value[1] for value in reports.values()) / max(1, len(reports))
        has_causal = bool(base["evidence_ids"])
        score = max(-1.0, min(1.0,
                             0.85 * base["score"] + 0.38 * report_score))
        evidence_ids = sorted(set(base["evidence_ids"]) | set(reports))
        return {**base, "score": score, "causal_score": base["score"],
                "report_score": report_score, "evidence_ids": evidence_ids,
                "report_evidence_ids": sorted(reports),
                "contradictory_axes": sorted(conflicts),
                "source_reliability": {v[2]: v[3] for v in reports.values()},
                "has_causal_evidence": has_causal,
                "evaluations": base["evaluations"] + checked}
