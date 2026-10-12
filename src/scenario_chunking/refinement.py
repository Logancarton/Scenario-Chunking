"""Evidence-grounded thought continuity; a refinement record is not a new episode.

The existing five-axis interpreter supplies evidence once per candidate. This owner
compares context-only and evidence-integrated interpretations without rereading
memory. Versions reflect changes in evidence or understanding, not repeated scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from .models import Action, RetrievalHit, EventKind, TruthStatus
from .state import CognitiveState


@dataclass(frozen=True)
class ThoughtVersion:
    number: int
    preferred_action: str
    alternatives: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    memory_chunk_ids: tuple[str, ...]
    contradictions: tuple[str, ...]
    uncertainty: float
    reason: str
    signature: tuple[object, ...]
    time: float
    hypotheses: tuple[dict[str, object], ...] = ()
    predicted_outcomes: tuple[tuple[str, float], ...] = ()

    def trace(self) -> dict[str, object]:
        return {
            "version": self.number, "preferred_action": self.preferred_action,
            "alternatives": list(self.alternatives),
            "evidence_ids": list(self.evidence_ids),
            "memory_chunk_ids": list(self.memory_chunk_ids),
            "contradictions": list(self.contradictions),
            "uncertainty": round(self.uncertainty, 6),
            "reason": self.reason, "time": self.time,
            "hypotheses": list(self.hypotheses),
            "predicted_outcomes": dict(self.predicted_outcomes),
        }


@dataclass
class ThoughtThread:
    thought_id: str
    initiating_goal: str
    initiating_episode: str
    versions: list[ThoughtVersion] = field(default_factory=list)
    outcomes: list[dict[str, object]] = field(default_factory=list)
    # Bounded pointers into the existing episodic store, never copied events.
    causal_anchors: dict[str, str] = field(default_factory=dict)
    activation_strength: float = 1.0
    activation_last_time: float = 0.0
    # E12: an investigation's open questions persist with its identity. Attention
    # (activation) may fade; neither the questions nor learned knowledge decay.
    open_reviews: list[dict[str, object]] = field(default_factory=list)
    review_history: list[dict[str, object]] = field(default_factory=list)
    last_outcome: dict[str, object] | None = None
    # E12 investigation: appraisals of HOW past decisions were reached, and the
    # advantages the scarce option offered (for pricing its future use).
    process_appraisals: list[dict[str, object]] = field(default_factory=list)
    option_advantages: list[float] = field(default_factory=list)

    def activation_at(self, now: float, decay_rate: float = 0.08) -> float:
        return self.activation_strength * math.exp(-decay_rate * max(0.0, now - self.activation_last_time))

    def status_at(self, now: float) -> str:
        value = self.activation_at(now)
        return 'focused' if value >= 1.0 else ('peripheral' if value >= 0.2 else 'dormant')


class ProgressiveRefiner:
    """At most two bounded interpretation stages per decision, with no extra detail reads.

    Stage 1 represents the current context without historical meaning scores.
    Stage 2 validates existing interpreted sources and weighs their applicability.
    The same signature never generates a redundant new version.
    """

    def __init__(self, learner, *, learning: bool = False, max_versions: int = 256,
                 max_focused: int = 2, max_active: int = 4) -> None:
        self.learner = learner
        self.learning = learning
        self.max_versions = max_versions
        self.max_focused = max_focused
        self.max_active = max_active
        if min(max_versions, max_focused, max_active) < 1 or max_focused > max_active:
            raise ValueError('thought capacity must be positive with focused <= active')
        self._active: dict[str, ThoughtThread] = {}
        self.threads: dict[tuple[str, str], ThoughtThread] = {}
        self.max_open_reviews = 2
        self.review_patience = 8

    def thread(self, state: CognitiveState) -> ThoughtThread:
        key = (getattr(state, "thought_key", None) or state.episode_id, state.goal_id)
        if key not in self.threads:
            self.threads[key] = ThoughtThread(
                thought_id=f"{key[0]}:{state.goal_id}",
                initiating_episode=state.episode_id,
                initiating_goal=state.goal_id,
            )
        return self.threads[key]

    def _focus(self, thread: ThoughtThread, now: float) -> None:
        # Only scan the bounded active set; dormant history remains indexed but
        # receives zero repeated processing until its specific thought is cued.
        thread.activation_strength = min(2.0, thread.activation_at(now) + 1.0)
        thread.activation_last_time = now
        self._active[thread.thought_id] = thread
        for tid, candidate in tuple(self._active.items()):
            if candidate.status_at(now) == 'dormant':
                del self._active[tid]
        focused = sorted(
            (t for t in self._active.values() if t.status_at(now) == 'focused'),
            key=lambda t: (t is not thread, -t.activation_at(now), t.thought_id),
        )
        for displaced in focused[self.max_focused:]:
            displaced.activation_strength = min(0.6, displaced.activation_at(now))
            displaced.activation_last_time = now
        active = sorted(self._active.values(),
                        key=lambda t: (t is not thread, -t.activation_at(now), t.thought_id))
        for displaced in active[self.max_active:]:
            displaced.activation_strength = 0.0
            displaced.activation_last_time = now
            del self._active[displaced.thought_id]

    @staticmethod
    def _uncertainty(ordered: list[tuple[float, Action, list[str], object]]) -> float:
        margin = ordered[0][0] - ordered[1][0] if len(ordered) > 1 else 4.0
        return 1.0 / (1.0 + max(0.0, margin))

    @staticmethod
    def _hypotheses(state: CognitiveState, scored, prepared) -> list[dict[str, object]]:
        """Build falsifiable key interpretations from already-indexed causal pairs.

        Historical outcomes transfer only to the same goal, target and key.
        Observed current clues are weighted by their actual source confidence.
        No new chunk events are read, and no speculative causal edge is created.
        """
        hypotheses = []
        for _, action, _, _ in scored:
            if action.kind != "try_key" or not action.key_name:
                continue
            positives: dict[str, tuple[float, str]] = {}
            negatives: dict[str, tuple[float, str]] = {}
            for source_id, agrees, conf, provenance in state.key_observations.get(action.key_name, ()):
                (positives if agrees else negatives)[source_id] = (
                    (2.8 if agrees else 3.1) * max(0.0, min(1.0, conf)),
                    provenance,
                )
            historical: dict[str, tuple[bool, float]] = {}
            if prepared is not None:
                for _, attempt, response in prepared.matches(action):
                    if (attempt.action != action or attempt.goal_id != state.goal_id
                            or response.kind.value != "outcome"
                            or response.truth_status.value != "outcome"
                            or response.cause_event_id != attempt.event_id
                            or response.outcome not in ("success", "failure")):
                        continue
                    source_id = str(response.metadata.get("source_event_id") or response.event_id)
                    if source_id in positives or source_id in negatives:
                        continue
                    historical[source_id] = (
                        response.outcome == "success",
                        max(0.0, min(1.0, response.confidence)),
                    )
            # Bound the influence of repeated historical examples and disallow
            # mere memory quantity from becoming certainty.
            historical_support = min(1.8, sum(1.8 * c for good, c in historical.values() if good))
            historical_against = min(1.8, sum(1.8 * c for good, c in historical.values() if not good))
            log_odds = (
                sum(weight for weight, _ in positives.values())
                - sum(weight for weight, _ in negatives.values())
                + historical_support - historical_against
            )
            log_odds = max(-6.0, min(6.0, log_odds))
            probability = 1.0 / (1.0 + math.exp(-log_odds))
            supportive = tuple(sorted(set(positives) | {sid for sid, (good, _) in historical.items() if good}))
            conflicting = tuple(sorted(set(negatives) | {sid for sid, (good, _) in historical.items() if not good}))
            sources = tuple(sorted(set(supportive) | set(conflicting)))
            hypotheses.append({
                "action": action.label,
                "claim": f"{action.key_name} opens {action.target}",
                "confidence": round(probability, 6),
                "supports": supportive,
                "contradicts": conflicting,
                "sources": sources,
                "provenance": tuple(sorted({
                    tag for _, tag in (*positives.values(), *negatives.values())
                } | ({"historical_causal_outcome"} if historical else set()))),
                "uncertain": 0.30 < probability < 0.78,
            })
        return hypotheses

    def refine(self, state: CognitiveState, scored, hits: list[RetrievalHit], *, prepared=None):
        if not scored:
            raise ValueError("refinement requires candidate actions")
        thread = self.thread(state)
        self._focus(thread, state.current_time)
        direct = sorted(scored, key=lambda x: (-x[0], x[1].label))
        hypotheses = self._hypotheses(state, scored, prepared)
        known = [h for h in hypotheses if h["sources"]]
        contenders = sorted(known, key=lambda h: (-h["confidence"], h["action"]))
        desired_label = None
        decision_basis = "one_pass_adequate"
        if contenders:
            strongest = contenders[0]
            runner = contenders[1]["confidence"] if len(contenders) > 1 else 0.5
            investigative = [item for item in direct if item[1].kind != "try_key"]
            current_cues = any(state.key_observations.values())
            # A past success of the same key and object is not proof the
            # relationship still holds. Check cheaply available current
            # evidence before committing to an uncorroborated old result.
            historical_only = (
                strongest["provenance"] == ("historical_causal_outcome",)
                and not current_cues and bool(investigative)
            )
            if historical_only:
                if direct[0][1].kind == "try_key":
                    desired_label = investigative[0][1].label
                decision_basis = "verify_historical_applicability"
            elif strongest["confidence"] >= 0.78 and strongest["confidence"] - runner >= 0.13:
                desired_label = strongest["action"]
                decision_basis = "evidence_supported_prediction"
            elif direct[0][1].kind == "try_key" and investigative:
                desired_label = investigative[0][1].label
                decision_basis = "resolve_competing_hypotheses"
        # A memory axis may show activity for a superficially similar
        # historical attempt, but if no candidate hypothesis has an applicable
        # causal source, that is NOT evidence for using the historical key.
        if not contenders and not any(state.key_observations.values()):
            investigative = [row for row in direct if row[1].kind != "try_key"]
            if investigative and direct[0][1].kind == "try_key":
                selected_frame = direct[0][3]
                if selected_frame and selected_frame.independent_sources:
                    desired_label = investigative[0][1].label
                    decision_basis = "reject_unverified_transfer"
        ordered = list(direct)
        if desired_label and direct[0][1].label != desired_label:
            chosen = next((row for row in ordered if row[1].label == desired_label), None)
            if chosen:
                ordered.remove(chosen)
                ordered.insert(0, (direct[0][0] + 0.001, chosen[1],
                                   chosen[2] + [f"thought={decision_basis}"], chosen[3]))
        preferred = ordered[0][1].label
        alternatives = tuple(item[1].label for item in ordered[1:4])
        contradictions = tuple(sorted(state.contradicted_keys))
        local_observations = tuple(sorted(
            (key, eid, good, round(conf, 6))
            for key, evidence in state.key_observations.items()
            for eid, good, conf, _ in evidence
        ))
        hypothesis_signature = tuple(
            (h["action"], h["supports"], h["contradicts"], h["confidence"])
            for h in sorted(hypotheses, key=lambda h: h["action"])
        )
        signature = (state.goal_id, state.candidate_key, contradictions,
                     local_observations, hypothesis_signature)
        memory_ids = tuple(sorted({h.chunk_id for h in hits}))
        evidence_ids = tuple(sorted({sid for h in hypotheses for sid in h["sources"]}))
        supported_hypothesis = next((h for h in hypotheses if h["action"] == preferred), None)
        selected_prediction = (supported_hypothesis["confidence"]
                               if supported_hypothesis and supported_hypothesis["sources"] else None)
        history_before = len(thread.versions)
        hypotheses_record = tuple({
            "claim": h["claim"], "action": h["action"],
            "supports": list(h["supports"]), "contradicts": list(h["contradicts"]),
            "sources": list(h["sources"]),
            "confidence": h["confidence"], "provenance": list(h["provenance"]),
        } for h in hypotheses)
        predictions = tuple((h["action"], h["confidence"]) for h in hypotheses)

        if not thread.versions:
            local = sorted(
                scored, key=lambda x: (-(x[0] - next(
                    (float(reason.split("=", 1)[1]) for reason in x[2]
                     if reason.startswith("memory=")), 0.0)), x[1].label))
            thread.versions.append(ThoughtVersion(
                number=1, preferred_action=local[0][1].label,
                alternatives=tuple(x[1].label for x in local[1:4]),
                evidence_ids=tuple(sorted(state.observed_evidence_ids)),
                memory_chunk_ids=(), contradictions=contradictions,
                uncertainty=self._uncertainty(local), reason="initial_context_hypothesis",
                signature=("initial",) if known else signature,
                time=state.current_time,
            ))
        previous = thread.versions[-1]
        # Only relevant evidence or changed interpretation causes a revision;
        # stable evidence cannot create thought versions by looping or RNG.
        if previous.signature != signature and len(thread.versions) < self.max_versions:
            reason = ("contradiction_resolved_or_exposed"
                      if contradictions != previous.contradictions else
                      "revised_from_grounded_evidence")
            thread.versions.append(ThoughtVersion(
                number=len(thread.versions) + 1, preferred_action=preferred,
                alternatives=alternatives, evidence_ids=evidence_ids,
                memory_chunk_ids=memory_ids, contradictions=contradictions,
                uncertainty=(1 - max((h["confidence"] for h in known), default=0.5)),
                reason=reason, signature=signature, time=state.current_time,
                hypotheses=hypotheses_record, predicted_outcomes=predictions,
            ))
        current = thread.versions[-1]
        selected_support = list(supported_hypothesis["supports"]) if supported_hypothesis else []
        return ordered, {
            "thought_id": thread.thought_id, "version": current.number,
            "activation_state": thread.status_at(state.current_time),
            "changed": len(thread.versions) != history_before,
            "reasoning_cycles": 2 if known else 1,
            "stopped_reason": ("no_relevant_new_evidence"
                               if len(thread.versions) == history_before else
                               "evidence_revision_complete"),
            "current": current.trace(), "history_length": len(thread.versions),
            "hypotheses": list(hypotheses_record),
            "decision_basis": decision_basis,
            "prediction": selected_prediction,
            "contributing_evidence_ids": selected_support,
            "meaningfully_changed_action": preferred != direct[0][1].label,
            "additional_detailed_reads": 0,
        }

    def refine_contextual(self, state, scored, hits, *, context: dict):
        """Revise the existing thought identity when verification changes evidence.

        Reflection uses observed confidence, consequences and learned information
        value. It may investigate, or stop immediately when evidence suffices.
        The world answer and hidden dependency are never consulted here.
        """
        if not scored:
            raise ValueError("cannot refine empty contextual choices")
        thread = self.thread(state)
        self._focus(thread, state.current_time)
        ordered = sorted(scored, key=lambda x: (-x[0], x[1].label))
        hypotheses = [row for row in ordered if row[1].kind == "resolve"]
        if not hypotheses:
            raise ValueError("no candidate procedures")
        first = hypotheses[0]
        second = hypotheses[1] if len(hypotheses) > 1 else hypotheses[0]
        margin = first[0] - second[0]
        uncertain = 1.0 / (1.0 + max(0.0, margin))
        verified = bool(context.get("verified_dependency"))
        consequence = float(context.get("consequence", .5))
        reliability = float(context.get("source_reliability", .5))
        audits = context.get("audits", {})
        reported_rank = sorted(
            hypotheses, key=lambda row: (
                -audits.get(row[1].label, {}).get("reported_causal_score", 0.0),
                row[1].label))
        alternative_rank = sorted(
            hypotheses, key=lambda row: (
                -(audits.get(row[1].label, {}).get("alternative_causal_score")
                  or 0.0), row[1].label))
        disagree = reported_rank[0][1] != alternative_rank[0][1]
        # Decision-contingent expected value of reducing uncertainty. Source
        # trust is verified from prior observed inspection outcomes; the
        # inspection-value prior never masquerades as proven goal success.
        expected_information_gain = (
            uncertain * (1.0 - reliability) * consequence *
            (0.5 + 0.5 * self.learner.observation_value()) *
            (1.4 if disagree else 0.85) *
            (1.2 if context.get("revision_notice") else 1.0)
        )
        inspection = next((row for row in ordered
                           if row[1].kind == "inspect_dependency"), None)
        should_inspect = bool(
            inspection is not None and not verified and
            expected_information_gain > float(context.get("inspection_cost", .25)))
        choice = inspection if should_inspect else first
        basis = ("verify_uncertain_dependency" if should_inspect else
                 "stop_after_sufficient_evidence" if margin > .5 else
                 "act_under_remaining_uncertainty")
        evidence = tuple(sorted(set(
            context.get("audits", {}).get(choice[1].label, {}).get(
                "evidence_ids", ()))))
        signature = (
            choice[1].label, evidence, verified,
            tuple(sorted(context.get("attention_cues", ()))),
            basis,
        )
        before = len(thread.versions)
        if not thread.versions or thread.versions[-1].signature != signature:
            if len(thread.versions) < self.max_versions:
                thread.versions.append(ThoughtVersion(
                    number=len(thread.versions) + 1,
                    preferred_action=choice[1].label,
                    alternatives=tuple(row[1].label for row in hypotheses[:3]
                                       if row[1].label != choice[1].label),
                    evidence_ids=evidence,
                    memory_chunk_ids=tuple(hit.chunk_id for hit in hits),
                    contradictions=("unverified_dependency",) if not verified else (),
                    uncertainty=uncertain,
                    reason=basis, signature=signature,
                    time=state.current_time,
                    hypotheses=tuple({
                        "action": row[1].label,
                        "relative_score": round(row[0], 5),
                    } for row in hypotheses[:4]),
                ))
        # Selection still occurs through ActionSelector; the refiner only
        # changes preferences within that existing scored action set.
        if should_inspect:
            ordered = [
                (max(row[0] for row in ordered) + .01, choice[1],
                 choice[2] + ["thought=targeted_inspection"], choice[3])
            ] + [row for row in ordered if row[1] != choice[1]]
        return ordered, {
            "thought_id": thread.thought_id,
            "version": thread.versions[-1].number,
            "changed": len(thread.versions) != before,
            "decision_basis": basis,
            "prediction": None,
            "reasoning_cycles": 2 if should_inspect else 1,
            "contributing_evidence_ids": list(evidence),
            "expected_information_gain": expected_information_gain,
            "dependency_hypotheses_disagree": disagree,
            "attention_cues": sorted(context.get("attention_cues", ())),
            "meaningfully_changed_action": choice[1] != ordered[0][1]
                                      if not should_inspect else
                                      choice[1] != first[1],
        }

    def refine_transfer(self, state, scored, hits, *, context: dict):
        """Evidence-linked refinement for the same shared action-selection pathway.

        Stage one is the current policy ranking. Stage two checks whether a
        previously trusted axis has been invalidated without new applicable
        evidence. It can choose a genuine information-seeking action; it never
        guesses which candidate is correct from the invalidation alone.
        """
        if not scored:
            raise ValueError("transfer refinement needs candidate actions")
        thread = self.thread(state)
        self._focus(thread, state.current_time)
        direct = sorted(scored, key=lambda row: (-row[0], row[1].label))
        decisions = context.get("audits", {})
        invalidated = set(getattr(state, "invalidated_since", {}))
        resolved = set(context.get("resolved_change_axes", ()))
        validity_unresolved = []
        for axis in sorted(invalidated - resolved):
            applicable = any(
                record.get("provenance", {}).get(cue)
                for record in decisions.values()
                for cue in record.get("categories", {})
                if cue.startswith(f"signal:{axis}:")
                and record.get("categories", {}).get(cue, 0) > 0
            )
            if not applicable:
                validity_unresolved.append(axis)
        # Inspection is a candidate affordance exposed to ALL conditions.
        # The refiner may activate it only when evidence shows the old
        # relationship has become inapplicable and current truth is unknown.
        investigate = next(
            (row for row in direct if row[1].kind in ("inspect_protocol", "inspect_dependencies")
             and row[1].key_name in validity_unresolved), None
        )
        reason = "evidence_already_sufficient"
        ordered = list(direct)
        if investigate is not None:
            reason = "verify_invalidated_relationship"
            ordered.remove(investigate)
            ordered.insert(0, (direct[0][0] + 0.001, investigate[1],
                               investigate[2] + [f"thought={reason}"], investigate[3]))
        preferred = ordered[0][1].label
        selected_evidence = tuple(sorted(set(
            decisions.get(preferred, {}).get("evidence_ids", ())
        )))
        evidence_signature = tuple(sorted(
            (label, tuple(sorted(record.get("evidence_ids", ()))),
             tuple(sorted(record.get("categories", {}).items())))
            for label, record in decisions.items()
        ))
        signature = (state.goal_id, preferred, tuple(validity_unresolved), evidence_signature)
        before = len(thread.versions)
        if not thread.versions:
            thread.versions.append(ThoughtVersion(
                number=1, preferred_action=direct[0][1].label,
                alternatives=tuple(row[1].label for row in direct[1:4]),
                evidence_ids=(), memory_chunk_ids=(),
                contradictions=tuple(validity_unresolved),
                uncertainty=self._uncertainty(direct),
                reason="initial_policy_hypothesis", signature=("initial",),
                time=state.current_time,
            ))
        if (thread.versions[-1].signature != signature and
                len(thread.versions) < self.max_versions):
            thread.versions.append(ThoughtVersion(
                number=len(thread.versions) + 1, preferred_action=preferred,
                alternatives=tuple(row[1].label for row in ordered[1:4]),
                evidence_ids=selected_evidence,
                memory_chunk_ids=tuple(sorted(hit.chunk_id for hit in hits)),
                contradictions=tuple(validity_unresolved),
                uncertainty=self._uncertainty(ordered),
                reason=reason, signature=signature,
                time=state.current_time,
            ))
        return ordered, {
            "thought_id": thread.thought_id,
            "version": thread.versions[-1].number,
            "activation_state": thread.status_at(state.current_time),
            "changed": len(thread.versions) != before,
            "reasoning_cycles": 2 if invalidated else 1,
            "decision_basis": reason,
            "current": thread.versions[-1].trace(),
            "contributing_evidence_ids": list(selected_evidence),
            "invalidated_axes_without_current_evidence": validity_unresolved,
            "meaningfully_changed_action": preferred != direct[0][1].label,
            "additional_detailed_reads": 0,
        }

    def record_transfer(self, state, *, preferred: str, evidence_ids,
                        alternatives=(), review_depth: int = 1) -> dict:
        """Keep one evidence-grounded thought identity across transfer decisions.

        Reflection revisions are recorded only when preference or supporting
        observations change; repeated thinking cannot validate a hypothesis.
        """
        thread = self.thread(state)
        self._focus(thread, state.current_time)
        sources = tuple(sorted(set(evidence_ids)))
        signature = (state.goal_id, preferred, sources)
        changed = not thread.versions or thread.versions[-1].signature != signature
        if changed and len(thread.versions) < self.max_versions:
            thread.versions.append(ThoughtVersion(
                number=len(thread.versions) + 1,
                preferred_action=preferred,
                alternatives=tuple(alternatives),
                evidence_ids=sources,
                memory_chunk_ids=(),
                contradictions=(),
                uncertainty=0.5,
                reason="observed_evidence_review",
                signature=signature,
                time=state.current_time,
            ))
        return {
            "thought_id": thread.thought_id,
            "version": thread.versions[-1].number if thread.versions else 0,
            "changed": changed,
            "review_depth": review_depth,
            "evidence_ids": list(sources),
            "activation_state": thread.status_at(state.current_time),
        }

    def attention_status(self, state: CognitiveState) -> str:
        """Read-only: the thread's attention on arrival, before it is refocused."""
        key = (getattr(state, "thought_key", None) or state.episode_id, state.goal_id)
        thread = self.threads.get(key)
        return "absent" if thread is None else thread.status_at(state.current_time)

    def pending_reviews(self, state: CognitiveState, *, scope, procedure: str) -> list[dict]:
        """Open questions this continuing thought carries for one scope/action."""
        thread = self.thread(state)
        return [dict(r) for r in thread.open_reviews
                if r["scope"] == list(scope) and r["procedure"] == procedure]

    def record_causal_revision(self, state: CognitiveState, update: dict, outcome, *,
                               now: float | None = None) -> dict:
        """Record the learner's revision in the SAME explanation thread.

        now is the simulated acquisition time of the outcome; processing ticks
        (state.current_time after apply_signal) are not elapsed time.
        """
        now = state.current_time if now is None else now
        thread = self.thread(state)
        self._focus(thread, now)
        hypotheses = tuple(update.get("hypotheses", ()))
        signature = (outcome.event_id, tuple(update["updated"]))
        if len(thread.versions) < self.max_versions:
            thread.versions.append(ThoughtVersion(
                number=len(thread.versions) + 1, preferred_action="",
                alternatives=(), evidence_ids=(outcome.event_id,), memory_chunk_ids=(),
                contradictions=tuple(sorted({eid for h in hypotheses for eid in h["contradictions"]})),
                uncertainty=1-min((h["confidence"] for h in hypotheses), default=0.0),
                reason=("explanation_revised" if update.get("revised")
                        else "observed_evidence_revision"), signature=signature,
                time=now, hypotheses=hypotheses))
        thread.outcomes.append({"episode_id": state.episode_id,
                                "event_id": outcome.event_id, "updated": update["updated"],
                                "explanation": update.get("explanation")})
        del thread.outcomes[:-self.max_versions]
        # Chronology for sequence/time meaning: what this investigation last did.
        thread.last_outcome = {"procedure": outcome.metadata.get("procedure"),
                               "outcome": outcome.outcome, "time": now,
                               "event_id": outcome.event_id}
        # The learner validated and resolved reviews; the thread only carries them.
        status = update.get("review_status") or {}
        still_open = []
        for review in thread.open_reviews:
            verdict = status.get(review["review_id"])
            age = review.get("experiences", 0) + (1 if verdict is not None else 0)
            if verdict == "resolved" or age > self.review_patience:
                thread.review_history.append({**review, "closed_by": outcome.event_id,
                    "status": "resolved" if verdict == "resolved" else "abandoned",
                    "time": now})
            else:
                still_open.append({**review, "experiences": age})
        new = update.get("review")
        if new is not None and status.get(new["review_id"]) != "resolved":
            still_open.append({**new, "experiences": 0, "opened_time": now})
        elif new is not None:
            thread.review_history.append({**new, "closed_by": outcome.event_id,
                                          "status": "resolved", "time": now})
        # Bounded: the oldest open question yields to a newer surprise.
        while len(still_open) > self.max_open_reviews:
            dropped = still_open.pop(0)
            thread.review_history.append({**dropped, "closed_by": outcome.event_id,
                                          "status": "displaced", "time": now})
        thread.open_reviews = still_open
        del thread.review_history[:-self.max_versions]
        return {"thought_id": thread.thought_id, "version": len(thread.versions),
                "updated": update["updated"], "evidence_id": outcome.event_id,
                "explanation_before": update.get("explanation_before"),
                "explanation": update.get("explanation"), "revised": update.get("revised", False),
                "open_reviews": [r["review_id"] for r in thread.open_reviews]}

    def assess_incremental_adequacy(self, state: CognitiveState, analysis: dict, *,
                                    permitted: frozenset[str], omitted: int) -> dict:
        """Predecision process appraisal using ONLY public predictions and values.

        Distinguishes restricted actions, unfavorable known choices, gaps in
        evidence and possible reasoning insufficiency. The last is a signal
        to consider an extra bounded pass, not a diagnosis of an omitted cause.
        """
        if not permitted:
            status = "safety"
            return {"status": status, "uncertainty": 0.0, "margin": 0.0}
        choices = [analysis["values"]["route:" + p] for p in sorted(permitted)]
        choices.append(analysis["values"]["route:defer"])
        ordered = sorted(choices, reverse=True)
        margin = ordered[0] - ordered[1]
        uncertainty = max(analysis["predictions"]["route:" + p]["uncertainty"]
                          for p in permitted)
        if not analysis["unknown"]:
            status = "adequate"
        elif ordered[0] == analysis["values"]["route:defer"] and uncertainty < 0.2:
            status = "negative_evidence"
        elif uncertainty < 0.2 and margin >= 0.25:
            status = "adequate"
        elif ordered[0] == analysis["values"]["route:defer"]:
            status = "world_uncertainty"
        elif omitted:
            status = "needs_reconsideration"
        else:
            status = "world_uncertainty"
        return {"status": status, "uncertainty": uncertainty, "margin": margin}

    def appraise_process(self, state: CognitiveState, *, record: dict, hindsight: dict,
                         unexplained: float, now: float, information_gain: dict | None = None,
                         association: dict | None = None) -> dict:
        """Appraise the cognitive process behind one decision, after its outcome.

        Not "was the prediction right?" but "how was the decision reached?":
        which variables were examined and how much each improved the prediction
        of what actually happened (prequential information gain), how strongly
        the outcomes examined so far depend on each variable (model-free
        association evidence supplied by the knowledge owner), which
        examinations current knowledge judges warranted in hindsight (value of
        information above cost, in the visible context the decision started
        from), whether an error is attributable to information that was
        available but not examined, whether the action followed from the values
        and respected the constraints, and what the reasoning cost. The thought
        owner records the appraisal; only the learner turns it into process
        learning.
        """
        verdicts = {}
        for axis, h in sorted(hindsight.items()):
            verdicts[axis] = {"information_value": h["voi"], "cost": h["cost"],
                              "warranted": h["voi"] > h["cost"], "competence": h["competence"],
                              "examined": axis in record["inspected"]}
        missed = [a for a, v in verdicts.items() if v["warranted"] and not v["examined"]]
        wasted = [a for a, v in verdicts.items() if v["examined"] and not v["warranted"]
                  and not record["inspected"][a]["changed_decision"]]
        route = record.get("route")
        error = None
        if route is not None and abs(float(route["success"]) - route["prediction"]) > .5:
            error = "missing_information" if missed else "knowledge_or_noise"
        appraisal = {"evidence_id": record.get("evidence_id"), "scope": list(record["scope"]),
                     "time": now, "hindsight": verdicts, "missed": missed, "wasted": wasted,
                     "information_gain": dict(information_gain or {}),
                     "association": dict(association or {}),
                     "error": error, "route": route, "unexplained": unexplained,
                     "decision_followed_values": record["followed_values"],
                     "constraints_respected": record["permissible"],
                     "effort": dict(record["effort"])}
        if record.get("recovery") is not None:
            # Opt-in adaptive policy only. A deferral has no fabricated route
            # outcome, but an unhelpful computation can still be recorded.
            appraisal["recovery"] = dict(record["recovery"])
            appraisal["inspection_feedback"] = {
                axis: {"gain": item.get("gain", 0.0)}
                for axis, item in appraisal["information_gain"].items()}
        thread = self.thread(state)
        thread.process_appraisals.append(appraisal)
        del thread.process_appraisals[:-self.max_versions]
        return appraisal

    def option_costs(self, state: CognitiveState, *, values: dict[str, float], scarce: str,
                     alternatives: list[str], horizon: int, minimum: int = 8) -> dict[str, float]:
        """Price the future use a scarce option would forfeit (multi-step consequence).

        Using ``scarce`` now makes it unavailable for the next ``horizon``
        decisions of this investigation. Its opportunity cost is the expected
        best advantage it would offer over that horizon, estimated from the
        advantages this continuing thought has actually seen. The refiner only
        prices; the selector still decides.
        """
        thread = self.thread(state)
        advantage = values[scarce] - max(values[a] for a in alternatives)
        history = sorted(max(0.0, a) for a in thread.option_advantages)
        cost = 0.0
        if horizon > 0 and len(history) >= minimum:
            n = len(history)
            cost = sum(x * (((i + 1) / n) ** horizon - (i / n) ** horizon)
                       for i, x in enumerate(history))
        thread.option_advantages.append(round(advantage, 9))
        del thread.option_advantages[:-64]
        return {scarce: round(cost, 9)} if cost > 0 else {}

    def diagnose_causal(self, state: CognitiveState, outcome, retrieved_events, *,
                        probability: float) -> dict:
        """Bounded same-action/scope contrast selection on observed surprise.

        The only inputs are the actual outcome and already activated events.
        Changed conditions suggest explanations, not verified causality.
        """
        cues = frozenset(outcome.metadata["context_cues"])
        procedure = outcome.metadata["procedure"]
        scope = self.learner.causal_scope(cues)
        success = outcome.outcome == "success"
        prediction_error = abs(float(success) - probability)
        candidates = []
        for previous in retrieved_events:
            old = frozenset(previous.metadata.get("context_cues", ()))
            if (previous.kind != EventKind.OUTCOME or previous.truth_status != TruthStatus.OUTCOME
                    or previous.outcome not in ("success", "failure") or not previous.cause_event_id
                    or previous.metadata.get("procedure") != procedure
                    or previous.goal_id != state.goal_id
                    or self.learner.causal_scope(old) != scope
                    or previous.outcome == outcome.outcome):
                continue
            # Two literal differences correspond to one changed axis.
            if len(old ^ cues) == 2:
                candidates.append(previous)
        candidates.sort(key=lambda e: (-e.occurrence_time, e.event_id))
        old = candidates[0] if candidates and prediction_error >= .25 else None
        thread = self.thread(state)
        # Existing bounded task register retains the latest outcome of each
        # available procedure; the router decides its later activation.
        thread.causal_anchors[procedure] = outcome.event_id
        while len(thread.causal_anchors) > 2:
            del thread.causal_anchors[next(iter(thread.causal_anchors))]
        return {"prediction_error": prediction_error,
                "examined_events": len(retrieved_events),
                "contrast": None if old is None else {
                    "cues": list(old.metadata["context_cues"]),
                    "procedure": procedure, "success": old.outcome == "success",
                    "evidence_id": old.event_id},
                "reason": "candidate_explanation" if old else "insufficient_comparable_evidence"}

    def refine_incremental(self, state: CognitiveState, scored, hits, *,
                           context: dict):
        """Single evidence-grounded, cost-sensitive reconsideration.

        The refiner cannot create evidence or select actions. It may revise the
        ranking using already-computed predictions and explicitly priced risk.
        """
        if not scored:
            raise ValueError("missing incremental candidates")
        direct = sorted(scored, key=lambda row: (-row[0], row[1].label))
        values = context["candidate_values"]
        predictions = context["predictions"]
        risk_price = max(0.0, float(context.get("risk_price", 0.0)))
        # Explicitly priced future options (E12 multi-step): using an action now
        # may forfeit it later. Absent, this is exactly the historical ranking.
        option_costs = context.get("option_costs") or {}
        assessed = {
            a.label: values[a.label] - risk_price *
            predictions[a.label]["uncertainty"] *
            (1.0 - predictions[a.label]["probability"]) - option_costs.get(a.label, 0.0)
            for _, a, _, _ in scored
        }
        best = min(scored, key=lambda row: (-assessed[row[1].label], row[1].label))
        thread = self.thread(state)
        self._focus(thread, state.current_time)
        evidence = tuple(sorted(set(
            eid for data in predictions.values() for eid in data["evidence_ids"])))
        signature = (state.episode_id, best[1].label, evidence,
                     tuple((key, round(value, 4)) for key, value in sorted(assessed.items())))
        changed = not thread.versions or thread.versions[-1].signature != signature
        if changed and len(thread.versions) < self.max_versions:
            thread.versions.append(ThoughtVersion(
                number=len(thread.versions) + 1,
                preferred_action=best[1].label,
                alternatives=tuple(row[1].label for row in direct if row[1] != best[1]),
                evidence_ids=evidence,
                memory_chunk_ids=tuple(hit.chunk_id for hit in hits),
                contradictions=tuple(sorted({eid for p in predictions.values()
                    for h in p.get("hypotheses", ()) for eid in h["contradictions"]})),
                hypotheses=tuple(h for p in predictions.values() for h in p.get("hypotheses", ())),
                uncertainty=predictions[best[1].label]["uncertainty"],
                reason=context.get("reason", "conditional_evidence_and_decision_risk"),
                signature=signature, time=state.current_time,
                predicted_outcomes=tuple(sorted(
                    (key, v["probability"]) for key, v in predictions.items())),
            ))
        ordered = [(max(row[0] for row in scored) + 1, best[1],
                    best[2] + ["incremental_value_refinement"], best[3])] + [
                        row for row in direct if row[1].label != best[1].label]
        return ordered, {
            "thought_id": thread.thought_id,
            "changed": changed, "version": len(thread.versions),
            "reasoning_cycles": 2,
            "meaningfully_changed_action": best[1] != direct[0][1],
            "contributing_evidence_ids": list(evidence),
            "additional_detailed_reads": 0,
            "decision_basis": context.get("reason", "evidence_risk_adjusted"),
            "prediction": predictions[best[1].label]["probability"],
        }

    def refine_temporal(self, state: CognitiveState, scored, hits, *, context: dict):
        """Reconsider a sourced expectation only after a relevant time/evidence change.

        Clock passage marks an expectation due, never false. A verification
        action is weighed against its observed prediction risk and inspection
        cost. ActionSelector remains the sole authority to select the action.
        """
        if not scored:
            raise ValueError("temporal refinement requires candidate actions")
        expectation = state.timed_expectations[context["expectation_id"]]
        thread = self.thread(state)
        self._focus(thread, state.world_time)
        direct = sorted(scored, key=lambda row: (-row[0], row[1].label))
        promised = expectation.due_at - expectation.started_at
        expected_duration, punctuality = self.learner.temporal_prediction(
            expectation.context, promised)
        lateness = max(0.0, state.world_time - expectation.due_at)
        consequence = max(0.0, float(context.get("consequence", 1.0)))
        inspection_cost = max(0.0, float(context.get("inspection_cost", .25)))
        information_value = (1.0 - punctuality) * consequence * (
            1.0 + min(1.0, lateness / max(1.0, promised))) * expectation.confidence
        choices = {
            "pending": "wait",
            "due_unverified": ("check_return" if information_value > inspection_cost else "wait"),
            "verified_absent": "follow_up",
            "fulfilled": "acknowledge_return",
        }
        desired_kind = choices[expectation.status]
        selected = next((row for row in scored if row[1].kind == desired_kind), None)
        if selected is None:
            raise ValueError("temporal action affordance missing: " + desired_kind)
        evidence_ids = tuple([expectation.source_event_id] + expectation.observation_ids)
        signature = (
            expectation.expectation_id, expectation.status, evidence_ids,
            desired_kind, round(information_value, 4) > inspection_cost,
        )
        before = len(thread.versions)
        if (not thread.versions or thread.versions[-1].signature != signature) and (
                len(thread.versions) < self.max_versions):
            reason = {
                "pending": "prediction_pending",
                "due_unverified": "due_without_confirmed_outcome",
                "verified_absent": "directly_observed_delay",
                "fulfilled": "directly_observed_return",
            }[expectation.status]
            thread.versions.append(ThoughtVersion(
                number=len(thread.versions) + 1,
                preferred_action=selected[1].label,
                alternatives=tuple(row[1].label for row in direct
                                   if row[1].label != selected[1].label)[:3],
                evidence_ids=evidence_ids,
                memory_chunk_ids=tuple(sorted(hit.chunk_id for hit in hits)),
                contradictions=(expectation.expectation_id,)
                               if expectation.status == "verified_absent" else (),
                uncertainty=1.0 - punctuality,
                reason=reason, signature=signature,
                time=state.world_time,
                predicted_outcomes=(("on_time", punctuality),),
            ))
        reordered = [(
            max(row[0] for row in scored) + 1.0,
            selected[1], selected[2] + ["temporal=" + expectation.status],
            selected[3],
        )] + [row for row in direct if row[1] != selected[1]]
        return reordered, {
            "thought_id": thread.thought_id,
            "version": thread.versions[-1].number,
            "changed": len(thread.versions) != before,
            "reasoning_cycles": 2 if expectation.status != "pending" else 1,
            "decision_basis": thread.versions[-1].reason,
            "activation_state": thread.status_at(state.world_time),
            "expectation_id": expectation.expectation_id,
            "expectation_status": expectation.status,
            "source_event_id": expectation.source_event_id,
            "contributing_evidence_ids": list(evidence_ids),
            "retrieved_chunk_ids": [hit.chunk_id for hit in hits],
            "predicted_on_time": punctuality,
            "learned_expected_duration": expected_duration,
            "expected_information_value": information_value,
            "inspection_cost": inspection_cost,
            "reasoning_stopped": len(thread.versions) == before,
            "meaningfully_changed_action": selected[1] != direct[0][1],
            "additional_detailed_reads": 0,
        }

    def refine_temporal_value(self, state: CognitiveState, scored, hits, *,
                              context: dict):
        """E10: only revise when a value calculation may change the decision.

        Reads the SAME sourced interpretation, not extra evidence. A thought
        version carries provenance and uncertainty, not additional votes.
        ActionSelector alone remains responsible for ranking final actions.
        """
        if not scored:
            raise ValueError("no temporal options")
        thread = self.thread(state)
        self._focus(thread, state.world_time)
        direct = sorted(scored, key=lambda x: (-x[0], x[1].label))
        values = context["candidate_values"]
        costs = float(context.get("thought_cost", .02))
        base = direct[0][1]
        ranked = sorted(scored, key=lambda x: (-values.get(x[1].label, -1e9),
                                                x[1].label))
        best = ranked[0][1]
        improved_value = values.get(best.label, -1e9) - values.get(base.label, -1e9)
        should_revise = (context.get("value_enabled", True)
                         and best != base and improved_value > costs)
        if context.get("clock_rule", False):
            due = [name for name, exp in state.timed_expectations.items()
                   if exp.status == "due_unverified"]
            desired = "check:" + sorted(due)[0].removeprefix("delivery:") if due else base.label
            candidate = next((r for r in scored if r[1].label == desired), None)
            best = candidate[1] if candidate is not None else base
            should_revise = best != base
        all_ids = list(context.get("evidence_ids", ()))
        evidence_ids = tuple(dict.fromkeys(all_ids))
        # Temporal threshold is meaningful only when a decision or probability
        # band changes, not simply because a thought remained activated.
        band = tuple(sorted((p, int(v * 8)) for p, v in
                            context.get("presence", {}).items()))
        signature = (evidence_ids, band, base.label,
                     best.label if should_revise else base.label)
        changed = (not thread.versions
                   or thread.versions[-1].signature != signature)
        if changed and len(thread.versions) < self.max_versions:
            thread.versions.append(ThoughtVersion(
                number=len(thread.versions) + 1,
                preferred_action=best.label if should_revise else base.label,
                alternatives=tuple(row[1].label for row in ranked[:3]
                                   if row[1] != best),
                evidence_ids=evidence_ids,
                memory_chunk_ids=tuple(hit.chunk_id for hit in hits),
                contradictions=(),
                uncertainty=max(0.0, 1.0 - max(context.get("presence", {}).values(),
                                               default=0.0)),
                reason="expected_decision_gain" if should_revise
                       else "insufficient_refinement_value",
                signature=signature, time=state.world_time,
                predicted_outcomes=tuple(sorted(
                    (p, float(v)) for p, v in context.get("presence", {}).items())),
            ))
        if not thread.versions:
            raise RuntimeError("temporal thought cap cannot suppress initial version")
        result = direct
        if should_revise:
            preferred = next(x for x in scored if x[1] == best)
            result = [(max(x[0] for x in scored) + 1.0, preferred[1],
                       preferred[2] + ["reconsidered_decision_value"],
                       preferred[3])] + [x for x in direct if x[1] != best]
        return result, {
            "thought_id": thread.thought_id, "version": thread.versions[-1].number,
            "changed": changed, "reasoning_cycles": 2 if should_revise else 1,
            "reasoning_stopped": not changed,
            "decision_basis": thread.versions[-1].reason,
            "activation_state": thread.status_at(state.world_time),
            "contributing_evidence_ids": list(evidence_ids),
            "meaningfully_changed_action": should_revise,
            "additional_detailed_reads": 0,
            "expected_gain_before_thought_cost": max(0.0, improved_value),
            "thought_cost": costs,
            "predicted_presence": dict(context.get("presence", {})),
        }

    def feedback(self, state: CognitiveState, decision, *, outcome: str | None,
                 outcome_event_ids: tuple[str, ...]) -> None:
        if not decision.thought_trace:
            return
        thread = self.thread(state)
        thread.outcomes.append({
            "version": decision.thought_trace["version"],
            "action": decision.action.label, "predicted": decision.predicted_success,
            "outcome": outcome, "outcome_event_ids": list(outcome_event_ids),
        })
        # Credit only an observed success/failure of that exact attempted action.
        # An informative inspection is not a goal-success label.
        if (self.learning and outcome in ("success", "failure")
                and decision.action.kind == "try_key"
                and decision.thought_trace.get("contributing_evidence_ids")
                and decision.meaning_frame and decision.meaning_frame.independent_sources):
            self.learner.learn_from_refinement(
                decision.meaning_frame.context, decision.action.kind,
                predicted=decision.predicted_success,
                outcome=(outcome == "success"),
            )
