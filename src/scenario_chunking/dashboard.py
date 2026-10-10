from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_dashboard(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    aggregate = payload["aggregate"]
    modes = list(aggregate)
    max_actions = max((aggregate[m]["actions_mean"] for m in modes), default=1) or 1
    max_details = max((aggregate[m]["active_detail_events_mean"] for m in modes), default=1) or 1

    rows = "".join(
        f"<tr><td>{html.escape(m)}</td><td>{aggregate[m]['completion_rate_mean']:.2f}</td>"
        f"<td>{aggregate[m]['actions_mean']:.1f}</td><td>{aggregate[m]['improvement_e1_to_e2_mean']:.2f}</td>"
        f"<td>{aggregate[m]['retrieval_recall_at_k_mean']:.2f}</td><td>{aggregate[m]['outdated_retrievals_mean']:.1f}</td>"
        f"<td>{aggregate[m]['delayed_credit_accuracy_mean']:.2f}</td>"
        f"<td>{aggregate[m]['recall_after_spreading_mean']:.2f}</td>"
        f"<td>{aggregate[m]['recovered_relevant_memories_mean']:.2f}</td>"
        f"<td>{aggregate[m]['spread_action_changes_mean']:.1f}</td>"
        f"<td>{aggregate[m]['active_detail_events_mean']:.0f}</td></tr>" for m in modes
    )
    action_bars = "".join(
        f"<div class='barrow'><span>{html.escape(m)}</span><div class='bar' style='width:{100*aggregate[m]['actions_mean']/max_actions:.1f}%'></div><b>{aggregate[m]['actions_mean']:.1f}</b></div>"
        for m in modes
    )
    cost_bars = "".join(
        f"<div class='barrow'><span>{html.escape(m)}</span><div class='bar' style='width:{100*aggregate[m]['active_detail_events_mean']/max_details:.1f}%'></div><b>{aggregate[m]['active_detail_events_mean']:.0f}</b></div>"
        for m in modes
    )

    structural = next((r for r in payload["runs"] if r["mode"] == "spreading"), payload["runs"][0])
    trace_rows = []
    for ep in structural["episodes"]:
        for item in ep["trace"][:6]:
            activated = ", ".join(item.get("initially_activated", [])) or "none"
            spread = ", ".join(item.get("spread_activated", [])) or "none"
            influence = "ACTION CHANGED" if item.get("spread_changed_action") else (
                "score changed" if item.get("spread_changed_score") else "no observed choice/score effect"
            )
            trace_rows.append(
                f"<tr><td>{html.escape(ep['episode_id'])}</td><td>{item['step']}</td><td>{html.escape(item['action'])}</td>"
                f"<td>{html.escape(activated)}</td><td>{html.escape(spread)}</td>"
                f"<td>{html.escape(influence)}</td><td>{html.escape('; '.join(item['signals'])[:180])}</td></tr>"
            )
    pref = structural.get("preference_snapshots", [])
    pref_json = html.escape(json.dumps(pref, indent=2))
    control = payload["recency_negative_control"]

    document = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Scenario-Chunking Results</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 18px;line-height:1.45}}h1,h2{{margin-top:1.4em}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #ccc;padding:7px;text-align:left;vertical-align:top}}.barrow{{display:grid;grid-template-columns:130px 1fr 70px;gap:10px;align-items:center;margin:8px 0}}.bar{{height:18px;background:#777;min-width:2px}}code,pre{{background:#f4f4f4;padding:10px;overflow:auto}}.note{{border-left:4px solid #777;padding-left:12px}}
</style></head><body>
<h1>Scenario-Chunking Cognitive Architecture Experiment</h1>
<p class='note'>Measured runtime latency and peak memory are reported separately from processing-count proxies. No transformer attention FLOPs are claimed.</p>
<h2>1. Learning improvement across episodes</h2><div>{action_bars}</div>
<table><thead><tr><th>Mode</th><th>Completion</th><th>Total actions</th><th>E1→E2 action improvement</th><th>Initial recall@K</th><th>Outdated retrievals</th><th>Delayed credit</th><th>Post-spread recall</th><th>Relevant recoveries</th><th>Choice changes</th><th>Detailed events</th></tr></thead><tbody>{rows}</tbody></table>
<h2>2. Thought/action trajectory</h2><p>Representative spreading-mode trace. Initial and newly activated memories are separate. Choice-change flags are paired-RNG counterfactual comparisons, not proof of better decisions.</p>
<table><thead><tr><th>Episode</th><th>Step</th><th>Selected action</th><th>Initial top-K</th><th>One-hop spread</th><th>Selection effect</th><th>Environment feedback</th></tr></thead><tbody>{''.join(trace_rows)}</tbody></table>
<h2>3. Memory activation and bounded spreading</h2><p>Budget: two additional chunks, 18 neighbor-header evaluations, 36 detailed events including the initial top-K, and one hop. Newly activated chunks cannot seed another expansion. The complete per-step justification and budget use are in event_traces.jsonl.</p>
<h2>3a. Existing misses and observed recovery</h2>
<pre>{html.escape(json.dumps([d for d in structural.get("recall_diagnostics", []) if d["initially_missed"]], indent=2)[:12000])}</pre>
<h2>4. Action preference changes after feedback</h2><pre>{pref_json}</pre>
<h2>5. Historical events versus current beliefs</h2><p>Episode 3 introduces a replaced lock. The current contradiction is carried in cognitive state and structural retrieval penalizes the old successful key without rewriting the old event.</p>
<p>Negative recency control: fixed-recency top-1 = <b>{html.escape(str(control['fixed_recency_top1']))}</b>; structural top-1 = <b>{html.escape(str(control['structural_top1']))}</b>; recency made the decision worse = <b>{control['fixed_recency_worse']}</b>.</p>
<h2>6. Computation/retrieval cost</h2><p>Active detailed-event volume (lower is better at equivalent task performance):</p>{cost_bars}
<p>Full numerical results, failure cases, per-step initial/spread activations, and counterfactual action impact are in <code>results.json</code> and <code>event_traces.jsonl</code>. Detail counts are proxies; runtime latency and Python peak memory are measured separately.</p>
</body></html>"""
    path.write_text(document, encoding="utf-8")
