from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import run_experiments
from .models import RetrievalMode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Scenario-Chunking cognitive architecture experiment.")
    parser.add_argument("--output", default="results", help="Directory for results.json, traces, and dashboard.html")
    parser.add_argument("--seeds", default="7,23,61,101,149", help="Comma-separated deterministic seeds")
    parser.add_argument("--modes", default="full,chunked,recency,structural,spreading", help="Comma-separated retrieval modes")
    parser.add_argument("--distractors", type=int, default=1000, help="Distractor chunks inserted before Episode 4")
    parser.add_argument("--activation", action="store_true",
                        help="Include experimental working-memory treatment")
    parser.add_argument("--meaning", action="store_true",
                        help="Run separate four-treatment learned meaning ablations")
    args = parser.parse_args()

    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    modes = tuple(RetrievalMode(x.strip()) for x in args.modes.split(",") if x.strip())
    if args.activation and RetrievalMode.ACTIVATION not in modes:
        modes += (RetrievalMode.ACTIVATION,)
    payload = run_experiments(Path(args.output), seeds=seeds, modes=modes, distractor_count=args.distractors, include_meaning=args.meaning)
    print(f"Wrote experiment outputs to {Path(args.output).resolve()}")
    for mode, metrics in payload["aggregate"].items():
        print(
            f"{mode:10s} completion={metrics['completion_rate_mean']:.2f} "
            f"actions={metrics['actions_mean']:.1f} recall@K={metrics['retrieval_recall_at_k_mean']:.2f} "
            f"post_recall={metrics['recall_after_spreading_mean']:.2f} " f"detail_events={metrics['active_detail_events_mean']:.0f}"
        )


if __name__ == "__main__":
    main()
