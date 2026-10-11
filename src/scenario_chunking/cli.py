from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import run_experiments
from .transfer import run_transfer_experiment
from .uncertainty import run_uncertainty_experiment
from .scaling import run_scaling_experiment, CONDITIONS_8
from .models import RetrievalMode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Scenario-Chunking cognitive architecture experiment.")
    parser.add_argument("--output", default="results", help="Directory for results.json, traces, and dashboard.html")
    parser.add_argument("--seeds", default="7,23,61,101,149", help="Comma-separated deterministic seeds")
    parser.add_argument("--modes", default="full,chunked,recency,structural,spreading", help="Comma-separated retrieval modes")
    parser.add_argument("--distractors", type=int, default=1000, help="Distractor chunks inserted before Episode 4")
    parser.add_argument("--refinement", action="store_true",
                        help="Run bounded progressive thought refinement ablations")
    parser.add_argument("--transfer", action="store_true", help="Run Experiment 6 category learning and stepwise transfer")
    parser.add_argument("--uncertainty", action="store_true", help="Run Experiment 7 adaptive uncertainty challenge")
    parser.add_argument("--scaling", action="store_true", help="Run Experiment 8 cumulative learning and focused retrieval")
    parser.add_argument("--training-stages", default="20,100,500,2000,10000")
    parser.add_argument("--memory-stages", default="100,1000,5000,20000", help="Independent added distractor-memory loads")
    parser.add_argument("--complexities", default="1,2,3", help="Independent task difficulty levels")
    parser.add_argument("--scaling-action-budget", type=int, default=3, help="Max actions per Experiment 8 held-out task")
    parser.add_argument("--scaling-conditions", default=",".join(CONDITIONS_8))
    parser.add_argument("--candidate-budget", type=int, default=48)
    parser.add_argument("--action-budget", type=int, default=9, help="Maximum actions per Experiment 7 novel task")
    parser.add_argument("--rounds", type=int, default=4, help="Novel rounds in transfer benchmark")
    parser.add_argument("--meaning", action="store_true",
                        help="Run separate four-treatment learned meaning ablations")
    args = parser.parse_args()

    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    modes = tuple(RetrievalMode(x.strip()) for x in args.modes.split(",") if x.strip())
    if args.scaling:
        payload = run_scaling_experiment(
            Path(args.output), seeds=seeds,
            stages=tuple(int(s) for s in args.training_stages.split(",")),
            memory_targets=tuple(int(s) for s in args.memory_stages.split(",")),
            conditions=tuple(s.strip() for s in args.scaling_conditions.split(",")),
            complexities=tuple(int(s) for s in args.complexities.split(",")),
            rounds=args.rounds, action_budget=args.scaling_action_budget,
            candidate_budget=args.candidate_budget)
        for stage, conditions in payload["stage_results"].items():
            for name, metrics in conditions.items():
                print(f"stage={stage} {name}: completion={metrics['completion_rate']:.2f} "
                      f"actions={metrics['mean_actions']:.2f} "
                      f"detail={metrics['mean_detail_reads']:.1f} "
                      f"headers={metrics['mean_header_evaluations']:.1f}")
        return
    if args.uncertainty:
        payload = run_uncertainty_experiment(
            Path(args.output), seeds=seeds, rounds=args.rounds,
            distractors=args.distractors, action_budget=args.action_budget)
        for name, summary in payload["conditions"].items():
            print(f"{name}: completion={summary['completion_rate']:.2f} "
                  f"actions={summary['mean_actions']:.2f} "
                  f"first={summary['first_action_accuracy']:.2f}")
        print("VERDICT:", payload["verdict"])
        return
    if args.transfer:
        payload = run_transfer_experiment(Path(args.output), seeds=seeds, rounds=args.rounds)
        for name, summary in payload["conditions"].items():
            print(f"{name}: completion={summary['completion_rate']:.2f} "
                  f"actions={summary['mean_actions']:.2f} "
                  f"initial_plans={summary['initial_plan_accuracy']:.2f}")
        return
    payload = run_experiments(Path(args.output), seeds=seeds, modes=modes, distractor_count=args.distractors, include_meaning=args.meaning,
                              include_refinement=args.refinement)
    print(f"Wrote experiment outputs to {Path(args.output).resolve()}")
    for mode, metrics in payload["aggregate"].items():
        print(
            f"{mode:10s} completion={metrics['completion_rate_mean']:.2f} "
            f"actions={metrics['actions_mean']:.1f} recall@K={metrics['retrieval_recall_at_k_mean']:.2f} "
            f"post_recall={metrics['recall_after_spreading_mean']:.2f} " f"detail_events={metrics['active_detail_events_mean']:.0f}"
        )


if __name__ == "__main__":
    main()
