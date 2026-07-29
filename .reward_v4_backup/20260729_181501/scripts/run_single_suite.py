from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ALL_VARIANTS = ["v3", "A", "B", "C", "D"]
ALL_SEEDS = [42, 43, 44, 45, 46, 47, 48]


def run_command(command: list[str], log_path: Path | None = None) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=True)


def valid_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and "completed_mean" in data


def common_train_args(args, variant: str, algo: str, seed: int, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/train.py",
        "--algo",
        algo,
        "--ablation",
        variant,
        "--episodes",
        str(args.train_episodes),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--run-dir",
        str(run_dir),
        "--train-every",
        str(args.train_every),
        "--learning-starts",
        str(args.learning_starts),
        "--eval-every",
        str(args.eval_every),
        "--eval-episodes",
        str(args.periodic_eval_episodes),
        "--periodic-eval-seed-base",
        str(args.periodic_eval_seed_base),
        "--max-steps",
        str(args.max_steps),
    ]
    if algo == "bdqn":
        command += [
            "--posterior-update-period",
            str(args.posterior_update_period),
            "--posterior-replay-size",
            str(args.posterior_replay_size),
            "--posterior-chunk-size",
            str(args.posterior_chunk_size),
            "--posterior-min-samples",
            str(args.posterior_min_samples),
            "--posterior-mode",
            "rebuild",
        ]
    return command


def final_eval_args(
    args,
    variant: str,
    algo: str,
    seed: int,
    checkpoint: Path,
    json_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate.py",
        "--algo",
        algo,
        "--ablation",
        variant,
        "--checkpoint",
        str(checkpoint),
        "--episodes",
        str(args.final_eval_episodes),
        "--eval-seed-base",
        str(args.final_eval_seed_base),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--max-steps",
        str(args.max_steps),
        "--json-out",
        str(json_path),
    ]


def run_jobs(args, *, stage: str, variants: list[str], algos: list[str], seeds: list[int]) -> None:
    run_root = Path(args.run_root) / stage
    log_root = Path(args.log_root) / stage

    for variant in variants:
        for algo in algos:
            for seed in seeds:
                tag = f"{algo}_seed{seed}_train{args.train_episodes}"
                run_dir = run_root / variant / tag
                variant_log_dir = log_root / variant
                train_log = variant_log_dir / f"{tag}_train.log"
                eval_json = variant_log_dir / f"{algo}_seed{seed}_eval{args.final_eval_episodes}.json"
                eval_log = variant_log_dir / f"{algo}_seed{seed}_eval{args.final_eval_episodes}.log"

                if not args.force and valid_json(eval_json):
                    print(f"SKIP completed: {eval_json}")
                    continue

                checkpoint = run_dir / "best.pt"
                if args.force or not checkpoint.exists():
                    run_command(
                        common_train_args(args, variant, algo, seed, run_dir),
                        train_log,
                    )
                else:
                    print(f"REUSE checkpoint: {checkpoint}")

                run_command(
                    final_eval_args(args, variant, algo, seed, checkpoint, eval_json),
                    eval_log,
                )

    aggregate_dir = log_root / "aggregate"
    run_command(
        [
            sys.executable,
            "scripts/aggregate_single_results.py",
            "--roots",
            str(log_root),
            "--output-dir",
            str(aggregate_dir),
        ]
    )


def run_baselines(args, variants: list[str]) -> None:
    for variant in variants:
        output_dir = Path(args.log_root) / "baselines"
        expected = output_dir / variant / f"baseline_oracle_eval{args.baseline_episodes}.json"
        if not args.force and valid_json(expected):
            print(f"SKIP baselines completed: {variant}")
            continue
        run_command(
            [
                sys.executable,
                "scripts/evaluate_baselines.py",
                "--ablation",
                variant,
                "--episodes",
                str(args.baseline_episodes),
                "--eval-seed-base",
                str(args.final_eval_seed_base),
                "--max-steps",
                str(args.max_steps),
                "--output-dir",
                str(output_dir),
            ]
        )


def aggregate_existing(args) -> None:
    run_command(
        [
            sys.executable,
            "scripts/aggregate_single_results.py",
            "--roots",
            "logs/v3_frontier",
            "logs/v3_frontier_new_bdqn",
            "--output-dir",
            str(Path(args.log_root) / "existing_v3_aggregate"),
        ]
    )


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-command single-UAV experiment runner with resume support."
    )
    parser.add_argument(
        "stage",
        choices=["smoke", "existing", "baselines", "screen", "confirm"],
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--run-root", default="runs/single_suite")
    parser.add_argument("--log-root", default="logs/single_suite")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--variants", default="")
    parser.add_argument("--algos", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--final-eval-episodes", type=int, default=None)
    parser.add_argument("--baseline-episodes", type=int, default=500)
    parser.add_argument("--periodic-eval-episodes", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--periodic-eval-seed-base", type=int, default=100_000)
    parser.add_argument("--final-eval-seed-base", type=int, default=200_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--posterior-update-period", type=int, default=500)
    parser.add_argument("--posterior-replay-size", type=int, default=8192)
    parser.add_argument("--posterior-chunk-size", type=int, default=512)
    parser.add_argument("--posterior-min-samples", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.stage == "smoke":
        run_command([sys.executable, "scripts/smoke_test_single_v4.py"])
        return
    if args.stage == "existing":
        aggregate_existing(args)
        return

    variants = parse_csv_strings(args.variants) if args.variants else ALL_VARIANTS
    unknown = sorted(set(variants) - set(ALL_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    if args.stage == "baselines":
        run_baselines(args, variants)
        return

    run_command([sys.executable, "scripts/smoke_test_single_v4.py"])
    run_baselines(args, variants)

    if args.stage == "screen":
        args.train_episodes = args.train_episodes or 400
        args.final_eval_episodes = args.final_eval_episodes or 300
        args.periodic_eval_episodes = args.periodic_eval_episodes or 20
        algos = parse_csv_strings(args.algos) if args.algos else ["ddqn"]
        seeds = parse_csv_ints(args.seeds) if args.seeds else [42, 43, 44]
        run_jobs(args, stage="screen", variants=variants, algos=algos, seeds=seeds)
        return

    # Confirmation should normally be run only after inspecting the screen summary.
    args.train_episodes = args.train_episodes or 1000
    args.final_eval_episodes = args.final_eval_episodes or 1000
    args.periodic_eval_episodes = args.periodic_eval_episodes or 50
    if not args.variants:
        variants = ["v3", "C", "D"]
    algos = parse_csv_strings(args.algos) if args.algos else ["ddqn", "bdqn"]
    seeds = parse_csv_ints(args.seeds) if args.seeds else ALL_SEEDS
    run_jobs(args, stage="confirm", variants=variants, algos=algos, seeds=seeds)


if __name__ == "__main__":
    main()
