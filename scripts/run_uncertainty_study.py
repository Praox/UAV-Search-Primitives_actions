from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SINGLE_ALGOS = ["ddqn", "bdqn"]
DEFAULT_MULTI_ALGOS = [
    "shared_ddqn",
    "shared_bdqn",
    "qmix_ddqn",
    "bayes_qmix_shared",
    "bayes_qmix_independent",
]


def parse_strings(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(item) for item in parse_strings(text)]


def parse_floats(text: str) -> list[float]:
    return [float(item) for item in parse_strings(text)]


def probability_slug(probability: float) -> str:
    return f"pdet_{float(probability):.2f}".replace(".", "p")


def valid_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and "completed_mean" in payload


def run(command: list[str], log_path: Path | None = None) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = log_path.read_text(errors="replace").splitlines()[-80:]
        print("\n".join(tail))
        raise SystemExit(result.returncode)


def add_metadata(path: Path, **metadata) -> None:
    payload = json.loads(path.read_text())
    payload.update(metadata)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")


def single_train_command(args, *, algo: str, seed: int, probability: float, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/train.py",
        "--algo", algo,
        "--ablation", args.single_ablation,
        "--episodes", str(args.train_episodes),
        "--device", args.device,
        "--seed", str(seed),
        "--run-dir", str(run_dir),
        "--train-every", str(args.single_train_every),
        "--learning-starts", str(args.learning_starts),
        "--eval-every", str(args.eval_every),
        "--eval-episodes", str(args.periodic_eval_episodes),
        "--periodic-eval-seed-base", str(args.single_eval_seed_base),
        "--max-steps", str(args.max_steps),
        "--detection-probability", str(probability),
    ]
    if algo == "bdqn":
        command.extend([
            "--posterior-update-period", str(args.posterior_update_period),
            "--posterior-replay-size", str(args.posterior_replay_size),
            "--posterior-chunk-size", str(args.posterior_chunk_size),
            "--posterior-min-samples", str(args.posterior_min_samples),
            "--posterior-mode", args.posterior_mode,
        ])
    return command


def single_eval_command(
    args,
    *,
    algo: str,
    seed: int,
    probability: float,
    checkpoint: Path,
    output: Path,
    sampled: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/evaluate.py",
        "--algo", algo,
        "--ablation", args.single_ablation,
        "--checkpoint", str(checkpoint),
        "--episodes", str(args.final_eval_episodes),
        "--eval-seed-base", str(args.single_eval_seed_base),
        "--device", args.device,
        "--seed", str(seed),
        "--max-steps", str(args.max_steps),
        "--detection-probability", str(probability),
        "--json-out", str(output),
    ]
    if sampled:
        command.append("--bdqn-sampled-eval")
    return command


def run_single(args, probabilities: list[float], seeds: list[int]) -> None:
    algos = parse_strings(args.single_algos)
    for probability in probabilities:
        slug = probability_slug(probability)
        scenario = f"single_{slug}"
        run_root = Path(args.run_root) / "single" / slug
        log_root = Path(args.log_root) / "single" / slug

        if args.include_baselines:
            baseline_dir = log_root / "baselines"
            expected = baseline_dir / args.single_ablation / f"baseline_frontier_eval{args.baseline_episodes}.json"
            if args.force or not valid_json(expected):
                run(
                    [
                        sys.executable,
                        "scripts/evaluate_baselines.py",
                        "--ablation", args.single_ablation,
                        "--baselines", "random", "frontier", "oracle",
                        "--episodes", str(args.baseline_episodes),
                        "--eval-seed-base", str(args.single_eval_seed_base),
                        "--detection-probability", str(probability),
                        "--max-steps", str(args.max_steps),
                        "--output-dir", str(baseline_dir),
                    ],
                    log_root / "baselines.log",
                )
            for path in baseline_dir.rglob("baseline_*_eval*.json"):
                add_metadata(
                    path,
                    scope="single",
                    scenario_label=scenario,
                    detection_probability=float(probability),
                    global_state_mode="not_applicable",
                    evaluation_policy="heuristic" if "frontier" in path.name or "oracle" in path.name else "random",
                )

        for algo in algos:
            for seed in seeds:
                tag = f"{algo}_seed{seed}_train{args.train_episodes}"
                run_dir = run_root / tag
                checkpoint = run_dir / "best.pt"
                mean_json = log_root / f"{tag}_mean_eval.json"
                train_log = log_root / f"{tag}_train.log"

                if not valid_json(mean_json) or args.force:
                    if args.force or not checkpoint.exists():
                        run(
                            single_train_command(
                                args,
                                algo=algo,
                                seed=seed,
                                probability=probability,
                                run_dir=run_dir,
                            ),
                            train_log,
                        )
                    else:
                        print(f"REUSE checkpoint: {checkpoint}")
                    run(
                        single_eval_command(
                            args,
                            algo=algo,
                            seed=seed,
                            probability=probability,
                            checkpoint=checkpoint,
                            output=mean_json,
                            sampled=False,
                        ),
                        log_root / f"{tag}_mean_eval.log",
                    )
                    add_metadata(
                        mean_json,
                        scope="single",
                        scenario_label=scenario,
                        detection_probability=float(probability),
                        global_state_mode="not_applicable",
                        evaluation_policy="posterior_mean" if algo == "bdqn" else "deterministic",
                        ablation=args.single_ablation,
                    )
                else:
                    print(f"SKIP completed: {mean_json}")

                if algo == "bdqn" and args.evaluate_bdqn_samples:
                    sampled_json = log_root / f"{tag}_sampled_eval.json"
                    if not valid_json(sampled_json) or args.force:
                        run(
                            single_eval_command(
                                args,
                                algo=algo,
                                seed=seed,
                                probability=probability,
                                checkpoint=checkpoint,
                                output=sampled_json,
                                sampled=True,
                            ),
                            log_root / f"{tag}_sampled_eval.log",
                        )
                        add_metadata(
                            sampled_json,
                            scope="single",
                            scenario_label=scenario,
                            detection_probability=float(probability),
                            global_state_mode="not_applicable",
                            evaluation_policy="posterior_sample_per_episode",
                            ablation=args.single_ablation,
                        )
                    else:
                        print(f"SKIP completed: {sampled_json}")


def run_multi_baselines(args, probability: float, log_root: Path, scenario: str) -> None:
    output_dir = log_root / "baselines"
    expected = output_dir / f"baseline_local_frontier_eval{args.baseline_episodes}.json"
    if valid_json(expected) and not args.force:
        print(f"SKIP completed: {expected}")
        return
    run(
        [
            sys.executable,
            "scripts/evaluate_multi_local_baselines.py",
            "--baselines", "random,local_frontier",
            "--episodes", str(args.baseline_episodes),
            "--eval-seed-base", str(args.multi_eval_seed_base),
            "--policy-seed", str(args.baseline_policy_seed),
            "--output-dir", str(output_dir),
            "--scenario-label", scenario,
            "--n-agents", str(args.n_agents),
            "--grid-size", str(args.grid_size),
            "--detection-probability", str(probability),
            "--max-steps", str(args.max_steps),
            "--global-state-mode", args.global_state_mode,
        ],
        log_root / "baselines.log",
    )


def run_multi(args, probabilities: list[float], seeds: list[int]) -> None:
    algos = parse_strings(args.multi_algos)
    for probability in probabilities:
        slug = probability_slug(probability)
        scenario = f"multi_{slug}__state_{args.global_state_mode}"
        run_root = Path(args.run_root) / "multi" / slug
        log_root = Path(args.log_root) / "multi" / slug

        if args.include_baselines:
            run_multi_baselines(args, probability, log_root, scenario)

        for algo in algos:
            for seed in seeds:
                tag = f"{algo}_seed{seed}_train{args.train_episodes}"
                run_dir = run_root / tag
                eval_json = log_root / f"{tag}_eval.json"
                if valid_json(eval_json) and not args.force:
                    print(f"SKIP completed: {eval_json}")
                    continue
                mode = "eval_only" if (run_dir / "best.pt").exists() and not args.force else "train_eval"
                command = [
                    sys.executable,
                    "scripts/train_multi_local.py",
                    "--algo", algo,
                    "--mode", mode,
                    "--seed", str(seed),
                    "--episodes", str(args.train_episodes),
                    "--eval-every", str(args.eval_every),
                    "--eval-episodes", str(args.periodic_eval_episodes),
                    "--final-eval-episodes", str(args.final_eval_episodes),
                    "--device", args.device,
                    "--run-dir", str(run_dir),
                    "--eval-json", str(eval_json),
                    "--n-agents", str(args.n_agents),
                    "--grid-size", str(args.grid_size),
                    "--max-steps", str(args.max_steps),
                    "--eval-seed-base", str(args.multi_eval_seed_base),
                    "--posterior-eval-seed-base", str(args.posterior_eval_seed_base),
                    "--train-every", str(args.multi_train_every),
                    "--learning-starts", str(args.learning_starts),
                    "--scenario-label", scenario,
                    "--detection-probability", str(probability),
                    "--global-state-mode", args.global_state_mode,
                    "--bayes-kl-weight", str(args.bayes_kl_weight),
                ]
                run(command, log_root / f"{tag}.log")
                add_metadata(eval_json, scope="multi")


def aggregate(args) -> None:
    root = Path(args.log_root)
    output = root / "aggregate"
    run(
        [
            sys.executable,
            "scripts/aggregate_uncertainty_study.py",
            "--roots", str(root),
            "--output-dir", str(output),
            "--reference-detection-probability", str(args.reference_detection_probability),
        ]
    )


def smoke() -> None:
    run([sys.executable, "scripts/smoke_test_single_v4.py"])
    run([sys.executable, "scripts/smoke_test_bayesian_qmix.py"])
    run(
        [
            sys.executable,
            "scripts/evaluate_multi_local_baselines.py",
            "--episodes", "2",
            "--output-dir", "logs/debug_uncertainty_baselines",
            "--detection-probability", "0.7",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automated uncertainty study for single-UAV and multi-UAV methods."
    )
    parser.add_argument(
        "mode",
        choices=[
            "smoke", "single-screen", "multi-screen", "full-screen",
            "single-confirm", "multi-confirm", "full-confirm", "aggregate",
        ],
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--probabilities", default="1.0,0.7")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--single-algos", default=",".join(DEFAULT_SINGLE_ALGOS))
    parser.add_argument("--multi-algos", default=",".join(DEFAULT_MULTI_ALGOS))
    parser.add_argument("--single-ablation", default="D")
    parser.add_argument("--run-root", default="runs/uncertainty_study")
    parser.add_argument("--log-root", default="logs/uncertainty_study")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-baselines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate-bdqn-samples", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--train-episodes", type=int, default=400)
    parser.add_argument("--final-eval-episodes", type=int, default=300)
    parser.add_argument("--periodic-eval-episodes", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--single-train-every", type=int, default=4)
    parser.add_argument("--multi-train-every", type=int, default=1)
    parser.add_argument("--single-eval-seed-base", type=int, default=200_000)
    parser.add_argument("--multi-eval-seed-base", type=int, default=100_000)
    parser.add_argument("--posterior-eval-seed-base", type=int, default=900_000)
    parser.add_argument("--baseline-episodes", type=int, default=500)
    parser.add_argument("--baseline-policy-seed", type=int, default=999)
    parser.add_argument("--reference-detection-probability", type=float, default=1.0)

    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument(
        "--global-state-mode",
        choices=["privileged_truth", "memory_union"],
        default="privileged_truth",
    )
    parser.add_argument("--bayes-kl-weight", type=float, default=1e-3)
    parser.add_argument("--posterior-update-period", type=int, default=500)
    parser.add_argument("--posterior-replay-size", type=int, default=8192)
    parser.add_argument("--posterior-chunk-size", type=int, default=512)
    parser.add_argument("--posterior-min-samples", type=int, default=1000)
    parser.add_argument("--posterior-mode", choices=["rebuild", "cumulative"], default="rebuild")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "smoke":
        smoke()
        return
    if args.mode == "aggregate":
        aggregate(args)
        return

    probabilities = parse_floats(args.probabilities)
    if any(probability <= 0.0 or probability > 1.0 for probability in probabilities):
        raise ValueError("Every detection probability must be in (0, 1].")
    seeds = parse_ints(args.seeds)

    if "confirm" in args.mode:
        if args.seeds == "42,43,44":
            seeds = list(range(42, 49))
        if args.train_episodes == 400:
            args.train_episodes = 1000
        if args.final_eval_episodes == 300:
            args.final_eval_episodes = 1000
        if args.periodic_eval_episodes == 30:
            args.periodic_eval_episodes = 50

    smoke()
    if args.mode in {"single-screen", "single-confirm", "full-screen", "full-confirm"}:
        run_single(args, probabilities, seeds)
    if args.mode in {"multi-screen", "multi-confirm", "full-screen", "full-confirm"}:
        run_multi(args, probabilities, seeds)
    aggregate(args)


if __name__ == "__main__":
    main()
