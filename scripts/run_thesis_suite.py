from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from uav_search_belief20.experiments.thesis_automation import (
    parse_csv_strings,
    parse_probabilities,
    parse_seeds,
    probability_label,
)


PRESETS = {
    # Diagnostic overfitting test: every learned policy sees exactly one world.
    # Repeated final episodes are useful for random baselines and posterior samples;
    # deterministic learned policies should be aggregated over training seeds.
    "fixed": {
        "seeds": [42, 43, 44, 45, 46],
        "probabilities": [1.0],
        "train_episodes": 4000,
        "validation_episodes": 1,
        "final_test_episodes": 100,
        "eval_every": 50,
        "learning_starts": 1000,
        "single_algos": ["ddqn", "bdqn"],
        "multi_algos": [            "shared_ddqn",
            "shared_bdqn",
            "qmix_ddqn",
            "bayes_qmix_shared",
            "bayes_qmix_independent",
        ],
        
    },

        """        "multi_algos": [
            "shared_ddqn",
            "shared_bdqn",
            "qmix_ddqn",
            "bayes_qmix_shared",
            "bayes_qmix_independent",
            ],"""


    "smoke": {
        "seeds": [42],
        "probabilities": [1.0],
        "train_episodes": 6,
        "validation_episodes": 2,
        "final_test_episodes": 4,
        "eval_every": 3,
        "learning_starts": 1,
        "single_algos": ["ddqn"],
        "multi_algos": ["shared_ddqn", "qmix_ddqn"],
    },
    "screen": {
        "seeds": [42, 43, 44],
        "probabilities": [1.0, 0.7],
        "train_episodes": 400,
        "validation_episodes": 100,
        "final_test_episodes": 300,
        "eval_every": 50,
        "learning_starts": 1000,
        "single_algos": ["ddqn", "bdqn"],
        "multi_algos": [
            "shared_ddqn",
            "shared_bdqn",
            "qmix_ddqn",
            "bayes_qmix_shared",
            "bayes_qmix_independent",
        ],
    },
    "confirm": {
        "seeds": list(range(45, 49)),
        "probabilities": [1.0, 0.7, 0.5],
        "train_episodes": 1000,
        "validation_episodes": 100,
        "final_test_episodes": 1000,
        "eval_every": 50,
        "learning_starts": 1000,
        "single_algos": ["ddqn", "bdqn"],
        "multi_algos": [
            "shared_ddqn",
            "qmix_ddqn",
            "bayes_qmix_shared",
            "bayes_qmix_independent",
        ],
    },
    "custom": {
        "seeds": [42],
        "probabilities": [1.0],
        "train_episodes": 100,
        "validation_episodes": 30,
        "final_test_episodes": 100,
        "eval_every": 25,
        "learning_starts": 1000,
        "single_algos": ["ddqn"],
        "multi_algos": ["shared_ddqn", "qmix_ddqn"],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automate corrected thesis training, paired evaluation, baselines, "
            "aggregation, plots and diagnostics."
        )
    )
    parser.add_argument(
        "mode",
        choices=["fixed", "smoke", "screen", "confirm", "custom", "evaluate", "aggregate", "status"],
    )
    parser.add_argument("--stage", default="", help="Output stage name; defaults to the mode.")
    parser.add_argument("--scope", choices=["single", "multi", "both"], default="both")
    parser.add_argument("--seeds", default="", help="Examples: 42,43,44 or 42-48")
    parser.add_argument("--probabilities", default="", help="Example: 1.0,0.7")
    parser.add_argument("--single-algos", default="")
    parser.add_argument("--multi-algos", default="")
    parser.add_argument("--train-episodes", type=int, default=0)
    parser.add_argument("--validation-episodes", type=int, default=0)
    parser.add_argument("--final-test-episodes", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--validation-seed-base", type=int, default=100_000)
    parser.add_argument("--final-test-seed-base", type=int, default=200_000)
    parser.add_argument("--posterior-eval-seed-base", type=int, default=900_000)

    parser.add_argument("--run-root", default="runs/thesis_v2")
    parser.add_argument("--log-root", default="logs/thesis_v2")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--force", action="store_true", help="Rerun completed jobs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")

    # Environment and objective.
    parser.add_argument(
        "--fixed-scenario",
        action="store_true",
        help="Train, validate and test on one exactly repeated world.",
    )
    parser.add_argument(
        "--scenario-seed",
        type=int,
        default=12_345,
        help="World seed used by --fixed-scenario.",
    )
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--n-value1-targets", type=int, default=3)
    parser.add_argument("--n-value2-targets", type=int, default=1)
    parser.add_argument("--sensor-radius", type=int, default=2)
    parser.add_argument("--teammate-visibility-radius", type=int, default=2)
    parser.add_argument("--track-radius", type=int, default=1)
    parser.add_argument("--track-required", type=int, default=3)
    parser.add_argument("--track-progress-decay", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--reward-mode", choices=["legacy", "task_potential"], default="task_potential")
    parser.add_argument("--global-state-mode", choices=["privileged_truth", "memory_union"], default="memory_union")
    parser.add_argument("--include-agent-id-map", action="store_true")
    parser.add_argument("--coverage-potential-scale", type=float, default=5.0)
    parser.add_argument("--detection-potential-scale", type=float, default=1.0)
    parser.add_argument("--progress-potential-scale", type=float, default=1.0)

    # Optimization.
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity-single", type=int, default=50_000)
    parser.add_argument("--replay-capacity-multi", type=int, default=100_000)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--target-update-period", type=int, default=500)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--learning-starts", type=int, default=-1)
    parser.add_argument("--train-every-single", type=int, default=4)
    parser.add_argument("--train-every-multi", type=int, default=1)

    # Bayesian controls.
    parser.add_argument("--posterior-update-period", type=int, default=500)
    parser.add_argument("--posterior-replay-size", type=int, default=8192)
    parser.add_argument("--posterior-min-samples", type=int, default=1000)
    parser.add_argument("--blr-lambda", type=float, default=1.0)
    parser.add_argument("--blr-noise-var", type=float, default=1.0)
    parser.add_argument("--adapt-bdqn-features", action="store_true")
    parser.add_argument("--bayes-prior-std", type=float, default=1.0)
    parser.add_argument("--bayes-initial-std", type=float, default=0.05)
    parser.add_argument("--bayes-kl-weight", type=float, default=1e-3)
    return parser


def _resolved(cli) -> dict:
    default_stage = cli.mode
    if cli.fixed_scenario:
        default_stage = f"{cli.mode}_scenario{int(cli.scenario_seed)}"

    preset_name = (
        cli.mode
        if cli.mode in PRESETS
        else cli.stage if cli.stage in PRESETS else "custom"
    )
    preset = PRESETS[preset_name]
    return {
        "stage": cli.stage or default_stage,
        "fixed_scenario": bool(cli.fixed_scenario),
        "scenario_seed": int(cli.scenario_seed),
        "seeds": parse_seeds(cli.seeds) if cli.seeds else list(preset["seeds"]),
        "probabilities": (
            parse_probabilities(cli.probabilities)
            if cli.probabilities
            else list(preset["probabilities"])
        ),
        "single_algos": (
            parse_csv_strings(cli.single_algos)
            if cli.single_algos
            else list(preset["single_algos"])
        ),
        "multi_algos": (
            parse_csv_strings(cli.multi_algos)
            if cli.multi_algos
            else list(preset["multi_algos"])
        ),
        "train_episodes": cli.train_episodes or int(preset["train_episodes"]),
        "validation_episodes": cli.validation_episodes or int(preset["validation_episodes"]),
        "final_test_episodes": cli.final_test_episodes or int(preset["final_test_episodes"]),
        "eval_every": cli.eval_every or int(preset["eval_every"]),
        "learning_starts": (
            cli.learning_starts
            if cli.learning_starts >= 0
            else int(preset["learning_starts"])
        ),
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, allow_nan=True) + "\n")


def run_command(
    command: list[str],
    *,
    log_path: Path,
    manifest_path: Path,
    dry_run: bool,
    continue_on_error: bool,
) -> bool:
    print("\n$ " + " ".join(command))
    if dry_run:
        return True

    log_path.parent.mkdir(parents=True, exist_ok=True)
    _append_manifest(
        manifest_path,
        {"time": _timestamp(), "event": "start", "command": command, "log": str(log_path)},
    )
    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        return_code = process.wait()

    _append_manifest(
        manifest_path,
        {
            "time": _timestamp(),
            "event": "finish",
            "command": command,
            "log": str(log_path),
            "return_code": return_code,
        },
    )
    if return_code != 0:
        print(f"ERROR: command failed ({return_code}); log: {log_path}")
        if not continue_on_error:
            raise SystemExit(return_code)
        return False
    return True


def _common_env_args(cli, probability: float) -> list[str]:
    args = [
        "--grid-size", str(cli.grid_size),
        "--n-value1-targets", str(cli.n_value1_targets),
        "--n-value2-targets", str(cli.n_value2_targets),
        "--sensor-radius", str(cli.sensor_radius),
        "--detection-probability", str(probability),
        "--track-radius", str(cli.track_radius),
        "--track-required", str(cli.track_required),
        "--track-progress-decay", str(cli.track_progress_decay),
        "--max-steps", str(cli.max_steps),
        "--reward-mode", cli.reward_mode,
        "--coverage-potential-scale", str(cli.coverage_potential_scale),
        "--detection-potential-scale", str(cli.detection_potential_scale),
        "--progress-potential-scale", str(cli.progress_potential_scale),
    ]
    if cli.fixed_scenario:
        args.extend(
            ["--fixed-scenario", "--scenario-seed", str(cli.scenario_seed)]
        )
    return args


def _common_optimization_args(cli, resolved: dict, replay_capacity: int, train_every: int) -> list[str]:
    return [
        "--gamma", str(cli.gamma),
        "--n-step", str(cli.n_step),
        "--lr", str(cli.lr),
        "--batch-size", str(cli.batch_size),
        "--replay-capacity", str(replay_capacity),
        "--target-tau", str(cli.target_tau),
        "--target-update-period", str(cli.target_update_period),
        "--huber-delta", str(cli.huber_delta),
        "--feature-dim", str(cli.feature_dim),
        "--epsilon-start", str(cli.epsilon_start),
        "--epsilon-end", str(cli.epsilon_end),
        "--epsilon-decay-steps", str(cli.epsilon_decay_steps),
        "--learning-starts", str(resolved["learning_starts"]),
        "--train-every", str(train_every),
        "--eval-every", str(resolved["eval_every"]),
        "--validation-episodes", str(resolved["validation_episodes"]),
        "--final-test-episodes", str(resolved["final_test_episodes"]),
        "--validation-seed-base", str(cli.validation_seed_base),
        "--final-test-seed-base", str(cli.final_test_seed_base),
        "--skip-final-eval",
    ]


def _bayes_args(cli) -> list[str]:
    args = [
        "--posterior-update-period", str(cli.posterior_update_period),
        "--posterior-replay-size", str(cli.posterior_replay_size),
        "--posterior-min-samples", str(cli.posterior_min_samples),
        "--blr-lambda", str(cli.blr_lambda),
        "--blr-noise-var", str(cli.blr_noise_var),
        "--bayes-prior-std", str(cli.bayes_prior_std),
        "--bayes-initial-std", str(cli.bayes_initial_std),
        "--bayes-kl-weight", str(cli.bayes_kl_weight),
    ]
    if cli.adapt_bdqn_features:
        args.append("--adapt-bdqn-features")
    return args


def _run_dir(run_root: Path, stage: str, probability: float, scope: str, algo: str, seed: int) -> Path:
    return run_root / stage / probability_label(probability) / scope / algo / f"seed{seed}"


def _train_single(cli, resolved, probability, seed, algo, run_root, log_root, manifest) -> bool:
    run_dir = _run_dir(run_root, resolved["stage"], probability, "single", algo, seed)
    eval_index = run_dir / "evaluation" / "final_test_evaluation_index.json"
    if eval_index.exists() and not cli.force:
        print(f"SKIP complete: {run_dir}")
        return True

    command = [
        sys.executable,
        "scripts/train_thesis_single.py",
        "--algo", algo,
        "--seed", str(seed),
        "--episodes", str(resolved["train_episodes"]),
        "--run-dir", str(run_dir),
        "--device", cli.device,
        *_common_env_args(cli, probability),
        *_common_optimization_args(
            cli, resolved, cli.replay_capacity_single, cli.train_every_single
        ),
        *_bayes_args(cli),
    ]
    if algo == "bdqn":
        source = _run_dir(run_root, resolved["stage"], probability, "single", "ddqn", seed) / "best.pt"
        if not source.exists() and not cli.dry_run:
            raise FileNotFoundError(
                f"BDQN needs its matching corrected DDQN checkpoint: {source}"
            )
        command.extend(["--warmstart-ddqn", str(source)])

    log_path = log_root / resolved["stage"] / probability_label(probability) / "single" / algo / f"seed{seed}_train.log"
    if not run_command(
        command,
        log_path=log_path,
        manifest_path=manifest,
        dry_run=cli.dry_run,
        continue_on_error=cli.continue_on_error,
    ):
        return False
    return _evaluate_one(cli, resolved, run_dir, "single", probability, seed, log_root, manifest)


def _train_multi(cli, resolved, probability, seed, algo, run_root, log_root, manifest) -> bool:
    run_dir = _run_dir(run_root, resolved["stage"], probability, "multi", algo, seed)
    eval_index = run_dir / "evaluation" / "final_test_evaluation_index.json"
    if eval_index.exists() and not cli.force:
        print(f"SKIP complete: {run_dir}")
        return True

    command = [
        sys.executable,
        "scripts/train_thesis_multi.py",
        "--algo", algo,
        "--seed", str(seed),
        "--episodes", str(resolved["train_episodes"]),
        "--run-dir", str(run_dir),
        "--device", cli.device,
        "--n-agents", str(cli.n_agents),
        "--teammate-visibility-radius", str(cli.teammate_visibility_radius),
        "--global-state-mode", cli.global_state_mode,
        *_common_env_args(cli, probability),
        *_common_optimization_args(
            cli, resolved, cli.replay_capacity_multi, cli.train_every_multi
        ),
        *_bayes_args(cli),
    ]
    if cli.include_agent_id_map:
        command.append("--include-agent-id-map")
    if algo == "shared_bdqn":
        source = _run_dir(
            run_root, resolved["stage"], probability, "multi", "shared_ddqn", seed
        ) / "best.pt"
        if not source.exists() and not cli.dry_run:
            raise FileNotFoundError(
                f"shared_bdqn needs its matching shared_ddqn checkpoint: {source}"
            )
        command.extend(["--warmstart-ddqn", str(source)])

    log_path = log_root / resolved["stage"] / probability_label(probability) / "multi" / algo / f"seed{seed}_train.log"
    if not run_command(
        command,
        log_path=log_path,
        manifest_path=manifest,
        dry_run=cli.dry_run,
        continue_on_error=cli.continue_on_error,
    ):
        return False
    return _evaluate_one(cli, resolved, run_dir, "multi", probability, seed, log_root, manifest)


def _evaluate_one(cli, resolved, run_dir: Path, scope: str, probability: float, seed: int, log_root: Path, manifest: Path) -> bool:
    index = run_dir / "evaluation" / "final_test_evaluation_index.json"
    if index.exists() and not cli.force:
        print(f"SKIP evaluation: {index}")
        return True
    script = "scripts/evaluate_thesis_single.py" if scope == "single" else "scripts/evaluate_thesis_multi.py"
    command = [
        sys.executable,
        script,
        "--run-dir", str(run_dir),
        "--checkpoint", "best",
        "--episodes", str(resolved["final_test_episodes"]),
        "--eval-seed-base", str(cli.final_test_seed_base),
        "--posterior-eval-seed-base", str(cli.posterior_eval_seed_base),
        "--device", cli.device,
    ]
    log_path = log_root / resolved["stage"] / probability_label(probability) / scope / run_dir.parent.name / f"seed{seed}_eval.log"
    return run_command(
        command,
        log_path=log_path,
        manifest_path=manifest,
        dry_run=cli.dry_run,
        continue_on_error=cli.continue_on_error,
    )


def _run_baselines(cli, resolved, log_root: Path, manifest: Path) -> bool:
    baseline_root = log_root / resolved["stage"] / "baselines"
    expected: list[Path] = []
    for probability in resolved["probabilities"]:
        scenario = probability_label(probability)
        if cli.scope in {"single", "both"}:
            expected.extend(
                baseline_root / scenario / "single" / f"baseline_{name}" / "final_test_deterministic_summary.json"
                for name in ("random", "frontier", "oracle")
            )
        if cli.scope in {"multi", "both"}:
            expected.extend(
                baseline_root / scenario / "multi" / f"baseline_{name}" / "final_test_deterministic_summary.json"
                for name in ("random", "local_frontier")
            )
    if expected and all(path.exists() for path in expected) and not cli.force:
        print(f"SKIP baselines: {len(expected)} summaries already exist.")
        return True

    command = [
        sys.executable,
        "scripts/evaluate_thesis_baselines.py",
        "--scope", cli.scope,
        "--probabilities", ",".join(str(value) for value in resolved["probabilities"]),
        "--episodes", str(resolved["final_test_episodes"]),
        "--eval-seed-base", str(cli.final_test_seed_base),
        "--output-root", str(log_root / resolved["stage"] / "baselines"),
        "--n-agents", str(cli.n_agents),
        "--grid-size", str(cli.grid_size),
        "--n-value1-targets", str(cli.n_value1_targets),
        "--n-value2-targets", str(cli.n_value2_targets),
        "--sensor-radius", str(cli.sensor_radius),
        "--teammate-visibility-radius", str(cli.teammate_visibility_radius),
        "--track-radius", str(cli.track_radius),
        "--track-required", str(cli.track_required),
        "--track-progress-decay", str(cli.track_progress_decay),
        "--max-steps", str(cli.max_steps),
        "--global-state-mode", cli.global_state_mode,
        "--reward-mode", cli.reward_mode,
        "--coverage-potential-scale", str(cli.coverage_potential_scale),
        "--detection-potential-scale", str(cli.detection_potential_scale),
        "--progress-potential-scale", str(cli.progress_potential_scale),
        "--gamma", str(cli.gamma),
    ]
    if cli.fixed_scenario:
        command.extend(
            ["--fixed-scenario", "--scenario-seed", str(cli.scenario_seed)]
        )
    if cli.include_agent_id_map:
        command.append("--include-agent-id-map")
    return run_command(
        command,
        log_path=log_root / resolved["stage"] / "baselines.log",
        manifest_path=manifest,
        dry_run=cli.dry_run,
        continue_on_error=cli.continue_on_error,
    )


def _analysis(cli, resolved, run_root: Path, log_root: Path, manifest: Path) -> None:
    aggregate_dir = log_root / resolved["stage"] / "aggregate"
    commands = [
        (
            [
                sys.executable,
                "scripts/aggregate_thesis_results.py",
                "--run-root", str(run_root / resolved["stage"]),
                "--baseline-root", str(log_root / resolved["stage"] / "baselines"),
                "--output-dir", str(aggregate_dir),
            ],
            log_root / resolved["stage"] / "aggregate.log",
        ),
        (
            [
                sys.executable,
                "scripts/plot_thesis_results.py",
                "--aggregate-dir", str(aggregate_dir),
                "--output-dir", str(log_root / resolved["stage"] / "plots"),
            ],
            log_root / resolved["stage"] / "plots.log",
        ),
        (
            [
                sys.executable,
                "scripts/inspect_thesis_runs.py",
                "--run-root", str(run_root / resolved["stage"]),
                "--log-root", str(log_root / resolved["stage"]),
                "--output-dir", str(log_root / resolved["stage"] / "diagnostics"),
            ],
            log_root / resolved["stage"] / "diagnostics.log",
        ),
    ]
    for command, log_path in commands:
        run_command(
            command,
            log_path=log_path,
            manifest_path=manifest,
            dry_run=cli.dry_run,
            continue_on_error=cli.continue_on_error,
        )


def main() -> None:
    cli = build_parser().parse_args()
    if cli.mode == "fixed":
        cli.fixed_scenario = True
    resolved = _resolved(cli)
    run_root = Path(cli.run_root)
    log_root = Path(cli.log_root)
    manifest = log_root / resolved["stage"] / "manifest.jsonl"

    print("Resolved experiment:")
    print(json.dumps(resolved, indent=2))

    if cli.mode == "status":
        command = [
            sys.executable,
            "scripts/inspect_thesis_runs.py",
            "--run-root", str(run_root / resolved["stage"]),
            "--log-root", str(log_root / resolved["stage"]),
            "--output-dir", str(log_root / resolved["stage"] / "diagnostics"),
        ]
        run_command(
            command,
            log_path=log_root / resolved["stage"] / "diagnostics.log",
            manifest_path=manifest,
            dry_run=cli.dry_run,
            continue_on_error=cli.continue_on_error,
        )
        return

    if cli.mode == "aggregate":
        _analysis(cli, resolved, run_root, log_root, manifest)
        return

    if cli.mode == "evaluate":
        search_root = run_root / resolved["stage"]
        for config_path in sorted(search_root.glob("**/run_config.json")):
            run_dir = config_path.parent
            config = json.loads(config_path.read_text())
            scope = "multi" if str(config.get("algo", "")).startswith(("shared_", "qmix_", "bayes_qmix_")) else "single"
            probability = float(config["detection_probability"])
            seed = int(config["seed"])
            _evaluate_one(cli, resolved, run_dir, scope, probability, seed, log_root, manifest)
        if not cli.skip_analysis:
            _analysis(cli, resolved, run_root, log_root, manifest)
        return

    if not cli.skip_tests:
        run_command(
            [sys.executable, "-m", "pytest", "-q", "tests/test_thesis_corrections.py", "tests/test_thesis_automation.py"],
            log_path=log_root / resolved["stage"] / "tests.log",
            manifest_path=manifest,
            dry_run=cli.dry_run,
            continue_on_error=cli.continue_on_error,
        )

    for probability in resolved["probabilities"]:
        for seed in resolved["seeds"]:
            if cli.scope in {"single", "both"}:
                # DDQN must precede BDQN because the corrected BDQN warm-starts
                # from the matching deterministic representation.
                ordered_single = sorted(
                    resolved["single_algos"], key=lambda name: 0 if name == "ddqn" else 1
                )
                for algo in ordered_single:
                    _train_single(
                        cli, resolved, probability, seed, algo, run_root, log_root, manifest
                    )
            if cli.scope in {"multi", "both"}:
                ordered_multi = sorted(
                    resolved["multi_algos"],
                    key=lambda name: (
                        0 if name == "shared_ddqn" else 1 if name == "shared_bdqn" else 2
                    ),
                )
                for algo in ordered_multi:
                    _train_multi(
                        cli, resolved, probability, seed, algo, run_root, log_root, manifest
                    )

    if not cli.skip_baselines:
        _run_baselines(cli, resolved, log_root, manifest)
    if not cli.skip_analysis:
        _analysis(cli, resolved, run_root, log_root, manifest)


if __name__ == "__main__":
    main()
