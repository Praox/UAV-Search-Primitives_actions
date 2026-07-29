from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_ALGOS = ["shared_ddqn", "shared_bdqn", "qmix_ddqn"]
BAYES_ALGOS = ["bayes_qmix_shared", "bayes_qmix_independent"]
BAYES_COMPARISON_ALGOS = ["qmix_ddqn", *BAYES_ALGOS]


def valid_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open() as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return all(key in payload for key in ("algo", "seed", "completed_mean"))


def run(command: list[str], log_path: Path | None = None) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = log_path.read_text(errors="replace").splitlines()[-60:]
        print("\n".join(tail))
        raise SystemExit(result.returncode)


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def scenario_slug(detection_probability: float, global_state_mode: str) -> str:
    probability = f"{float(detection_probability):.2f}".replace(".", "p")
    state = "privileged" if global_state_mode == "privileged_truth" else "memory_union"
    return f"pdet_{probability}__state_{state}"


def aggregate(
    log_root: Path,
    *,
    reference_roots: list[Path] | None = None,
) -> None:
    command = [
        sys.executable,
        "scripts/aggregate_multi_local.py",
        "--input-root",
        str(log_root),
    ]
    for reference in reference_roots or []:
        if reference.exists():
            command.extend(["--input-root", str(reference)])
        else:
            print(f"reference root ignored because it does not exist: {reference}")
    command.extend(["--output-dir", str(log_root / "aggregate")])
    run(command)


def launch(
    args,
    *,
    algos: list[str],
    seeds: list[int],
    label: str,
    scenario_label: str,
    detection_probability: float,
    global_state_mode: str,
    reference_roots: list[Path] | None = None,
) -> None:
    run_root = Path(args.run_root) / label
    log_root = Path(args.log_root) / label
    for algo in algos:
        for seed in seeds:
            tag = f"{algo}_seed{seed}_train{args.train_episodes}"
            run_dir = run_root / tag
            eval_json = log_root / f"{tag}_eval.json"
            if valid_json(eval_json) and not args.force:
                print(f"skip completed: {eval_json}")
                continue
            mode = (
                "eval_only"
                if (run_dir / "best.pt").exists() and not args.force
                else "train_eval"
            )
            command = [
                sys.executable,
                "scripts/train_multi_local.py",
                "--algo",
                algo,
                "--mode",
                mode,
                "--seed",
                str(seed),
                "--episodes",
                str(args.train_episodes),
                "--eval-every",
                str(args.eval_every),
                "--eval-episodes",
                str(args.eval_episodes),
                "--final-eval-episodes",
                str(args.final_eval_episodes),
                "--device",
                args.device,
                "--run-dir",
                str(run_dir),
                "--eval-json",
                str(eval_json),
                "--n-agents",
                str(args.n_agents),
                "--grid-size",
                str(args.grid_size),
                "--max-steps",
                str(args.max_steps),
                "--feature-dim",
                str(args.feature_dim),
                "--batch-size",
                str(args.batch_size),
                "--replay-capacity",
                str(args.replay_capacity),
                "--target-update-period",
                str(args.target_update_period),
                "--torch-threads",
                str(args.torch_threads),
                "--eval-seed-base",
                str(args.eval_seed_base),
                "--posterior-eval-seed-base",
                str(args.posterior_eval_seed_base),
                "--train-every",
                str(args.train_every),
                "--learning-starts",
                str(args.learning_starts),
                "--scenario-label",
                scenario_label,
                "--detection-probability",
                str(detection_probability),
                "--global-state-mode",
                global_state_mode,
                "--bayes-prior-std",
                str(args.bayes_prior_std),
                "--bayes-initial-std",
                str(args.bayes_initial_std),
                "--bayes-kl-weight",
                str(args.bayes_kl_weight),
                "--bayes-epsilon-start",
                str(args.bayes_epsilon_start),
                "--bayes-epsilon-end",
                str(args.bayes_epsilon_end),
            ]
            run(command, log_root / f"{tag}.log")
    aggregate(log_root, reference_roots=reference_roots)


def run_all_smoke_tests() -> None:
    # The Bayesian smoke test is cumulative: it includes the original local-memory
    # and CTDE contracts plus posterior-sampling and gradient checks. Running one
    # Python process is much faster and avoids repeated Torch allocator startup.
    run([sys.executable, "scripts/smoke_test_bayesian_qmix.py"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "smoke",
            "shared-screen",
            "qmix-screen",
            "screen",
            "confirm",
            "bayes-smoke",
            "bayes-kl-screen",
            "bayes-screen",
            "bayes-confirm",
            "bayes-noise-screen",
            "bayes-state-screen",
        ],
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--algos", default=",".join(DEFAULT_ALGOS))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--target-update-period", type=int, default=500)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--eval-seed-base", type=int, default=100_000)
    parser.add_argument("--posterior-eval-seed-base", type=int, default=900_000)
    parser.add_argument("--train-episodes", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--final-eval-episodes", type=int, default=300)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--run-root", default="runs/multi_local")
    parser.add_argument("--log-root", default="logs/multi_local")
    parser.add_argument("--reference-root", action="append", default=[])
    parser.add_argument("--detection-probabilities", default="1.0,0.8,0.6")
    parser.add_argument(
        "--state-modes", default="privileged_truth,memory_union"
    )
    parser.add_argument("--bayes-kl-weights", default="0.0001,0.001,0.01")
    parser.add_argument("--bayes-prior-std", type=float, default=1.0)
    parser.add_argument("--bayes-initial-std", type=float, default=0.05)
    parser.add_argument("--bayes-kl-weight", type=float, default=1e-3)
    parser.add_argument("--bayes-epsilon-start", type=float, default=0.0)
    parser.add_argument("--bayes-epsilon-end", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.mode in {"smoke", "bayes-smoke"}:
        run_all_smoke_tests()
        return

    run_all_smoke_tests()
    seeds = [int(seed) for seed in parse_list(args.seeds)]
    explicit_references = [Path(path) for path in args.reference_root]

    if args.mode == "shared-screen":
        launch(
            args,
            algos=["shared_ddqn", "shared_bdqn"],
            seeds=seeds,
            label="shared_screen",
            scenario_label="deterministic_privileged",
            detection_probability=1.0,
            global_state_mode="privileged_truth",
        )
        return

    if args.mode == "qmix-screen":
        launch(
            args,
            algos=["qmix_ddqn"],
            seeds=seeds,
            label="qmix_screen",
            scenario_label="deterministic_privileged",
            detection_probability=1.0,
            global_state_mode="privileged_truth",
        )
        return

    if args.mode == "screen":
        launch(
            args,
            algos=parse_list(args.algos),
            seeds=seeds,
            label="screen",
            scenario_label="deterministic_privileged",
            detection_probability=1.0,
            global_state_mode="privileged_truth",
        )
        return

    if args.mode == "confirm":
        if args.seeds == "42,43,44":
            seeds = list(range(42, 49))
        if args.train_episodes == 400:
            args.train_episodes = 1000
        if args.final_eval_episodes == 300:
            args.final_eval_episodes = 1000
        launch(
            args,
            algos=parse_list(args.algos),
            seeds=seeds,
            label="confirm",
            scenario_label="deterministic_privileged",
            detection_probability=1.0,
            global_state_mode="privileged_truth",
        )
        return

    if args.mode == "bayes-kl-screen":
        original_kl = args.bayes_kl_weight
        original_train = args.train_episodes
        original_eval = args.final_eval_episodes
        if args.train_episodes == 400:
            args.train_episodes = 250
        if args.final_eval_episodes == 300:
            args.final_eval_episodes = 150
        for weight_text in parse_list(args.bayes_kl_weights):
            weight = float(weight_text)
            args.bayes_kl_weight = weight
            slug = f"kl_{weight:.0e}".replace("+", "")
            launch(
                args,
                algos=["bayes_qmix_shared"],
                seeds=seeds,
                label=f"bayesian_qmix/kl_screen/{slug}",
                scenario_label=f"kl_screen_{weight:.0e}",
                detection_probability=1.0,
                global_state_mode="privileged_truth",
            )
        args.bayes_kl_weight = original_kl
        args.train_episodes = original_train
        args.final_eval_episodes = original_eval
        aggregate(Path(args.log_root) / "bayesian_qmix/kl_screen")
        return

    if args.mode == "bayes-screen":
        references = explicit_references or [Path(args.log_root) / "screen"]
        launch(
            args,
            algos=BAYES_ALGOS,
            seeds=seeds,
            label="bayesian_qmix/screen/deterministic_privileged",
            scenario_label="deterministic_privileged",
            detection_probability=1.0,
            global_state_mode="privileged_truth",
            reference_roots=references,
        )
        return

    if args.mode == "bayes-confirm":
        if args.seeds == "42,43,44":
            seeds = list(range(42, 49))
        if args.train_episodes == 400:
            args.train_episodes = 1000
        if args.final_eval_episodes == 300:
            args.final_eval_episodes = 1000
        references = explicit_references or [Path(args.log_root) / "confirm"]
        launch(
            args,
            algos=BAYES_ALGOS,
            seeds=seeds,
            label="bayesian_qmix/confirm/deterministic_privileged",
            scenario_label="deterministic_privileged",
            detection_probability=1.0,
            global_state_mode="privileged_truth",
            reference_roots=references,
        )
        return

    if args.mode == "bayes-noise-screen":
        for probability_text in parse_list(args.detection_probabilities):
            probability = float(probability_text)
            slug = scenario_slug(probability, "privileged_truth")
            launch(
                args,
                algos=BAYES_COMPARISON_ALGOS,
                seeds=seeds,
                label=f"bayesian_qmix/noise_screen/{slug}",
                scenario_label=f"noise_pdet_{probability:.2f}",
                detection_probability=probability,
                global_state_mode="privileged_truth",
            )
        aggregate(Path(args.log_root) / "bayesian_qmix/noise_screen")
        return

    if args.mode == "bayes-state-screen":
        for state_mode in parse_list(args.state_modes):
            if state_mode not in {"privileged_truth", "memory_union"}:
                raise ValueError(f"Unknown state mode: {state_mode}")
            slug = scenario_slug(1.0, state_mode)
            launch(
                args,
                algos=BAYES_COMPARISON_ALGOS,
                seeds=seeds,
                label=f"bayesian_qmix/state_screen/{slug}",
                scenario_label=f"state_{state_mode}",
                detection_probability=1.0,
                global_state_mode=state_mode,
            )
        aggregate(Path(args.log_root) / "bayesian_qmix/state_screen")
        return

    raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
