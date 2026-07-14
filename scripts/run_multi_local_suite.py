from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_ALGOS = ["shared_ddqn", "shared_bdqn", "qmix_ddqn"]


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
        tail = log_path.read_text(errors="replace").splitlines()[-40:]
        print("\n".join(tail))
        raise SystemExit(result.returncode)


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def launch(args, algos: list[str], seeds: list[int], label: str) -> None:
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
                "--algo", algo,
                "--mode", mode,
                "--seed", str(seed),
                "--episodes", str(args.train_episodes),
                "--eval-every", str(args.eval_every),
                "--eval-episodes", str(args.eval_episodes),
                "--final-eval-episodes", str(args.final_eval_episodes),
                "--device", args.device,
                "--run-dir", str(run_dir),
                "--eval-json", str(eval_json),
                "--n-agents", str(args.n_agents),
                "--train-every", str(args.train_every),
                "--learning-starts", str(args.learning_starts),
            ]
            run(command, log_root / f"{tag}.log")
    run([
        sys.executable,
        "scripts/aggregate_multi_local.py",
        "--input-root", str(log_root),
        "--output-dir", str(log_root / "aggregate"),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["smoke", "shared-screen", "qmix-screen", "screen", "confirm"],
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--algos", default=",".join(DEFAULT_ALGOS))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--train-episodes", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--final-eval-episodes", type=int, default=300)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--run-root", default="runs/multi_local")
    parser.add_argument("--log-root", default="logs/multi_local")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.mode == "smoke":
        run([sys.executable, "scripts/smoke_test_multi_local.py"])
        run([sys.executable, "scripts/validate_ctde.py"])
        return

    run([sys.executable, "scripts/smoke_test_multi_local.py"])
    run([sys.executable, "scripts/validate_ctde.py"])
    seeds = [int(seed) for seed in parse_list(args.seeds)]
    if args.mode == "shared-screen":
        algos, label = ["shared_ddqn", "shared_bdqn"], "shared_screen"
    elif args.mode == "qmix-screen":
        algos, label = ["qmix_ddqn"], "qmix_screen"
    elif args.mode == "screen":
        algos, label = parse_list(args.algos), "screen"
    else:
        algos, label = parse_list(args.algos), "confirm"
        if args.seeds == "42,43,44":
            seeds = list(range(42, 49))
        if args.train_episodes == 400:
            args.train_episodes = 1000
        if args.final_eval_episodes == 300:
            args.final_eval_episodes = 1000
    launch(args, algos, seeds, label)


if __name__ == "__main__":
    main()
