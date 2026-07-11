import argparse
import subprocess
import os
from pathlib import Path


def run_and_check(cmd, log_path):
    print(" ".join(cmd))
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"\nERROR: command failed with code {result.returncode}")
        print(f"Log file: {log_path}")
        print("\nLast 40 log lines:")
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
            for line in lines[-40:]:
                print(line.rstrip())
        except FileNotFoundError:
            pass
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reward_version", nargs="?", default="v2_frontier")
    parser.add_argument("algo_arg", nargs="?", default="ddqn")
    parser.add_argument("seed_arg", nargs="?", default="43")
    args = parser.parse_args()

    reward_version = args.reward_version
    algo_arg = args.algo_arg
    seed_arg = args.seed_arg

    algos = ["dqn", "ddqn", "bdqn"]
    seeds = ["42", "43", "44"]

    if algo_arg != "all":
        if algo_arg not in algos:
            raise ValueError(f"Unknown algo: {algo_arg}. Expected one of {algos} or 'all'.")
        algos = [algo_arg]

    if seed_arg != "all":
        if seed_arg not in seeds:
            raise ValueError(f"Unknown seed: {seed_arg}. Expected one of {seeds} or 'all'.")
        seeds = [seed_arg]

    Path(f"runs/{reward_version}").mkdir(parents=True, exist_ok=True)
    Path(f"logs/{reward_version}").mkdir(parents=True, exist_ok=True)

    for algo in algos:
        for seed in seeds:
            run_dir = f"runs/{reward_version}/{algo}_seed{seed}_1000"
            train_log = f"logs/{reward_version}/{algo}_seed{seed}_train.log"
            eval_log = f"logs/{reward_version}/{algo}_seed{seed}_eval1000.log"

            extra_args = []
            if algo == "bdqn":
                extra_args += ["--posterior-update-period", "500"]

            print(f"\n=== TRAIN {reward_version} {algo} seed {seed} ===")

            train_cmd = [
                "python", "scripts/train.py",
                "--algo", algo,
                "--episodes", "1000",
                "--device", "mps",
                "--reward-version", reward_version,
                "--train-every", "4",
                "--learning-starts", "1000",
                "--eval-every", "50",
                "--eval-episodes", "10",
                "--seed", seed,
                "--run-dir", run_dir,
            ] + extra_args

            run_and_check(train_cmd, train_log)

            print(f"\n=== EVAL {reward_version} {algo} seed {seed} ===")

            eval_cmd = [
                "python", "scripts/evaluate.py",
                "--algo", algo,
                "--checkpoint", f"{run_dir}/best.pt",
                "--episodes", "1000",
                "--reward-version", reward_version,
                "--device", "mps",
                "--json-out", f"logs/{reward_version}/{algo}_seed{seed}_eval1000.json",
            ]

            run_and_check(eval_cmd, eval_log)

            print(f"Done: {reward_version} {algo} seed {seed}")


if __name__ == "__main__":
    main()