import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reward_version", nargs="?", default="v2_frontier")
    parser.add_argument("algo_arg", nargs="?", default="ddqn")
    parser.add_argument("seed_arg", nargs="?", default="43")
    args = parser.parse_args()

    #reward_version = args.reward_version
    algo_arg = args.algo_arg
    seed_arg = args.seed_arg

    ALGOS = ["dqn", "ddqn", "bdqn"]
    SEEDS = ["42", "43", "44"]

    if algo_arg != "all":
        ALGOS = [algo_arg]

    if seed_arg != "all":
        SEEDS = [seed_arg]

    os.makedirs(f"runs/{reward_version}", exist_ok=True)
    os.makedirs(f"logs/{reward_version}", exist_ok=True)

    for algo in ALGOS:
        for seed in SEEDS:
            run_dir = f"runs/{reward_version}/{algo}_seed{seed}_1000"
            train_log = f"logs/{reward_version}/{algo}_seed{seed}_train.log"
            eval_log = f"logs/{reward_version}/{algo}_seed{seed}_eval1000.log"

            extra_args = []
            if algo == "bdqn":
                extra_args += ["--posterior-update-period", "500"]

            print(f"=== TRAIN {reward_version} {algo} seed {seed} ===")

            train_cmd = [
                "python", "scripts/train.py",
                "--algo", algo,
                "--episodes", "1000",
                "--device", "cuda",
                "--train-every", "4",
                "--learning-starts", "1000",
                "--eval-every", "50",
                "--eval-episodes", "10",
                "--seed", seed,
                "--run-dir", run_dir,
            ] + extra_args

            with open(train_log, "w") as f:
            
                subprocess.run(train_cmd, stdout=f, stderr=subprocess.STDOUT)

            print(f"=== EVAL {reward_version} {algo} seed {seed} ===")

            eval_cmd = [
                "python", "scripts/evaluate.py",
                "--algo", algo,
                "--checkpoint", f"{run_dir}/best.pt",
                "--episodes", "1000",
            ]

            with open(eval_log, "w") as f:
                subprocess.run(eval_cmd, stdout=f, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
