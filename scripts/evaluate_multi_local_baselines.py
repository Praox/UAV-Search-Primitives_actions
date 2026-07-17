from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from uav_search_belief20.envs.multi_drone_local_env import (
    MultiDroneLocalEnvConfig,
    MultiDroneLocalMemoryEnv,
)
from uav_search_belief20.evaluation_multi_local import evaluate_multi_local_policy
from uav_search_belief20.marl.multi_local_baselines import make_multi_local_baseline


def scenario_slug(detection_probability: float, global_state_mode: str) -> str:
    probability = f"{float(detection_probability):.2f}".replace(".", "p")
    state = "privileged" if global_state_mode == "privileged_truth" else "memory_union"
    return f"pdet_{probability}__state_{state}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate no-learning multi-UAV local-memory baselines."
    )
    parser.add_argument("--baselines", default="random,local_frontier")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-seed-base", type=int, default=100_000)
    parser.add_argument("--policy-seed", type=int, default=999)
    parser.add_argument("--output-dir", default="logs/multi_local/baselines")
    parser.add_argument("--scenario-label", default="")

    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--n-value1-targets", type=int, default=3)
    parser.add_argument("--n-value2-targets", type=int, default=1)
    parser.add_argument("--sensor-radius", type=int, default=2)
    parser.add_argument("--teammate-visibility-radius", type=int, default=2)
    parser.add_argument("--detection-probability", type=float, default=1.0)
    parser.add_argument("--track-radius", type=int, default=1)
    parser.add_argument("--track-required", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--collision-penalty", type=float, default=-0.02)
    parser.add_argument(
        "--global-state-mode",
        choices=["privileged_truth", "memory_union"],
        default="privileged_truth",
    )
    parser.add_argument("--reward-version", default="multi_local_v1_from_single_D")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < float(args.detection_probability) <= 1.0:
        raise ValueError("detection_probability must be in (0, 1].")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = args.scenario_label or scenario_slug(
        args.detection_probability, args.global_state_mode
    )
    baseline_names = [
        item.strip() for item in str(args.baselines).split(",") if item.strip()
    ]
    rows: list[dict] = []

    for baseline_name in baseline_names:
        policy_object = make_multi_local_baseline(
            baseline_name, seed=int(args.policy_seed)
        )
        holder: dict[str, MultiDroneLocalMemoryEnv] = {}

        def env_factory(world_seed: int) -> MultiDroneLocalMemoryEnv:
            env = MultiDroneLocalMemoryEnv(
                MultiDroneLocalEnvConfig(
                    grid_size=args.grid_size,
                    n_agents=args.n_agents,
                    n_value1_targets=args.n_value1_targets,
                    n_value2_targets=args.n_value2_targets,
                    sensor_radius=args.sensor_radius,
                    teammate_visibility_radius=args.teammate_visibility_radius,
                    detection_probability=args.detection_probability,
                    track_radius=args.track_radius,
                    track_required=args.track_required,
                    max_steps=args.max_steps,
                    seed=int(world_seed),
                    reward_version=args.reward_version,
                    collision_penalty=args.collision_penalty,
                    global_state_mode=args.global_state_mode,
                )
            )
            holder["env"] = env
            policy_object.begin_episode(int(world_seed))
            return env

        def policy(obs_all, masks):
            env = holder.get("env")
            if env is None:
                raise RuntimeError("Baseline policy called before environment creation.")
            return policy_object.act(env, obs_all, masks)

        metrics = evaluate_multi_local_policy(
            policy=policy,
            env_factory=env_factory,
            episodes=int(args.episodes),
            eval_seed_base=int(args.eval_seed_base),
        )
        normalized = str(baseline_name).strip().lower().replace("-", "_")
        algo = "baseline_local_frontier" if normalized in {"frontier", "local_frontier"} else "baseline_random"
        metrics.update(
            {
                "scope": "multi",
                "algo": algo,
                "baseline": normalized,
                "seed": int(args.policy_seed),
                "scenario_label": scenario,
                "detection_probability": float(args.detection_probability),
                "global_state_mode": args.global_state_mode,
                "evaluation_policy": "heuristic" if "frontier" in algo else "random",
                "posterior_sampling": "none",
            }
        )
        rows.append(metrics)

        output = output_dir / f"{algo}_eval{int(args.episodes)}.json"
        output.write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")
        print(f"Saved {output}")

    summary_path = output_dir / "baseline_summary.csv"
    fields = sorted({key for row in rows for key in row if key != "action_counts"})
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
