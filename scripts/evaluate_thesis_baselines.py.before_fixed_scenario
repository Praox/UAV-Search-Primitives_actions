from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from types import SimpleNamespace

from scripts.train_thesis_multi import make_env as make_multi_env
from scripts.train_thesis_single import make_env as make_single_env
from uav_search_belief20.baselines import make_baseline
from uav_search_belief20.experiments.thesis_automation import (
    parse_csv_strings,
    parse_probabilities,
    probability_label,
    write_csv,
    write_json,
)
from uav_search_belief20.experiments.thesis_evaluation import (
    evaluate_multi_detailed,
    evaluate_single_detailed,
)
from uav_search_belief20.marl.multi_local_baselines import make_multi_local_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate random/frontier/oracle baselines in corrected thesis environments."
    )
    parser.add_argument("--scope", choices=["single", "multi", "both"], default="both")
    parser.add_argument("--probabilities", default="1.0,0.7")
    parser.add_argument("--single-baselines", default="random,frontier,oracle")
    parser.add_argument("--multi-baselines", default="random,local_frontier")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-seed-base", type=int, default=200_000)
    parser.add_argument("--policy-seed", type=int, default=999)
    parser.add_argument("--output-root", default="logs/thesis_v2/baselines")

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
    parser.add_argument("--global-state-mode", choices=["privileged_truth", "memory_union"], default="memory_union")
    parser.add_argument("--include-agent-id-map", action="store_true")
    parser.add_argument("--reward-mode", choices=["legacy", "task_potential"], default="task_potential")
    parser.add_argument("--coverage-potential-scale", type=float, default=5.0)
    parser.add_argument("--detection-potential-scale", type=float, default=1.0)
    parser.add_argument("--progress-potential-scale", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser


def _base_namespace(cli, probability: float) -> SimpleNamespace:
    return SimpleNamespace(
        seed=cli.policy_seed,
        n_agents=cli.n_agents,
        grid_size=cli.grid_size,
        n_value1_targets=cli.n_value1_targets,
        n_value2_targets=cli.n_value2_targets,
        sensor_radius=cli.sensor_radius,
        teammate_visibility_radius=cli.teammate_visibility_radius,
        detection_probability=float(probability),
        track_radius=cli.track_radius,
        track_required=cli.track_required,
        track_progress_decay=cli.track_progress_decay,
        max_steps=cli.max_steps,
        include_agent_id_map=cli.include_agent_id_map,
        global_state_mode=cli.global_state_mode,
        reward_mode=cli.reward_mode,
        coverage_potential_scale=cli.coverage_potential_scale,
        detection_potential_scale=cli.detection_potential_scale,
        progress_potential_scale=cli.progress_potential_scale,
        gamma=cli.gamma,
    )


def _save_result(
    *,
    output_dir: Path,
    scope: str,
    algo: str,
    baseline: str,
    probability: float,
    cli,
    rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "final_test_deterministic_episodes.csv"
    summary_path = output_dir / "final_test_deterministic_summary.json"
    write_csv(episode_path, rows)
    write_json(
        summary_path,
        {
            "schema_version": 1,
            "scope": scope,
            "algo": algo,
            "baseline": baseline,
            "is_baseline": True,
            "training_seed": int(cli.policy_seed),
            "detection_probability": float(probability),
            "reward_mode": cli.reward_mode,
            "global_state_mode": cli.global_state_mode if scope == "multi" else None,
            "policy_mode": "deterministic",
            "eval_seed_base": int(cli.eval_seed_base),
            "episodes": int(cli.episodes),
            "episode_csv": str(episode_path),
            "summary": summary,
        },
    )
    print(
        f"[{scope} {baseline} pD={probability:.2f}] "
        f"reward={summary.get('reward_mean', float('nan')):.3f} "
        f"completed={summary.get('completed_mean', float('nan')):.3f}"
    )
    print(f"Saved {summary_path}")


def main() -> None:
    cli = build_parser().parse_args()
    output_root = Path(cli.output_root)
    probabilities = parse_probabilities(cli.probabilities)
    single_baselines = parse_csv_strings(cli.single_baselines)
    multi_baselines = parse_csv_strings(cli.multi_baselines)

    for probability in probabilities:
        env_args = _base_namespace(cli, probability)
        scenario = probability_label(probability)

        if cli.scope in {"single", "both"}:
            for baseline_name in single_baselines:
                holder: dict[str, object] = {}

                def start_single(episode_index: int, world_seed: int) -> None:
                    del episode_index
                    holder["policy"] = make_baseline(
                        baseline_name,
                        seed=cli.policy_seed + 1_000_003 * world_seed,
                    )

                def single_policy(env, obs, episode_index):
                    return holder["policy"].act(env, obs, episode_index)

                rows, summary = evaluate_single_detailed(
                    env_factory=lambda seed: make_single_env(env_args, seed),
                    policy=single_policy,
                    episodes=cli.episodes,
                    seed_base=cli.eval_seed_base,
                    on_episode_start=start_single,
                )
                _save_result(
                    output_dir=output_root / scenario / "single" / f"baseline_{baseline_name}",
                    scope="single",
                    algo=f"baseline_{baseline_name}",
                    baseline=baseline_name,
                    probability=probability,
                    cli=cli,
                    rows=rows,
                    summary=summary,
                )

        if cli.scope in {"multi", "both"}:
            for baseline_name in multi_baselines:
                holder = {}

                def start_multi(episode_index: int, world_seed: int) -> None:
                    del episode_index
                    holder["policy"] = make_multi_local_baseline(
                        baseline_name,
                        seed=cli.policy_seed + 1_000_003 * world_seed,
                    )

                def multi_policy(env, obs_all, masks, episode_index):
                    del episode_index
                    return holder["policy"].act(env, obs_all, masks)

                rows, summary = evaluate_multi_detailed(
                    env_factory=lambda seed: make_multi_env(env_args, seed),
                    policy=multi_policy,
                    episodes=cli.episodes,
                    seed_base=cli.eval_seed_base,
                    on_episode_start=start_multi,
                )
                _save_result(
                    output_dir=output_root / scenario / "multi" / f"baseline_{baseline_name}",
                    scope="multi",
                    algo=f"baseline_{baseline_name}",
                    baseline=baseline_name,
                    probability=probability,
                    cli=cli,
                    rows=rows,
                    summary=summary,
                )


if __name__ == "__main__":
    main()
