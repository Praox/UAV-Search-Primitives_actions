from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace


# ---------------------------------------------------------------------
# Make repository imports work when the script is launched directly.
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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

from uav_search_belief20.marl.multi_local_baselines import (
    make_multi_local_baseline,
)


# =====================================================================
# CLI
# =====================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate thesis heuristic baselines on the same final-test "
            "worlds used by the learned policies."
        )
    )

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------

    parser.add_argument(
        "--scope",
        choices=["single", "multi", "both"],
        default="both",
    )

    parser.add_argument(
        "--probabilities",
        default="1.0,0.7",
        help="Comma-separated detection probabilities.",
    )

    parser.add_argument(
        "--single-baselines",
        default="random,frontier,oracle",
    )

    parser.add_argument(
        "--multi-baselines",
        default="random,local_frontier",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--eval-seed-base",
        type=int,
        default=200_000,
        help=(
            "First final-test world seed. With 1000 episodes this gives "
            "worlds 200000 to 200999."
        ),
    )

    parser.add_argument(
        "--policy-seed",
        type=int,
        default=999,
        help=(
            "Seed controlling random tie-breaking inside the heuristic. "
            "This is NOT the world seed."
        ),
    )

    parser.add_argument(
        "--output-root",
        default="logs/final/baselines",
    )

    # -----------------------------------------------------------------
    # Scenario generation
    # -----------------------------------------------------------------

    parser.add_argument(
        "--fixed-scenario",
        action="store_true",
        help="Use exactly the same scenario at every episode.",
    )

    parser.add_argument(
        "--scenario-seed",
        type=int,
        default=12_345,
    )

    # -----------------------------------------------------------------
    # Environment
    # -----------------------------------------------------------------

    parser.add_argument(
        "--n-agents",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--n-value1-targets",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--n-value2-targets",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--sensor-radius",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--teammate-visibility-radius",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--track-radius",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--track-required",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--track-progress-decay",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=150,
    )

    # IMPORTANT:
    # make_single_env() and make_multi_env() both require this field.
    parser.add_argument(
        "--observation-frame",
        choices=["global", "egocentric"],
        default="egocentric",
    )

    parser.add_argument(
        "--global-state-mode",
        choices=["privileged_truth", "memory_union"],
        default="memory_union",
    )

    parser.add_argument(
        "--include-agent-id-map",
        action="store_true",
    )

    # -----------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------

    parser.add_argument(
        "--reward-mode",
        choices=["legacy", "task_potential"],
        default="legacy",
    )

    # These are ignored by the legacy reward but are still passed because
    # the environment constructor accepts them.
    parser.add_argument(
        "--coverage-potential-scale",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--detection-potential-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--progress-potential-scale",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    return parser


# =====================================================================
# Environment argument namespace
# =====================================================================


def _base_namespace(
    cli: argparse.Namespace,
    probability: float,
) -> SimpleNamespace:

    return SimpleNamespace(
        # Generic seed
        seed=int(cli.policy_seed),

        # Scenario
        fixed_scenario=bool(cli.fixed_scenario),
        scenario_seed=int(cli.scenario_seed),

        # Multi-agent
        n_agents=int(cli.n_agents),

        # World
        grid_size=int(cli.grid_size),
        n_value1_targets=int(cli.n_value1_targets),
        n_value2_targets=int(cli.n_value2_targets),

        # Observation
        observation_frame=str(cli.observation_frame),

        # Sensor
        sensor_radius=int(cli.sensor_radius),
        teammate_visibility_radius=int(
            cli.teammate_visibility_radius
        ),
        detection_probability=float(probability),

        # Tracking
        track_radius=int(cli.track_radius),
        track_required=int(cli.track_required),
        track_progress_decay=int(cli.track_progress_decay),

        # Episode
        max_steps=int(cli.max_steps),

        # Multi-agent observation/state
        include_agent_id_map=bool(cli.include_agent_id_map),
        global_state_mode=str(cli.global_state_mode),

        # Reward
        reward_mode=str(cli.reward_mode),
        coverage_potential_scale=float(
            cli.coverage_potential_scale
        ),
        detection_potential_scale=float(
            cli.detection_potential_scale
        ),
        progress_potential_scale=float(
            cli.progress_potential_scale
        ),

        gamma=float(cli.gamma),
    )


# =====================================================================
# Save results
# =====================================================================


def _save_result(
    *,
    output_dir: Path,
    scope: str,
    baseline: str,
    probability: float,
    cli: argparse.Namespace,
    rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    episode_path = (
        output_dir
        / "final_test_deterministic_episodes.csv"
    )

    summary_path = (
        output_dir
        / "final_test_deterministic_summary.json"
    )

    write_csv(
        episode_path,
        rows,
    )

    payload = {
        "schema_version": 2,

        "scope": scope,
        "algo": f"baseline_{baseline}",
        "baseline": baseline,
        "is_baseline": True,

        "policy_seed": int(cli.policy_seed),

        "detection_probability": float(probability),
        "observation_frame": str(cli.observation_frame),
        "reward_mode": str(cli.reward_mode),

        "global_state_mode": (
            str(cli.global_state_mode)
            if scope == "multi"
            else None
        ),

        "policy_mode": "deterministic",

        "eval_seed_base": int(cli.eval_seed_base),
        "episodes": int(cli.episodes),

        "fixed_scenario": bool(cli.fixed_scenario),

        "scenario_seed": (
            int(cli.scenario_seed)
            if cli.fixed_scenario
            else None
        ),

        "episode_csv": str(episode_path),

        "summary": summary,
    }

    write_json(
        summary_path,
        payload,
    )

    print()
    print(
        f"[{scope.upper()}] "
        f"baseline={baseline} "
        f"pD={probability:.2f} "
        f"policy_seed={cli.policy_seed}"
    )

    print(
        "  completed = "
        f"{float(summary.get('completed_mean', float('nan'))):.3f}"
    )

    print(
        "  completed value = "
        f"{float(summary.get('completed_value_mean', float('nan'))):.3f}"
    )

    if scope == "single":
        print(
            "  coverage = "
            f"{float(summary.get('sensor_coverage_ratio_mean', float('nan'))):.3f}"
        )
    else:
        print(
            "  team coverage = "
            f"{float(summary.get('team_coverage_ratio_mean', float('nan'))):.3f}"
        )

        print(
            "  overlap = "
            f"{float(summary.get('coverage_overlap_ratio_mean', float('nan'))):.3f}"
        )

        print(
            "  collision = "
            f"{float(summary.get('collision_agent_ratio', float('nan'))):.3f}"
        )

    print(f"  saved -> {summary_path}")


# =====================================================================
# Single-UAV evaluation
# =====================================================================


def evaluate_single_baselines(
    *,
    cli: argparse.Namespace,
    probability: float,
    env_args: SimpleNamespace,
    output_root: Path,
    scenario: str,
    baseline_names: list[str],
) -> None:

    for baseline_name in baseline_names:

        holder: dict[str, object] = {}

        # New independent heuristic instance for every test world.
        def start_single(
            episode_index: int,
            world_seed: int,
        ) -> None:

            del episode_index

            baseline_seed = (
                int(cli.policy_seed)
                + 1_000_003 * int(world_seed)
            )

            holder["policy"] = make_baseline(
                baseline_name,
                seed=baseline_seed,
            )

        def single_policy(
            env,
            obs,
            episode_index,
        ):

            return holder["policy"].act(
                env,
                obs,
                episode_index,
            )

        rows, summary = evaluate_single_detailed(
            env_factory=lambda seed: make_single_env(
                env_args,
                seed,
            ),
            policy=single_policy,
            episodes=int(cli.episodes),
            seed_base=int(cli.eval_seed_base),
            on_episode_start=start_single,
        )

        _save_result(
            output_dir=(
                output_root
                / scenario
                / "single"
                / f"baseline_{baseline_name}"
            ),
            scope="single",
            baseline=baseline_name,
            probability=probability,
            cli=cli,
            rows=rows,
            summary=summary,
        )


# =====================================================================
# Multi-UAV evaluation
# =====================================================================


def evaluate_multi_baselines(
    *,
    cli: argparse.Namespace,
    probability: float,
    env_args: SimpleNamespace,
    output_root: Path,
    scenario: str,
    baseline_names: list[str],
) -> None:

    for baseline_name in baseline_names:

        holder: dict[str, object] = {}

        def start_multi(
            episode_index: int,
            world_seed: int,
        ) -> None:

            del episode_index

            baseline_seed = (
                int(cli.policy_seed)
                + 1_000_003 * int(world_seed)
            )

            holder["policy"] = make_multi_local_baseline(
                baseline_name,
                seed=baseline_seed,
            )

        def multi_policy(
            env,
            obs_all,
            masks,
            episode_index,
        ):

            del episode_index

            return holder["policy"].act(
                env,
                obs_all,
                masks,
            )

        rows, summary = evaluate_multi_detailed(
            env_factory=lambda seed: make_multi_env(
                env_args,
                seed,
            ),
            policy=multi_policy,
            episodes=int(cli.episodes),
            seed_base=int(cli.eval_seed_base),
            on_episode_start=start_multi,
        )

        _save_result(
            output_dir=(
                output_root
                / scenario
                / "multi"
                / f"baseline_{baseline_name}"
            ),
            scope="multi",
            baseline=baseline_name,
            probability=probability,
            cli=cli,
            rows=rows,
            summary=summary,
        )


# =====================================================================
# Main
# =====================================================================


def main() -> None:

    cli = build_parser().parse_args()

    output_root = Path(cli.output_root)

    probabilities = parse_probabilities(
        cli.probabilities
    )

    single_baselines = parse_csv_strings(
        cli.single_baselines
    )

    multi_baselines = parse_csv_strings(
        cli.multi_baselines
    )

    print("=" * 70)
    print("THESIS BASELINE FINAL EVALUATION")
    print("=" * 70)

    print(
        f"scope              : {cli.scope}"
    )
    print(
        f"observation frame  : {cli.observation_frame}"
    )
    print(
        f"reward mode        : {cli.reward_mode}"
    )
    print(
        f"detection probs    : {probabilities}"
    )
    print(
        f"evaluation worlds  : "
        f"{cli.eval_seed_base} -> "
        f"{cli.eval_seed_base + cli.episodes - 1}"
    )
    print(
        f"policy seed        : {cli.policy_seed}"
    )
    print(
        f"output root        : {output_root}"
    )
    print("=" * 70)

    for probability in probabilities:

        env_args = _base_namespace(
            cli,
            probability,
        )

        scenario = probability_label(
            probability
        )

        print()
        print(
            f"--- Detection probability "
            f"pD={probability:.2f} ---"
        )

        if cli.scope in {
            "single",
            "both",
        }:

            evaluate_single_baselines(
                cli=cli,
                probability=probability,
                env_args=env_args,
                output_root=output_root,
                scenario=scenario,
                baseline_names=single_baselines,
            )

        if cli.scope in {
            "multi",
            "both",
        }:

            evaluate_multi_baselines(
                cli=cli,
                probability=probability,
                env_args=env_args,
                output_root=output_root,
                scenario=scenario,
                baseline_names=multi_baselines,
            )


if __name__ == "__main__":
    main()