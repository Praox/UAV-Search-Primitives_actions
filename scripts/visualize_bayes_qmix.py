from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_thesis_multi import actions_for, make_agent, make_env
from uav_search_belief20.actions import ACTION_NAMES
from uav_search_belief20.experiments.thesis_automation import load_json, resolve_checkpoint
from uav_search_belief20.marl.thesis_qmix import ThesisBayesianQMIXAgent
from uav_search_belief20.utils import pick_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize one bayes_qmix_independent rollout on the trained world."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--posterior-seed", type=int, default=7)
    parser.add_argument(
        "--scenario-seed",
        type=int,
        default=None,
        help="Optional override; otherwise uses run_config.json.",
    )
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--hide-undetected-targets", action="store_true")
    parser.add_argument("--export-deployment-policy", action="store_true")
    return parser


def fresh_agent(config: dict, checkpoint: Path, device: str):
    values = dict(config)
    values["device"] = device
    values["warmstart_ddqn"] = ""
    values.setdefault("fixed_scenario", False)
    values.setdefault("scenario_seed", int(values.get("seed", 42)))
    args = SimpleNamespace(**values)
    env = make_env(args, int(args.seed))
    agent = make_agent(args, env, device)
    agent.load(str(checkpoint))
    if not isinstance(agent, ThesisBayesianQMIXAgent):
        raise TypeError(f"Expected ThesisBayesianQMIXAgent, got {type(agent).__name__}")
    return args, env, agent


def snapshot(env, info: dict, actions, reward: float, cumulative_reward: float) -> dict:
    return {
        "step": int(info["t"]),
        "drone_pos": np.asarray(info["drone_pos"], dtype=np.int64).copy(),
        "target_pos": np.asarray(info["target_pos"], dtype=np.int64).copy(),
        "target_values": np.asarray(info["target_values"], dtype=np.int64).copy(),
        "detected_flags": np.asarray(info["detected_flags"], dtype=bool).copy(),
        "completed_flags": np.asarray(info["completed_flags"], dtype=bool).copy(),
        "track_progress": np.asarray(info["track_progress"], dtype=np.int64).copy(),
        "team_visited": np.asarray(env.team_visited, dtype=np.float32).copy(),
        "actions": None if actions is None else np.asarray(actions, dtype=np.int64).copy(),
        "action_names": [
            "initial" for _ in range(env.cfg.n_agents)
        ] if actions is None else [ACTION_NAMES[int(action)] for action in actions],
        "reward": float(reward),
        "cumulative_reward": float(cumulative_reward),
        "detected": int(info["detected"]),
        "completed": int(info["completed"]),
        "collision_count": int(info.get("last_collision_count", 0)),
        "tracking_agents": np.asarray(info.get("last_tracking_progress", []), dtype=bool).copy(),
    }


def rollout(agent, env, posterior_seed: int, max_steps: int) -> list[dict]:
    # One independent posterior draw per agent, sampled once for the whole episode.
    agent.cfg.posterior_sampling = "independent"
    seed_everything(int(posterior_seed))
    agent.resample_policy()

    obs_all, info = env.reset()
    frames = [snapshot(env, info, None, 0.0, 0.0)]
    cumulative_reward = 0.0
    done = False

    while not done and int(info["t"]) < max_steps:
        masks = env.action_mask()
        actions = actions_for(
            agent,
            "bayes_qmix_independent",
            obs_all,
            masks,
            train=False,
            sampled=True,
        )
        obs_all, reward, terminated, truncated, info = env.step(actions)
        cumulative_reward += float(reward)
        frames.append(snapshot(env, info, actions, reward, cumulative_reward))
        done = bool(terminated or truncated)
    return frames


def save_trajectory(path: Path, frames: list[dict]) -> None:
    fields = [
        "step", "agent_id", "row", "col", "action", "reward",
        "cumulative_reward", "detected", "completed", "collision_count", "tracking",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in frames:
            for agent_id, (row, col) in enumerate(frame["drone_pos"]):
                tracking = (
                    False if frame["tracking_agents"].size == 0
                    else bool(frame["tracking_agents"][agent_id])
                )
                writer.writerow({
                    "step": frame["step"],
                    "agent_id": agent_id,
                    "row": int(row),
                    "col": int(col),
                    "action": frame["action_names"][agent_id],
                    "reward": frame["reward"],
                    "cumulative_reward": frame["cumulative_reward"],
                    "detected": frame["detected"],
                    "completed": frame["completed"],
                    "collision_count": frame["collision_count"],
                    "tracking": tracking,
                })


def save_summary(path: Path, run_dir: Path, checkpoint: Path, config: dict,
                 frames: list[dict], posterior_seed: int) -> None:
    first, final = frames[0], frames[-1]
    payload = {
        "algo": "bayes_qmix_independent",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "training_seed": int(config["seed"]),
        "fixed_scenario": bool(config.get("fixed_scenario", False)),
        "scenario_seed": int(config.get("scenario_seed", config["seed"])),
        "posterior_seed": int(posterior_seed),
        "initial_drone_positions": first["drone_pos"].tolist(),
        "target_positions": first["target_pos"].tolist(),
        "target_values": first["target_values"].tolist(),
        "episode_steps": int(final["step"]),
        "cumulative_reward": float(final["cumulative_reward"]),
        "detected": int(final["detected"]),
        "completed": int(final["completed"]),
        "completed_flags": final["completed_flags"].tolist(),
        "final_track_progress": final["track_progress"].tolist(),
        "final_drone_positions": final["drone_pos"].tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_deployment_policy(path: Path, agent: ThesisBayesianQMIXAgent) -> None:
    # Mixer and optimizer are training-only. Decentralized execution uses the
    # shared feature extractor plus one independently sampled Bayesian head per agent.
    torch.save({
        "algo": "bayes_qmix_independent",
        "cfg": agent.cfg.__dict__,
        "action_names": ACTION_NAMES,
        "feature_net": agent.feature_net.state_dict(),
        "head": agent.head.state_dict(),
        "posterior_sampling": "independent",
    }, path)


def animate(frames: list[dict], grid_size: int, track_required: int,
            gif_path: Path, final_png_path: Path, fps: int,
            hide_undetected_targets: bool) -> None:
    n_agents = frames[0]["drone_pos"].shape[0]
    fig, ax = plt.subplots(figsize=(9, 8))
    background = ax.imshow(
        frames[0]["team_visited"], origin="upper", vmin=0.0, vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title(
        "Bayesian-QMIX independent: one sampled policy per agent\n"
        "Targets shown in truth-view for diagnosis"
    )

    trails, points, labels = [], [], []
    for agent_id in range(n_agents):
        trail, = ax.plot([], [], linewidth=1.5, label=f"agent {agent_id}")
        point, = ax.plot([], [], marker="o", linestyle="None", markersize=9)
        label = ax.text(0, 0, f"A{agent_id}", ha="center", va="bottom")
        trails.append(trail)
        points.append(point)
        labels.append(label)

    target_artists = {
        "hidden": ax.scatter([], [], marker="x", s=70, label="target hidden"),
        "detected": ax.scatter([], [], marker="*", s=120, label="target detected"),
        "completed": ax.scatter([], [], marker="P", s=100, label="target completed"),
    }
    target_labels = []
    status = ax.text(
        0.01, 0.99, "", transform=ax.transAxes, ha="left", va="top",
        bbox={"boxstyle": "round", "alpha": 0.8},
    )
    ax.legend(loc="lower right")

    def update(frame_index: int):
        nonlocal target_labels
        frame = frames[frame_index]
        background.set_data(frame["team_visited"])
        history = np.asarray([x["drone_pos"] for x in frames[:frame_index + 1]])
        for agent_id in range(n_agents):
            rows = history[:, agent_id, 0]
            cols = history[:, agent_id, 1]
            trails[agent_id].set_data(cols, rows)
            row, col = frame["drone_pos"][agent_id]
            points[agent_id].set_data([col], [row])
            labels[agent_id].set_position((col, row - 0.3))

        for item in target_labels:
            item.remove()
        target_labels = []
        groups = {"hidden": [], "detected": [], "completed": []}
        for target_id, (row, col) in enumerate(frame["target_pos"]):
            if frame["completed_flags"][target_id]:
                state = "completed"
            elif frame["detected_flags"][target_id]:
                state = "detected"
            else:
                state = "hidden"
            if hide_undetected_targets and state == "hidden":
                continue
            groups[state].append([col, row])
            target_labels.append(ax.text(
                col + 0.2, row + 0.2,
                f"T{target_id}/v{int(frame['target_values'][target_id])}"
                f"/k{int(frame['track_progress'][target_id])}/{track_required}",
                fontsize=8,
            ))

        for state, artist in target_artists.items():
            offsets = np.asarray(groups[state], dtype=np.float32)
            if offsets.size == 0:
                offsets = np.empty((0, 2), dtype=np.float32)
            artist.set_offsets(offsets)

        actions = ", ".join(
            f"A{i}:{name}" for i, name in enumerate(frame["action_names"])
        )
        status.set_text(
            f"step={frame['step']}\n"
            f"actions={actions}\n"
            f"reward={frame['reward']:.3f}  cum={frame['cumulative_reward']:.3f}\n"
            f"detected={frame['detected']}  completed={frame['completed']}\n"
            f"collisions={frame['collision_count']}"
        )
        return [background, *trails, *points, *labels, *target_artists.values(), status, *target_labels]

    animation = FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / max(1, fps),
        blit=False, repeat=True,
    )
    animation.save(gif_path, writer=PillowWriter(fps=max(1, fps)))
    update(len(frames) - 1)
    fig.savefig(final_png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cli = build_parser().parse_args()
    run_dir = Path(cli.run_dir)
    config = load_json(run_dir / "run_config.json")
    if str(config.get("algo")) != "bayes_qmix_independent":
        raise ValueError(
            "This script only accepts a bayes_qmix_independent run; "
            f"found {config.get('algo')!r}."
        )

    if cli.scenario_seed is not None:
        config["fixed_scenario"] = True
        config["scenario_seed"] = int(cli.scenario_seed)

    checkpoint = resolve_checkpoint(run_dir, cli.checkpoint)
    device = pick_device() if cli.device == "auto" else cli.device
    output_dir = Path(cli.output_dir) if cli.output_dir else run_dir / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    args, env, agent = fresh_agent(config, checkpoint, device)
    frames = rollout(agent, env, cli.posterior_seed, int(cli.max_steps or args.max_steps))

    trajectory_path = output_dir / "trajectory.csv"
    summary_path = output_dir / "episode_summary.json"
    gif_path = output_dir / "behavior.gif"
    final_png_path = output_dir / "final_frame.png"

    save_trajectory(trajectory_path, frames)
    save_summary(summary_path, run_dir, checkpoint, config, frames, cli.posterior_seed)
    animate(
        frames,
        grid_size=int(args.grid_size),
        track_required=int(args.track_required),
        gif_path=gif_path,
        final_png_path=final_png_path,
        fps=cli.fps,
        hide_undetected_targets=cli.hide_undetected_targets,
    )
    if cli.export_deployment_policy:
        export_deployment_policy(output_dir / "deployment_policy.pt", agent)

    print("Initial UAV positions:", frames[0]["drone_pos"].tolist())
    print("Target positions:", frames[0]["target_pos"].tolist())
    print("Target values:", frames[0]["target_values"].tolist())
    print("Final completed:", frames[-1]["completed"])
    print("Final cumulative reward:", f"{frames[-1]['cumulative_reward']:.3f}")
    for path in (gif_path, final_png_path, trajectory_path, summary_path):
        print("Saved:", path)


if __name__ == "__main__":
    main()
