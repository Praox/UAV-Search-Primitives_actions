from __future__ import annotations

from dataclasses import dataclass

from uav_search_belief20.envs.primitive_search_env import EnvConfig


@dataclass(frozen=True)
class SingleAblationSpec:
    name: str
    description: str
    use_boundary_action_mask: bool
    include_track_progress_map: bool
    track_progress_reward_scale: float = 1.0


ABLATIONS: dict[str, SingleAblationSpec] = {
    "v3": SingleAblationSpec(
        name="v3",
        description="Historical v3: no boundary mask, no tracking-progress map.",
        use_boundary_action_mask=False,
        include_track_progress_map=False,
        track_progress_reward_scale=1.0,
    ),
    "A": SingleAblationSpec(
        name="A",
        description="Boundary action mask only.",
        use_boundary_action_mask=True,
        include_track_progress_map=False,
        track_progress_reward_scale=1.0,
    ),
    "B": SingleAblationSpec(
        name="B",
        description="Tracking-progress map only.",
        use_boundary_action_mask=False,
        include_track_progress_map=True,
        track_progress_reward_scale=1.0,
    ),
    "C": SingleAblationSpec(
        name="C",
        description="Boundary mask and tracking-progress map.",
        use_boundary_action_mask=True,
        include_track_progress_map=True,
        track_progress_reward_scale=1.0,
    ),
    "D": SingleAblationSpec(
        name="D",
        description=(
            "Boundary mask + tracking-progress map + 1.5x tracking-progress reward."
        ),
        use_boundary_action_mask=True,
        include_track_progress_map=True,
        track_progress_reward_scale=1.5,
    ),
}


def get_ablation(name: str) -> SingleAblationSpec:
    normalized = str(name).strip()
    if normalized not in ABLATIONS:
        valid = ", ".join(ABLATIONS)
        raise ValueError(f"Unknown single-UAV ablation {name!r}; expected one of {valid}.")
    return ABLATIONS[normalized]


def build_env_config(
    args,
    *,
    seed: int,
) -> EnvConfig:
    """Create an EnvConfig from argparse-like attributes and an ablation preset."""

    spec = get_ablation(args.ablation)
    scale_override = getattr(args, "track_progress_scale", None)
    scale = (
        spec.track_progress_reward_scale
        if scale_override is None
        else float(scale_override)
    )
    return EnvConfig(
        grid_size=int(args.grid_size),
        n_value1_targets=int(args.n_value1_targets),
        n_value2_targets=int(args.n_value2_targets),
        sensor_radius=int(args.sensor_radius),
        detection_probability=float(args.detection_probability),
        track_radius=int(args.track_radius),
        track_required=int(args.track_required),
        max_steps=int(args.max_steps),
        seed=int(seed),
        reward_version=str(args.reward_version),
        ablation_name=spec.name,
        use_boundary_action_mask=spec.use_boundary_action_mask,
        include_track_progress_map=spec.include_track_progress_map,
        track_progress_value1_bonus=0.20 * scale,
        track_progress_value2_bonus=0.60 * scale,
    )


def describe_ablation(name: str) -> str:
    return get_ablation(name).description
