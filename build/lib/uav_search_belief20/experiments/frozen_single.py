from __future__ import annotations

from dataclasses import dataclass

from uav_search_belief20.envs.primitive_search_env import EnvConfig


@dataclass(frozen=True)
class FrozenSingleUAVSpec:
    name: str = "single_v4_D"
    source_ablation: str = "D"
    use_boundary_action_mask: bool = True
    include_track_progress_map: bool = True
    track_progress_value1_bonus: float = 0.30
    track_progress_value2_bonus: float = 0.90


FROZEN_SINGLE = FrozenSingleUAVSpec()


def build_frozen_single_env_config(
    *,
    seed: int,
    grid_size: int = 20,
    n_value1_targets: int = 3,
    n_value2_targets: int = 1,
    sensor_radius: int = 2,
    detection_probability: float = 1.0,
    track_radius: int = 1,
    track_required: int = 3,
    max_steps: int = 150,
) -> EnvConfig:
    """Build the exact single-UAV formulation used as the multi-UAV source."""
    return EnvConfig(
        grid_size=grid_size,
        n_value1_targets=n_value1_targets,
        n_value2_targets=n_value2_targets,
        sensor_radius=sensor_radius,
        detection_probability=detection_probability,
        track_radius=track_radius,
        track_required=track_required,
        max_steps=max_steps,
        seed=int(seed),
        reward_version=FROZEN_SINGLE.name,
        ablation_name=FROZEN_SINGLE.source_ablation,
        use_boundary_action_mask=FROZEN_SINGLE.use_boundary_action_mask,
        include_track_progress_map=FROZEN_SINGLE.include_track_progress_map,
        track_progress_value1_bonus=FROZEN_SINGLE.track_progress_value1_bonus,
        track_progress_value2_bonus=FROZEN_SINGLE.track_progress_value2_bonus,
    )
