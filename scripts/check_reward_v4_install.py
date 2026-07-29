from __future__ import annotations

import ast
from pathlib import Path

from uav_search_belief20.envs.multi_drone_env import MultiDroneEnvConfig
from uav_search_belief20.envs.primitive_search_env import EnvConfig
from uav_search_belief20.rewards.potential_reward import PotentialRewardConfig


def check_config(name: str, cfg) -> None:
    expected = {
        "reward_version": "v4_potential_simple",
        "step_penalty": -0.01,
        "boundary_penalty": -0.10,
        "new_cell_bonus": 0.0,
        "revisit_penalty": 0.0,
        "new_observed_cell_bonus": 0.0,
        "detect_value1_bonus": 0.0,
        "detect_value2_bonus": 0.0,
        "track_progress_value1_bonus": 0.0,
        "track_progress_value2_bonus": 0.0,
        "complete_value1_bonus": 5.0,
        "complete_value2_bonus": 10.0,
        "all_targets_bonus": 5.0,
        "reward_gamma": 0.99,
        "coverage_weight": 10.0,
        "detection_weight": 1.0,
        "progress_weight": 2.0,
    }
    for key, value in expected.items():
        actual = getattr(cfg, key)
        if actual != value:
            raise AssertionError(f"{name}.{key}: expected {value!r}, got {actual!r}")


def metric_keys(node: ast.AST) -> set[str]:
    keys: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript):
            continue
        if not isinstance(child.value, ast.Name) or child.value.id != "metrics":
            continue
        index = child.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, str):
            keys.add(index.value)
    return keys


def check_checkpoint_scores(root: Path) -> list[str]:
    checked: list[str] = []
    for path in sorted((root / "scripts").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "checkpoint_score":
                keys = metric_keys(node)
                if keys != {"reward_mean"}:
                    raise AssertionError(
                        f"{path.relative_to(root)} checkpoint_score uses {sorted(keys)}, "
                        "expected only reward_mean"
                    )
                checked.append(str(path.relative_to(root)))
    return checked


def main() -> None:
    check_config("EnvConfig", EnvConfig())
    multi = MultiDroneEnvConfig()
    check_config("MultiDroneEnvConfig", multi)
    if multi.collision_penalty != -0.05:
        raise AssertionError(f"collision_penalty expected -0.05, got {multi.collision_penalty}")
    PotentialRewardConfig().validate()

    root = Path(__file__).resolve().parents[1]
    checkpoint_files = check_checkpoint_scores(root)
    print("Reward v4 import/config check: OK")
    print("Checkpoint score uses reward_mean only in:")
    for path in checkpoint_files or ["(no checkpoint_score function found)"]:
        print(f"  - {path}")
    manifest = root / ".reward_v4_install.json"
    print(f"Install manifest: {manifest if manifest.exists() else 'not found'}")


if __name__ == "__main__":
    main()
