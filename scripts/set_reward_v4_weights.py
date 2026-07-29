from __future__ import annotations

import argparse
import re
from pathlib import Path


def replace_field(text: str, field: str, value: float) -> str:
    pattern = re.compile(
        rf"^(\s*{re.escape(field)}\s*:\s*[^=\n]+?=\s*)([^#\n]+)(\s*(?:#.*)?)$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {field!r} field, found {len(matches)}")
    match = matches[0]
    return text[: match.start()] + match.group(1) + repr(float(value)) + match.group(3) + text[match.end() :]


def update(path: Path, values: dict[str, float]) -> None:
    text = path.read_text(encoding="utf-8")
    for field, value in values.items():
        text = replace_field(text, field, value)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set the same reward-v4 weights in single and multi envs.")
    parser.add_argument("--coverage", type=float, required=True)
    parser.add_argument("--detection", type=float, default=1.0)
    parser.add_argument("--progress", type=float, default=2.0)
    parser.add_argument("--completion-scale", type=float, default=5.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    values = {
        "coverage_weight": args.coverage,
        "detection_weight": args.detection,
        "progress_weight": args.progress,
        "complete_value1_bonus": args.completion_scale,
        "complete_value2_bonus": 2.0 * args.completion_scale,
    }
    paths = [
        root / "src/uav_search_belief20/envs/primitive_search_env.py",
        root / "src/uav_search_belief20/envs/multi_drone_env.py",
    ]
    for path in paths:
        update(path, values)
        print(f"Updated {path.relative_to(root)}")
    print("Active reward:")
    print(
        f"  Phi = {args.coverage:g}*C + {args.detection:g}*D + "
        f"{args.progress:g}*P"
    )
    print(
        f"  completion value-1/value-2 = "
        f"{args.completion_scale:g}/{2.0 * args.completion_scale:g}"
    )


if __name__ == "__main__":
    main()
