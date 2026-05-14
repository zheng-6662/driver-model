# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    if value == "":
        return ""
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _fallback_yaml_load(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if ":" not in raw_line:
            continue
        key, value = raw_line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data or {}
    except Exception:
        return _fallback_yaml_load(path)


@dataclass
class Paths:
    project_root: Path
    rebuild_root: Path
    output_dir: Path
    raw_vehicle_glob: str
    scene_triggers: Path | None
    old_anchors: Path | None
    v05_candidates: Path | None
    module_segments: Path | None


def resolve_paths(config: dict[str, Any]) -> Paths:
    project_root = Path(str(config.get("project_root", "F:/data_set_process/data_process")))
    rebuild_root = Path(str(config.get("rebuild_root", project_root / "05_rebuild_from_raw_20260511")))
    output_dir = Path(str(config.get("output_dir", "outputs/event_episodes_v0_6")))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    context = config.get("context_tables", {}) or {}

    def optional_path(key: str) -> Path | None:
        value = context.get(key)
        if not value:
            return None
        path = Path(str(value))
        if not path.is_absolute():
            path = project_root / path
        return path

    return Paths(
        project_root=project_root,
        rebuild_root=rebuild_root,
        output_dir=output_dir,
        raw_vehicle_glob=str(config.get("raw_vehicle_glob", "01_datasets/数据预处理/原始车辆数据/**/*.csv")),
        scene_triggers=optional_path("scene_triggers"),
        old_anchors=optional_path("old_anchors"),
        v05_candidates=optional_path("v05_candidates"),
        module_segments=optional_path("module_segments"),
    )

