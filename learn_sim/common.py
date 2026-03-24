from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
LEARN_SIM_ROOT = REPO_ROOT / "learn_sim"
OUTPUT_ROOT = LEARN_SIM_ROOT / "output"


def get_repo_root() -> Path:
    return REPO_ROOT


def get_output_dir(name: str) -> Path:
    path = OUTPUT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_depthnav_dataset_path(require_exists: bool = True) -> Optional[Path]:
    try:
        from depthnav.utils.paths import get_depthnav_dataset_path

        path = Path(get_depthnav_dataset_path(require_exists=require_exists))
    except Exception:
        env_path = os.environ.get("DEPTHNAV_DATASET_PATH")
        if env_path:
            path = Path(env_path).expanduser().resolve()
        else:
            env_root = os.environ.get("DEPTHNAV_DATASETS_ROOT")
            if env_root:
                path = Path(env_root).expanduser().resolve() / "depthnav_dataset"
            else:
                path = REPO_ROOT / "datasets" / "depthnav_dataset"

    if require_exists and not path.exists():
        return None
    return path


def resolve_scene_dataset_config(
    dataset_config: Optional[str] = None,
    dataset_root: Optional[str] = None,
) -> Optional[Path]:
    if dataset_config:
        path = Path(dataset_config).expanduser().resolve()
        return path if path.exists() else None

    root = (
        Path(dataset_root).expanduser().resolve()
        if dataset_root
        else resolve_depthnav_dataset_path(require_exists=False)
    )
    if root is None or not root.exists():
        return None

    matches = sorted(root.glob("*.scene_dataset_config.json"))
    if matches:
        return matches[0]

    fallback = root / "depthnav_dataset.scene_dataset_config.json"
    if fallback.exists():
        return fallback
    return None


def discover_scene_path(
    scene: Optional[str] = None,
    dataset_root: Optional[str] = None,
) -> Optional[Path]:
    if scene:
        path = Path(scene).expanduser().resolve()
        return path if path.exists() else None

    root = (
        Path(dataset_root).expanduser().resolve()
        if dataset_root
        else resolve_depthnav_dataset_path(require_exists=False)
    )
    if root is None or not root.exists():
        return None

    candidates = []
    for pattern in ("*.basis.glb", "*.glb", "*.ply"):
        candidates.extend(root.rglob(pattern))

    filtered = []
    for path in sorted(candidates):
        parts = set(path.parts)
        name_lower = path.name.lower()
        if "configs" in parts or "agents" in parts:
            continue
        if "agent" in name_lower:
            continue
        if ".semantic." in name_lower:
            continue
        filtered.append(path)

    return filtered[0] if filtered else None


def save_color_png(rgba: np.ndarray, path: Path) -> None:
    rgb = rgba[..., :3]
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def save_depth_outputs(depth: np.ndarray, png_path: Path, npy_path: Path) -> None:
    np.save(npy_path, depth)

    finite = np.isfinite(depth)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(finite):
        values = depth[finite]
        lo = float(np.min(values))
        hi = float(np.percentile(values, 95.0))
        if hi <= lo:
            hi = float(np.max(values))
        if hi > lo:
            scaled = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
            preview = (scaled * 255.0).astype(np.uint8)

    Image.fromarray(preview).save(png_path)
