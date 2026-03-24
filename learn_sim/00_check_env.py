#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from common import discover_scene_path, resolve_depthnav_dataset_path, resolve_scene_dataset_config


def print_line(label: str, value: object) -> None:
    print(f"{label:24}: {value}")


def main() -> int:
    print("=== learn_sim / step 00: environment check ===")
    print("This script is safe for servers. It does not open a window.")
    print()

    failures = []

    print_line("python_executable", sys.executable)
    print_line("python_version", sys.version.split()[0])
    print_line("platform", platform.platform())
    print_line("repo_root", Path(__file__).resolve().parents[1])
    print_line("DISPLAY", os.environ.get("DISPLAY", "<unset>"))
    print_line("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    print_line("DEPTHNAV_DATASET_PATH", os.environ.get("DEPTHNAV_DATASET_PATH", "<unset>"))
    print_line("DEPTHNAV_DATASETS_ROOT", os.environ.get("DEPTHNAV_DATASETS_ROOT", "<unset>"))
    print()

    try:
        import torch

        print_line("torch_version", torch.__version__)
        print_line("torch_cuda_available", torch.cuda.is_available())
        print_line("torch_cuda_device_count", torch.cuda.device_count())
        if torch.cuda.is_available():
            print_line("torch_cuda_device_0", torch.cuda.get_device_name(0))
    except Exception as exc:
        failures.append("torch import failed")
        print_line("torch_status", f"FAILED: {exc}")
    print()

    try:
        import habitat_sim

        print_line("habitat_sim_status", "OK")
        print_line("habitat_sim_module", habitat_sim.__file__)
    except Exception as exc:
        failures.append("habitat_sim import failed")
        print_line("habitat_sim_status", f"FAILED: {exc}")
    print()

    dataset_root = resolve_depthnav_dataset_path(require_exists=False)
    print_line("resolved_dataset_root", dataset_root if dataset_root else "<none>")
    print_line("dataset_root_exists", bool(dataset_root and dataset_root.exists()))

    dataset_config = resolve_scene_dataset_config(dataset_root=str(dataset_root) if dataset_root else None)
    scene_path = discover_scene_path(dataset_root=str(dataset_root) if dataset_root else None)

    print_line("scene_dataset_config", dataset_config if dataset_config else "<not found>")
    print_line("auto_discovered_scene", scene_path if scene_path else "<not found>")
    print()

    if dataset_root is None or not dataset_root.exists():
        failures.append("dataset root not found")

    print("=== summary ===")
    if failures:
        print("Status: NOT READY")
        for item in failures:
            print(f"- {item}")
        print()
        print("Next suggestion:")
        print("- If habitat_sim is missing, install it in the server Python environment first.")
        print("- For this repo, there is a source-build example in README.md.")
        print("- On a server, prefer a headless/EGL-capable habitat-sim build.")
        print("- If dataset paths are missing, set DEPTHNAV_DATASET_PATH or DEPTHNAV_DATASETS_ROOT.")
        return 1

    print("Status: READY FOR STEP 01")
    print()
    print("Next command:")
    print("python learn_sim/01_headless_smoke.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
