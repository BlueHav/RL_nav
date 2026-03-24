#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    discover_scene_path,
    get_output_dir,
    resolve_depthnav_dataset_path,
    resolve_scene_dataset_config,
    save_color_png,
    save_depth_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless Habitat-Sim smoke test: load a scene and render one RGB/depth frame."
    )
    parser.add_argument("--scene", type=str, default=None, help="Absolute path to a scene mesh, e.g. .glb")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Dataset root. If omitted, uses the same lookup logic as depthnav.",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional scene dataset config file.",
    )
    parser.add_argument("--width", type=int, default=640, help="Rendered image width.")
    parser.add_argument("--height", type=int, default=480, help="Rendered image height.")
    parser.add_argument("--sensor-height", type=float, default=1.5, help="Sensor height in meters.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device id for Habitat-Sim.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save color/depth outputs. Defaults to learn_sim/output/01_headless_smoke",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        import habitat_sim
    except Exception as exc:
        print("habitat_sim import failed.")
        print(f"Error: {exc}")
        print("Run `python learn_sim/00_check_env.py` first and fix the environment.")
        return 1

    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root
        else resolve_depthnav_dataset_path(require_exists=False)
    )
    dataset_config = resolve_scene_dataset_config(
        dataset_config=args.dataset_config,
        dataset_root=str(dataset_root) if dataset_root else None,
    )
    scene_path = discover_scene_path(
        scene=args.scene,
        dataset_root=str(dataset_root) if dataset_root else None,
    )

    if scene_path is None:
        print("Could not resolve a scene file.")
        print("Please pass --scene /abs/path/to/scene.glb")
        return 1

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else get_output_dir("01_headless_smoke")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== learn_sim / step 01: headless smoke test ===")
    print(f"scene          : {scene_path}")
    print(f"dataset_root   : {dataset_root if dataset_root else '<none>'}")
    print(f"dataset_config : {dataset_config if dataset_config else '<none>'}")
    print(f"output_dir     : {output_dir}")

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = args.gpu_id
    sim_cfg.scene_id = str(scene_path)
    sim_cfg.enable_physics = False
    sim_cfg.create_renderer = True
    sim_cfg.random_seed = 0
    sim_cfg.requires_textures = True
    if dataset_config is not None:
        sim_cfg.scene_dataset_config_file = str(dataset_config)

    def make_camera(uuid: str, sensor_type) -> "habitat_sim.CameraSensorSpec":
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [args.height, args.width]
        spec.position = [0.0, args.sensor_height, 0.0]
        spec.orientation = [0.0, 0.0, 0.0]
        spec.near = 0.01
        spec.far = 50.0
        return spec

    agent_cfg = habitat_sim.agent.AgentConfiguration(
        sensor_specifications=[
            make_camera("color_sensor", habitat_sim.SensorType.COLOR),
            make_camera("depth_sensor", habitat_sim.SensorType.DEPTH),
        ]
    )

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    sim = None
    try:
        sim = habitat_sim.Simulator(cfg)
    except Exception as exc:
        message = str(exc)
        print("Failed to create Habitat-Sim simulator.")
        print(f"Error: {message}")
        print()
        print("Likely fixes on a server:")
        print("- Make sure habitat-sim was built or installed with headless/EGL support.")
        print("- If DISPLAY is unset, that is fine only when Habitat-Sim has headless support.")
        print("- If needed, rerun step 00 and share the full output with me.")
        return 1

    try:
        agent = sim.initialize_agent(0)
        state = habitat_sim.AgentState()

        start_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if sim.pathfinder.is_loaded:
            start_position = sim.pathfinder.get_random_navigable_point()
            print(f"pathfinder      : loaded")
        else:
            print(f"pathfinder      : not loaded, using origin")

        state.position = start_position
        agent.set_state(state)

        observations = sim.get_sensor_observations()
        color = observations["color_sensor"]
        depth = observations["depth_sensor"]

        color_path = output_dir / "color.png"
        depth_png_path = output_dir / "depth_preview.png"
        depth_npy_path = output_dir / "depth.npy"
        meta_path = output_dir / "meta.json"

        save_color_png(color, color_path)
        save_depth_outputs(depth, depth_png_path, depth_npy_path)

        meta = {
            "scene": str(scene_path),
            "dataset_root": str(dataset_root) if dataset_root else None,
            "dataset_config": str(dataset_config) if dataset_config else None,
            "color_shape": list(color.shape),
            "depth_shape": list(depth.shape),
            "agent_start_position": np.asarray(start_position).tolist(),
            "pathfinder_loaded": bool(sim.pathfinder.is_loaded),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print()
        print("Saved outputs:")
        print(f"- {color_path}")
        print(f"- {depth_png_path}")
        print(f"- {depth_npy_path}")
        print(f"- {meta_path}")
        print()
        print("Step 01 finished successfully.")
        print("Please send me the terminal output and, if helpful, the generated meta.json content.")
        return 0
    finally:
        if sim is not None:
            sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
