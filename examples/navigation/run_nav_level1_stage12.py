from depthnav.scripts.runner import run_experiment

if __name__ == "__main__":
    config_keys = (
        "env.scene_kwargs.load_geodesics",
        "env.scene_kwargs.path",
        "train_bptt.iterations",
    )

    # Stage12 policy requires geodesic caches for every curriculum stage.
    run_params = {
        "level0_stage12": (True, "configs/box_2", 500),
        "level1_stage12": (True, "configs/level_1", 20000),
    }
    base_config_files = [
        "examples/navigation/train_cfg/nav_empty_stage12.yaml",
        "examples/navigation/train_cfg/nav_levelX_stage12.yaml",
    ]
    run_experiment(
        script="depthnav/scripts/train_bptt.py",
        experiment_dir="examples/navigation/logs/level1_stage12",
        config_keys=config_keys,
        run_params=run_params,
        base_config_files=base_config_files,
        policy_config_file="examples/navigation/policy_cfg/stage12_yaw_geodesic.yaml",
        eval_configs=[
            "examples/navigation/eval_cfg/nav_level1_stage12.yaml",
        ],
        eval_csvs=[
            "examples/navigation/logs/level1_stage12/nav_level_1_stage12.csv",
        ],
        curriculum=True,
        max_retries=5,
    )
