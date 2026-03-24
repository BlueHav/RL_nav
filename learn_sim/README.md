# learn_sim

这个目录专门用来一步一步学习和运行 `habitat-sim` 的服务器端 demo。

约定：

- 后续新的学习脚本都放在这个目录里。
- 默认按“服务器 / headless / 不弹窗”的方式写，不依赖本地图形界面。
- 每一步都尽量做到“能单独运行、能单独定位问题”。

当前学习路线：

1. `00_check_env.py`
   作用：检查当前 Python、CUDA、`habitat_sim`、数据集路径是否准备好。
2. `01_headless_smoke.py`
   作用：无窗口加载一个场景，渲染一帧 `color` 和 `depth`，保存到磁盘。

建议顺序：

```bash
python learn_sim/00_check_env.py
```

如果上一步基本通过，再运行：

```bash
python learn_sim/01_headless_smoke.py --scene /abs/path/to/scene.glb
```

如果你的场景来自 scene dataset，可以额外传：

```bash
python learn_sim/01_headless_smoke.py \
  --scene /abs/path/to/scene.glb \
  --dataset-config /abs/path/to/xxx.scene_dataset_config.json
```

脚本输出默认会保存到：

```text
learn_sim/output/
```

你每跑完一步，把终端输出贴给我就行。我会根据结果继续带你走下一步，或者直接帮你定位问题。
