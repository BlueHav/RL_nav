# DepthNav

修改和优化了代码、模型内容

https://github.com/user-attachments/assets/1e379ef5-6bd3-4e3e-9459-0f89c0350a19



1.  **Clone the repository:**
    ```bash
    git clone git@github.com:rislab/depthnav.git --recursive
    cd depthnav/
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python3.9 -m venv .venv # install python3.9 if you do not have it on the machine
    source .venv/bin/activate
    pip install --upgrade pip
    ```

3.  **Install system-wide dependencies:**
    ```bash
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
    sudo apt-get install -y libcgal-dev
    ```

4.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Verify PyTorch installation:**
    Ensure that PyTorch is installed with CUDA support.
    ```bash
    python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list()); print(torch.randn(1).cuda())"
    ```
    You should see an output similar to:
    ```
    2.2.1+cu121
    ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
    tensor([0.6627], device='cuda:0')
    ```

6.  **Build and install `habitat-sim`:**
    ```bash
    cd habitat-sim
    python setup.py install --with-cuda --build-type Release --cmake-args="-DPYTHON_EXECUTABLE=$(which python) -DCMAKE_CXX_FLAGS_RELEASE='-Ofast -march=native'"
    cd ..
    ```
7.  Install the `depthnav` module
    ```bash
    pip install -e .
    ```

## Dataset Path
By default, the code now looks for datasets in this order:

1. `DEPTHNAV_DATASET_PATH`
2. `DEPTHNAV_DATASETS_ROOT/depthnav_dataset`
3. `/root/gpufree-data/datasets/depthnav_dataset`
4. `./datasets/depthnav_dataset` inside the repository

On a server where datasets were moved to `/root/gpufree-data/datasets`, you can optionally make the path explicit:

```bash
export DEPTHNAV_DATASETS_ROOT=/root/gpufree-data/datasets
```

## Usage
Download dataset:
```bash
cd /root/gpufree-data/datasets
./get_dataset.sh
```

### Training

We train the policy using `run_nav_level1.py`. The policy is trained in two
"levels". In the first level, "level0", the agent is trained for 500 iterations
in an empty environment with no collision loss. This helps the policy learn to
fly and navigate to the target without any obstacle avoidance. Next the policy
is trained in "level1" for 20K iterations with random obstacles and all loss
terms enabled. The training set includes 50 environment instances of randomly
generated cuboid obstacles.

```bash
python examples/navigation/run_nav_level1.py
```

You can follow the training progress by running tensorboard in another terminal.
Then opening `https://localhost:6006` in a web browser. Model checkpoints are 
saved in `examples/navigation/logs` every 500 iterations.

```bash
source .venv/bin/activate
tensorboard --logdir examples/navigation/logs/
```

Here is an example of what the training success rate should look like:

![tensorboard_success_rate](docs/tensorboard_success_rate.png)

### Evaluation

Rollouts of the trained policy can be visualized with the `eval_visual.py`
script.  The script runs a batch of `--num_envs` agents for `--num_rollouts`
with the policy specified by the `--weight` path. A video of all the rollouts is
saved to the `--save_name` path (default is the `--weight` parent directory).

The evaluation environments are 10 random configurations of cuboid obstacles
that have been held out from the training set.

```bash
python examples/navigation/eval_visual.py \
    --weight "examples/navigation/logs/level1/level1_1.pth" \
    --render \
    --num_envs 4 \
    --num_rollouts 10
```

You should see an output like this:


https://github.com/user-attachments/assets/fd2fac5c-4fbe-4c32-b600-7a0c2df15aac



Success rate and other evaluation statistics can be obtained by running the 
`eval_logger.py` script. It will run a batch of `--num_envs` agents for 
`--num_rollouts` and append a summary of statistics to a csv file.

```bash
python depthnav/scripts/eval_logger.py \
    --weight examples/navigation/logs/level1/level1_1.pth \
    --num_envs 4 \
    --num_rollouts 10
```

## License
MIT

## Citation
If you use [this work](https://arxiv.org/abs/2509.08177) in your research, kindly consider citing us:
```bibtex
@inproceedings{lee_quadrotor_2026,
  title = {{{Quadrotor Navigation}} using {{Reinforcement Learning}} with {{Privileged Information}}},
  author = {Lee, Jonathan and Rathod, Abhishek and Goel, Kshitij and Stecklein, John and Tabib, Wennie},
  year = {2025},
  month= {Sept},
  url={https://arxiv.org/abs/2509.08177},
  doi={10.48550/arXiv.2509.08177}
} 
```

