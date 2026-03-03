# Official Codebase for EfficientZero-Multitask (EZ-M)

EZ-M is a multi-task extension of EfficientZero series focused on humanoid locomotion tasks. 
We also reserved support for Atari (single-task) and DMControl single-task training. 
We recommend multi-task training in the future RL due to superior overall sample efficiency. More details could refer to:

[Project page](https://liushaohuai5.github.io/_projects/ez_m/),
[Paper link](https://arxiv.org/pdf/2603.01452)

The training pipeline is built around Hydra configuration, Ray-based workers, PyTorch models, and a custom C++/Cython Gumbel search backend. The main entry points are:

- `ez/train.py` for training
- `ez/eval.py` for evaluation

## Features

- MuZero-style training and evaluation workflow
- Support for both discrete and continuous action spaces
- Environment presets under `ez/config/exp/`
- Distributed execution
- Optional experiment tracking with Weights & Biases
- Support multi-task training with higher overall sample efficiency

## Requirements

- Python 3.8 or later
- CUDA-enabled GPU environment recommended for training
- Dependencies from `requirements.txt` or `requirements_py310.txt`

Install dependencies with:

```bash
pip install -r requirements.txt
```

If you are using Python 3.10, you can use:

```bash
pip install -r requirements_py310.txt
```

## Build the MCTS Extension

Before training or evaluation, compile the C++/Cython MCTS module:

```bash
cd ez/mcts/ctree_v2
bash make.sh
cd -
```

## Results
Reported scores could be found in results/ez-m-results.json

## Training

Run training with one of the experiment configs in `ez/config/exp/`:

```bash
python ez/train.py exp_config=ez/config/exp/dmc_state.yaml
```

You can also use the provided shell script:

```bash
bash scripts/train.sh
```

Example experiment configs:

- `ez/config/exp/atari.yaml`
- `ez/config/exp/dmc_image.yaml`
- `ez/config/exp/dmc_state.yaml`
- `ez/config/exp/maniskill_state.yaml`
- `ez/config/exp/humanoid_bench_state.yaml`

## Evaluation

Run evaluation with:

```bash
python ez/eval.py exp_config=ez/config/exp/dmc_image.yaml
```

Or use the script:

```bash
bash scripts/eval.sh
```

## Project Structure

```text
ez/
  agents/      Agent definitions and model implementations
  config/      Global and experiment-specific Hydra configs
  data/        Replay buffer, trajectory, and data processing utilities
  envs/        Environment wrappers and integrations
  mcts/        Python and C++/Cython MCTS implementations
  utils/       Training utilities and helper functions
  worker/      Distributed worker logic for training and evaluation
scripts/       Example launch scripts
```

## Notes

- Some training scripts include environment-specific settings such as `MUJOCO_GL`, `CUDA_VISIBLE_DEVICES`, and `wandb login`. Adjust them before running on your machine.
- The repository contains multiple experiment presets; choose the one that matches your target environment and observation type.

## Cite
```aiignore
@article{liu2026ezm,
  title={Scaling Tasks, Not Samples: Mastering Humanoid Control through Multi-Task Model-Based Reinforcement Learning},
  author={Liu, Shaohuai and Ye, Weirui and Du, Yilun and Xie, Le},
  journal={arXiv preprint},
  year={2026}
}
```