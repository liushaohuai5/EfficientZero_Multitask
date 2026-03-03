#!/bin/bash
set -ex
ulimit -n 65535
ulimit -Sn 65535
#export OMP_NUM_THREADS=1
#export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=2,3
#export CUDA_VISIBLE_DEVICES=0,3
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_IGNORE_DISABLED_P2P=1
export HYDRA_FULL_ERROR=1
export MASTER_PORT='12399'
export MUJOCO_GL=egl
#export MUJOCO_EGL_DEVICE_ID=0
#export MUJOCO_GL=glfw
#export MUJOCO_GL=osmesa
#export WANDB_BASE_URL=https://api.bandw.top

cd ./ez/mcts/ctree_v2
bash make.sh
cd -

#python ez/train.py exp_config=ez/config/exp/atari.yaml #> profile.txt
#python ez/train.py exp_config=ez/config/exp/dmc_image.yaml #> profile.txt
#python ez/train.py exp_config=ez/config/exp/dmc_image_human.yaml #> profile.txt
python ez/train.py exp_config=ez/config/exp/dmc_state.yaml #> profile.txt
#python ez/train.py exp_config=ez/config/exp/humanoid_bench_state.yaml #> profile.txt
#python ez/train.py exp_config=ez/config/exp/humanoidbench_state_single.yaml #> profile.txt
#python ez/train.py exp_config=ez/config/exp/maniskill_state.yaml