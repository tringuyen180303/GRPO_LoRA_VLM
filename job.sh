#!/bin/bash
#SBATCH -A r00043
#SBATCH --job-name=grpo_lora
#SBATCH --nodes=1
#SBATCH -p gpu --gpus 1
#SBATCH -t 48:00:00
#SBATCH --mem=128G
cd /N/slate/trihnguy/GRPO_LoRA_VLM

mkdir -p logs
mkdir -p /N/slate/trihnguy/hf/hub
mkdir -p /N/slate/trihnguy/hf/datasets
mkdir -p /N/slate/trihnguy/cache
mkdir -p /N/slate/trihnguy/tmp

module load python/gpu/3.10.10

export PYTHONNOUSERSITE=1
export PYTHONPATH=/N/slate/trihnguy/GRPO_LoRA_VLM/.pkgs
export HF_HOME=/N/slate/trihnguy/hf
export HF_HUB_CACHE=/N/slate/trihnguy/hf/hub
export HF_DATASETS_CACHE=/N/slate/trihnguy/hf/datasets
export XDG_CACHE_HOME=/N/slate/trihnguy/cache
export TMPDIR=/N/slate/trihnguy/tmp
export HF_HUB_DISABLE_XET=1

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

srun python train_grpo_raw.py
