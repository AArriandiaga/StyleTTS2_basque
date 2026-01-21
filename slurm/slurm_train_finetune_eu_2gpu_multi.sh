#!/bin/bash
#SBATCH --job-name=styletts2_ft_eu_2gpu_multi
#SBATCH --partition=gpu-H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=20-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ander.arriandiaga@ehu.eus
#SBATCH --output=logs/styletts2_ft_eu_2gpu_multi.%j.out
#SBATCH --error=logs/styletts2_ft_eu_2gpu_multi.%j.err

# Prevent color codes in logs
export NO_COLOR=1

# Performance / CUDA / NCCL tuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker

# reduce CPU thread oversubscription
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# dataloader workers
export DATALOADER_NUM_WORKERS=0
export PYTHONFAULTHANDLER=1

# go to repo
cd /scratch/ander.arriandiaga/StyleTTS2_hyperion

# activate environment
source ~/.bashrc
conda activate styletts2   # or your venv activation

# ensure logs dir exists
mkdir -p logs

# optional: show which GPUs were allocated by Slurm (useful for debugging)
echo "SLURM assigned devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits


# Run training and pass accumulation steps to the script
python train_finetune_eu.py -p Configs/config_ft_eu_multi_2gpu.yml