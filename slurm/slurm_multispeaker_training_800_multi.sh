#!/bin/bash
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --chdir=/scratch/anderarrigandiaga/StyleTTS2_basque
#SBATCH --job-name=styletts2_multispeaker_wavlm_short
#SBATCH --gres=gpu:2                 # 2 GPUs
#SBATCH --cpus-per-gpu=2             # 2 CPUs per GPU
#SBATCH --mem-per-gpu=16GB           # 16GB memory per GPU
#SBATCH --time=25-00:00:00            # 25 days for 4M steps
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ander.arriandiaga@ehu.eus
#SBATCH --output=logs/styletts2_multispeaker_wavlm_short%j.out
#SBATCH --error=logs/styletts2_multispeaker_wavlm_short_%j.err

export NO_COLOR=1

# Performance and CUDA environment tweaks for multi-GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker

cd /scratch/anderarrigandiaga/StyleTTS2_basque

# Activate conda env
source ~/.bashrc
conda activate styletts2

# Avoid tokenizer multiprocessing warnings
export TOKENIZERS_PARALLELISM=false

# Multispeaker configuration
CONFIG_FILE="Configs/config_basque_multispeaker_phoneme_wavlm_800.yml"
EXPERIMENT_NAME="multispeaker_phoneme_wavlm_800$(date +%Y%m%d)"
WANDB_PROJECT="StyleTTS2-Basque"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "========================================================="
echo "🚀 StyleTTS2 MULTISPEAKER TRAINING (MULTI-GPU) - WavLM + Phoneme178"
echo "========================================================="
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  GPUs requested: $SLURM_GPUS_ON_NODE"
echo "  Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "  Working directory: $(pwd)"
echo "  Date: $(date)"
echo "  Config file: $CONFIG_FILE ✅"
echo ""
echo "🎯 FEATURES:"
echo "  • Text Model: WavLM + AlBERT phoneme-178 pipeline"
echo "  • Audio Length: max_len=800 (10 seconds)"
echo "  • Schedule: Standard 50+30 epochs"
echo "  • Multi-GPU: torchrun / Distributed Data Parallel (via accelerate)"
echo "  • Memory Optimized: batch_size=2 (per-process), grad_acc=8"
echo "  • Decoder: HiFi-GAN"
echo "  • Multispeaker: true"
echo ""
echo "========================================================="

NGPUS=${SLURM_GPUS_ON_NODE:-2}
echo "Launching with NGPUS=$NGPUS"

echo "🖥️ GPU Status:";
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
echo ""

echo "🎯 Starting Complete Multispeaker Training (First + Second Stage) on $NGPUS GPUs..."
echo "========================================================"

# Use torchrun so the in-script Accelerator detects distributed mode
TORCHRUN_CMD="torchrun --nproc_per_node=$NGPUS --nnodes=1"

echo "Running first stage with: $TORCHRUN_CMD train_first_clean_wandb.py --config_path \"$CONFIG_FILE\""
${TORCHRUN_CMD} train_first_clean_wandb.py --config_path "$CONFIG_FILE"
FIRST_STAGE_EXIT_CODE=$?

if [ $FIRST_STAGE_EXIT_CODE -eq 0 ]; then
    echo "✅ First stage completed successfully!"
    echo ""
    echo "🥈 Starting Second Stage Training (single-process on GPU 0)..."
    # `train_second_clean_wandb.py` is not compatible with multi-process DDP.
    # Run it as a single process pinned to GPU 0 so it runs correctly on a multi-GPU node.
    CUDA_VISIBLE_DEVICES=0 python3 train_second_clean_wandb.py --config_path "$CONFIG_FILE"
    SECOND_STAGE_EXIT_CODE=$?
    if [ $SECOND_STAGE_EXIT_CODE -eq 0 ]; then
        echo "✅ Multispeaker complete training finished successfully!"
        FINAL_EXIT_CODE=0
    else
        echo "❌ Second stage training failed with exit code: $SECOND_STAGE_EXIT_CODE"
        FINAL_EXIT_CODE=$SECOND_STAGE_EXIT_CODE
    fi
else
    echo "❌ First stage training failed with exit code: $FIRST_STAGE_EXIT_CODE"
    FINAL_EXIT_CODE=$FIRST_STAGE_EXIT_CODE
fi

echo "========================================================"
echo "End time: $(date)"
echo "========================================================"

exit $FINAL_EXIT_CODE
