#!/bin/bash
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --chdir=/scratch/anderarrigandiaga/StyleTTS2_basque
#SBATCH --job-name=styletts2_multispeaker_wavlm_short_second_stage
#SBATCH --gres=gpu:1                 # single GPU for second stage
#SBATCH --cpus-per-gpu=2             # 2 CPUs per GPU
#SBATCH --mem-per-gpu=32GB           # 32GB memory for safety
#SBATCH --time=10-00:00:00           # 10 days
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ander.arriandiaga@ehu.eus
#SBATCH --output=logs/styletts2_multispeaker_wavlm_short_second_stage_%j.out
#SBATCH --error=logs/styletts2_multispeaker_wavlm_short_second_stage_%j.err

export NO_COLOR=1

cd /scratch/anderarrigandiaga/StyleTTS2_basque

# Activate conda env (robust)
source "$HOME/.bashrc" || true
conda activate styletts2

# Diagnostics
echo "DEBUG: which python: $(which python || true)"
echo "DEBUG: python -V: $(python -V 2>&1 || true)"
echo "DEBUG: CONDA_PREFIX=${CONDA_PREFIX:-unset}"

# Avoid tokenizer multiprocessing warnings
export TOKENIZERS_PARALLELISM=false

# Use the second-stage config (adjust if you use a different config)
CONFIG_FILE="Configs/config_basque_multispeaker_phoneme_wavlm_800_2nd.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "========================================================="
echo "🚀 Running StyleTTS2 SECOND STAGE (single-GPU)" 
echo "  Config: $CONFIG_FILE"
echo "  Working dir: $(pwd)"
echo "========================================================="

# Prefer environment python if available
PYTHON_CMD="python3"
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_CMD="${CONDA_PREFIX}/bin/python"
fi

# Run second stage pinned to GPU 0
CUDA_VISIBLE_DEVICES=0 "$PYTHON_CMD" train_second_clean_wandb.py --config_path "$CONFIG_FILE"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Second stage finished successfully"
else
    echo "❌ Second stage failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
