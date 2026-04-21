#!/bin/bash
#SBATCH --job-name=dima-gen-lmhead
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=GPU-shared
#SBATCH --qos=gpu_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --gpus=v100-32:1
#SBATCH --time=01:00:00
#SBATCH -A cis260039p

set -euo pipefail

echo "======================================================"
echo "DiMA Generation with lm_head Decoder (Poly-L Fix)"
echo "======================================================"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

DIMA_DIR="${PROJECT_ROOT}/DiMA"

CONDA_ENV_NAME="${CONDA_ENV:-chiu-lab}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$HOME/.conda/envs/$CONDA_ENV_NAME}"

if [[ -d "$CONDA_ENV_PREFIX" ]]; then
  export PATH="$CONDA_ENV_PREFIX/bin:$PATH"
  export CONDA_PREFIX="$CONDA_ENV_PREFIX"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_NAME"
else
  echo "[ERROR] Conda environment not found: $CONDA_ENV_PREFIX"
  exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export DISABLE_WANDB=1
export WANDB_MODE=disabled

RUN_ROOT="$PROJECT_ROOT/DiMA"
STATS_SRC="$PROJECT_ROOT/DiMA/checkpoints/statistics/encodings-ESM2-3B.pth"
STATS_DST_DIR="$RUN_ROOT/checkpoints/statistics"
HF_CACHE_DIR="/local/dima_lmhead_gen_hf_cache"

mkdir -p "$STATS_DST_DIR" "$HF_CACHE_DIR/transformers" "$HF_CACHE_DIR/hub" "$HF_CACHE_DIR/xdg"
if [[ -f "$STATS_SRC" ]]; then
  STATS_DST_FILE="$STATS_DST_DIR/$(basename "$STATS_SRC")"
  if [[ "$(readlink -f "$STATS_SRC")" == "$(readlink -f "$STATS_DST_FILE")" ]]; then
    echo "[INFO] Statistics file already in place: $STATS_DST_FILE"
  else
    cp -f "$STATS_SRC" "$STATS_DST_DIR/"
  fi
fi

export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export XDG_CACHE_HOME="$HF_CACHE_DIR/xdg"

cd "$DIMA_DIR"

echo "[*] Python: $(which python)"
echo "[*] PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "[*] Project: $PROJECT_ROOT"
echo "[*] Working dir: $(pwd)"
echo ""

# Run generation with reference checkpoint
echo "[*] Starting generation with reference checkpoint..."
echo "[*] Using ESM lm_head decoder (transformer decoder disabled)"
echo ""

python train_diffusion.py \
  ++training.init_se="${PROJECT_ROOT}/artifacts/reference_dima_eval/checkpoints/diffusion_checkpoints/reference_dima_eval/5000.pth" \
  ++training.eval_only=true \
  ++generation.num_gen_samples=256 \
  ++generation.N_steps=500 \
  ++project.wandb_project=offline_eval \
  ++decoder.decoder_path=/nonexistent/disabled_decoder.pth \
  ++project.path="$RUN_ROOT" \
  ++project.checkpoints_prefix=dima_lmhead_generation \
  ++ddp.enabled=false

echo ""
echo "[✓] Generation complete!"
echo "[*] Results saved to ${RUN_ROOT}/generated_sequences/"
echo ""

echo ""
echo "======================================================"
echo "Job completed at $(date)"
echo "======================================================"
