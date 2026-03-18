#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIMA_ROOT="$PROJECT_ROOT/DiMA"

if [[ -f "$PROJECT_ROOT/configs/baseline.env" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/configs/baseline.env"
fi

RUN_NAME="${RUN_NAME:-decoder_pretrain}"
WANDB_PROJECT="${WANDB_PROJECT:-dima-course}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"
DATASET_NAME="${DATASET_NAME:-AFDB-v2}"
DECODER_NUM_WORKERS="${DECODER_NUM_WORKERS:-4}"
DECODER_BATCH_SIZE_PER_GPU="${DECODER_BATCH_SIZE_PER_GPU:-2}"
DECODER_MAX_STEPS="${DECODER_MAX_STEPS:-50000}"
DECODER_CKPT_INTERVAL="${DECODER_CKPT_INTERVAL:-5000}"

if [[ -n "${LOCAL:-}" && -d "${LOCAL:-}" ]]; then
  DEFAULT_HF_CACHE_ROOT="$LOCAL/hf_cache"
else
  DEFAULT_HF_CACHE_ROOT="$PROJECT_ROOT/artifacts/hf_cache"
fi
HF_CACHE_ROOT="${HF_CACHE_ROOT:-$DEFAULT_HF_CACHE_ROOT}"
mkdir -p "$HF_CACHE_ROOT"

export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_CACHE_ROOT/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_CACHE_ROOT/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HF_CACHE_ROOT/xdg}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Prefer conda-shipped C++ runtime.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6"

DATA_DIR="${DATA_DIR:-$DIMA_ROOT/data/$DATASET_NAME}"
PROJECT_PATH="${PROJECT_PATH:-$DIMA_ROOT}"
DECODER_CHECKPOINT_DIR="${DECODER_CHECKPOINT_DIR:-$PROJECT_ROOT/artifacts/$RUN_NAME/decoder_checkpoints}"
DECODER_RESUME_PATH="${DECODER_RESUME_PATH:-}"

if [[ ! -d "$DATA_DIR/train" || ! -d "$DATA_DIR/test" ]]; then
  echo "[ERROR] Dataset split missing under $DATA_DIR"
  echo "Run baseline launcher once or download dataset first."
  exit 1
fi

STATS_PATH_DEFAULT="$PROJECT_PATH/checkpoints/statistics/encodings-ESM2-3B.pth"
STATS_PATH="${STATS_PATH:-$STATS_PATH_DEFAULT}"
if [[ ! -f "$STATS_PATH" ]]; then
  echo "[ERROR] Missing encoder statistics at $STATS_PATH"
  echo "Run: sbatch --export=ALL,CONDA_ENV=chiu-lab,DATASET_NAME=$DATASET_NAME slurm/calculate_statistics_single_gpu.sbatch"
  exit 1
fi

cd "$DIMA_ROOT"
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export DECODER_NUM_WORKERS
export DECODER_BATCH_SIZE_PER_GPU
export DECODER_MAX_STEPS
export DECODER_CKPT_INTERVAL
export DECODER_CHECKPOINT_DIR
export DECODER_RESUME_PATH
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

DISABLE_WANDB="$DISABLE_WANDB" python -m src.preprocessing.train_decoder \
  --config_path ../configs \
  --project_path "$PROJECT_PATH" \
  --data_dir "$DATA_DIR"
