#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIMA_ROOT="$PROJECT_ROOT/DiMA"

if [[ -f "$PROJECT_ROOT/configs/baseline.env" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/configs/baseline.env"
fi

RUN_NAME="${RUN_NAME:-ablation_ft_lastn}"
WANDB_PROJECT="${WANDB_PROJECT:-dima-course}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"
TRAINING_ITERS="${TRAINING_ITERS:-20000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GEN_STEPS="${GEN_STEPS:-500}"
GEN_SAMPLES="${GEN_SAMPLES:-256}"
INIT_SE="${INIT_SE:-}"

PERSIST_ARTIFACT_ROOT="$PROJECT_ROOT/artifacts/$RUN_NAME"
mkdir -p "$PERSIST_ARTIFACT_ROOT"

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

DATASET_NAME="${DATASET_NAME:-AFDB-v2}"
DATA_ROOT_PERSIST="${DATA_ROOT_PERSIST:-$DIMA_ROOT/data}"
DATASET_DIR="$DATA_ROOT_PERSIST/$DATASET_NAME"
LENGTH_DISTRIB_PATH="$DATA_ROOT_PERSIST/distributions/$DATASET_NAME.npy"
REPLAY_DATASET_NAME="${REPLAY_DATASET_NAME:-}"
REPLAY_DATA_DIR="${REPLAY_DATA_DIR:-}"
REPLAY_RATIO="${REPLAY_RATIO:-0}"
REPLAY_SEED="${REPLAY_SEED:-42}"
FT_MODE="${FT_MODE:-last_n}"
FT_LAST_N_LAYERS="${FT_LAST_N_LAYERS:-4}"
USE_AMP="${USE_AMP:-1}"
EVAL_ONLY="${EVAL_ONLY:-0}"
OPT_LR="${OPT_LR:-}"
MODEL_USE_SELF_COND="${MODEL_USE_SELF_COND:-1}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-}"

if [[ -z "$REPLAY_DATA_DIR" && -n "$REPLAY_DATASET_NAME" ]]; then
  REPLAY_DATA_DIR="$DATA_ROOT_PERSIST/$REPLAY_DATASET_NAME"
fi

if [[ ! -d "$DATASET_DIR/train" || ! -d "$DATASET_DIR/test" ]]; then
  echo "[INFO] Dataset split not found at $DATASET_DIR. Downloading from Hugging Face..."
  python - <<PY
from datasets import load_dataset
dataset_name = "bayes-group-diffusion/${DATASET_NAME}"
target_dir = "${DATASET_DIR}"
ds = load_dataset(dataset_name)
ds.save_to_disk(target_dir)
print(f"Saved {dataset_name} to {target_dir}")
PY
fi

if [[ ! -f "$LENGTH_DISTRIB_PATH" ]]; then
  echo "[ERROR] Missing length distribution file: $LENGTH_DISTRIB_PATH"
  echo "Run: python -m src.helpers.prepare_length_distribution --config_path src/configs/config.yaml"
  exit 1
fi

if [[ -n "${LOCAL:-}" && -d "${LOCAL:-}" ]]; then
  RUN_ROOT="$LOCAL/$RUN_NAME"
  mkdir -p "$RUN_ROOT"
else
  RUN_ROOT="$PERSIST_ARTIFACT_ROOT"
fi

cleanup() {
  if [[ "$RUN_ROOT" == "$PERSIST_ARTIFACT_ROOT" ]]; then
    return
  fi

  mkdir -p "$PERSIST_ARTIFACT_ROOT"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$RUN_ROOT/" "$PERSIST_ARTIFACT_ROOT/"
  else
    cp -a "$RUN_ROOT/." "$PERSIST_ARTIFACT_ROOT/"
  fi
}
trap cleanup EXIT

cd "$DIMA_ROOT"
export HYDRA_FULL_ERROR=1

EXTRA_OVERRIDES=()
if [[ -n "$REPLAY_DATA_DIR" ]]; then
  if [[ ! -d "$REPLAY_DATA_DIR/train" ]]; then
    echo "[ERROR] REPLAY_DATA_DIR missing train split: $REPLAY_DATA_DIR/train"
    exit 1
  fi

  EXTRA_OVERRIDES+=("training.replay_data_dir=$REPLAY_DATA_DIR")
  EXTRA_OVERRIDES+=("training.replay_ratio=$REPLAY_RATIO")
  EXTRA_OVERRIDES+=("training.replay_seed=$REPLAY_SEED")
fi

if [[ -n "$OPT_LR" ]]; then
  EXTRA_OVERRIDES+=("optimizer.lr=$OPT_LR")
fi

if [[ -n "$GRAD_CLIP_NORM" ]]; then
  EXTRA_OVERRIDES+=("training.grad_clip_norm=$GRAD_CLIP_NORM")
fi

DISABLE_WANDB="$DISABLE_WANDB" python "$PROJECT_ROOT/scripts/run_dima_train.py" \
  ddp.enabled=false \
  project.path="$RUN_ROOT" \
  datasets.data_name="$DATASET_NAME" \
  datasets.data_dir="$DATASET_DIR" \
  datasets.length_distribution="$LENGTH_DISTRIB_PATH" \
  project.wandb_project="$WANDB_PROJECT" \
  project.checkpoints_prefix="$RUN_NAME" \
  training.training_iters="$TRAINING_ITERS" \
  training.eval_interval="$EVAL_INTERVAL" \
  training.save_interval="$SAVE_INTERVAL" \
  training.init_se="$INIT_SE" \
  training.batch_size="$BATCH_SIZE" \
  training.batch_size_per_gpu="$BATCH_SIZE_PER_GPU" \
  dataloader.num_workers="$NUM_WORKERS" \
  generation.N_steps="$GEN_STEPS" \
  generation.num_gen_samples="$GEN_SAMPLES" \
  training.ft_mode="$FT_MODE" \
  training.ft_last_n_layers="$FT_LAST_N_LAYERS" \
  training.use_amp="$USE_AMP" \
  training.eval_only="$EVAL_ONLY" \
  model.config.use_self_cond="$MODEL_USE_SELF_COND" \
  project.decoder_checkpoints_folder="$RUN_ROOT/checkpoints/decoder_checkpoints" \
  "${EXTRA_OVERRIDES[@]}"