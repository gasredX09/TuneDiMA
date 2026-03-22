#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIMA_ROOT="$PROJECT_ROOT/DiMA"

if [[ -f "$PROJECT_ROOT/configs/baseline.env" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/configs/baseline.env"
fi

RUN_NAME="${RUN_NAME:-baseline_full_ft}"
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
  project.decoder_checkpoints_folder="$RUN_ROOT/checkpoints/decoder_checkpoints"
