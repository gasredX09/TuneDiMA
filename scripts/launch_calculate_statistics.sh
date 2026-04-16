#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIMA_ROOT="$PROJECT_ROOT/DiMA"

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

# Prefer conda-shipped C++ runtime to avoid GLIBCXX mismatch.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6"

DATASET_NAME="${DATASET_NAME:-AFDB-v2}"
DATA_DIR="${DATA_DIR:-$DIMA_ROOT/data/$DATASET_NAME}"
PROJECT_PATH="${PROJECT_PATH:-$DIMA_ROOT}"

cd "$DIMA_ROOT"
python -m src.preprocessing.calculate_statistics \
  --config_path ../configs \
  --project_path "$PROJECT_PATH" \
  --data_dir "$DATA_DIR"
