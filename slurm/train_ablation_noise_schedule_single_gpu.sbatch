#!/bin/bash
#SBATCH --job-name=dima-ablate-noise-1g
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=GPU-shared
#SBATCH --qos=gpu_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --gpus=v100-32:1
#SBATCH --time=36:00:00
#SBATCH --time-min=08:00:00
#SBATCH -A cis260039p

set -euo pipefail

# Fixed project root for this workspace
PROJECT_ROOT="/ocean/projects/cis260039p/sdhanuka/TuneDiMA"
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="${CONDA_ENV:-nndl}"
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

python -c "import torch; print('Using Python:', __import__('sys').executable)"
python -c "import datasets"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

export RUN_NAME="${RUN_NAME:-ablation_noise_linear}"
export WANDB_PROJECT="${WANDB_PROJECT:-dima-course}"
export NOISE_SCHEDULE="${NOISE_SCHEDULE:-linear}"

if [[ -n "${LOCAL:-}" && -d "${LOCAL:-}" ]]; then
  RUN_ROOT="${LOCAL}/${RUN_NAME}"
else
  RUN_ROOT="${PROJECT_ROOT}/artifacts/${RUN_NAME}"
fi

STATS_SRC="${PROJECT_ROOT}/DiMA/checkpoints/statistics/encodings-ESM2-3B.pth"
if [[ -f "$STATS_SRC" && -n "${LOCAL:-}" && -d "${LOCAL:-}" ]]; then
  STATS_DIR="${LOCAL}/${RUN_NAME}/checkpoints/statistics"
  mkdir -p "$STATS_DIR"
  cp -v "$STATS_SRC" "$STATS_DIR/" 2>/dev/null || true
fi

if [[ -n "${REF_CKPT_SRC:-}" ]]; then
  if [[ ! -f "${REF_CKPT_SRC}" ]]; then
    echo "[ERROR] REF_CKPT_SRC does not exist: ${REF_CKPT_SRC}"
    exit 1
  fi

  REF_CKPT_NAME="${REF_CKPT_NAME:-5000}"
  REF_DST_DIR="${RUN_ROOT}/checkpoints/diffusion_checkpoints/${RUN_NAME}"
  REF_DST_FILE="${REF_DST_DIR}/${REF_CKPT_NAME}.pth"

  mkdir -p "${REF_DST_DIR}"
  if [[ "$(readlink -f "${REF_CKPT_SRC}")" == "$(readlink -f "${REF_DST_FILE}")" ]]; then
    echo "[INFO] Reference checkpoint already in place: ${REF_DST_FILE}"
  else
    cp -v "${REF_CKPT_SRC}" "${REF_DST_FILE}"
  fi

  if [[ ! -s "${REF_DST_FILE}" ]]; then
    echo "[ERROR] Staged reference checkpoint is missing/empty: ${REF_DST_FILE}"
    exit 1
  fi
  echo "[INFO] Reference checkpoint staged: ${REF_DST_FILE}"
fi

if [[ -n "${INIT_SE:-}" && "${INIT_SE}" =~ ^[0-9]+$ ]]; then
  INIT_SE_PATH="${RUN_ROOT}/checkpoints/diffusion_checkpoints/${RUN_NAME}/${INIT_SE}.pth"
  if [[ -f "${INIT_SE_PATH}" ]]; then
    export INIT_SE="${INIT_SE_PATH}"
    echo "[INFO] INIT_SE step resolved to checkpoint path: ${INIT_SE}"
  else
    echo "[ERROR] INIT_SE is numeric (${INIT_SE}) but checkpoint file not found: ${INIT_SE_PATH}"
    exit 1
  fi
fi

if [[ -n "${DECODER_CKPT_SRC:-}" ]]; then
  if [[ ! -f "${DECODER_CKPT_SRC}" ]]; then
    echo "[ERROR] DECODER_CKPT_SRC does not exist: ${DECODER_CKPT_SRC}"
    exit 1
  fi

  DECODER_CKPT_NAME="${DECODER_CKPT_NAME:-transformer-decoder-ESM2-3B.pth}"
  DECODER_DST_DIR="${RUN_ROOT}/checkpoints/decoder_checkpoints"
  DECODER_DST_FILE="${DECODER_DST_DIR}/${DECODER_CKPT_NAME}"

  mkdir -p "${DECODER_DST_DIR}"
  if [[ "$(readlink -f "${DECODER_CKPT_SRC}")" == "$(readlink -f "${DECODER_DST_FILE}")" ]]; then
    echo "[INFO] Decoder checkpoint already in place: ${DECODER_DST_FILE}"
  else
    cp -v "${DECODER_CKPT_SRC}" "${DECODER_DST_FILE}"
  fi

  if [[ ! -s "${DECODER_DST_FILE}" ]]; then
    echo "[ERROR] Staged decoder checkpoint is missing/empty: ${DECODER_DST_FILE}"
    exit 1
  fi
  echo "[INFO] Decoder checkpoint staged: ${DECODER_DST_FILE}"
fi

bash "/ocean/projects/cis260039p/sdhanuka/TuneDiMA/scripts/launch_ablation_noise_schedule.sh"
