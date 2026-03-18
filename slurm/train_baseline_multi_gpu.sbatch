#!/bin/bash
#SBATCH --job-name=dima-base-4g
#SBATCH --output=/ocean/projects/cis260039p/aguda1/nndl/project/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis260039p/aguda1/nndl/project/logs/%x-%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gpus=v100-32:4
#SBATCH --time=24:00:00
#SBATCH --time-min=08:00:00
#SBATCH --chdir=/ocean/projects/cis260039p/aguda1/nndl/project

set -euo pipefail

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

python -c "import torch; print('Using Python:', __import__('sys').executable)"

# Prefer conda-shipped shared libraries over older system libstdc++.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Optional run overrides
export RUN_NAME="${RUN_NAME:-baseline_full_ft_multi_gpu}"
export WANDB_PROJECT="${WANDB_PROJECT:-dima-course}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
export MASTER_PORT="${MASTER_PORT:-29511}"

bash scripts/launch_baseline_multi_gpu.sh
