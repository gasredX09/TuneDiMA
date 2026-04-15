#!/bin/bash
#SBATCH --job-name=dima-stats-1g
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --gpus=v100-32:1
#SBATCH --time=06:00:00
#SBATCH --time-min=01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
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

bash "$PROJECT_ROOT/scripts/launch_calculate_statistics.sh"
