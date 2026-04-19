#!/bin/bash
#SBATCH --job-name="ligand_pocket_train"
#SBATCH --account=cis260039p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module load cuda/12.4.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nn

# Required by PyTorch deterministic mode on CUDA (see runtime error in slurm log).
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

PROJECT_PARENT=/ocean/projects/cis260039p/mjaju
cd "$PROJECT_PARENT"

echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Python: $(which python)"
echo "PWD: $(pwd)"

srun python -u -m projects.train_model \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-3 \
  --device cuda \
  --seed 42