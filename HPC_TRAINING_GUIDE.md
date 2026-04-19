# Training on HPC - Quick Start

## Data Ready ✅

**Training manifest prepared:** `data/processed/training_manifest.csv`
- **Total samples:** 9,348
- **Train/valid/test split:** 6,543 / 1,402 / 1,402
- **Columns:** `ligand_smiles`, `pdb_path`, `label_value`

## Step 1: Setup on HPC

```bash
cd /ocean/projects/cis260039p/mjaju/projects

# Activate your environment
conda activate nn

# Install dependencies (if needed)
pip install torch torch_geometric rdkit biopython pandas
```

## Step 2: Run Training

### Option A: Default settings (10 epochs, batch=32, lr=0.001)
```bash
python train_model.py
```

### Option B: Custom settings
```bash
python train_model.py \
  --epochs 20 \
  --batch-size 16 \
  --lr 0.0005 \
  --device cuda
```

### Available flags:
```
--epochs N           Number of training epochs (default: 10)
--batch-size N       Batch size (default: 32)
--lr FLOAT           Learning rate (default: 0.001)
--device cuda|cpu    Device to use (default: cuda)
--seed INT           Random seed (default: 42)
```

## What the Script Does

1. Loads `data/processed/training_manifest.csv`
2. Splits into 70/15/15 train/valid/test
3. Creates `LigandPocketDataset` with ligand + pocket graphs
4. Trains `LigandPocketNet` model for regression
5. Saves best model checkpoint
6. Prints epoch metrics and final validation loss

## Output

Model checkpoint will be saved automatically. Check the training script or `training.py` for checkpoint location.

## Alternative: Batch/Slurm Job

Create `train_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=train_ligand_pocket
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --gpus=1

cd /ocean/projects/cis260039p/mjaju/projects
conda activate nn
python train_model.py --epochs 20 --batch-size 16 --device cuda
```

Then submit:
```bash
sbatch train_job.sh
```

## Questions?

The model architecture is defined in `projects/models/ligand_pocket.py`.
The dataset loader is in `projects/dataset.py`.
Training logic is in `projects/training.py`.
