# NNDL Final Project Workspace

This directory contains two streams of work that were merged together:

1. DiMA protein diffusion experimentation on HPC.
2. A ligand-pocket GNN training pipeline for kinase/structure datasets.

The merge brought useful code from both branches, but also duplicated docs and stale absolute paths from a previous clone. This README is the consolidated, working reference for this repository path.

## 1) Repository Layout

- `DiMA/`: upstream-style diffusion model code, configs, decoder/statistics pipeline.
- `scripts/`: launchers for DiMA workflows and chemistry manifest utilities.
- `slurm/`: sbatch wrappers for baseline, decoder, and ablation runs.
- `artifacts/`: persisted DiMA run artifacts and checkpoints.
- `logs/`: SLURM logs for DiMA/ablation workflows.
- `data/`: chemistry/structure manifests and raw/processed data used by the GNN pipeline.
- `models/`, `dataset.py`, `training.py`, `evaluate.py`, `train_model.py`: ligand-pocket GNN training stack.
- `docs/`: project audit notes, metrics reports, and handoff docs.

## 2) Current Merge Status

### What is now fixed

- README conflict content was consolidated into one document.
- `.gitignore` duplication/conflict was cleaned.
- Stale import paths like `from projects...` were replaced with local imports.
- Legacy absolute PDB paths from older clone roots are now remapped to this local workspace at runtime.

### What to know

- Many historical `.err` / `.out` files are preserved but moved under `archive/root_slurm_logs/`.
- The DiMA and chemistry tracks are both kept; neither was removed.

## 3) DiMA Workflow (Main Branch Baseline)

### Single GPU baseline

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sbatch slurm/train_baseline_single_gpu.sbatch
```

### Multi GPU baseline

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sbatch slurm/train_baseline_multi_gpu.sbatch
```

### Decoder pretraining

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sbatch slurm/train_decoder_single_gpu.sbatch
```

### Statistics generation

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sbatch slurm/calculate_statistics_single_gpu.sbatch
```

## 4) Chemistry / Ligand-Pocket Workflow (Mudit Branch Additions)

### Setup

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
conda activate nn
python -m pip install -r requirements.txt
```

### Train the ligand-pocket model

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
python train_model.py --epochs 10 --batch-size 32 --lr 0.001 --device cuda
```

### Evaluate helpers

Use functions in `evaluate.py` for RMSE/MAE and classification metrics.

### Manifest prep helpers

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
python scripts/build_training_manifest.py
```

## 5) Important Compatibility Notes

- Legacy manifests may include absolute paths such as `/ocean/projects/cis260039p/mjaju/projects/...`.
- The path resolver now maps those to this repository root when possible.
- If a PDB file is truly missing locally, training scripts still skip those rows.

## 6) Recommended Working Practice

- Keep new run outputs in `artifacts/` and `logs/`.
- Keep generated raw-heavy data out of Git (already covered by `.gitignore`).
- Use `docs/` for experiment summaries and final figures.

## 7) Branch Intent Summary

- `main` branch work: stable DiMA HPC training/evaluation pipeline, ablation infrastructure, run tracking and metrics reporting.
- `mudit` branch work: chemistry-oriented ligand-pocket dataset and GNN training stack, plus large HKPocket raw data/manifests and local training logs.

This merged workspace now supports both tracks side-by-side with corrected paths.
