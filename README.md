# DiMA Protein Diffusion Course Project

This repository is a reproducible HPC-focused training/evaluation wrapper for DiMA-based protein sequence generation experiments.

The project currently supports:

- Stable single-GPU and multi-GPU training on Bridges-2.
- Domain-adaptive denoiser fine-tuning with resume.
- Decoder pretraining and checkpointed recovery.
- Local scratch execution with artifact sync back to persistent storage.
- Controlled reference checkpoint evaluation for apples-to-apples comparison.

## 1) Project Objectives

Primary objective for the course project:

- Establish a reliable baseline and adaptation pipeline for DiMA under realistic HPC constraints.
- Compare adapted models against a reference checkpoint in the same evaluation path.
- Report quality/diversity/novelty/structure proxies while tracking training cost and stability.

Practical research direction currently implemented:

- Baseline reproducibility and infrastructure hardening.
- Domain-adaptive denoiser fine-tuning.
- Decoder reliability and evaluation robustness fixes.

## 2) Repository Structure

- `DiMA/`: model/training source and upstream-style configs/checkpoints layout.
- `scripts/`: local launch entrypoints for baseline, decoder training, and helpers.
- `slurm/`: sbatch submit files for single-GPU, multi-GPU, decoder, and statistics jobs.
- `configs/`: default experiment env settings (`baseline.env`).
- `artifacts/`: persisted run outputs (checkpoints, generated sequences, selected models).
- `logs/`: SLURM out/err logs.
- `runs/`: run-note templates and per-run records.
- `reports/`: summarized experiment comparisons.
- `README_PROGRESS.md`: live milestone log and current status notes.

## 3) Compute and Environment

Target platform:

- Bridges-2 `GPU-shared` partition.
- V100 32GB profile in single-GPU script (`--gpus=v100-32:1`).

Environment assumptions:

- Conda environment name defaults to `nndl`.
- Conda activation is handled inside sbatch scripts with robust fallback logic.
- `.venv` is intentionally not used.

Required behavior already baked in:

- Prefer conda C++ runtime libraries (`LD_LIBRARY_PATH`/`LD_PRELOAD` where relevant).
- Disable W&B online mode for non-interactive HPC jobs.
- Enable full Hydra tracebacks for easier debugging.

## 4) Data and Caching Strategy

Dataset defaults:

- `DATASET_NAME=AFDB-v2`
- Data expected under `DiMA/data/AFDB-v2` with `train/` and `test/` splits.

Auto-download behavior:

- Baseline launchers will download AFDB data from Hugging Face if split directories are missing.

Length distribution requirement:

- `DiMA/data/distributions/AFDB-v2.npy` must exist.
- If missing, generate via:

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project/DiMA
python -m src.helpers.prepare_length_distribution --config_path src/configs/config.yaml
```

Caching policy:

- Cache root is moved away from home quota (prefers `${LOCAL}/hf_cache` when available).
- Exports include `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, and `XDG_CACHE_HOME`.

## 5) Run Artifacts Layout

Each run writes into:

- Local scratch run root (`${LOCAL}/${RUN_NAME}`) when available, then syncs to:
- Persistent root: `artifacts/${RUN_NAME}`

Typical outputs:

- Diffusion checkpoints: `artifacts/<run>/checkpoints/diffusion_checkpoints/<run>/*.pth`
- Decoder checkpoints: `artifacts/<run>/decoder_checkpoints/*.pth`
- Generated sequences: `artifacts/<run>/generated_sequences/<run>/<step>.json`
- Structure outputs (pLDDT eval path): `artifacts/<run>/pdb_files/<run>/*.pdb`

## 6) Core Workflows

### 6.1 Single-GPU baseline training

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sbatch slurm/train_baseline_single_gpu.sbatch
```

Key controls (override via `--export=ALL,...`):

- `RUN_NAME`, `TRAINING_ITERS`, `EVAL_INTERVAL`, `SAVE_INTERVAL`
- `BATCH_SIZE`, `BATCH_SIZE_PER_GPU`, `NUM_WORKERS`
- `GEN_STEPS`, `GEN_SAMPLES`
- `INIT_SE` (resume/init from checkpoint)

### 6.2 Multi-GPU baseline training

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sbatch slurm/train_baseline_multi_gpu.sbatch
```

Additional multi-GPU controls:

- `GPUS_PER_NODE`
- `MASTER_PORT`
- `BATCH_SIZE_PER_GPU`

### 6.3 Decoder pretraining

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sbatch slurm/train_decoder_single_gpu.sbatch
```

Useful decoder controls:

- `DECODER_BATCH_SIZE_PER_GPU`
- `DECODER_NUM_WORKERS`
- `DECODER_MAX_STEPS`
- `DECODER_CKPT_INTERVAL`
- `DECODER_RESUME_PATH`

### 6.4 Encoder statistics generation

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sbatch slurm/calculate_statistics_single_gpu.sbatch
```

Expected output:

- `DiMA/checkpoints/statistics/encodings-ESM2-3B.pth`

## 7) Reference Evaluation Workflow

To evaluate an official/reference checkpoint in the exact same local pipeline:

1. Stage reference checkpoint via `REF_CKPT_SRC` (fail-fast if missing).
2. Optionally stage decoder checkpoint via `DECODER_CKPT_SRC`.
3. Run single-GPU baseline script with `RUN_NAME=reference_dima_eval`.

Example:

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sbatch --export=ALL,CONDA_ENV=nndl,DISABLE_WANDB=1,RUN_NAME=reference_dima_eval,DATASET_NAME=AFDB-v2,TRAINING_ITERS=5000,EVAL_INTERVAL=5000,SAVE_INTERVAL=5000,GEN_SAMPLES=64,REF_CKPT_SRC=/ocean/projects/cis260039p/aguda1/nndl/project/artifacts/reference_dima_eval/checkpoints/diffusion_checkpoints/reference_dima_eval/5000.pth,DECODER_CKPT_SRC=/ocean/projects/cis260039p/aguda1/nndl/project/DiMA/checkpoints/decoder_checkpoints/transformer-decoder-ESM2-3B.pth slurm/train_baseline_single_gpu.sbatch
```

## 8) Metrics and Interpretation

Current evaluation logs report:

- `fid`: lower is better.
- `mmd`: lower is better.
- `esm_pppl`: language-model proxy of protein plausibility.
- `plddt`: structure-confidence proxy from ESMFold.

Important practical notes:

- Empty/invalid decoded sequences can corrupt `esm_pppl`/`plddt` signals.
- The code now sanitizes sequences before structure/perplexity metrics.
- If decoder checkpoint is unavailable, decode now falls back to ESM lm-head instead of using random transformer decoder weights.

## 9) Current Status Snapshot

From `README_PROGRESS.md` and run artifacts:

- Baseline, statistics, decoder pretraining, and sanity retraining are complete.
- Domain-adaptive denoiser run recovered from OOM and completed 5000 steps.
- Selected adaptation checkpoint promoted at step 4500.
- Corrected reference-eval pipeline includes checkpoint staging and checkpoint-load verification.

Latest promoted adaptation artifacts:

- `artifacts/domain_adaptive_denoiser/selected/best_by_fid_mmd_step4500.pth`
- `artifacts/domain_adaptive_denoiser/selected/best_by_fid_mmd_step4500_samples.json`

## 10) Logging and Monitoring

- Main SLURM logs: `logs/*.out`, `logs/*.err`
- Queue checks:

```bash
squeue -u $USER
```

- Watch a run:

```bash
tail -f /ocean/projects/cis260039p/aguda1/nndl/project/logs/<job-log>.out
tail -f /ocean/projects/cis260039p/aguda1/nndl/project/logs/<job-log>.err
```

## 11) Troubleshooting Guide

### Conda environment not found

- Ensure `CONDA_ENV` and/or `CONDA_ENV_PREFIX` points to a valid env.

### Missing length distribution

- Generate `AFDB-v2.npy` with `prepare_length_distribution` command above.

### OOM during eval/generation

- Reduce `GEN_SAMPLES` first.
- Optionally reduce batch settings and/or number of workers.

### Reference run accidentally retrains from scratch

- Always pass `REF_CKPT_SRC` and verify log line:
	- `Checkpoint is loaded from ...`

### Decoder not loaded

- Pass `DECODER_CKPT_SRC` in sbatch export for scratch-based runs.
- Verify log line indicating decoder checkpoint staged.

## 12) Reproducibility and Experiment Hygiene

- Keep one primary change per run.
- Store run notes using `runs/RUN_TEMPLATE.md` copied to dated files.
- Record:
	- launch command
	- key hyperparameters
	- checkpoint used
	- final and best metric snapshots
	- stability incidents (OOM/NaN)

## 13) Suggested Semester Execution Plan

1. Lock baselines and reference comparability.
2. Run full denoiser fine-tune adaptation.
3. Run one parameter-efficient adaptation variant (LoRA/adapters).
4. Add exactly one secondary ablation axis (self-conditioning toggle or schedule variant).
5. Finalize side-by-side table in `reports/` with compute cost and metric deltas.

## 14) Quick Command Reference

Single GPU baseline:

```bash
sbatch slurm/train_baseline_single_gpu.sbatch
```

Multi GPU baseline:

```bash
sbatch slurm/train_baseline_multi_gpu.sbatch
```

Decoder train:

```bash
sbatch slurm/train_decoder_single_gpu.sbatch
```

Statistics:

```bash
sbatch slurm/calculate_statistics_single_gpu.sbatch
```

---

If you update job scripts or add new experiment types, keep this README and `README_PROGRESS.md` synchronized so new teammates can reproduce results end-to-end without tribal knowledge.
# Chemistry-Aware Neural Network Pipeline

This folder implements the full pipeline described in the PDF:
- data collection and curation for kinase and glutamate receptor binding assays
- ligand feature extraction with RDKit
- protein pocket feature extraction with BioPython and pocket residue graphs
- PyTorch Geometric models: LigandGNN and LigandPocketNet
- training, evaluation, and screening utilities

## Setup

```bash
cd /ocean/projects/cis260039p/mjaju/projects
conda env create -f environment.yml
conda activate projects-env
python -m pip install --upgrade pip
```

If you already have an environment such as `nn`, activate it instead:

```bash
conda activate nn
```

Then install the Python dependencies inside that environment:

```bash
python -m pip install -r requirements.txt
```

## Download datasets

The repository includes a download script for the main sources.

```bash
cd /ocean/projects/cis260039p/mjaju/projects
python scripts/download_datasets.py
```

This will create `data/raw/`, fetch broad assay archives, download a few example kinase PDB structures, and create a kinase-complex scaffold:

- `data/raw/klifs/`
- `data/raw/kinase_complexes/`
- `data/raw/manifests/kinase_complex_manifest.csv`

For a real kinase bound-complex dataset, use structure sources such as KLIFS/PDB for complexes and ChEMBL/BindingDB for labels. The default downloader is a starting scaffold, not a complete kinase-complex curation pipeline.

## Data preparation

After download, prepare the raw assay and structure files for modeling. The current implementation expects raw inputs such as:

- `data/raw/chembl/`
- `data/raw/bindingdb/`
- `data/raw/pubchem/`
- `data/raw/pdb/`
- `data/raw/klifs/`
- `data/raw/manifests/kinase_complex_manifest.csv`

The code in `projects/data/curation.py` standardizes units, removes duplicates, maps to UniProt IDs, and computes p-values. For kinase complex modeling, those labels should be joined to structure-backed complexes rather than treated as standalone assay rows.

To build a first-pass kinase training manifest from KLIFS + ChEMBL, run:

```bash
cd /ocean/projects/cis260039p/mjaju/projects
python scripts/build_kinase_manifest.py
```

This writes:

- `data/processed/kinase_activity_manifest.csv`
- `data/processed/kinase_activity_manifest_summary.txt`
- `data/raw/manifests/kinase_complex_manifest.csv`

The current join uses a representative KLIFS structure per kinase target together with ChEMBL ligand activities for that same target. That matches the repo's `(SMILES, pocket PDB, label)` setup, but it does not guarantee that the ChEMBL ligand is the same ligand crystallized in the KLIFS template structure.

## Training

Run the example training script:

```bash
cd /ocean/projects/cis260039p/mjaju/projects
python -m projects.training
```

That script uses the PDF’s model architecture and training loop structure.

## Evaluation

Use the evaluation module to compute regression and classification metrics:

```bash
python -m projects.evaluate
```

## Screening

A library filtering implementation is available in `projects/screening.py`:

```bash
python -m projects.screening
```

## Project layout

- `projects/config.py`: constants and data paths
- `projects/data/curation.py`: assay standardization and schema enforcement
- `projects/features/ligand.py`: RDKit ligand featurization
- `projects/features/pocket.py`: pocket residue extraction and graph building
- `projects/dataset.py`: PyTorch Geometric dataset wrappers
- `projects/models/gnn.py`: ligand-only GCN model
- `projects/models/ligand_pocket.py`: ligand+pocket dual-branch model
- `projects/training.py`: training loop, optimizer, scheduler
- `projects/evaluate.py`: metrics calculator
- `projects/screening.py`: Lipinski and PAINS filters
- `projects/scripts/download_datasets.py`: dataset fetching scripts

## Notes

This implementation follows the PDF exactly, including:
- 3-layer GCN architecture
- BatchNorm + ReLU + Dropout
- global mean pooling
- Adam optimizer with `lr=1e-3` and `weight_decay=1e-5`
- loss examples for MSE and BCEWithLogits
- ROC-AUC, PR-AUC, RMSE, enrichment evaluation
