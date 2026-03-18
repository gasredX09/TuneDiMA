# Project Progress Log

This document summarizes what has been completed, where the project stands now, and what to do next.

## Scope

Goal: run a reliable DiMA baseline, then move to high-return improvements (decoder reliability + domain-adaptive fine-tuning).

## Work Completed So Far

### 1) Infrastructure and Job Setup

- Organized project workspace under `project/`.
- Added/updated SLURM jobs for Bridges-2 `GPU-shared` with V100 syntax.
- Standardized single-GPU and multi-GPU launch flow.
- Added robust conda environment activation in SLURM scripts.
- Removed accidental `.venv` usage and switched to conda-only workflow.

### 2) Caching and Disk Quota Fixes

- Moved Hugging Face caches to project/local scratch paths to avoid home quota failures.
- Added cache-related env vars in launch scripts (`HF_HOME`, `HUGGINGFACE_HUB_CACHE`, etc.).
- Added artifact sync behavior from local scratch to persistent project artifacts.

### 3) Baseline Training Debugging (Major Fixes)

- Fixed missing dataset path handling and auto-download of AFDB dataset when absent.
- Fixed non-DDP trainer bugs (`ddp_score_estimator` assumptions in single-GPU mode).
- Fixed V100 mixed precision compatibility (bf16 -> fp16 fallback when bf16 unsupported).
- Fixed multiple non-DDP distributed API crashes in evaluation/validation code paths.
- Added resume support in baseline launcher (`INIT_SE`).

### 4) Encoder Normalization and Statistics

- Added robust fallback behavior in encoder normalizer for missing stats.
- Generated encoder statistics successfully:
  - `DiMA/checkpoints/statistics/encodings-ESM2-3B.pth`

### 5) Decoder Training Pipeline Hardening

- Added decoder launcher and SLURM path with conda/cache/library safeguards.
- Reduced decoder memory usage:
  - no gradients through frozen encoder
  - reduced decoder batch size/workers via env overrides
  - allocator setting for fragmentation mitigation
- Fixed device mismatch in normalization (`cpu` vs `cuda`) for decoder training.
- Added decoder run controls:
  - periodic checkpointing (`DECODER_CKPT_INTERVAL`)
  - step cap (`DECODER_MAX_STEPS`)
  - resume support (`DECODER_RESUME_PATH`)

## Current Status (Now)

- Baseline diffusion run reached 2000 steps and completed.
- Statistics generation run completed and saved stats file.
- Long decoder run was canceled intentionally because it would exceed practical walltime and had no useful intermediate saves before patching.
- Decoder code now supports periodic saves and restartable capped runs.

## Recommended Next Step (Immediate)

Run decoder training with capped steps and periodic checkpoints:

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sbatch --export=ALL,CONDA_ENV=chiu-lab,DISABLE_WANDB=1,RUN_NAME=decoder_pretrain_steps50k,DATASET_NAME=AFDB-v2,DECODER_BATCH_SIZE_PER_GPU=2,DECODER_NUM_WORKERS=2,DECODER_MAX_STEPS=50000,DECODER_CKPT_INTERVAL=5000 slurm/train_decoder_single_gpu.sbatch
```

Optional resume from a saved decoder checkpoint:

```bash
sbatch --export=ALL,CONDA_ENV=chiu-lab,DISABLE_WANDB=1,RUN_NAME=decoder_pretrain_steps50k,DATASET_NAME=AFDB-v2,DECODER_BATCH_SIZE_PER_GPU=2,DECODER_NUM_WORKERS=2,DECODER_MAX_STEPS=50000,DECODER_CKPT_INTERVAL=5000,DECODER_RESUME_PATH=/ocean/projects/cis260039p/aguda1/nndl/project/artifacts/decoder_pretrain_steps50k/decoder_checkpoints/decoder_last.pth slurm/train_decoder_single_gpu.sbatch
```

## Next Milestones After Decoder

1. Run a short sanity diffusion retrain (e.g., 500 iterations) with the improved decoder path.
2. Verify stable metrics (no NaN loss, finite `esm_pppl`, non-degenerate quality indicators).
3. Launch domain-adaptive denoiser fine-tuning on the target protein subset.
4. Compare baseline vs adapted model with identical evaluation settings.

## Operational Notes

- Use conda env `chiu-lab` for all SLURM jobs.
- Ignore stale shell snippets that source `.venv`; the `.venv` was removed.
- Main logs are under `project/logs/`.
- Persisted artifacts/checkpoints are under `project/artifacts/`.
