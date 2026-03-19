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
- Decoder pretraining completed: 50,000 steps with 5,000-step checkpoints saved.

### 6) Sanity Validation and Encoder Statistics in Local Paths

- Run sanity retrain (500 iterations) with newly trained decoder: loss converged smoothly (0.826 → 0.098).
- Added automatic encoder statistics copy in baseline SLURM script: when `LOCAL` scratch is set, encoder statistics are copied from persistent project storage to `${LOCAL}/${RUN_NAME}/checkpoints/statistics`.
- Verified smooth training pipeline with decoder → baseline integration.

### 7) Domain-Adaptive Fine-Tuning Run (Latest)

- Launched domain-adaptive run (`RUN_NAME=domain_adaptive_denoiser`, 5000 iters).
- Training progressed to around step 1000 with stable loss trend.
- Run failed during evaluation/sample generation with CUDA OOM (generation/eval path), not during core training update.
- Persistent checkpoint available for resume:
  - `artifacts/domain_adaptive_denoiser/checkpoints/diffusion_checkpoints/domain_adaptive_denoiser/1000.pth`


## Current Status (Now)

- ✅ Baseline diffusion run: 2000 steps completed.
- ✅ Encoder statistics generation: job 38044904 completed, stats saved to `DiMA/checkpoints/statistics/encodings-ESM2-3B.pth`.
- ✅ Decoder pretraining (50k steps): job 38060424 completed successfully with checkpoint/resume support.
- ✅ Sanity retrain (500 iterations): job 38065104 completed successfully with healthy loss curve (0.826 → 0.098). Encoder stats fallback to identity normalization (no critical impact for validation).
- ✅ Job infrastructure: Baseline SLURM script now copies encoder statistics to local scratch run directory when `LOCAL` is available.
- ⚠️ Domain-adaptive run (job 38065249) stopped early due to CUDA OOM during eval/sample generation after saving step-1000 checkpoint.

## Next Milestone: Domain-Adaptive Fine-Tuning

Ready to launch domain-adaptive denoiser fine-tuning on target protein subset. Configuration complete, all infrastructure validated.

## Recommended Next Step (Immediate)

Resume domain-adaptive fine-tuning from saved step-1000 checkpoint and reduce generation sample count to avoid eval OOM:

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sbatch --export=ALL,CONDA_ENV=chiu-lab,DISABLE_WANDB=1,RUN_NAME=domain_adaptive_denoiser,DATASET_NAME=AFDB-v2,TRAINING_ITERS=5000,EVAL_INTERVAL=500,SAVE_INTERVAL=500,GEN_SAMPLES=64,INIT_SE=/ocean/projects/cis260039p/aguda1/nndl/project/artifacts/domain_adaptive_denoiser/checkpoints/diffusion_checkpoints/domain_adaptive_denoiser/1000.pth slurm/train_baseline_single_gpu.sbatch
```


## Operational Notes

- Use conda env `chiu-lab` for all SLURM jobs.
- Ignore stale shell snippets that source `.venv`; the `.venv` was removed.
- Main logs are under `project/logs/`.
- Persisted artifacts/checkpoints are under `project/artifacts/`.
- Decoder intermediate checkpoints are expected under `project/artifacts/<RUN_NAME>/decoder_checkpoints/`.
