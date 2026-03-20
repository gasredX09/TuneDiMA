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

### 8) Domain-Adaptive Recovery + Completion

- Resumed from step-1000 checkpoint with reduced generation load (`GEN_SAMPLES=64`) and completed full 5000-step schedule.
- Recovery run job: `38067100` (state: COMPLETED, elapsed: 05:32:08).
- No CUDA OOM during eval/generation after generation sample reduction.
- Final step-5000 metrics:
  - `fid: 8.80376`
  - `mmd: 1.49501`
  - `esm_pppl: 244.63275`
  - `plddt: 35.59927`
- Best observed checkpoint by FID/MMD tradeoff in this run: step `4500`.
- Promoted best model artifact for downstream comparison:
  - `artifacts/domain_adaptive_denoiser/selected/best_by_fid_mmd_step4500.pth`
  - `artifacts/domain_adaptive_denoiser/selected/best_by_fid_mmd_step4500_samples.json`
- Accidental comparability submission (`job 38073338`) was cancelled to avoid unintended retraining.

### 9) Reference DiMA Comparability (In Progress)

- First reference attempt (`job 38073350`) completed but was invalid for comparison because it did not load a checkpoint (`training.init_se` empty and no `Checkpoint is loaded from ...` line).
- Added fail-fast reference checkpoint staging in `slurm/train_baseline_single_gpu.sbatch`:
  - New env controls: `REF_CKPT_SRC`, `REF_CKPT_NAME`
  - If `REF_CKPT_SRC` is missing, job exits early.
  - If present, checkpoint is copied into the exact run folder used by `load_checkpoint()`.
- Launched corrected reference run: `job 38075574`.
- Verified from log that this corrected run is using the staged checkpoint (not scratch retraining):
  - `[INFO] Reference checkpoint staged: /local/reference_dima_eval/checkpoints/diffusion_checkpoints/reference_dima_eval/5000.pth`
  - `Checkpoint is loaded from /local/reference_dima_eval/checkpoints/diffusion_checkpoints/reference_dima_eval/5000.pth`
  - `Evaluation of loaded checkpoint`


## Current Status (Now)

- ✅ Baseline diffusion run: 2000 steps completed.
- ✅ Encoder statistics generation: job 38044904 completed, stats saved to `DiMA/checkpoints/statistics/encodings-ESM2-3B.pth`.
- ✅ Decoder pretraining (50k steps): job 38060424 completed successfully with checkpoint/resume support.
- ✅ Sanity retrain (500 iterations): job 38065104 completed successfully with healthy loss curve (0.826 → 0.098). Encoder stats fallback to identity normalization (no critical impact for validation).
- ✅ Job infrastructure: Baseline SLURM script now copies encoder statistics to local scratch run directory when `LOCAL` is available.
- ✅ Domain-adaptive run recovered and completed: job 38067100 reached step 5000 with stable eval metrics and persisted checkpoints (`500` ... `5000`).
- ✅ Selected checkpoint promoted for downstream use: `artifacts/domain_adaptive_denoiser/selected/best_by_fid_mmd_step4500.pth`.
- ✅ Official DiMA reference checkpoint staged at `artifacts/reference_dima_eval/checkpoints/diffusion_checkpoints/reference_dima_eval/5000.pth`.
- ⚠️ Reference attempt `38073350`: completed but invalid comparison run (checkpoint not loaded).
- ⏳ Corrected reference comparability run `38075574` is running with checkpoint-load confirmed in logs.

## Next Milestone: Apples-to-Apples DiMA Comparability

Finalize corrected reference run (`38075574`), validate metric health (finite `esm_pppl`, non-zero `plddt`, non-empty generated sequences), then report side-by-side deltas against selected checkpoint `best_by_fid_mmd_step4500.pth`.

## Recommended Next Step (Immediate)

Evaluate the selected checkpoint against an official DiMA reference in the same local pipeline:

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
# 1) place the official DiMA checkpoint at:
#    artifacts/reference_dima_eval/checkpoints/diffusion_checkpoints/reference_dima_eval/5000.pth
# 2) run evaluation in the same local pipeline (checkpoint must exist first)
sbatch --export=ALL,CONDA_ENV=chiu-lab,DISABLE_WANDB=1,RUN_NAME=reference_dima_eval,DATASET_NAME=AFDB-v2,TRAINING_ITERS=5000,EVAL_INTERVAL=5000,SAVE_INTERVAL=5000,GEN_SAMPLES=64 slurm/train_baseline_single_gpu.sbatch
```

If the checkpoint in step (1) is missing, do not submit step (2): it will start a fresh training run.


## Operational Notes

- Use conda env `chiu-lab` for all SLURM jobs.
- Ignore stale shell snippets that source `.venv`; the `.venv` was removed.
- Main logs are under `project/logs/`.
- Persisted artifacts/checkpoints are under `project/artifacts/`.
- Decoder intermediate checkpoints are expected under `project/artifacts/<RUN_NAME>/decoder_checkpoints/`.
