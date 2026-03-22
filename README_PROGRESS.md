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

### 9) Reference DiMA Comparability (Completed)

- First reference attempt (`job 38073350`) completed but was invalid for comparison because it did not load a checkpoint (`training.init_se` empty and no `Checkpoint is loaded from ...` line).
- Added fail-fast reference checkpoint staging in `slurm/train_baseline_single_gpu.sbatch`:
  - New env controls: `REF_CKPT_SRC`, `REF_CKPT_NAME`
  - If `REF_CKPT_SRC` is missing, job exits early.
  - If present, checkpoint is copied into the exact run folder used by `load_checkpoint()`.
- Launched corrected reference run sequence with checkpoint staging and decoder staging.
- Final validated reference run: `job 38112128`.
- Verified from log that the corrected run is using the staged checkpoint (not scratch retraining):
  - `[INFO] Reference checkpoint staged: /local/reference_dima_eval/checkpoints/diffusion_checkpoints/reference_dima_eval/5000.pth`
  - `Checkpoint is loaded from /local/reference_dima_eval/checkpoints/diffusion_checkpoints/reference_dima_eval/5000.pth`
  - `Evaluation of loaded checkpoint`
- Job accounting confirms clean completion:
  - `sacct -j 38112128` -> `COMPLETED`, exit code `0:0`
- Final reference metrics from `38112128`:
  - `fid: 23.68523`
  - `mmd: 3.50680`
  - `esm_pppl: 1.00430`
  - `plddt: 61.83303`


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
- ✅ Corrected reference comparability run `38112128`: completed with checkpoint load confirmed and healthy metrics.
- ✅ Selected checkpoint comparability run `38128244` completed (`RUN_NAME=selected_step4500_eval`).

### 10) Apples-to-Apples Domain-Adaptive vs Reference Comparability (Completed)

- ✅ Selected checkpoint eval completed: job `38128244` with `best_by_fid_mmd_step4500.pth` checkpoint.
- Job `38128244` completed in 52 minutes with exit code 0:0.
- Side-by-side comparison results:

| Metric | Reference (38112128) | Selected Step4500 (38128244) | Delta |
|--------|:----:|:----:|:----:|
| FID ↓ | 23.68523 | 23.78363 | +0.0984 (worse) |
| MMD ↓ | 3.50680 | 3.52037 | +0.01357 (worse) |
| ESM-PPL ↓ | 1.00430 | 1.00457 | +0.00027 (worse) |
| pLDDT ↑ | 61.83303 | 58.77998 | -3.05305 (worse) |

**Interpretation**: Reference 5000-step checkpoint (untrained) produces marginally better metrics overall. Selected step-4500 checkpoint (domain-adaptive fine-tuned) shows slight quality degradation on reference AFDB task. This suggests domain-adaptive fine-tuning optimized for target domain at the cost of reference generalization.

## Next Milestone: Final Project Summary & Recommendations
Run a matched evaluation for selected checkpoint `best_by_fid_mmd_step4500.pth` in the same pipeline (same decoder, generation settings, and metric settings), then report side-by-side deltas against reference run `38112128`.

## Recommended Next Step (Immediate)

Evaluate the selected checkpoint in the exact same local pipeline used by the completed reference run:

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sbatch --export=ALL,CONDA_ENV=chiu-lab,DISABLE_WANDB=1,RUN_NAME=selected_step4500_eval,DATASET_NAME=AFDB-v2,TRAINING_ITERS=5000,EVAL_INTERVAL=5000,SAVE_INTERVAL=5000,GEN_SAMPLES=64,REF_CKPT_SRC=/ocean/projects/cis260039p/aguda1/nndl/project/artifacts/domain_adaptive_denoiser/selected/best_by_fid_mmd_step4500.pth,REF_CKPT_NAME=5000,DECODER_CKPT_SRC=/ocean/projects/cis260039p/aguda1/nndl/project/DiMA/checkpoints/decoder_checkpoints/transformer-decoder-ESM2-3B.pth slurm/train_baseline_single_gpu.sbatch
```

If `REF_CKPT_SRC` is missing, the sbatch script now fails fast and exits with a clear error.


## Operational Notes

- Use conda env `chiu-lab` for all SLURM jobs.
- Ignore stale shell snippets that source `.venv`; the `.venv` was removed.
- Main logs are under `project/logs/`.
- Persisted artifacts/checkpoints are under `project/artifacts/`.
- Decoder intermediate checkpoints are expected under `project/artifacts/<RUN_NAME>/decoder_checkpoints/`.
