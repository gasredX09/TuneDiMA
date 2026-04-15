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
- ✅ Partial fine-tuning controls added for denoiser ablation (`FT_MODE`, `FT_LAST_N_LAYERS`).
- ✅ Replay-mix controls added (`REPLAY_DATA_DIR`, `REPLAY_RATIO`, `REPLAY_SEED`) for future anti-forgetting experiments.
- ⚠️ Initial partial fine-tuning runs failed (jobs `38131626`, `38131636`, `38131675`, `38131676`, `38131699`) and were used to fix pipeline issues.
- 🚧 One-at-a-time corrected rerun is pending: job `38131704` (`RUN_NAME=ft_last2_from_ref_5k_r4`).

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

### 11) Partial Fine-Tuning Ablation Setup + Debugging (In Progress)

Goal: run controlled one-at-a-time ablations from the validated reference checkpoint using partial denoiser fine-tuning.

Implemented:

- Added fine-tuning controls in training config/launch path:
  - `training.ft_mode` (`full` or `last_n`)
  - `training.ft_last_n_layers`
- Added optional replay controls for later anti-forgetting experiments:
  - `training.replay_data_dir`, `training.replay_ratio`, `training.replay_seed`
- Added numeric `INIT_SE` resolution in SLURM launcher:
  - if `INIT_SE=5000`, it resolves to `${RUN_ROOT}/checkpoints/diffusion_checkpoints/${RUN_NAME}/5000.pth`

Failure chain and fixes:

1) `torch.load(int)` crash when `INIT_SE` was passed as a number.
- Fix: resolve numeric `INIT_SE` to checkpoint path in `slurm/train_baseline_single_gpu.sbatch`.

2) Optimizer state mismatch when auto-resume loaded checkpoint after partial freezing.
- Fix: skip automatic `load_checkpoint()` when `training.init_se` is explicitly set.

3) EMA shape mismatch after changing trainable parameter set.
- Fix: recreate EMA after applying fine-tune masking so shadow params match `requires_grad` parameters.

Current status:

- One-at-a-time rerun submitted with all fixes:
  - Job `38131704`, run `ft_last2_from_ref_5k_r4`, state `PENDING`.

## Next Milestone: Partial Fine-Tuning Results

Complete and evaluate the one-at-a-time `last_n=2` run (`38131704`). If stable, launch `last_n=4` using the same corrected flow and compare both against reference `38112128`.

## Recommended Next Step (Immediate)

Monitor job `38131704` to completion and extract the four metrics (FID, MMD, ESM-PPL, pLDDT). Then decide whether to launch the second one-at-a-time ablation (`last_n=4`).

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sacct -j 38131704 --format=JobID,State,ExitCode,Elapsed
```

### 12) Environment Setup and Dependency Validation (Completed)

- Created and activated `nndl` conda environment from `DiMA/environment.yaml`.
- Verified core dependencies: torch 2.1.2, hydra-core 1.3.2, wandb 0.25.1, numpy, pandas, transformers, etc.
- Confirmed CUDA availability on compute nodes (not login node).
- Resolved import issues: hydra, wandb now loadable in `nndl` env.

### 13) Ablation Script Development (Completed)

- Created three new ablation launch scripts in `scripts/`:
  - `launch_ablation_selfcond.sh`: Varies `model.config.use_self_cond` (0/1) to test self-conditioning impact.
  - `launch_ablation_noise_schedule.sh`: Varies `generation.noise_schedule` (cosine/linear) to test noise schedule effects.
  - `launch_ablation_ft_lastn.sh`: Varies `training.ft_last_n_layers` (2/4/8) with `training.ft_mode=last_n` for partial fine-tuning ablations.
- Scripts are based on `launch_baseline_single_gpu.sh`, with added env vars and Hydra overrides.
- All scripts support SLURM submission and artifact sync.

### 14) Ablation Study Planning (In Progress)

- Identified key ablation axes from DiMA paper: self-conditioning, noise schedule, partial FT layers, replay ratio, etc.
- Planned controlled experiments: one variable at a time, compare FID/MMD/ESM-PPL/pLDDT against reference (job 38112128).
- Next: Run first ablation (e.g., self-cond off) on compute node, collect metrics, update evaluation summary.

## Next Steps: Ablation Execution

1. Submit first ablation job using new scripts (e.g., `launch_ablation_selfcond.sh` with `MODEL_USE_SELF_COND=0`).
2. Monitor via `sacct`, extract metrics from logs.
3. Compare to reference in `EVALUATION_SUMMARY.md`.
4. Iterate: noise schedule, then FT layers, then replay.

## Operational Notes


## Experiments You Can Run (Quick Reference)

### 1. Baseline DiMA Diffusion Run
- **Purpose:** Establishes a reference for all future experiments.
- **How:** Use `slurm/train_baseline_single_gpu.sbatch` with default config.
- **Status:** Completed (2000 steps, metrics logged).

### 2. Encoder Statistics Generation
- **Purpose:** Precompute encoder normalization stats.
- **How:** Run encoder stats job (see scripts/ or slurm/).
- **Status:** Completed, stats at `DiMA/checkpoints/statistics/encodings-ESM2-3B.pth`.

### 3. Decoder Pretraining
- **Purpose:** Pretrain the decoder for 50k steps.
- **How:** Use decoder training SLURM script.
- **Status:** Completed, checkpoints every 5k steps.

### 4. Sanity Retrain
- **Purpose:** Quick 500-iteration retrain to verify pipeline.
- **How:** Use baseline script with `training_iters=500`.
- **Status:** Completed, healthy loss curve.

### 5. Domain-Adaptive Fine-Tuning
- **Purpose:** Adapt model to a new domain.
- **How:** Use baseline script, set `RUN_NAME=domain_adaptive_denoiser`, resume from checkpoint if needed.
- **Status:** Completed, best checkpoint at step 4500.

### 6. Reference Checkpoint Evaluation
- **Purpose:** Apples-to-apples comparison with official reference.
- **How:** Use baseline script, set `training.init_se` to reference checkpoint.
- **Status:** Completed, metrics logged.

### 7. Selected Checkpoint Evaluation
- **Purpose:** Compare best domain-adapted model to reference.
- **How:** Use baseline script, set `training.init_se` to selected checkpoint.
- **Status:** Completed, metrics logged.

### 8. Partial Fine-Tuning Ablations
- **Purpose:** Fine-tune only last N layers for ablation study.
- **How:** Use baseline script, set `ft_mode=last_n`, `ft_last_n_layers=N`.
- **Status:** In progress; last-2-layers run submitted, last-4-layers recommended next.

### 9. Eval-Only/Hardening Runs
- **Purpose:** Test model stability, NaN/Inf safety, and eval-only mode.
- **How:** Use baseline script with `eval_only=1`, optionally disable AMP/self-cond.
- **Status:** Completed; non-finite logits warning persists but is handled.

---

**See the README_PROGRESS.md for full details, job IDs, and recommended next steps.**
For any experiment, check the corresponding SLURM script and set the right environment variables as described above.
- Decoder intermediate checkpoints are expected under `project/artifacts/<RUN_NAME>/decoder_checkpoints/`.
