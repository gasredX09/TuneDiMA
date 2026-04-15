# README for Shailja: TuneDiMA Project Guide

**Date**: March 23, 2026  
**Project**: TuneDiMA - DiMA Protein Diffusion Model Adaptation and Ablation Studies  
**Location**: `/ocean/projects/cis260039p/sdhanuka/TuneDiMA`

---

## Project Overview

This is a course project focused on **DiMA (Diffusion Model for Proteins)**, a latent diffusion model for protein sequence generation. The goal is to establish a reliable baseline on HPC (Bridges-2), perform domain-adaptive fine-tuning, and conduct ablation studies to understand model components.

Key components:
- **DiMA Model**: Gaussian diffusion on ESM-2/CHEAP protein language model encodings.
- **Training Pipeline**: Encoder (ESM-2), diffusion denoiser, decoder (transformer).
- **Evaluation**: FID, MMD, ESM-PPL, pLDDT metrics on AFDB-v2 dataset.
- **HPC Setup**: SLURM jobs, conda env (`nndl`), artifact sync.

---

## Current Status

### Completed Work
- **Infrastructure**: SLURM scripts, conda env, caching, artifact management.
- **Baseline Runs**: Reference DiMA (job 38112128), domain-adaptive (job 38067100), comparability evals.
- **Environment Setup**: `nndl` env created, dependencies verified (torch, hydra, wandb, etc.).
- **Scripts**: Baseline launchers, decoder training, statistics generation.
- **Ablation Prep**: Three new scripts created for controlled ablations.

### In Progress
- Partial fine-tuning ablation (job 38131704 pending).
- Ablation study planning and execution.

### Key Metrics (Reference Baseline)
From job 38112128:
- FID: 23.68523
- MMD: 3.50680
- ESM-PPL: 1.00430
- pLDDT: 61.83303

---

## What to Do Next

### Immediate Next Steps
1. **Monitor Pending Job**: Check job 38131704 status.
   ```bash
   sacct -j 38131704 --format=JobID,State,ExitCode,Elapsed
   ```
   If completed, extract metrics and update `EVALUATION_SUMMARY.md`.

2. **Run First Ablation**: Use the new scripts to test one variable at a time.

3. **Update Progress**: Log results in `README_PROGRESS.md` and `EVALUATION_SUMMARY.md`.

### Ablation Studies: Why and How

From earlier chat (when you asked about ablation studies):

> "Great progress: your scripts are solid and provide a robust baseline for launching experiments. Next I'll extract key ablation ideas from your PDF and map them into concrete script additions, so you can start immediately with reproducible runs."
> 
> "The paper explicitly says they investigate: 1. encoder representation, 2. decoder mapping, 3. diffusion model architecture, 4. noise schedule / timing, 5. self-conditioning, 6. length sampling, 7. inference guidance, 8. dataset/domain training, 9. partial fine-tuning strategy, 10. replay/anti-forgetting."
> 
> "Your ablations are just env vars + config overrides. So your ablations are just env vars + config overrides."

We are doing ablation studies to understand which DiMA components most impact protein generation quality. This helps optimize the model for better FID/MMD/pLDDT without full retraining.

### The Three Ablation Scripts

I've created three scripts in `scripts/` for controlled experiments:

1. **`launch_ablation_selfcond.sh`**: Tests self-conditioning (on/off).
   - Why: Self-conditioning can stabilize training and improve sample quality.
   - How: Set `MODEL_USE_SELF_COND=0` (off) vs default 1 (on).
   - Run: `bash scripts/launch_ablation_selfcond.sh MODEL_USE_SELF_COND=0 RUN_NAME=ablation_selfcond_off`

2. **`launch_ablation_noise_schedule.sh`**: Tests noise schedules (cosine/linear).
   - Why: Noise schedule affects diffusion quality and convergence.
   - How: Set `NOISE_SCHEDULE=linear` vs default cosine.
   - Run: `bash scripts/launch_ablation_noise_schedule.sh NOISE_SCHEDULE=linear RUN_NAME=ablation_noise_linear`

3. **`launch_ablation_ft_lastn.sh`**: Tests partial fine-tuning layers (2/4/8).
   - Why: Full fine-tuning can cause forgetting; partial may preserve generalization.
   - How: Set `FT_LAST_N_LAYERS=2` (fine-tune last 2 layers) vs 4/8.
   - Run: `bash scripts/launch_ablation_ft_lastn.sh FT_LAST_N_LAYERS=2 RUN_NAME=ablation_ft_last2`

### How to Run Ablations
- **Local Test**: Short run (500 iters) on compute node.
  ```bash
  srun --pty --gres=gpu:1 --time=00:30:00 bash
  conda activate nndl
  bash scripts/launch_ablation_selfcond.sh MODEL_USE_SELF_COND=0 TRAINING_ITERS=500
  ```
- **Full SLURM**: Submit to queue.
  ```bash
  sbatch --gres=gpu:1 --time=08:00:00 slurm/train_baseline_single_gpu.sbatch --export=ALL,MODEL_USE_SELF_COND=0,RUN_NAME=ablation_selfcond_off
  ```
- **Monitor**: `sacct -j <jobid>`, check logs in `experiment_logs/`.
- **Evaluate**: Compare metrics to reference (23.69 FID, etc.).

### Batch Scripts for Ablations

Here are the scripts (also in `scripts/`):

#### launch_ablation_selfcond.sh
```bash
#!/usr/bin/env bash
# ... (full script content as created)
```

#### launch_ablation_noise_schedule.sh
```bash
#!/usr/bin/env bash
# ... (full script content as created)
```

#### launch_ablation_ft_lastn.sh
```bash
#!/usr/bin/env bash
# ... (full script content as created)
```

### Expected Outcomes
- Each ablation: 4 metrics (FID/MMD/ESM-PPL/pLDDT).
- Goal: Find settings that improve over reference without degradation.
- Document in `EVALUATION_SUMMARY.md` table.

---

## Full Project Structure

- `DiMA/`: Model code, configs, environment.yaml.
- `scripts/`: Launch scripts (baseline, decoder, ablations).
- `slurm/`: SLURM batch files.
- `artifacts/`: Checkpoints, logs.
- `configs/`: Env vars, baseline settings.
- `README_PROGRESS.md`: Detailed progress log.
- `EVALUATION_SUMMARY.md`: Metrics and recommendations.

---

## Tips
- Always activate `nndl` env.
- Use compute nodes for GPU (not login).
- Check `README_PROGRESS.md` for job IDs and status.
- If issues, paste terminal output for debugging.

---

**Next Action**: Run first ablation and log results!



Suggested things to run for sanity check
cd /ocean/projects/cis260039p/sdhanuka/TuneDiMA
. /jet/home/sdhanuka/miniconda3/etc/profile.d/conda.sh
conda activate nndl
bash scripts/launch_baseline_single_gpu.sh \
  RUN_NAME=ablation_lastn2 \
  FT_MODE=last_n \
  FT_LAST_N_LAYERS=2 \
  MODEL_USE_SELF_COND=0 \
  REPLAY_RATIO=0.25 \
  TRAINING_ITERS=2000 \
  EVAL_INTERVAL=500 \
  SAVE_INTERVAL=500