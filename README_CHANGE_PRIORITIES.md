# DiMA Project Change Roadmap (Priority x Expected Return)

This document lists practical changes to improve your DiMA project, ordered by expected return on time.

## Current Baseline Status

- Single-GPU training run reached 2000/2000 and completed.
- Checkpoints and generated sequences were produced at 500, 1000, 1500, 2000.
- Current metrics indicate weak quality signals (`esm_pppl: nan`, `plddt: 0.0`), so metric trust is low until data/normalization/decoder issues are fixed.

## Priority Legend

- Priority: P0 (critical) -> P3 (nice-to-have)
- Return: Very High / High / Medium / Low
- Effort: S (hours), M (1-2 days), L (multi-day)

## P0: Critical Fixes Before Any Scientific Conclusions

### 1) Compute encoder normalization statistics (replace identity fallback)

- Priority: P0
- Return: Very High
- Effort: S-M
- Why: Training currently runs with identity normalization because stats file is missing; this can distort latent scale and destabilize optimization.
- Action:
  - Generate stats with `src.preprocessing.calculate_statistics`.
  - Ensure `encoder.config.statistics_path` points to saved stats for all runs.

### 2) Initialize/train decoder properly

- Priority: P0
- Return: Very High
- Effort: M
- Why: Logs show "Decoder wasn't initialized"; sequence reconstruction quality and downstream metrics can be invalid without a trained decoder.
- Action:
  - Train decoder stage once using `src.preprocessing.train_decoder`.
  - Save and wire `decoder.decoder_path` into training config.

### 3) Stabilize NaN loss behavior

- Priority: P0
- Return: Very High
- Effort: S-M
- Why: `total_loss: nan` appeared during training. A completed run with NaN loss is not scientifically reliable.
- Action:
  - Add explicit NaN/Inf guard in train loop (skip/break on invalid loss and log step).
  - Lower LR for debug runs (e.g., 4e-4 -> 1e-4 or 5e-5).
  - Keep grad clipping enabled and log gradient norm anomalies.
  - Optional: add GradScaler when using fp16.

## P1: Highest Modeling Return

### 4) Domain-adaptive fine-tuning of denoiser (best first project modification)

- Priority: P1
- Return: Very High
- Effort: M
- Why: This is the most direct path to improved in-domain generation quality for your course objective.
- Action:
  - Build a domain subset (family/fold/function-specific).
  - Fine-tune denoiser from baseline checkpoint.
  - Compare against baseline on same metrics and same sample budget.

### 5) Improve evaluation protocol and metric reliability

- Priority: P1
- Return: High
- Effort: M
- Why: Current `esm_pppl: nan` and `plddt: 0.0` indicate metric pipeline fragility.
- Action:
  - Add metric sanity checks (input validity, finite-value assertions).
  - Cache per-sequence metric outputs for failure triage.
  - Report confidence intervals across multiple random seeds.

### 6) Resume-first workflow and reproducibility hardening

- Priority: P1
- Return: High
- Effort: S
- Why: You already needed multiple restarts. Fast resume and deterministic tracking will save substantial time.
- Action:
  - Always set `training.init_se` from latest checkpoint for retries.
  - Keep run metadata (git commit, config hash, seed, job id) alongside checkpoints.

## P2: Strong Upside, Moderate Complexity

### 7) Parameter-efficient adapters (LoRA) on denoiser blocks

- Priority: P2
- Return: High
- Effort: M-L
- Why: Lets you iterate faster across domains with less overfitting and lower compute.
- Action:
  - Add LoRA to attention and/or MLP projection layers.
  - Tune rank and alpha for best quality/compute tradeoff.

### 8) Noise schedule and self-conditioning ablations

- Priority: P2
- Return: Medium-High
- Effort: M
- Why: Diffusion quality can be sensitive to scheduler and self-conditioning weight.
- Action:
  - Compare tanh schedule variants and step counts.
  - Run ablation with/without self-conditioning for stability.

### 9) Decoder fine-tuning on noisy latent targets

- Priority: P2
- Return: Medium-High
- Effort: M
- Why: Better decode robustness can improve generated sequence quality from imperfect denoiser latents.
- Action:
  - Train decoder with latent perturbation curriculum.
  - Validate with reconstruction and generation quality metrics.

## P3: Infrastructure and Throughput Improvements

### 10) Multi-GPU DDP scaling for throughput

- Priority: P3
- Return: Medium
- Effort: M
- Why: Good for speed, not first-order quality gains.
- Action:
  - Move stable single-GPU recipe to 4-GPU DDP.
  - Confirm identical metrics at small scale before long jobs.

### 11) Data pipeline optimization

- Priority: P3
- Return: Medium
- Effort: S-M
- Why: Improves wall-clock efficiency and reduces intermittent IO stalls.
- Action:
  - Tune dataloader workers/prefetch/persistent workers.
  - Keep HF cache and temporary artifacts on local scratch with robust sync back.

## Recommended Execution Order (Fastest Path to Credible Results)

1. P0.1 encoder stats
2. P0.2 decoder initialization/training
3. P0.3 NaN stabilization
4. Re-run baseline short debug (200-500 iters) and verify finite losses + non-degenerate metrics
5. Re-run baseline full training
6. P1.4 domain-adaptive fine-tuning
7. P1.5 metric reliability checks and baseline-vs-domain report
8. P2/P3 upgrades as time allows

## Success Criteria for "Good Baseline"

- Training loss finite throughout run (no NaN/Inf).
- `esm_pppl` is finite.
- `plddt` is non-zero and stable across eval checkpoints.
- Checkpoint resume works from any save interval.
- Repeated run with same seed gives close metrics.
