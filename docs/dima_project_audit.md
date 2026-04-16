# TuneDiMA Audit (Evidence-Driven)

Date: 2026-04-14
Repository root audited: /ocean/projects/cis260039p/aguda1/nndl/project
Baseline reference commit used for diffs: c6edb19 ("add base files")

## A. Executive Summary

Current status:
- The repository is a hardened DiMA training/evaluation pipeline for HPC (single-GPU and multi-GPU launchers, checkpoint staging, cache relocation, decoder/statistics side workflows).
- The strongest verified comparison result in this repo is the validated reference checkpoint run (reference_dima_eval, job 38112128): fid 23.68523, mmd 3.50680, esm_pppl 1.00430, plddt 61.83303.
- Domain-adaptive run completed to step 5000, but in the apples-to-apples comparison the selected domain-adaptive checkpoint (selected_step4500_eval) was slightly worse than reference on all 4 metrics.

Definitely implemented:
- Baseline diffusion training/evaluation pipeline and metrics path.
- Decoder pretraining pipeline with resume/checkpoint intervals.
- Encoder statistics generation.
- Fine-tuning controls (ft_mode=full/last_n), replay-mix controls, eval_only mode, init checkpoint handling fixes.
- Evaluation robustness fixes (sequence sanitization + decoder fallback + non-finite guards).
- Ablation launchers for last-N FT, self-conditioning toggle, and noise-schedule (script exists; see caveat below).

Definitely run (with evidence in logs/artifacts):
- Decoder pretraining to 50k (with intermediate checkpoints).
- Sanity retrain, domain-adaptive runs, multiple reference-eval retries, selected-step comparability eval.
- Partial FT attempts (several failed/partial, one completed to 5000 with metrics).
- Eval-only nan-guard validation runs.

Biggest confirmed gap:
- No completed/validated run from the newly added dedicated ablation sbatch scripts in slurm/train_ablation_*.sbatch. Existing successful runs appear to come from baseline single-GPU path with env overrides.

## B. Repository Structure Relevant To This Project

Core code:
- DiMA/src/models
- DiMA/src/diffusion
- DiMA/src/encoders
- DiMA/src/metrics
- DiMA/src/configs

Training and execution entrypoints:
- DiMA/train_diffusion.py
- scripts/run_dima_train.py
- scripts/launch_baseline_single_gpu.sh
- scripts/launch_baseline_multi_gpu.sh
- scripts/launch_train_decoder.sh
- scripts/launch_calculate_statistics.sh
- scripts/launch_ablation_ft_lastn.sh
- scripts/launch_ablation_selfcond.sh
- scripts/launch_ablation_noise_schedule.sh

Schedulers and job wrappers:
- slurm/train_baseline_single_gpu.sbatch
- slurm/train_baseline_multi_gpu.sbatch
- slurm/train_decoder_single_gpu.sbatch
- slurm/calculate_statistics_single_gpu.sbatch
- slurm/train_ablation_ft_lastn_single_gpu.sbatch
- slurm/train_ablation_selfcond_single_gpu.sbatch
- slurm/train_ablation_noise_schedule_single_gpu.sbatch

Outputs and evidence:
- artifacts/*
- logs/*.out, logs/*.err
- EVALUATION_SUMMARY.md
- README_PROGRESS.md
- experiment_logs/*

Notebook evidence:
- DiMA/example.ipynb

## C. Baseline DiMA Summary (Original Code Path)

Entry and orchestration:
- DiMA/train_diffusion.py constructs BaseDiffusionTrainer and calls train().

Baseline model architecture:
- Score estimator: DiMA/src/models/score_estimator.py, class ScoreEstimator.
- Transformer blocks/attention: DiMA/src/models/blocks.py, classes BertBlock, BertAttention, FlashAttention.

Diffusion settings and sampling:
- Dynamics: DiMA/src/diffusion/dynamic.py, class DynamicSDE.
- Schedulers: DiMA/src/diffusion/schedulers.py, classes Linear, Tanh.
- Solvers: DiMA/src/diffusion/solvers.py, class EulerDiffEqSolver (default in config).

Conditioning:
- Time conditioning via timestep_embedding and time MLP in ScoreEstimator.
- Optional self-conditioning via model.config.use_self_cond in ScoreEstimator/TransformerEncoder.
- Cross-attention exists in block implementation but default model config sets add_cross_attention: false.

Losses:
- Diffusion training loss is MSE over latent reconstructions in BaseDiffusionTrainer.calc_loss using mse_loss from DiMA/src/utils/training_utils.py.
- Decoder pretraining reconstruction loss uses reconstruction_loss in DiMA/src/utils/training_utils.py.

Pretrained/loading behavior:
- Auto-resume from latest checkpoint folder in BaseDiffusionTrainer.load_checkpoint().
- Explicit init checkpoint in BaseDiffusionTrainer.init_checkpoint() via training.init_se path.

Default config/pipeline:
- DiMA/src/configs/config.yaml with defaults: model=bert_base, encoder=esm2, decoder=transformer, scheduler=tanh, dynamic=sde, solver=euler, metrics=default.

## D. Implemented Modifications (Compared To Baseline Commit c6edb19)

Status tags used:
- [Implemented+Run]
- [Implemented-NotRun]
- [Run-MetricsMissing]
- [Partial]
- [Dead/Stale]

### D1) Fine-tuning mode controls (last-N layer training)
- What changed: Added ft_mode and ft_last_n_layers support, applies requires_grad masking to selected output/time/self-cond layers and output projector.
- Where: DiMA/src/diffusion/base_trainer.py, methods _apply_finetune_mode(), _setup_training_utils(). Also config fields in DiMA/src/configs/config.yaml.
- Active path: Yes (called every train() via _setup_training_utils()).
- Classification: [Implemented+Run] (jobs 38134120/38134328/38134593 attempt/use these settings).

### D2) Replay-mix training data support
- What changed: Optional replay dataset concatenation with configurable ratio and seed.
- Where: DiMA/src/diffusion/base_trainer.py, method _setup_train_data_generator(); config fields replay_data_dir, replay_ratio, replay_seed in DiMA/src/configs/config.yaml; launcher wiring in scripts/launch_baseline_single_gpu.sh and ablation scripts.
- Active path: Yes when replay args set.
- Classification: [Implemented-NotRun] (no clear log evidence in current logs of replay mix enabled line).

### D3) Eval-only mode and init-se resume behavior fix
- What changed: Added eval_only mode to run validate()+training_estimation() without optimizer updates; skip automatic latest-checkpoint resume when init_se is explicitly set.
- Where: DiMA/src/diffusion/base_trainer.py, method train().
- Active path: Yes.
- Classification: [Implemented+Run] (jobs 38147436 and 38147492 show "Eval-only mode enabled" and explicit init checkpoint).

### D4) Non-finite safeguards in training and decoding path
- What changed:
  - Detect/abort on non-finite total_loss.
  - nan_to_num sanitization for clean_X, x_t, x_0.
  - Decoder-side non-finite logits cleanup before token decode.
- Where:
  - DiMA/src/diffusion/base_trainer.py: train(), calc_loss().
  - DiMA/src/encoders/esm2.py: _decode_logits_to_sequences().
- Active path: Yes.
- Classification: [Implemented+Run] (non-finite warning present in 38147436/38147492; non-finite loss abort in 38134328/38145569).

### D5) Decoder fallback and robust sequence decoding
- What changed:
  - If transformer decoder checkpoint missing, fallback to ESM lm_head decoder.
  - Mask special tokens before argmax decode.
  - Empty-sequence fallback from transformer decode to lm_head decode.
- Where: DiMA/src/encoders/esm2.py.
- Active path: Yes for generation/eval.
- Classification: [Implemented+Run].

### D6) Metric sanitization for invalid amino-acid outputs
- What changed: Added _sanitize_sequences to filter invalid chars and empties before plddt and esm_pppl.
- Where: DiMA/src/metrics/metric.py.
- Active path: Yes in compute_ddp_metric().
- Classification: [Implemented+Run].

### D7) Attention mask length fix for sampled sequence lengths
- What changed: Added +2 token adjustment (CLS/EOS) and simpler vectorized mask filling.
- Where: DiMA/src/encoders/base.py, get_attention_mask_for_lens().
- Active path: Yes for sample generation.
- Classification: [Implemented+Run].

### D8) EncNormalizer device/dtype safety fix
- What changed: enc_mean and enc_std moved to input device/dtype before normalize/denormalize operations.
- Where: DiMA/src/encoders/enc_normalizer.py.
- Active path: Yes.
- Classification: [Implemented+Run].

### D9) Decoder training hardening
- What changed:
  - Frozen encoder no_grad during decoder training.
  - AMP dtype fallback bfloat16->float16.
  - Resume support and periodic checkpointing (decoder_step_N, decoder_last).
  - Env-driven worker/batch/max-step overrides.
  - Optional wandb disable.
- Where: DiMA/src/preprocessing/train_decoder.py; scripts/launch_train_decoder.sh; slurm/train_decoder_single_gpu.sbatch.
- Active path: Yes.
- Classification: [Implemented+Run] (38060424 reached DECODER_MAX_STEPS=50000).

### D10) Statistics generation hardening
- What changed: AMP fallback and explicit CLI args for project/data path.
- Where: DiMA/src/preprocessing/calculate_statistics.py; scripts/launch_calculate_statistics.sh; slurm/calculate_statistics_single_gpu.sbatch.
- Active path: Yes.
- Classification: [Implemented+Run] (statistics file present at DiMA/checkpoints/statistics/encodings-ESM2-3B.pth and staged into runs).

### D11) Baseline launcher and SLURM hardening
- What changed:
  - Cache relocation (HF_HOME, datasets, transformers caches).
  - scratch-to-artifacts sync.
  - stats copy to local run folder.
  - optional reference checkpoint staging, decoder checkpoint staging.
  - numeric INIT_SE to path resolution.
  - expanded override surface (use_amp, eval_only, ft_mode, replay, etc.).
- Where:
  - scripts/launch_baseline_single_gpu.sh
  - slurm/train_baseline_single_gpu.sbatch
  - scripts/run_dima_train.py (WandB disable patch wrapper)
- Active path: Yes.
- Classification: [Implemented+Run].

### D12) Ablation launchers and sbatch scripts
- What changed: Added dedicated scripts/sbatch for ft_lastn, selfcond, noise_schedule ablations.
- Where:
  - scripts/launch_ablation_*.sh
  - slurm/train_ablation_*_single_gpu.sbatch
- Active path:
  - Launcher code exists.
  - No definitive evidence in current logs that these sbatch files themselves were used successfully.
- Classification: [Partial].

### D13) Noise schedule ablation wiring issue
- What changed: launch_ablation_noise_schedule.sh uses override token "++generation.noise_schedule", but generation.noise_schedule is not part of the baseline config path and this key is not used in trainer/solver code.
- Where: scripts/launch_ablation_noise_schedule.sh.
- Active path: Potentially broken/ineffective.
- Classification: [Dead/Stale] until validated.

### D14) Alternate project-path sbatch scripts likely stale in this clone
- What changed: ablation sbatch scripts point to /ocean/projects/cis260039p/sdhanuka/TuneDiMA instead of this repo path.
- Where:
  - slurm/train_ablation_ft_lastn_single_gpu.sbatch
  - slurm/train_ablation_selfcond_single_gpu.sbatch
  - slurm/train_ablation_noise_schedule_single_gpu.sbatch
- Active path: Not valid for this clone without edits.
- Classification: [Dead/Stale].

### D15) Non-core/stale additions
- File named "=1.3" appears to contain pip-install transcript text, not executable project code.
- DiMA/src/diffusion/dima.py defines DiMAModel used by README/example notebook import path, but not used in the training launch path (train_diffusion.py + run_dima_train.py).
- Classification:
  - =1.3 -> [Dead/Stale]
  - DiMA/src/diffusion/dima.py -> [Partial] (API/demo path only)

## E. Experiments Completed So Far

### E1) Decoder pretraining series
- Jobs: 38051400, 38052727, 38060424
- Evidence:
  - logs/dima-decoder-1g-38051400.err (cancelled)
  - logs/dima-decoder-1g-38052727.err (time limit)
  - logs/dima-decoder-1g-38060424.out (reached 50000)
  - artifacts/decoder_pretrain_steps50k/decoder_checkpoints/*
- Result:
  - Final successful run resumed from step 25000 and reached DECODER_MAX_STEPS=50000.
- Classification: [Implemented+Run]

### E2) Sanity decoder retrain
- Job: 38065104
- Evidence: logs/dima-base-1g-38065104.out/.err; artifacts/sanity_decoder_retrain/checkpoints/.../250.pth
- Result:
  - Training started (500 iters), checkpoint at 250; err file reports CUDA OOM.
- Classification: [Partial]

### E3) Domain-adaptive denoiser
- Jobs: 38065249 then resumed/completed 38067100
- Evidence:
  - logs/dima-base-1g-38065249.err (OOM)
  - logs/dima-base-1g-38067100.out (checkpoints 2500..5000 saved)
  - artifacts/domain_adaptive_denoiser/checkpoints/.../500..5000.pth
  - artifacts/domain_adaptive_denoiser/generated_sequences/.../500..5000.json
- Final metrics (from 38067100): fid 8.80376, mmd 1.49501, esm_pppl 244.63275, plddt 35.59927.
- Classification: [Implemented+Run]

### E4) Reference eval attempts (multiple retries)
- Jobs: 38073350, 38075574, 38075630, 38101345, 38105384, 38106895, 38107929, 38110284, 38112128
- Evidence: logs/dima-base-1g-*.out and artifacts/reference_dima_eval/*
- Notes:
  - Earlier runs showed degenerate metrics (esm_pppl nan, plddt 0).
  - Later validated runs show stable metrics around fid~23.69, mmd~3.51, esm_pppl~1.004, plddt~61.83.
  - 38112128 is the validated reference metric set used in summary docs.
- Classification:
  - Early attempts: [Run-MetricsMissing] or [Partial] quality-wise
  - Final validated reference: [Implemented+Run]

### E5) Selected checkpoint comparability eval
- Job: 38128244 (run name selected_step4500_eval)
- Evidence:
  - logs/dima-base-1g-38128244.out
  - artifacts/selected_step4500_eval/checkpoints/.../5000.pth
  - artifacts/selected_step4500_eval/generated_sequences/.../4500.json and 5000.json
- Metrics (final in log): fid 23.78363, mmd 3.52037, esm_pppl 1.00457, plddt 58.77998.
- Classification: [Implemented+Run]

### E6) Partial FT from reference (r6/r7/r8 path)
- Jobs:
  - 38134120 (shape mismatch runtime error)
  - 38134328 (non-finite loss step 1)
  - 38134593 (completed to 5000 checkpoint)
- Evidence:
  - logs/dima-base-1g-38134120.err
  - logs/dima-base-1g-38134328.err
  - logs/dima-base-1g-38134593.out and artifacts/ft_last2_from_ref_5k_r8_noamp_lr3e6_noselfcond_safe/*
- Metrics available in 38134593 log: fid 23.68523, mmd 3.50680, esm_pppl 1.00430, plddt 61.83303.
- Classification:
  - r6/r7: [Partial]
  - r8 safe: [Implemented+Run]

### E7) Eval-only nan-guard runs
- Jobs: 38147436 (v2), 38147492 (v3_safe)
- Evidence:
  - logs/dima-base-1g-38147436.out
  - logs/dima-base-1g-38147492.out
  - experiment_logs/dima-base-1g-38147492.out
  - artifacts/eval_only_nan_guard_5000_v2 and artifacts/eval_only_nan_guard_5000_v3_safe
- Result:
  - Both produced stable reference-like metrics and demonstrate eval_only path + sanitization behavior.
- Classification: [Implemented+Run]

### E8) Baseline full FT long run
- Job: 38145569 (run baseline_full_ft_single_gpu)
- Evidence: logs/dima-base-1g-38145569.out/.err
- Result:
  - Started 20000-iters run, aborted at step 398 due to non-finite total_loss.
- Classification: [Partial]

### E9) Artifact directories with run signs but missing full log evidence in this repo clone
- Directories:
  - artifacts/ft_last2_from_ref_5k
  - artifacts/ft_last2_from_ref_5k_r2
  - artifacts/ft_last2_from_ref_5k_r3
  - artifacts/ft_last2_from_ref_5k_r4
  - artifacts/ft_last2_from_ref_5k_r5_noamp_lr1e5
  - artifacts/ft_last4_from_ref_5k
  - artifacts/ft_last4_from_ref_5k_r2
- Observation:
  - Some have step-5000 checkpoints but no matching complete logs in logs/ for metric extraction.
- Classification: [Run-MetricsMissing] or [Uncertain -> treat as Run-MetricsMissing]

## F. Metrics Summary Table (Best Available Evidence)

| Run / Job | Variant | fid | mmd | esm_pppl | plddt | Status |
|---|---|---:|---:|---:|---:|---|
| domain_adaptive_denoiser / 38067100 | domain-adaptive completed | 8.80376 | 1.49501 | 244.63275 | 35.59927 | [Implemented+Run] |
| reference_dima_eval / 38112128 | validated reference eval | 23.68523 | 3.50680 | 1.00430 | 61.83303 | [Implemented+Run] |
| selected_step4500_eval / 38128244 | selected domain-adapted ckpt eval | 23.78363 | 3.52037 | 1.00457 | 58.77998 | [Implemented+Run] |
| ft_last2_from_ref_5k_r8_noamp_lr3e6_noselfcond_safe / 38134593 | partial-FT safe rerun | 23.68523 | 3.50680 | 1.00430 | 61.83303 | [Implemented+Run] |
| eval_only_nan_guard_5000_v2 / 38147436 | eval-only guard run | 23.68523 | 3.50680 | 1.00430 | 61.83303 | [Implemented+Run] |
| eval_only_nan_guard_5000_v3_safe / 38147492 | eval-only guard run (safe) | 23.68523 | 3.50680 | 1.00430 | 61.83303 | [Implemented+Run] |
| reference_dima_eval early retries / 38073350..38106895 | pre-sanitization unstable retries | ~6.91203 | ~1.18165 | nan | 0.0 | [Run-MetricsMissing] |

Important comparability note:
- Metric magnitudes differ across experiment contexts (e.g., domain-adaptive run vs reference-eval runs). Use only explicitly matched setup pairs for conclusions.

## G. Missing Experiments / Incomplete Branches

1) Dedicated ablation sbatch scripts not validated in this clone
- slurm/train_ablation_*_single_gpu.sbatch contain fixed project path pointing to sdhanuka workspace, not this repo clone.
- Status: [Dead/Stale] until path-corrected and executed.

2) Noise schedule ablation likely ineffective/broken wiring
- scripts/launch_ablation_noise_schedule.sh passes "++generation.noise_schedule", but no active consumer in trainer/config path.
- Status: [Dead/Stale].

3) Partial FT branches with checkpoint-only evidence
- Several artifact dirs contain checkpoint(s) but no complete logs/metrics in logs/.
- Status: [Run-MetricsMissing].

4) Baseline 20k full FT currently unstable
- Non-finite loss failure at step 398 (job 38145569).
- Status: [Partial].

5) Multi-GPU baseline path exists but no clear completed run evidence in current logs set.
- scripts/launch_baseline_multi_gpu.sh and slurm/train_baseline_multi_gpu.sbatch are present.
- Status: [Implemented-NotRun] in this evidence set.

## H. Recommended Next Best 3-5 Experiments (Evidence-Constrained)

### 1) Controlled self-conditioning ablation on validated reference checkpoint
- Why next: self_cond is active in baseline and already parameterized in launchers; high explanatory value for report.
- Minimal setup: run with model.config.use_self_cond=0 and =1 under identical eval_only and generation settings.
- Expected value: isolates one architectural behavior with low engineering overhead.

### 2) Last-N sweep with clean stability recipe (N=1,2,4)
- Why next: ft_mode path is implemented and partially exercised; current evidence shows instability in some settings.
- Minimal setup: keep reference init checkpoint and decoder fixed, use conservative LR/clip/use_amp settings from stable run.
- Expected value: directly tests PEFT-like adaptation effect and produces clear ablation figure/table.

### 3) Replay-mix anti-forgetting test (small ratio)
- Why next: replay code is implemented but not evidenced as run.
- Minimal setup: replay_ratio in {0.05, 0.1} with same eval protocol as reference/selected runs.
- Expected value: strongest novelty-per-cost opportunity because it addresses observed generalization drop after adaptation.

### 4) Complete and validate true noise schedule ablation only after wiring fix
- Why next: currently likely dead path; fixing it first prevents wasted compute.
- Expected value: medium, but only after ensuring override actually changes scheduler behavior.

### 5) Optional multi-GPU throughput sanity run (short)
- Why next: infrastructure exists; useful for report reproducibility/engineering depth.
- Expected value: low scientific novelty, medium engineering evidence.

## I. Resume-Ready Project Summary

CV bullets (4):
- Built and hardened a reproducible HPC pipeline for protein latent diffusion (DiMA), including single-GPU/multi-GPU SLURM launch paths, cache relocation, and robust checkpoint staging/resume.
- Implemented evaluation robustness fixes for protein generation (decoder fallback, sequence sanitization, non-finite guards), stabilizing pLDDT and ESM-perplexity computation.
- Added parameter-efficient adaptation controls (last-N-layer fine-tuning, replay-mix hooks, eval-only mode) and debugged failure modes including OOM, shape mismatch, and NaN training.
- Executed and documented end-to-end comparability experiments between reference and adapted checkpoints with artifact-traceable metrics and run logs.

Progress-report bullets (3):
- Completed decoder pretraining (50k steps) and validated checkpointed recovery flow.
- Completed domain-adaptive denoiser training and reference/selected checkpoint comparability evaluations.
- Established evidence-based experiment registry from logs/artifacts, including failure analysis and run classification.

Novelty statement (1-2 sentences):
- The project extends a baseline DiMA protein diffusion workflow with practical robustness and adaptation controls targeted at real HPC constraints. The main novelty is an evidence-backed adaptation framework (partial FT + replay hooks + eval hardening) that turns unstable trial-and-error into reproducible, comparable experiments.

## Chronology (Git Reconstruction)

Commit order and meaning:
- c6edb19: base DiMA files and initial project scaffolding.
- ad3cd62: decoder training/statistics scripts and launcher additions.
- a3dda7c: sanity retrain updates.
- a8495a3: readme and metric/encoder updates.
- 13502b0: evaluation files added.
- a02f0ba: further trainer/config/esm2 changes and log docs.
- 1a392c4: ablation scripts and sbatch additions.
- 6a69175: merge of ablation branch into main.

Evidence files:
- git log graph from this repository
- git diff c6edb19..HEAD

