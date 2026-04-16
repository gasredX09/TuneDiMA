# Experiment Registry (Audit-Derived)

Date: 2026-04-14
Evidence sources: logs/*.out, logs/*.err, artifacts/*, README_PROGRESS.md, EVALUATION_SUMMARY.md

Legend:
- [Implemented+Run]
- [Implemented-NotRun]
- [Run-MetricsMissing]
- [Partial]
- [Dead/Stale]

## 1) Job-Backed Runs (Logs Present)

| Job | Run Name | Variant | Key Config Signals | Last Metrics Found | Outcome | Class |
|---|---|---|---|---|---|---|
| 38051400 | decoder_pretrain_steps50k (early) | decoder pretrain | decoder job | none | cancelled by scheduler | [Partial] |
| 38052727 | decoder_pretrain_steps50k (retry) | decoder pretrain | decoder job | none | timed out | [Partial] |
| 38060424 | decoder_pretrain_steps50k | decoder pretrain | DECODER_MAX_STEPS=50000 | decoder ckpts saved | completed | [Implemented+Run] |
| 38065104 | sanity_decoder_retrain | sanity retrain | training_iters=500 | no final metric line | checkpoint 250 saved, then OOM | [Partial] |
| 38065249 | domain_adaptive_denoiser | domain adaptive | training_iters=5000 | fid 6.81547, mmd 1.10004, esm_pppl 567.46172, plddt 32.89318 | partial then OOM | [Partial] |
| 38067100 | domain_adaptive_denoiser | domain adaptive resume | init_se=.../1000.pth | fid 8.80376, mmd 1.49501, esm_pppl 244.63275, plddt 35.59927 | completed to 5000 ckpt | [Implemented+Run] |
| 38073338 | unknown | cancelled submission | n/a | none | cancelled | [Partial] |
| 38073350 | reference_dima_eval | reference eval early | training_iters=5000 | fid 6.91203, mmd 1.18165, esm_pppl nan, plddt 0.0 | not valid quality | [Run-MetricsMissing] |
| 38075574 | reference_dima_eval | reference eval retry | loaded checkpoint | fid 6.91203, mmd 1.18165, esm_pppl nan, plddt 0.0 | degenerate metrics | [Run-MetricsMissing] |
| 38075630 | reference_dima_eval | reference eval retry | loaded checkpoint | fid 6.91203, mmd 1.18165, esm_pppl nan, plddt 0.0 | degenerate metrics | [Run-MetricsMissing] |
| 38101345 | reference_dima_eval | reference eval retry | loaded checkpoint | fid 6.91203, mmd 1.18165, esm_pppl nan, plddt 0.0 | degenerate metrics | [Run-MetricsMissing] |
| 38105384 | reference_dima_eval | reference eval retry | loaded checkpoint | fid 6.91203, mmd 1.18165, esm_pppl nan, plddt 0.0 | degenerate metrics | [Run-MetricsMissing] |
| 38106895 | reference_dima_eval | reference eval retry | loaded checkpoint | fid 6.91203, mmd 1.18165, esm_pppl nan, plddt 0.0 | degenerate metrics | [Run-MetricsMissing] |
| 38107929 | reference_dima_eval | reference eval improved | loaded checkpoint | fid 23.65436, mmd 3.50310, esm_pppl 1.00419, plddt 61.97135 | valid | [Implemented+Run] |
| 38110284 | reference_dima_eval | reference eval validated | loaded checkpoint | fid 23.68523, mmd 3.50680, esm_pppl 1.00430, plddt 61.83303 | valid | [Implemented+Run] |
| 38112128 | reference_dima_eval | reference eval validated | loaded checkpoint | fid 23.68523, mmd 3.50680, esm_pppl 1.00430, plddt 61.83303 | valid | [Implemented+Run] |
| 38128244 | selected_step4500_eval | selected ckpt eval | loaded checkpoint | fid 23.78363, mmd 3.52037, esm_pppl 1.00457, plddt 58.77998 | valid | [Implemented+Run] |
| 38134120 | ft_last2_from_ref_5k_r6_noamp_lr3e6_noselfcond | partial FT | ft_mode=last_n, ft_last_n_layers=2, use_amp=0 | none | runtime shape mismatch | [Partial] |
| 38134328 | ft_last2_from_ref_5k_r7_noamp_lr3e6_noselfcond | partial FT | ft_mode=last_n, ft_last_n_layers=2, use_amp=0 | none | non-finite total_loss at step 1 | [Partial] |
| 38134593 | ft_last2_from_ref_5k_r8_noamp_lr3e6_noselfcond_safe | partial FT safe | ft_mode=last_n, ft_last_n_layers=2, use_amp=0 | fid 23.68523, mmd 3.50680, esm_pppl 1.00430, plddt 61.83303 | completed 5000 | [Implemented+Run] |
| 38145569 | baseline_full_ft_single_gpu | long baseline retrain | training_iters=20000 | none | non-finite total_loss at step 398 | [Partial] |
| 38147436 | eval_only_nan_guard_5000_v2 | eval-only guard | eval_only=1, init_se set | fid 23.68523, mmd 3.50680, esm_pppl 1.00430, plddt 61.83303 | completed eval | [Implemented+Run] |
| 38147492 | eval_only_nan_guard_5000_v3_safe | eval-only guard safe | eval_only=1, use_amp=0, init_se set | fid 23.68523, mmd 3.50680, esm_pppl 1.00430, plddt 61.83303 | completed eval | [Implemented+Run] |

## 2) Artifact-Only Runs (No Complete Matching Log Evidence In Current logs/)

| Run Directory | Evidence In Artifacts | Notes | Class |
|---|---|---|---|
| debug_1g | checkpoints at 500,1000,1500,2000 + generated jsons | has outputs but no matching retained full run narrative in logs set | [Run-MetricsMissing] |
| ft_last2_from_ref_5k | 5000 checkpoint only | no metrics log found in logs/ | [Run-MetricsMissing] |
| ft_last2_from_ref_5k_r2 | 5000 checkpoint only | no metrics log found in logs/ | [Run-MetricsMissing] |
| ft_last2_from_ref_5k_r3 | staged stats/decoder only | no diffusion checkpoint | [Partial] |
| ft_last2_from_ref_5k_r4 | minimal artifacts only | appears not completed here | [Partial] |
| ft_last2_from_ref_5k_r5_noamp_lr1e5 | staged stats/decoder only | appears not completed here | [Partial] |
| ft_last4_from_ref_5k | 5000 checkpoint only | no metrics log found in logs/ | [Run-MetricsMissing] |
| ft_last4_from_ref_5k_r2 | 5000 checkpoint only | no metrics log found in logs/ | [Run-MetricsMissing] |

## 3) Run Infrastructure Presence (No Verified Completed Run In Current Evidence Set)

| Path | Purpose | Class |
|---|---|---|
| scripts/launch_baseline_multi_gpu.sh | multi-GPU training launcher | [Implemented-NotRun] |
| slurm/train_baseline_multi_gpu.sbatch | multi-GPU sbatch wrapper | [Implemented-NotRun] |
| slurm/train_ablation_ft_lastn_single_gpu.sbatch | dedicated ablation sbatch | [Dead/Stale] (path points to sdhanuka workspace) |
| slurm/train_ablation_selfcond_single_gpu.sbatch | dedicated ablation sbatch | [Dead/Stale] (path points to sdhanuka workspace) |
| slurm/train_ablation_noise_schedule_single_gpu.sbatch | dedicated ablation sbatch | [Dead/Stale] (path points to sdhanuka workspace) |

