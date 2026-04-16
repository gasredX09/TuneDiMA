# Next Experiments (Evidence-Prioritized)

Date: 2026-04-14
Constraint basis: current repository state, logs, artifacts, and completed metrics in docs/metrics_summary.csv.

## Priority 1: Must Do Next

### P1.1 Self-conditioning on/off controlled ablation from validated reference checkpoint
- Why now:
  - Existing code supports model.config.use_self_cond toggle in launchers.
  - Current best stable baseline is reference_dima_eval (38112128 style setup).
- Minimal design:
  - Two eval-only runs with identical init checkpoint, decoder, and generation settings:
    - use_self_cond=1
    - use_self_cond=0
- Expected novelty:
  - Medium-high (architectural insight from one switch with low confounders).
- Effort:
  - Low (launcher/env override only).
- Compute cost:
  - Low (~single eval pass each).
- Story impact:
  - Professor evaluation: High
  - Final report: High
  - Resume/PhD/internship: Medium

### P1.2 Last-N fine-tuning sweep with stabilized recipe
- Why now:
  - FT path exists and was partially unstable (r6/r7 failed, r8 succeeded).
  - High value to show controlled adaptation vs forgetting tradeoff.
- Minimal design:
  - N in {1,2,4}, fixed init checkpoint (reference 5000), same decoder and eval settings.
  - Keep use_amp and LR to known-stable settings from successful r8-safe lineage.
- Expected novelty:
  - High (direct PEFT-style adaptation study on DiMA).
- Effort:
  - Medium (3 runs + table).
- Compute cost:
  - Medium.
- Story impact:
  - Professor evaluation: High
  - Final report: High
  - Resume/PhD/internship: High

### P1.3 Replay-mix anti-forgetting experiment
- Why now:
  - Replay controls implemented but no clear run evidence.
  - Addresses observed reference-generalization drop after adaptation.
- Minimal design:
  - replay_ratio in {0.05, 0.1}, same FT settings as stable run.
- Expected novelty:
  - High (methodological contribution beyond pure hyperparameter tuning).
- Effort:
  - Medium.
- Compute cost:
  - Medium.
- Story impact:
  - Professor evaluation: High
  - Final report: High
  - Resume/PhD/internship: High

## Priority 2: Nice To Have

### P2.1 True noise schedule ablation after wiring fix
- Why not must-do yet:
  - Current script appears to pass an override key not consumed by active pipeline.
- Prerequisite:
  - Correct scheduler override path and validate via printed config in logs.
- Expected novelty:
  - Medium.
- Effort:
  - Medium (fix + run).
- Compute cost:
  - Medium.
- Story impact:
  - Professor evaluation: Medium
  - Final report: Medium
  - Resume/PhD/internship: Medium

### P2.2 Multi-GPU throughput sanity benchmark
- Why:
  - multi-GPU path exists but not clearly evidenced by completed logs in this clone.
- Expected novelty:
  - Low scientific, medium engineering.
- Effort:
  - Low-medium.
- Compute cost:
  - Medium.
- Story impact:
  - Professor evaluation: Medium
  - Final report: Medium
  - Resume/PhD/internship: Medium

## Priority 3: Avoid / Low ROI (for now)

### P3.1 Large broad hyperparameter sweeps before completing core ablations
- Rationale:
  - Existing evidence still has unresolved stability and missing ablation coverage.
- Novelty contribution:
  - Low relative to compute.
- Effort/cost:
  - High compute, low interpretability gain.

### P3.2 New architecture branch (major block redesign) before finishing controlled comparisons
- Rationale:
  - Current project already has strong story potential via robust controlled adaptation and anti-forgetting tests.
- Novelty contribution:
  - Potentially high but high risk for timeline.
- Effort/cost:
  - High implementation and debugging cost.

## Suggested Immediate Execution Order

1. Self-cond on/off eval-only pair from reference checkpoint.
2. Last-N sweep (N=1,2,4) with stable settings.
3. Replay-ratio sweep (0.05, 0.1) on best Last-N setting.
4. Optional: noise-schedule ablation only after override wiring is confirmed in logs.

