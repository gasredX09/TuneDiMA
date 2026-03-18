# DiMA Course Project Wrapper

This directory is a professional wrapper around `DiMA/` for reproducible experiments on HPC.

No source code changes are required inside `DiMA/` to run baseline experiments from this wrapper.

## Directory Layout

- `DiMA/`: upstream model repository (kept unchanged)
- `scripts/`: local launch scripts used by both interactive and SLURM runs
- `slurm/`: sbatch job files
- `configs/`: experiment-level environment defaults
- `runs/`: per-run notes
- `reports/`: evaluation summaries
- `artifacts/`: checkpoints and generated outputs per run
- `logs/`: SLURM stdout/stderr

## First Run (Single GPU via SLURM)

1. Edit resource directives in `slurm/train_baseline_single_gpu.sbatch` if needed.
2. Submit:

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sbatch slurm/train_baseline_single_gpu.sbatch
```

## First Run (Multi GPU via SLURM)

```bash
cd /ocean/projects/cis260039p/aguda1/nndl/project
sbatch slurm/train_baseline_multi_gpu.sbatch
```

## Step-by-step workflow

1. Run baseline.
2. Record metrics in `runs/RUN_TEMPLATE.md` (copy to a dated file).
3. Compare against previous best.
4. Apply one modification at a time.
