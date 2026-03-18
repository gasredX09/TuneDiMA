#!/usr/bin/env python3
import os
import runpy
import sys


def _should_disable_wandb() -> bool:
    val = os.getenv("DISABLE_WANDB", "1").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _patch_wandb() -> None:
    import wandb

    original_init = wandb.init

    def patched_init(*args, **kwargs):
        # DiMA hardcodes mode='online'; force disabled mode for no-login HPC runs.
        kwargs["mode"] = "disabled"
        return original_init(*args, **kwargs)

    wandb.init = patched_init


if __name__ == "__main__":
    # Keep behavior close to `python train_diffusion.py` by ensuring cwd (DiMA root)
    # is importable for `src.*` modules.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    if _should_disable_wandb():
        _patch_wandb()

    # The launchers change into DiMA root before calling this script.
    runpy.run_path("train_diffusion.py", run_name="__main__")
