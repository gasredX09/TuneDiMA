from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"

DEFAULTS = {
    "hidden_dim": 128,
    "num_layers": 3,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 20,
}

SEED = 42
