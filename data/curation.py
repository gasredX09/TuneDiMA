import numpy as np
import pandas as pd
from pathlib import Path
from ..config import RAW_ROOT

UNIT_CONVERSION = {
    "M": 1e9,
    "mM": 1e6,
    "µM": 1e3,
    "uM": 1e3,
    "nM": 1.0,
}


def to_nM(value, unit):
    if pd.isna(value) or pd.isna(unit):
        return None
    unit = str(unit).strip()
    if unit in UNIT_CONVERSION:
        return float(value) * UNIT_CONVERSION[unit]
    return None


def compute_p_value(value_nM):
    return -np.log10(value_nM * 1e-9)


def standardize_assay_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["qualifier"].isin(["=", "<", ">"])]
    df = df.dropna(subset=["value", "unit"])
    df["value_nM"] = df.apply(lambda r: to_nM(r["value"], r["unit"]), axis=1)
    df = df.dropna(subset=["value_nM"])
    df["p_value"] = df["value_nM"].apply(compute_p_value)
    return df


def drop_duplicate_compound_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("p_value", ascending=False)
    return df.drop_duplicates(subset=["ligand_id", "target_id"], keep="first")


def map_targets_to_uniprot(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    df = df.copy()
    df["target_uniprot"] = df["target_id"].map(mapping)
    return df.dropna(subset=["target_uniprot"])


def load_assay_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported assay format: {path}")
