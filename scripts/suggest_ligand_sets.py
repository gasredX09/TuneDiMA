#!/usr/bin/env python3
"""Build small, diverse ligand seed sets for a specific protein or target class.

The script pulls candidates from an existing training manifest, filters by target keywords,
ranks by label value, and writes top-N diverse SMILES files (20/50/100 by default).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "training_manifest.csv"

PROTEIN_CLASS_KEYWORDS = {
    "kinase": ["KINASE", "CDK", "MAPK", "ERK", "EGFR", "FGFR", "JAK", "ABL", "AKT", "PI3K", "BTK", "SRC", "RAF", "MEK"],
    "glutamate": ["GLUT", "MGLU", "GRM", "NMDA", "AMPA", "KAINATE"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small ligand seed sets from training manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Input manifest with ligand_smiles/target_name/label_value")
    parser.add_argument("--target-query", action="append", default=[], help="Target keyword filter. Can be passed multiple times.")
    parser.add_argument("--protein-class", choices=["kinase", "glutamate"], default=None, help="Apply predefined target-name keywords.")
    parser.add_argument("--label-direction", choices=["higher", "lower"], default="higher", help="Whether higher or lower label means stronger binders.")
    parser.add_argument("--preselect", type=int, default=1000, help="Number of top-ranked rows to consider before diversity selection.")
    parser.add_argument("--set-sizes", type=int, nargs="+", default=[20, 50, 100], help="Output set sizes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for tie-break behavior.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed", help="Directory for output files.")
    parser.add_argument("--prefix", type=str, default="ligand_set", help="Output filename prefix.")
    return parser.parse_args()


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def choose_diverse(smiles_list: list[str], k: int, seed: int) -> list[str]:
    uniq = list(dict.fromkeys(smiles_list))
    if len(uniq) <= k:
        return uniq

    mols = [Chem.MolFromSmiles(s) for s in uniq]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m is not None else None for m in mols]
    valid = [(s, fp) for s, fp in zip(uniq, fps) if fp is not None]

    if len(valid) <= k:
        return [s for s, _ in valid]

    rng = np.random.default_rng(seed)
    remaining = list(range(len(valid)))
    picked = [int(rng.choice(remaining))]
    remaining.remove(picked[0])

    while remaining and len(picked) < k:
        best_idx = None
        best_dist = -1.0
        for idx in remaining:
            fp_i = valid[idx][1]
            max_sim = 0.0
            for p in picked:
                sim = DataStructs.TanimotoSimilarity(fp_i, valid[p][1])
                if sim > max_sim:
                    max_sim = sim
            dist = 1.0 - max_sim
            if dist > best_dist:
                best_dist = dist
                best_idx = idx
        picked.append(best_idx)
        remaining.remove(best_idx)

    return [valid[i][0] for i in picked]


def build_keyword_list(args: argparse.Namespace) -> list[str]:
    keywords = [k.strip().upper() for k in args.target_query if k.strip()]
    if args.protein_class:
        keywords.extend(PROTEIN_CLASS_KEYWORDS.get(args.protein_class, []))
    return list(dict.fromkeys(keywords))


def filter_manifest(frame: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    out = frame.dropna(subset=["ligand_smiles", "target_name", "label_value"]).copy()
    out["label_value"] = pd.to_numeric(out["label_value"], errors="coerce")
    out = out[np.isfinite(out["label_value"].to_numpy())]

    if keywords:
        target_upper = out["target_name"].astype(str).str.upper()
        mask = target_upper.apply(lambda text: any(k in text for k in keywords))
        out = out[mask]

    return out


def aggregate_best_by_smiles(frame: pd.DataFrame, direction: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["canonical_smiles"] = frame["ligand_smiles"].astype(str).map(canonicalize_smiles)
    frame = frame.dropna(subset=["canonical_smiles"])

    ascending = direction == "lower"
    frame = frame.sort_values("label_value", ascending=ascending)

    # Keep one best row per canonical SMILES for traceability.
    best = frame.groupby("canonical_smiles", as_index=False).first()
    best = best.sort_values("label_value", ascending=ascending)
    return best


def write_outputs(best: pd.DataFrame, args: argparse.Namespace, keywords: list[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / f"{args.prefix}_summary.csv"
    best.to_csv(summary_path, index=False)

    all_smiles = best["canonical_smiles"].tolist()
    if not all_smiles:
        raise ValueError("No valid ligands left after filtering and canonicalization")

    for n in sorted(set(args.set_sizes)):
        picked = choose_diverse(all_smiles[: min(len(all_smiles), args.preselect)], k=min(n, len(all_smiles)), seed=args.seed)
        out_file = args.output_dir / f"{args.prefix}_top{len(picked)}.smi"
        out_file.write_text("\n".join(picked) + "\n")

    info_path = args.output_dir / f"{args.prefix}_info.txt"
    info_lines = [
        f"manifest: {args.manifest}",
        f"rows_total: {len(best)}",
        f"protein_class: {args.protein_class}",
        f"target_keywords: {';'.join(keywords) if keywords else '(none)'}",
        f"label_direction: {args.label_direction}",
        f"preselect: {args.preselect}",
        f"set_sizes: {','.join(map(str, sorted(set(args.set_sizes))))}",
    ]
    info_path.write_text("\n".join(info_lines) + "\n")



def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    frame = pd.read_csv(args.manifest)
    keywords = build_keyword_list(args)
    filtered = filter_manifest(frame, keywords)
    best = aggregate_best_by_smiles(filtered, args.label_direction)
    write_outputs(best, args, keywords)

    print(f"Wrote ligand seed sets in: {args.output_dir}")
    print(f"Candidate unique ligands: {len(best)}")


if __name__ == "__main__":
    main()
