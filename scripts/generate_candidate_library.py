#!/usr/bin/env python3
"""Generate candidate small molecules from known binders.

This script builds a novel candidate library using BRICS recombination from a seed set
extracted from training_manifest.csv, then applies the existing drug-likeness filters.
Output can be passed into design_and_screen.py as --ligand-library.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, BRICS

from screening import filter_library

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "training_manifest.csv"


PROTEIN_CLASS_KEYWORDS = {
    "kinase": ["KINASE", "CDK", "MAPK", "ERK", "EGFR", "FGFR", "JAK", "ABL", "AKT", "PI3K", "BTK", "SRC", "RAF", "MEK"],
    "glutamate": ["GLUT", "MGLU", "GRM", "NMDA", "AMPA", "KAINATE"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate molecules from known binders")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Input manifest with ligand_smiles/target_name/label_value")
    parser.add_argument("--target-query", action="append", default=[], help="Target keyword filter. Can be passed multiple times.")
    parser.add_argument("--protein-class", choices=["kinase", "glutamate"], default=None, help="Apply predefined target-name keywords.")
    parser.add_argument("--label-direction", choices=["higher", "lower"], default="higher", help="Whether higher or lower label means stronger binders.")
    parser.add_argument("--seed-top-k", type=int, default=300, help="Top ranked canonical seed ligands to use for BRICS.")
    parser.add_argument("--max-generated", type=int, default=3000, help="Maximum BRICS-generated molecules.")
    parser.add_argument("--max-library", type=int, default=5000, help="Maximum final deduplicated filtered library size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prefix", type=str, default="generated_candidates", help="Output filename prefix")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed", help="Output directory")
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

    rng = random.Random(seed)
    remaining = list(range(len(valid)))
    first = rng.choice(remaining)
    picked = [first]
    remaining.remove(first)

    while remaining and len(picked) < k:
        best_idx = None
        best_min_dist = -1.0
        for idx in remaining:
            fp_i = valid[idx][1]
            max_sim = 0.0
            for p in picked:
                sim = DataStructs.TanimotoSimilarity(fp_i, valid[p][1])
                if sim > max_sim:
                    max_sim = sim
            min_dist = 1.0 - max_sim
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        picked.append(best_idx)
        remaining.remove(best_idx)

    return [valid[i][0] for i in picked]


def build_keywords(target_queries: list[str], protein_class: str | None) -> list[str]:
    keywords = [k.strip().upper() for k in target_queries if k.strip()]
    if protein_class:
        keywords.extend(PROTEIN_CLASS_KEYWORDS.get(protein_class, []))
    return list(dict.fromkeys(keywords))


def load_seed_smiles(manifest: Path, keywords: list[str], label_direction: str, top_k: int) -> list[str]:
    frame = pd.read_csv(manifest)
    req = {"ligand_smiles", "label_value", "target_name"}
    missing = req.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    frame = frame.dropna(subset=["ligand_smiles", "label_value", "target_name"]).copy()
    frame["label_value"] = pd.to_numeric(frame["label_value"], errors="coerce")
    frame = frame[np.isfinite(frame["label_value"].to_numpy())]

    if keywords:
        target_upper = frame["target_name"].astype(str).str.upper()
        mask = target_upper.apply(lambda text: any(k in text for k in keywords))
        frame = frame[mask]

    ascending = label_direction == "lower"
    frame = frame.sort_values("label_value", ascending=ascending)

    raw = [str(x).strip() for x in frame["ligand_smiles"].tolist() if str(x).strip()]
    canonical = [canonicalize_smiles(s) for s in raw]
    canonical = [s for s in canonical if s is not None]
    dedup = list(dict.fromkeys(canonical))

    if not dedup:
        raise ValueError("No valid seed ligands found after filtering")

    return dedup[: min(len(dedup), top_k)]


def generate_brics(seed_smiles: list[str], max_generated: int, seed: int) -> list[str]:
    if max_generated <= 0 or not seed_smiles:
        return []

    fragments: list[Chem.Mol] = []
    for smi in seed_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            frag_set = BRICS.BRICSDecompose(mol)
        except Exception:
            continue
        for frag in frag_set:
            frag_mol = Chem.MolFromSmiles(frag)
            if frag_mol is not None:
                fragments.append(frag_mol)

    if not fragments:
        return []

    rng = random.Random(seed)
    rng.shuffle(fragments)
    fragments = fragments[: min(len(fragments), 2000)]

    seen = set(seed_smiles)
    generated: list[str] = []
    try:
        for mol in BRICS.BRICSBuild(fragments, maxDepth=3):
            smi = Chem.MolToSmiles(mol, canonical=True)
            if smi in seen:
                continue
            seen.add(smi)
            generated.append(smi)
            if len(generated) >= max_generated:
                break
    except Exception:
        pass

    return generated


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    keywords = build_keywords(args.target_query, args.protein_class)
    seed_smiles = load_seed_smiles(args.manifest, keywords, args.label_direction, args.seed_top_k)
    generated = generate_brics(seed_smiles, args.max_generated, args.seed)

    combined = list(dict.fromkeys(seed_smiles + generated))
    combined = filter_library(combined)
    if len(combined) > args.max_library:
        combined = choose_diverse(combined, args.max_library, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_smi = args.output_dir / f"{args.prefix}.smi"
    out_info = args.output_dir / f"{args.prefix}_info.txt"

    out_smi.write_text("\n".join(combined) + ("\n" if combined else ""))

    info_lines = [
        f"manifest: {args.manifest}",
        f"protein_class: {args.protein_class}",
        f"keywords: {';'.join(keywords) if keywords else '(none)'}",
        f"label_direction: {args.label_direction}",
        f"seed_top_k: {args.seed_top_k}",
        f"seed_count_used: {len(seed_smiles)}",
        f"generated_count: {len(generated)}",
        f"final_library_count: {len(combined)}",
        f"output_smi: {out_smi}",
    ]
    out_info.write_text("\n".join(info_lines) + "\n")

    print(f"Wrote generated candidate library: {out_smi}")
    print(f"Final library size: {len(combined)}")


if __name__ == "__main__":
    main()
