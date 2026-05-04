#!/usr/bin/env python3
"""Create design_and_screen targets CSV from FASTA IDs and a structure directory.

This helper links sequence IDs to existing PDB files. It does not predict structures.
Use AlphaFold/ESMFold/homology modeling first, then point this script to the folder
containing generated PDB files.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ALLOWED_EXTENSIONS = [".pdb", ".ent"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map FASTA IDs to PDB paths for screening")
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA file with sequence IDs")
    parser.add_argument("--pdb-dir", type=Path, required=True, help="Directory containing predicted/known PDB files")
    parser.add_argument("--output", type=Path, default=Path("data/processed/new_targets.csv"), help="Output CSV path")
    parser.add_argument("--unmatched", type=Path, default=Path("data/processed/new_targets_unmatched.txt"), help="Output text file for IDs without PDB")
    return parser.parse_args()


def read_fasta_ids(fasta_path: Path) -> list[str]:
    ids = []
    for line in fasta_path.read_text().splitlines():
        if not line.startswith(">"):
            continue
        header = line[1:].strip()
        if not header:
            continue
        ids.append(header.split()[0])
    return list(dict.fromkeys(ids))


def build_pdb_index(pdb_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for ext in ALLOWED_EXTENSIONS:
        for path in pdb_dir.rglob(f"*{ext}"):
            stem = path.stem
            variants = {
                stem,
                stem.lower(),
                stem.upper(),
                stem.replace("pdb", ""),
                stem.replace("PDB", ""),
                stem.lower().replace("pdb", ""),
            }
            for key in variants:
                if key and key not in index:
                    index[key] = path
    return index


def lookup_pdb(seq_id: str, pdb_index: dict[str, Path]) -> Path | None:
    candidates = [
        seq_id,
        seq_id.lower(),
        seq_id.upper(),
        seq_id.replace("|", "_"),
        seq_id.split("|")[-1],
        seq_id.split("_")[0],
    ]
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate in pdb_index:
            return pdb_index[candidate]
    return None


def main() -> None:
    args = parse_args()
    if not args.fasta.exists():
        raise FileNotFoundError(f"FASTA not found: {args.fasta}")
    if not args.pdb_dir.exists():
        raise FileNotFoundError(f"PDB directory not found: {args.pdb_dir}")

    target_ids = read_fasta_ids(args.fasta)
    pdb_index = build_pdb_index(args.pdb_dir)

    matched_rows = []
    unmatched = []
    for target_id in target_ids:
        match = lookup_pdb(target_id, pdb_index)
        if match is None:
            unmatched.append(target_id)
            continue
        matched_rows.append({"target_id": target_id, "pdb_path": str(match.resolve())})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_id", "pdb_path"])
        writer.writeheader()
        writer.writerows(matched_rows)

    args.unmatched.parent.mkdir(parents=True, exist_ok=True)
    args.unmatched.write_text("\n".join(unmatched) + ("\n" if unmatched else ""))

    print(f"Matched targets: {len(matched_rows)}")
    print(f"Unmatched targets: {len(unmatched)}")
    print(f"Targets CSV: {args.output}")
    print(f"Unmatched IDs: {args.unmatched}")


if __name__ == "__main__":
    main()
