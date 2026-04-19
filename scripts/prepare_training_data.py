"""Prepare training data from combined manifest. No dependencies besides pandas."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_MANIFEST = PROCESSED_DIR / "combined_structure_ligand_manifest_all.csv"
OUTPUT_MANIFEST = PROCESSED_DIR / "training_manifest.csv"


def prepare_training_data():
    """Load combined manifest and create training-ready CSV."""
    rows = []
    with INPUT_MANIFEST.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                label = float(row.get("label_value", 0))
            except (ValueError, TypeError):
                continue

            pdb_path = row.get("pdb_path", "").strip()
            if not pdb_path or not Path(pdb_path).exists():
                continue

            # Use ligand residue name as placeholder SMILES
            ligand_res = row.get("ligand_resname", "").strip() or "LIG"
            smiles = f"[{ligand_res}]"

            rows.append({
                "ligand_smiles": smiles,
                "pdb_path": pdb_path,
                "label_value": label,
            })

    fieldnames = ["ligand_smiles", "pdb_path", "label_value"]
    with OUTPUT_MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Training manifest ready: {OUTPUT_MANIFEST}")
    print(f"   Rows: {len(rows)}")
    print(f"   Split: {int(0.7*len(rows))} train, {int(0.15*len(rows))} valid, {int(0.15*len(rows))} test")


if __name__ == "__main__":
    prepare_training_data()
