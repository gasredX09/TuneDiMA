"""Prepare training data from combined manifest and train the model immediately."""

from __future__ import annotations

import csv
import random
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import resolve_pdb_file

INPUT_MANIFEST = PROCESSED_DIR / "combined_structure_ligand_manifest_all.csv"
OUTPUT_MANIFEST = PROCESSED_DIR / "training_manifest.csv"


def prepare_training_data() -> None:
    """Load combined manifest and prepare for training.
    
    Uses existing label values (quality scores, activity data) and PDB paths.
    For SMILES, uses residue names as placeholders (can be enhanced to extract from PDB).
    """
    rows = []
    with INPUT_MANIFEST.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                label = float(row.get("label_value", 0))
            except (ValueError, TypeError):
                continue

            pdb_path = row.get("pdb_path", "").strip()
            if not pdb_path:
                continue

            resolved_pdb_path = resolve_pdb_file(pdb_path)
            if not Path(resolved_pdb_path).exists():
                continue

            # Use ligand residue name as placeholder SMILES (or generate simple one)
            ligand_res = row.get("ligand_resname", "").strip() or "LIG"
            # Create a simple but deterministic SMILES placeholder
            smiles = f"[{ligand_res}]"

            rows.append({
                "complex_id": row.get("complex_id", ""),
                "ligand_smiles": smiles,
                "pdb_path": resolved_pdb_path,
                "label_value": label,
            })

    fieldnames = ["complex_id", "ligand_smiles", "pdb_path", "label_value"]
    with OUTPUT_MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prepared {len(rows)} training samples")
    print(f"Wrote training manifest to {OUTPUT_MANIFEST}")


def train_model() -> None:
    """Load training data and run a quick training loop."""
    from dataset import LigandPocketDataset, load_manifest_records, collate_fn
    from models.ligand_pocket import LigandPocketNet
    from training import train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training records
    records = load_manifest_records(OUTPUT_MANIFEST)
    print(f"Loaded {len(records)} records")

    if len(records) < 10:
        print("Not enough training samples. Skipping training.")
        return

    # Random split
    random.shuffle(records)
    n_train = int(0.7 * len(records))
    n_valid = int(0.15 * len(records))
    train_records = records[:n_train]
    valid_records = records[n_train : n_train + n_valid]

    print(f"Train: {len(train_records)}, Valid: {len(valid_records)}")

    # Create datasets
    train_dataset = LigandPocketDataset(train_records, use_3d=False)
    valid_dataset = LigandPocketDataset(valid_records, use_3d=False)

    # Build model
    model = LigandPocketNet(
        lig_in=16,
        poc_in=25,
        int_in=25,
        edge_dim=10,
        int_edge_dim=3,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        mol_desc_dim=17,
    ).to(device)

    # Train
    print("Starting training...")
    try:
        train(
            model,
            train_dataset,
            valid_dataset,
            epochs=5,
            lr=1e-3,
            batch_size=16,
            device=str(device),
            mode="regression",
            patience=3,
        )
        print("Training complete!")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    prepare_training_data()
    train_model()
