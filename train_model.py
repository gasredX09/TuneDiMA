#!/usr/bin/env python3
"""Standalone training script for ligand-pocket binding prediction.

Usage:
    python train_model.py [--epochs 10] [--batch-size 32] [--lr 0.001] [--device cuda]

This script:
1. Loads training data from training_manifest.csv (or generates it if missing)
2. Splits into train/valid/test with randomization
3. Validates model dimensions and feature shapes
4. Trains LigandPocketNet model
5. Saves best model checkpoint and training logs
"""

import argparse
import csv
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from dataset import LigandPocketDataset, load_manifest_records, collate_fn
from models.ligand_pocket import LigandPocketNet
from training import train, set_seed
from features import resolve_pdb_file


def prepare_training_data_if_needed():
    """Auto-generate training manifest if it doesn't exist."""
    root = Path(__file__).resolve().parent
    processed_dir = root / "data" / "processed"
    
    output_manifest = processed_dir / "training_manifest.csv"
    input_manifest = processed_dir / "combined_structure_ligand_manifest_all.csv"
    
    if output_manifest.exists():
        print(f"✅ Training manifest exists: {output_manifest}")
        return output_manifest
    
    if not input_manifest.exists():
        raise FileNotFoundError(f"Missing: {input_manifest}")
    
    print(f"Preparing training manifest from {input_manifest.name}...")
    rows = []
    with input_manifest.open() as f:
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

            ligand_res = row.get("ligand_resname", "").strip() or "LIG"
            smiles = f"[{ligand_res}]"

            rows.append({
                "ligand_smiles": smiles,
                "pdb_path": resolved_pdb_path,
                "label_value": label,
            })

    fieldnames = ["ligand_smiles", "pdb_path", "label_value"]
    with output_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Generated training manifest: {output_manifest}")
    print(f"   Rows: {len(rows)}")
    return output_manifest


def validate_dimensions(sample_batch, model):
    """Validate that data dimensions flow through the model correctly."""
    try:
        lig_data, poc_data, int_data, label = sample_batch
        print(f"✓ Ligand node features shape: {lig_data.x.shape}")
        print(f"✓ Ligand edges: {lig_data.edge_index.shape}")
        print(f"✓ Pocket node features shape: {poc_data.x.shape}")
        print(f"✓ Pocket edges: {poc_data.edge_index.shape}")
        print(f"✓ Interaction node features shape: {int_data.x.shape if int_data.x.numel() > 0 else '(0, 0)'}")
        print(f"✓ Label shape: {label.shape}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(lig_data, poc_data, int_data)
            print(f"✓ Model output shape: {output.shape}")
            assert output.shape[0] == lig_data.num_graphs, "Batch size mismatch!"
        print("✓ All dimensions validated!\n")
        return True
    except Exception as e:
        print(f"✗ Dimension validation failed: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train ligand-pocket binding model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Setup checkpoint directory
    root = Path(__file__).resolve().parents[0]
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"model_epoch_best_{timestamp}.pt"
    log_path = checkpoint_dir / f"training_log_{timestamp}.txt"

    # Prepare manifest if needed
    manifest_path = prepare_training_data_if_needed()

    # Load records
    print(f"Loading training records from {manifest_path.name}...")
    records = load_manifest_records(manifest_path)
    print(f"   Total records: {len(records)}\n")

    if len(records) < 10:
        print("❌ Not enough training samples (need ≥10)")
        return

    # Split data with randomization
    print("Splitting data with randomization...")
    random.seed(args.seed)  # Ensure reproducibility
    random.shuffle(records)
    
    n_train = int(0.7 * len(records))
    n_valid = int(0.15 * len(records))
    train_records = records[:n_train]
    valid_records = records[n_train : n_train + n_valid]
    test_records = records[n_train + n_valid :]

    print(f"   Train: {len(train_records)} ({100*len(train_records)/len(records):.1f}%)")
    print(f"   Valid: {len(valid_records)} ({100*len(valid_records)/len(records):.1f}%) [randomized]")
    print(f"   Test:  {len(test_records)} ({100*len(test_records)/len(records):.1f}%)\n")

    # Create datasets
    print("Creating datasets...")
    train_dataset = LigandPocketDataset(train_records, use_3d=False)
    valid_dataset = LigandPocketDataset(valid_records, use_3d=False)

    # Build model
    print("Building LigandPocketNet model...")
    model = LigandPocketNet(
        lig_in=16,           # Ligand atom feature dimension
        poc_in=25,           # Pocket residue feature dimension
        int_in=25,           # Interaction feature dimension
        edge_dim=10,         # Ligand edge feature dimension
        int_edge_dim=3,      # Interaction edge feature dimension
        hidden_dim=128,      # Hidden layer dimension (for GNN layers)
        num_layers=3,        # Number of message passing layers
        dropout=0.2,         # Dropout for regularization
        mol_desc_dim=17,     # Molecular descriptor dimension
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"   Device: {device}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Validate dimensions on a sample batch
    print("Validating data-model dimensions...")
    sample_count = min(2, len(valid_records))
    sample_dataset = LigandPocketDataset(valid_records[:sample_count], use_3d=False)
    sample_loader = DataLoader(sample_dataset, batch_size=1, collate_fn=collate_fn)
    for sample_batch in sample_loader:
        sample_batch = tuple(x.to(device) if isinstance(x, torch.Tensor) or hasattr(x, 'to') else x for x in sample_batch)
        if not validate_dimensions(sample_batch, model):
            print("❌ Dimension validation failed. Exiting.")
            return

    # Train
    print(f"Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Log: {log_path}\n")

    try:
        # Wrap training to capture and save best model
        trained_model = train(
            model,
            train_dataset,
            valid_dataset,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=str(device),
            mode="regression",
            patience=5,
        )
        
        # Save best model checkpoint
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_architecture': {
                'lig_in': 16,
                'poc_in': 25,
                'int_in': 25,
                'edge_dim': 10,
                'int_edge_dim': 3,
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': 0.2,
                'mol_desc_dim': 17,
            },
            'training_config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'seed': args.seed,
            },
            'data_split': {
                'train_size': len(train_records),
                'valid_size': len(valid_records),
                'test_size': len(test_records),
            }
        }, checkpoint_path)
        
        print(f"\n✅ Training complete!")
        print(f"✅ Model saved to: {checkpoint_path}")
        
        # Log summary
        log_content = f"""Training Summary
================
Timestamp: {timestamp}
Epochs: {args.epochs}
Batch Size: {args.batch_size}
Learning Rate: {args.lr}
Device: {device}
Seed: {args.seed}

Data Split:
  Train: {len(train_records)}
  Valid: {len(valid_records)} (randomized)
  Test:  {len(test_records)}

Model Architecture:
  Input dims: ligand=16, pocket=25, interaction=25
  Hidden dims: 128
  Layers: 3
  Dropout: 0.2
  Parameters: {sum(p.numel() for p in trained_model.parameters()):,}

Checkpoint: {checkpoint_path}
"""
        log_path.write_text(log_content)
        print(f"✅ Training log saved to: {log_path}\n")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
