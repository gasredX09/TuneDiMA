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
import os
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# Import from projects package
from projects.dataset import LigandPocketDataset, load_manifest_records, collate_fn
from projects.models.ligand_pocket import LigandPocketNet
from projects.training import train, set_seed


def prepare_training_data_if_needed():
    """Ensure a real-molecule training manifest exists and is non-empty."""
    root = Path(__file__).resolve().parent
    processed_dir = root / "data" / "processed"
    
    output_manifest = processed_dir / "training_manifest.csv"
    
    if output_manifest.exists():
        print(f"✅ Training manifest exists: {output_manifest}")
        with output_manifest.open() as f:
            row_count = sum(1 for _ in f) - 1
        if row_count <= 0:
            raise ValueError(
                "training_manifest.csv exists but has zero rows. Add a source with real ligand_smiles before training."
            )
        return output_manifest

    raise FileNotFoundError(
        "training_manifest.csv is missing. Build it with scripts/build_training_manifest.py to extract real ligands first."
    )


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
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (higher can be faster but uses more RAM)")
    parser.add_argument("--cache-graphs", action="store_true", help="Cache featurized graphs in RAM (fastest with --num-workers 0)")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training (slower)")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Gradient clipping max norm (e.g., 1.0). None to disable")
    parser.add_argument("--warmup", action="store_true", help="Enable learning rate warmup")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--cosine-anneal", action="store_true", help="Enable cosine annealing learning rate schedule")
    parser.add_argument("--ligand-use-3d", dest="ligand_use_3d", action="store_true", help="Use 3D ligand conformers for interaction graph construction")
    parser.add_argument("--no-ligand-use-3d", dest="ligand_use_3d", action="store_false", help="Disable 3D ligand conformers (2D-only ligand geometry)")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Training split fraction (default: 0.7)")
    parser.add_argument("--valid-frac", type=float, default=0.15, help="Validation split fraction (default: 0.15)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience in epochs")
    parser.set_defaults(ligand_use_3d=True)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.train_frac <= 0.0 or args.valid_frac <= 0.0:
        raise ValueError("train-frac and valid-frac must be > 0")
    if args.train_frac + args.valid_frac > 1.0:
        raise ValueError("train-frac + valid-frac must be <= 1.0")
    if args.patience <= 0:
        raise ValueError("patience must be > 0")

    # Setup checkpoint directory
    root = Path(__file__).resolve().parents[0]
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = os.getenv("SLURM_JOB_ID") or f"pid{os.getpid()}"
    checkpoint_path = checkpoint_dir / f"model_epoch_best_{timestamp}_{run_id}.pt"
    log_path = checkpoint_dir / f"training_log_{timestamp}_{run_id}.txt"

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
    
    n_train = int(args.train_frac * len(records))
    n_valid = int(args.valid_frac * len(records))
    train_records = records[:n_train]
    valid_records = records[n_train : n_train + n_valid]
    test_records = records[n_train + n_valid :]

    print(f"   Train: {len(train_records)} ({100*len(train_records)/len(records):.1f}%)")
    print(f"   Valid: {len(valid_records)} ({100*len(valid_records)/len(records):.1f}%) [randomized]")
    print(f"   Test:  {len(test_records)} ({100*len(test_records)/len(records):.1f}%)\n")

    # Create datasets
    print("Creating datasets...")
    # Worker processes each hold their own dataset object; caching with multiple workers can OOM.
    cache_graphs = args.cache_graphs and args.num_workers == 0
    train_dataset = LigandPocketDataset(train_records, use_3d=args.ligand_use_3d, cache_graphs=cache_graphs)
    valid_dataset = LigandPocketDataset(valid_records, use_3d=args.ligand_use_3d, cache_graphs=cache_graphs)

    # Build model
    print("Building LigandPocketNet model...")
    lig_in_dim = 19 if args.ligand_use_3d else 16
    model = LigandPocketNet(
        lig_in=lig_in_dim,   # Ligand atom feature dimension
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
    sample_dataset = LigandPocketDataset(valid_records[:sample_count], use_3d=args.ligand_use_3d, cache_graphs=False)
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
    print(f"   DataLoader workers: {args.num_workers}")
    print(f"   Cache graphs: {cache_graphs}")
    print(f"   Deterministic: {args.deterministic}")
    print(f"   Max grad norm: {args.max_grad_norm}")
    print(f"   LR Warmup: {args.warmup} ({args.warmup_epochs} epochs)")
    print(f"   Cosine Annealing: {args.cosine_anneal}")
    print(f"   Ligand 3D conformers: {args.ligand_use_3d}")
    print(f"   Early-stopping patience: {args.patience}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Log: {log_path}\n")

    # Create log file immediately so users can tail it during long first epochs.
    start_log = f"""Training Start
==============
Timestamp: {timestamp}
Epochs: {args.epochs}
Batch Size: {args.batch_size}
Learning Rate: {args.lr}
Device: {device}
Seed: {args.seed}
DataLoader workers: {args.num_workers}
Cache graphs: {cache_graphs}
Deterministic: {args.deterministic}
Max Grad Norm: {args.max_grad_norm}
LR Warmup: {args.warmup} ({args.warmup_epochs} epochs)
Cosine Annealing: {args.cosine_anneal}
Ligand 3D conformers: {args.ligand_use_3d}
Train fraction: {args.train_frac}
Valid fraction: {args.valid_frac}
Early-stopping patience: {args.patience}

Data Split:
    Train: {len(train_records)}
    Valid: {len(valid_records)} (randomized)
    Test:  {len(test_records)}

Checkpoint (best): {checkpoint_path}
\n"""
    log_path.write_text(start_log)

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
            patience=args.patience,
            checkpoint_path=str(checkpoint_path),
            log_path=str(log_path),
            num_workers=args.num_workers,
            deterministic=args.deterministic,
            seed=args.seed,
            max_grad_norm=args.max_grad_norm,
            use_warmup=args.warmup,
            warmup_epochs=args.warmup_epochs,
            use_cosine_anneal=args.cosine_anneal,
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
        log_content = f"""\nTraining Summary
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
        with log_path.open("a") as f:
            f.write(log_content)
        print(f"✅ Training log saved to: {log_path}\n")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        with log_path.open("a") as f:
            f.write(f"\nTraining error: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
