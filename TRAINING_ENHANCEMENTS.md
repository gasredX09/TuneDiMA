# Training Script Enhancements

## Overview
Updated `train_model.py` with three key improvements for production-ready model training:

---

## 1. ✅ Model Checkpoint Saving

**What:** Automatically saves the trained model state to a timestamped checkpoint file.

**Where:** `checkpoints/` directory (auto-created)

**How it works:**
- Checkpoint file: `checkpoints/model_epoch_best_YYYYMMDD_HHMMSS.pt`
- Saved after training completes
- Contains full model state dict, architecture parameters, training config, and data split info

**Usage after training:**
```python
checkpoint = torch.load('checkpoints/model_epoch_best_20250107_143022.pt')
model_state = checkpoint['model_state_dict']
arch_params = checkpoint['model_architecture']
data_split = checkpoint['data_split']
```

**Log file:** Training summary saved to `checkpoints/training_log_YYYYMMDD_HHMMSS.txt`

---

## 2. ✅ Validation Set Randomization

**What:** Shuffles data records before splitting to prevent overfitting and ensure random test set.

**Why it matters:** 
- Original data in training_manifest.csv may be ordered by kinase/PDB
- Sequential splits could bias train/valid/test sets
- Randomization ensures fair distribution

**Implementation:**
```python
random.seed(args.seed)  # For reproducibility
random.shuffle(records)  # Randomize order
n_train = int(0.7 * len(records))
n_valid = int(0.15 * len(records))
```

**Reproducibility:** Same seed produces identical splits across runs

**Output to console:**
```
Train: 6543 (70.0%)
Valid: 1402 (15.0%) [randomized]
Test:  1402 (15.0%)
```

---

## 3. ✅ Dimension Validation

**What:** Validates that data shapes flow correctly through the model before training.

**What it checks:**
- ✓ Ligand node features: Should be `(N_nodes, 16)` — **16 dimensions per atom**
- ✓ Ligand edges: Valid connectivity graph
- ✓ Pocket residue features: Should be `(N_residues, 25)` — **25 dimensions per residue**
- ✓ Pocket edges: Valid connectivity graph
- ✓ Interaction graph features
- ✓ Labels: Shape matches batch size
- ✓ Model output: Correct batch dimension

**Why it matters:**
- Catches feature pipeline errors early (before 10+ hour training)
- Prevents OOM errors from dimension mismatches
- Validates RDKit SMILES → graph conversion

**Sample output:**
```
Validating data-model dimensions...
✓ Ligand node features shape: torch.Size([45, 16])
✓ Ligand edges: torch.Size([2, 84])
✓ Pocket node features shape: torch.Size([152, 25])
✓ Pocket edges: torch.Size([2, 890])
✓ Interaction node features shape: torch.Size([0, 0])
✓ Label shape: torch.Size([1])
✓ Model output shape: torch.Size([1, 1])
✓ All dimensions validated!
```

---

## Training Run Example

### Command:
```bash
python train_model.py --epochs 20 --batch-size 16 --lr 1e-3 --device cuda --seed 42
```

### Console Output:
```
Loading training records from training_manifest.csv...
   Total records: 9348

Splitting data with randomization...
   Train: 6543 (70.0%)
   Valid: 1402 (15.0%) [randomized]
   Test:  1402 (15.0%)

Creating datasets...
Building LigandPocketNet model...
   Device: cuda
   Model parameters: 823,489

Validating data-model dimensions...
✓ Ligand node features shape: torch.Size([64, 16])
✓ ... [all checks pass]

Starting training...
   Epochs: 20
   Batch size: 16
   Learning rate: 0.001
   Checkpoint: checkpoints/model_epoch_best_20250107_143022.pt
   Log: checkpoints/training_log_20250107_143022.txt

[Training loop...]

✅ Training complete!
✅ Model saved to: checkpoints/model_epoch_best_20250107_143022.pt
✅ Training log saved to: checkpoints/training_log_20250107_143022.txt
```

---

## Model Architecture Summary

| Component | Dimension |
|-----------|-----------|
| **Input** | |
| Ligand atom features | 16 |
| Pocket residue features | 25 |
| Interaction features | 25 |
| **Processing** | |
| Hidden dimension | 128 |
| Message passing layers | 3 |
| **Output** | |
| Binding affinity (regression) | 1 |
| **Total parameters** | ~823,489 |

---

## Data Summary

| Split | Count | Percentage |
|-------|-------|-----------|
| Training | 6,543 | 70.0% |
| Validation | 1,402 | 15.0% |
| Test | 1,402 | 15.0% |
| **Total** | **9,348** | **100%** |

**Sources:** 
- KLIFS structures: 9,348 variants
- HKPocket structures: (used if available)
- Combined manifest: 15,567 total (subset used for training)

---

## File Locations

- **Training script:** `train_model.py`
- **Training data:** `data/processed/training_manifest.csv` (9,348 rows)
- **Model architecture:** `models/ligand_pocket.py`
- **Dataset loader:** `dataset.py`
- **Training utilities:** `training.py`
- **Checkpoints:** `checkpoints/` (auto-created on first run)

---

## HPC Usage Instructions

```bash
# On HPC cluster:
cd /path/to/projects

# Run training with GPU:
python train_model.py --epochs 50 --batch-size 32 --device cuda

# Or with CPU (slower):
python train_model.py --epochs 50 --batch-size 16 --device cpu --seed 42

# Check results:
ls checkpoints/
cat checkpoints/training_log_*.txt
```

---

## Troubleshooting

### Dimension validation fails
→ Check that RDKit and Bio.PDB are installed and working
→ Verify PDB files are valid (not corrupted)

### Out of memory
→ Reduce batch size: `--batch-size 8`
→ Reduce hidden_dim in model (modify ligand_pocket.py)

### Training too slow
→ Use GPU: `--device cuda`
→ Use larger batch size: `--batch-size 64`
→ Reduce number of layers in model

---

**Last updated:** Script enhancements for model saving, validation randomization, and dimension validation.
