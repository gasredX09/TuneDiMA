#!/usr/bin/env python
"""
Diagnostic script to test if decoder is the problem.
Regenerates samples using lm_head decoder (fallback) instead of transformer decoder.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DiMA'))

import torch
import json
from pathlib import Path
from hydra import compose, initialize_config_dir
from src.diffusion.base_trainer import BaseDiffusionTrainer


def test_with_lm_head():
    """Test generation using ESM lm_head decoder instead of transformer decoder."""
    
    config_dir = Path('/ocean/projects/cis260039p/aguda1/nndl/project/DiMA/src/configs').absolute()
    
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name='config')
    
    # Force use of lm_head by pointing to non-existent decoder path
    print("[*] Creating trainer with transformer decoder DISABLED...")
    config.decoder.decoder_path = "/nonexistent/fake/path.pth"
    
    trainer = BaseDiffusionTrainer(config)
    
    # Load reference checkpoint
    print(f"[*] Loading checkpoint: {config.training.init_se}")
    trainer.load_checkpoint(config.training.init_se)
    
    print(f"[*] Using ESM lm_head decoder: {not trainer.encoder._use_transformer_decoder}")
    
    # Generate samples
    print("[*] Generating 64 samples with lm_head decoder...")
    sequences = trainer.generate_samples(num_texts=64)
    
    # Analyze results
    from collections import Counter
    chars = Counter(''.join(sequences))
    
    print(f"\n=== GENERATION RESULTS (lm_head decoder) ===")
    print(f"num_seqs: {len(sequences)}")
    print(f"unique_chars: {sorted(chars.keys())}")
    print(f"top_10_counts: {chars.most_common(10)}")
    print(f"\nFirst 3 sequences:")
    for i, seq in enumerate(sequences[:3]):
        print(f"  {i}: {seq[:100]}... (len={len(seq)})")
    
    # Save diagnostic output
    output = {
        'decoder_type': 'ESM_lm_head',
        'num_samples': len(sequences),
        'unique_amino_acids': sorted(chars.keys()),
        'sample_size': len(sequences[0]) if sequences else 0,
        'composition': dict(chars),
        'first_sample': sequences[0] if sequences else None,
    }
    
    with open('/tmp/dima_lmhead_test.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[✓] Saved diagnostic output to /tmp/dima_lmhead_test.json")
    
    return sequences


if __name__ == '__main__':
    test_with_lm_head()
