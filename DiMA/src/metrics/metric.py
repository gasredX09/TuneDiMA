
import torch
import numpy as np
from typing import Optional, List
import re

from src.metrics.plddt import calculate_plddt
from src.metrics.fid import calculate_fid_for_lists
from src.metrics.mmd import calculate_mmd_for_lists
from src.metrics.esmpppl import calculate_pppl
from src.utils.ddp_utils import reduce_tensor


_VALID_AA = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


def _sanitize_sequences(sequences: List[str]) -> List[str]:
    cleaned = []
    for seq in sequences:
        seq = (seq or "").upper()
        seq = _VALID_AA.sub("", seq)
        if seq:
            cleaned.append(seq)
    return cleaned


def compute_ddp_metric(metric_name: str, predictions: List[str], references: List[str], 
                       max_len: int, device: str, rank: int = 0, world_size: int = 1, pdb_path: str = "") -> float:
    if metric_name in ["plddt", "esm_pppl"]:
        # Split predictions and references across GPUs
        num_samples = len(predictions) // world_size
        if rank < len(predictions) % world_size:
            num_samples += 1
        predictions = predictions[rank * num_samples: (rank + 1) * num_samples]
        references = references[rank * num_samples: (rank + 1) * num_samples]
        index_list = list(range(rank * num_samples, (rank + 1) * num_samples))

    if metric_name == "plddt":  
        print(f"Calculating plddt for {len(predictions)} texts")  
        predictions = _sanitize_sequences(predictions)
        if not predictions:
            return 0.0
        index_list = list(range(len(predictions)))
        plddt_result = calculate_plddt(predictions=predictions, index_list=index_list, device=device, pdb_path=pdb_path)
        value = np.mean(list(plddt_result.values()))

    if metric_name == "fid":
        value = calculate_fid_for_lists(predictions=predictions, references=references, max_len=max_len, device=device)

    if metric_name == "mmd":
        value = calculate_mmd_for_lists(predictions=predictions, references=references, max_len=max_len, device=device)

    if metric_name == "esm_pppl":
        print(f"Calculating esm_pppl for {len(predictions)} texts")
        predictions = _sanitize_sequences(predictions)
        if not predictions:
            return float("nan")
        pppl_result = calculate_pppl(predictions=predictions, max_len=max_len, device=device)
        value = np.mean(pppl_result)

    if metric_name in ["plddt", "esm_pppl"] and world_size > 1:
        value = torch.tensor([value], device=device)
        value = reduce_tensor(value)
        value = value.item()

    return value