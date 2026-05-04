#!/usr/bin/env python3
"""Convert FASTA to AlphaFold3 input JSON.

Creates either:
- a single AlphaFold3 job JSON (first sequence), or
- a batch JSON list (all sequences).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id = None
    current_seq: list[str] = []

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records.append((current_id, "".join(current_seq).upper()))
            header = line[1:].strip()
            current_id = header.split()[0] if header else f"seq{len(records)+1}"
            current_seq = []
            continue
        current_seq.append(line)

    if current_id is not None:
        records.append((current_id, "".join(current_seq).upper()))

    return records


def make_job(name: str, sequence: str, seed: int) -> dict:
    return {
        "name": name,
        "modelSeeds": [seed],
        "sequences": [
            {
                "proteinChain": {
                    "sequence": sequence,
                    "count": 1,
                }
            }
        ],
        "dialect": "alphafold3",
        "version": 1,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert FASTA to AlphaFold3 JSON")
    p.add_argument("--input", type=Path, required=True, help="Input FASTA")
    p.add_argument("--output", type=Path, required=True, help="Output JSON")
    p.add_argument("--seed", type=int, default=1, help="Model seed")
    p.add_argument(
        "--batch",
        action="store_true",
        help="Write all sequences as a JSON list (batch). Default: first sequence only.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input FASTA not found: {args.input}")

    records = parse_fasta(args.input)
    if not records:
        raise ValueError(f"No FASTA records found in: {args.input}")

    jobs = [make_job(name=seq_id, sequence=seq, seed=args.seed) for seq_id, seq in records]
    payload = jobs if args.batch else jobs[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    mode = "batch" if args.batch else "single"
    print(f"Wrote {mode} AlphaFold3 JSON: {args.output}")
    print(f"Input sequences: {len(records)}")
    print(f"Included in output: {len(jobs) if args.batch else 1}")


if __name__ == "__main__":
    main()
