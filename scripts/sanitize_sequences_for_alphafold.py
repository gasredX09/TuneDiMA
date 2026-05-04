#!/usr/bin/env python3
"""Sanitize protein sequences for AlphaFold/ESMFold input.

This utility reads either a FASTA file or a plain text file containing one
sequence per line and writes a FASTA file with only standard amino acids.

Default behavior:
- replace unknown/nonstandard residues with 'A'
- keep sequence length unchanged
- optionally drop sequences with too many unknowns

Why this exists:
AlphaFold can accept unknown residues in some workflows, but downstream use
is usually cleaner if the input is limited to the 20 canonical amino acids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
# Map common ambiguous/nonstandard codes to a deterministic fallback.
# X is the default unknown token from the decoder pipeline.
REPLACEMENTS = {
    "B": "D",  # Asp/Asn -> Asp
    "Z": "E",  # Glu/Gln -> Glu
    "J": "L",  # Leu/Ile -> Leu
    "U": "C",  # Selenocysteine -> Cys
    "O": "K",  # Pyrrolysine -> Lys
    "X": "A",  # Unknown -> Ala
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize protein sequences for AlphaFold input")
    parser.add_argument("--input", type=Path, required=True, help="Input FASTA or plain-text sequence file")
    parser.add_argument("--output", type=Path, required=True, help="Output FASTA path")
    parser.add_argument(
        "--drop-if-unknown-frac-above",
        type=float,
        default=1.0,
        help="Drop sequences with unknown/nonstandard fraction above this threshold (default: keep all)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Drop sequences shorter than this length after sanitization",
    )
    return parser.parse_args()


def read_sequences(path: Path) -> list[tuple[str, str]]:
    text = path.read_text().splitlines()
    if any(line.startswith(">") for line in text):
        records: list[tuple[str, str]] = []
        current_id = None
        current_seq_parts: list[str] = []
        for line in text:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq_parts)))
                current_id = line[1:].strip().split()[0] or f"seq{len(records)+1}"
                current_seq_parts = []
            else:
                current_seq_parts.append(line.strip().upper())
        if current_id is not None:
            records.append((current_id, "".join(current_seq_parts)))
        return records

    # Plain text: one sequence per line.
    records = []
    for idx, line in enumerate(text, start=1):
        seq = line.strip().upper()
        if seq:
            records.append((f"seq{idx}", seq))
    return records


def sanitize_sequence(sequence: str) -> tuple[str, int, int]:
    sanitized_chars = []
    unknown_count = 0
    for residue in sequence.upper():
        if residue in STANDARD_AA:
            sanitized_chars.append(residue)
        elif residue in REPLACEMENTS:
            sanitized_chars.append(REPLACEMENTS[residue])
            unknown_count += 1
        else:
            sanitized_chars.append("A")
            unknown_count += 1
    sanitized = "".join(sanitized_chars)
    return sanitized, unknown_count, len(sequence)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    records = read_sequences(args.input)
    kept = []
    dropped = []

    for seq_id, sequence in records:
        sanitized, unknown_count, original_len = sanitize_sequence(sequence)
        unknown_frac = (unknown_count / original_len) if original_len else 1.0
        if original_len < args.min_length or unknown_frac > args.drop_if_unknown_frac_above:
            dropped.append((seq_id, original_len, unknown_count, unknown_frac))
            continue
        kept.append((seq_id, sanitized, original_len, unknown_count, unknown_frac))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for seq_id, sanitized, original_len, unknown_count, unknown_frac in kept:
            handle.write(f">{seq_id} len={original_len} unknowns={unknown_count} frac={unknown_frac:.3f}\n")
            handle.write(f"{sanitized}\n")

    summary_path = args.output.with_suffix(".summary.txt")
    summary_lines = [
        f"input: {args.input}",
        f"output: {args.output}",
        f"input_records: {len(records)}",
        f"kept_records: {len(kept)}",
        f"dropped_records: {len(dropped)}",
        f"drop_if_unknown_frac_above: {args.drop_if_unknown_frac_above}",
        f"min_length: {args.min_length}",
    ]
    if dropped:
        summary_lines.append("dropped_ids: " + ",".join(seq_id for seq_id, *_ in dropped[:20]))
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"Wrote sanitized FASTA: {args.output}")
    print(f"Summary: {summary_path}")
    print(f"Kept {len(kept)} / {len(records)} sequences")


if __name__ == "__main__":
    main()
