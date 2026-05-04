#!/usr/bin/env python3
"""Extract a filtered SMILES library from a local ChEMBL SQLite database.

Outputs a .smi file suitable for design_and_screen.py --ligand-library.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from screening import filter_library


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SMILES from ChEMBL SQLite")
    parser.add_argument(
        "--chembl-db",
        type=Path,
        default=ROOT / "data" / "raw" / "chembl" / "chembl_36" / "chembl_36_sqlite" / "chembl_36.db",
        help="Path to ChEMBL sqlite db",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "chembl36_druglike.smi",
        help="Output .smi file",
    )
    parser.add_argument("--max-molecules", type=int, default=200000, help="Max molecules to keep after filtering")
    parser.add_argument("--no-filter", action="store_true", help="Skip Lipinski/PAINS filtering and export full unique SMILES set")
    parser.add_argument("--seed", type=int, default=42, help="Unused placeholder for reproducibility compatibility")
    return parser.parse_args()


def fetch_smiles_streaming(db_path: Path, output_file: Path, checkpoint_interval: int = 1000) -> int:
    """Stream SMILES from DB and write incrementally to file. Returns total count."""
    # Prefer parent_molregno where possible; fall back to direct canonical structures.
    query_parent = """
        SELECT DISTINCT cs.canonical_smiles
        FROM molecule_dictionary md
        JOIN molecule_hierarchy mh ON mh.molregno = md.molregno
        JOIN compound_structures cs ON cs.molregno = mh.parent_molregno
        WHERE cs.canonical_smiles IS NOT NULL
          AND LENGTH(TRIM(cs.canonical_smiles)) > 0
    """

    query_direct = """
        SELECT DISTINCT canonical_smiles
        FROM compound_structures
        WHERE canonical_smiles IS NOT NULL
          AND LENGTH(TRIM(canonical_smiles)) > 0
    """

    con = sqlite3.connect(str(db_path))
    # SQLite performance optimizations
    con.execute("PRAGMA query_only = ON")      # Read-only mode
    con.execute("PRAGMA cache_size = 100000")  # Large cache (100K pages ~400MB)
    con.execute("PRAGMA synchronous = OFF")    # No sync (safe for reads)
    con.execute("PRAGMA journal_mode = OFF")   # No journal
    con.execute("PRAGMA temp_store = MEMORY")  # Temp tables in RAM
    con.execute("PRAGMA mmap_size = 30000000") # Memory-mapped I/O (30MB)
    con.execute("PRAGMA query_only = ON")      # Ensure read-only
    
    cur = con.cursor()
    smiles_set = set()
    batch_size = 10000
    existing_count = 0

    if output_file.exists():
        with open(output_file) as f_in:
            for line in f_in:
                smi = line.strip()
                if smi:
                    smiles_set.add(smi)
        existing_count = len(smiles_set)
        print(f"Resuming from existing output with {existing_count} SMILES already written")

    # Open output file for append so a rerun extends the current library.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output_file.exists() else "w"
    new_written = 0
    total_seen = existing_count

    with open(output_file, mode) as f_out:
        try:
            cur.execute(query_parent)
            print("Fetching parent molecules...")
            while True:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break
                for row in batch:
                    if row and row[0]:
                        smi = row[0].strip()
                        if smi not in smiles_set:
                            smiles_set.add(smi)
                            f_out.write(smi + "\n")
                            new_written += 1
                            total_seen += 1
                            if new_written % checkpoint_interval == 0:
                                f_out.flush()
                                print(f"  Checkpoint: {total_seen} unique molecules written")
        except sqlite3.Error as e:
            print(f"Warning: parent query failed ({e}), falling back to direct query")

        if new_written == 0 and existing_count == 0:
            print("No parent molecules found, trying direct query...")
            cur.execute(query_direct)
            while True:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break
                for row in batch:
                    if row and row[0]:
                        smi = row[0].strip()
                        if smi not in smiles_set:
                            smiles_set.add(smi)
                            f_out.write(smi + "\n")
                            new_written += 1
                            total_seen += 1
                            if new_written % checkpoint_interval == 0:
                                f_out.flush()
                                print(f"  Checkpoint: {total_seen} unique molecules written")

        f_out.flush()

    con.close()
    return total_seen


def fetch_smiles(db_path: Path) -> list[str]:
    """Legacy function kept for compatibility."""
    query_parent = """
        SELECT DISTINCT cs.canonical_smiles
        FROM molecule_dictionary md
        JOIN molecule_hierarchy mh ON mh.molregno = md.molregno
        JOIN compound_structures cs ON cs.molregno = mh.parent_molregno
        WHERE cs.canonical_smiles IS NOT NULL
          AND LENGTH(TRIM(cs.canonical_smiles)) > 0
    """

    query_direct = """
        SELECT DISTINCT canonical_smiles
        FROM compound_structures
        WHERE canonical_smiles IS NOT NULL
          AND LENGTH(TRIM(canonical_smiles)) > 0
    """

    con = sqlite3.connect(str(db_path))
    # SQLite performance optimizations
    con.execute("PRAGMA query_only = ON")      # Read-only mode
    con.execute("PRAGMA cache_size = 100000")  # Large cache (100K pages ~400MB)
    con.execute("PRAGMA synchronous = OFF")    # No sync (safe for reads)
    con.execute("PRAGMA journal_mode = OFF")   # No journal
    con.execute("PRAGMA temp_store = MEMORY")  # Temp tables in RAM
    con.execute("PRAGMA mmap_size = 30000000") # Memory-mapped I/O (30MB)
    con.execute("PRAGMA query_only = ON")      # Ensure read-only
    
    cur = con.cursor()
    smiles_set = set()
    batch_size = 10000

    try:
        cur.execute(query_parent)
        while True:
            batch = cur.fetchmany(batch_size)
            if not batch:
                break
            smiles_set.update(row[0].strip() for row in batch if row and row[0])
    except sqlite3.Error as e:
        print(f"Warning: parent query failed ({e}), falling back to direct query")

    if not smiles_set:
        cur.execute(query_direct)
        while True:
            batch = cur.fetchmany(batch_size)
            if not batch:
                break
            smiles_set.update(row[0].strip() for row in batch if row and row[0])

    con.close()
    return list(smiles_set)


def main() -> None:
    args = parse_args()
    if not args.chembl_db.exists():
        raise FileNotFoundError(f"ChEMBL DB not found: {args.chembl_db}")

    print(f"Reading ChEMBL DB: {args.chembl_db}")
    print(f"Output file: {args.output}")
    print(f"Mode: {'No filtering (full unique SMILES)' if args.no_filter else 'Lipinski+PAINS filter'}")
    print(f"Incremental checkpoints every 1000 molecules\n")
    
    raw_count = 0
    filtered_count = 0

    if args.no_filter:
        # Use streaming mode with incremental checkpoints
        print("Starting streaming extraction with checkpoints...")
        total_count = fetch_smiles_streaming(args.chembl_db, args.output, checkpoint_interval=1000)
        raw_count = total_count
        filtered_count = total_count
        print(f"\nTotal unique SMILES written: {total_count}")
    else:
        # Use legacy memory-based mode for filtering
        raw = fetch_smiles(args.chembl_db)
        raw_count = len(raw)
        print(f"Raw unique SMILES: {raw_count}")
        filtered = filter_library(raw)
        filtered_count = len(filtered)
        print(f"After Lipinski+PAINS filters: {filtered_count}")
        
        if args.max_molecules > 0:
            filtered = filtered[: min(args.max_molecules, len(filtered))]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(filtered) + ("\n" if filtered else ""))
        filtered_count = len(filtered)
        total_count = filtered_count

    info = args.output.with_suffix(".info.txt")
    info.write_text(
        "\n".join(
            [
                f"chembl_db: {args.chembl_db}",
                f"raw_unique_smiles: {raw_count}",
                f"filtered_smiles: {filtered_count}",
                f"filtering_enabled: {not args.no_filter}",
                f"max_molecules: {args.max_molecules}",
                f"output: {args.output}",
            ]
        )
        + "\n"
    )

    print(f"Wrote: {args.output}")
    print(f"Info: {info}")


if __name__ == "__main__":
    main()
