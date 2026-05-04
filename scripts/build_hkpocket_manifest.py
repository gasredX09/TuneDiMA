"""Build a HKPocket structure manifest from downloaded HKPocket PDB archives.

This script scans extracted HKPocket pocket-residue PDB files and emits a
manifest CSV consumable by build_combined_structure_ligand_manifest.py.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HKPOCKET_ROOT = ROOT / "data" / "raw" / "hkpocket"
HKPOCKET_PDB_EXPANDED = HKPOCKET_ROOT / "pdb" / "groups_expanded"
OUTPUT_MANIFEST = HKPOCKET_ROOT / "hkpocket_manifest.csv"

# Common non-drug HET residues to ignore when selecting a likely ligand residue.
EXCLUDED_HET_RESNAMES = {
    "HOH",
    "WAT",
    "DOD",
    "SO4",
    "PO4",
    "CL",
    "NA",
    "K",
    "MG",
    "CA",
    "ZN",
    "MN",
    "CU",
    "CD",
    "IOD",
    "BR",
    "EDO",
    "GOL",
    "PEG",
    "ACT",
    "ACE",
    "FMT",
    "EOH",
    "TRS",
    "SEP",
    "TPO",
    "PTR",
    "MSE",
}


def first_chain_from_pdb(pdb_path: Path) -> str:
    try:
        with pdb_path.open("r", errors="ignore") as handle:
            for line in handle:
                if line.startswith("ATOM") and len(line) >= 22:
                    chain = line[21].strip()
                    return chain or "A"
    except OSError:
        return ""
    return ""


def best_ligand_resname(full_pdb_path: Path) -> str:
    counts: Counter[str] = Counter()
    try:
        with full_pdb_path.open("r", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("HETATM") or len(line) < 20:
                    continue
                resname = line[17:20].strip().upper()
                if not resname or resname in EXCLUDED_HET_RESNAMES:
                    continue
                counts[resname] += 1
    except OSError:
        return ""

    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def parse_pocket_metadata(pocket_path: Path) -> tuple[str, str, str, str]:
    # Expected shape:
    # .../<group>/<pdbid>/<pdbid>_P_..._res.pdb
    pdb_id = pocket_path.stem.split("_P_")[0].lower()
    pocket_id = pocket_path.stem.replace("_res", "")
    group = pocket_path.parts[-3] if len(pocket_path.parts) >= 3 else ""
    return pdb_id, pocket_id, group, str(pocket_path)


def build_manifest_rows() -> list[dict]:
    rows = []
    pattern = "**/*_res.pdb"
    for pocket_path in HKPOCKET_PDB_EXPANDED.glob(pattern):
        pdb_id, pocket_id, group, pocket_path_str = parse_pocket_metadata(pocket_path)
        parent_dir = pocket_path.parent
        full_pdb_path = parent_dir / f"{pdb_id}.pdb"

        chain = first_chain_from_pdb(pocket_path)
        ligand_resname = best_ligand_resname(full_pdb_path) if full_pdb_path.exists() else ""

        rows.append(
            {
                "complex_id": f"hkpocket_{pocket_id}",
                "pdb_id": pdb_id,
                "protein_chain": chain,
                "ligand_resname": ligand_resname,
                "ligand_smiles": "",
                "target_name": "",
                "species": "HOMO SAPIENS",
                "uniprot_id": "",
                "label_type": "HKPocket_structure",
                "label_value": "",
                "label_unit": "",
                "activity_type": "structure_only",
                "n_measurements": 1,
                "molecule_chembl_ids": "",
                "template_structure_id": pocket_id,
                "template_quality_score": "",
                "template_resolution": "",
                "template_ligand_resname": ligand_resname,
                "data_source": f"HKPocket {group}" if group else "HKPocket",
                "pdb_path": pocket_path_str,
            }
        )

    rows.sort(key=lambda r: (r["pdb_id"], r["template_structure_id"]))
    return rows


def main() -> None:
    if not HKPOCKET_PDB_EXPANDED.exists():
        raise FileNotFoundError(
            "HKPocket extracted PDB directory not found at "
            f"{HKPOCKET_PDB_EXPANDED}. Download and extract HKPocket first."
        )

    rows = build_manifest_rows()
    if not rows:
        raise ValueError("No HKPocket pocket files were found.")

    fieldnames = list(rows[0].keys())
    with OUTPUT_MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    unique_pdbs = len({row["pdb_id"] for row in rows})
    print(f"Wrote HKPocket manifest: {OUTPUT_MANIFEST}")
    print(f"Rows: {len(rows)}")
    print(f"Unique PDB IDs: {unique_pdbs}")


if __name__ == "__main__":
    main()
