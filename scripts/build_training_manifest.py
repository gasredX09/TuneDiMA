"""Build a training manifest from real ligand SMILES.

This script extracts real ligand molecules from PDB files using ligand residue names
and writes a training manifest for kinase and glutamate-family targets.
It can also run a small audit subset first to identify problematic rows.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

from Bio.PDB import PDBIO, PDBParser, Select
import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import build_pocket_edge_attr, extract_pocket_features, featurize_ligand

PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_MANIFEST = PROCESSED_DIR / "combined_structure_ligand_manifest_all.csv"
OUTPUT_MANIFEST = PROCESSED_DIR / "training_manifest.csv"
OUTPUT_SUMMARY = PROCESSED_DIR / "training_manifest_summary.txt"

_NON_LIGAND_RESNAMES = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "CL", "NA", "K", "MG", "CA", "ZN",
    "MN", "CU", "CD", "IOD", "BR", "EDO", "GOL", "PEG", "ACT", "ACE", "FMT",
    "EOH", "TRS", "SEP", "TPO", "PTR", "MSE",
}


def is_target_of_interest(target_name: str, data_source: str) -> bool:
    target = (target_name or "").upper()
    source = (data_source or "").upper()

    glutamate_keywords = ["GLUT", "MGLU", "GRM", "NMDA", "AMPA", "KAINATE"]
    if any(keyword in target for keyword in glutamate_keywords):
        return True

    kinase_keywords = [
        "KINASE", "CDK", "MAPK", "ERK", "EGFR", "FGFR", "JAK", "ABL", "AKT", "PI3K",
        "PIM", "VRK", "ALK", "MET", "EPH", "TRK", "CLK", "CHK", "DYRK", "MELK",
        "MER", "PKC", "BTK", "SRC", "RAF", "MEK", "GSK", "CK1", "CK2",
    ]
    if any(keyword in target for keyword in kinase_keywords):
        return True

    # KLIFS/HKPocket datasets in this project are kinase-focused.
    return "KLIFS" in source or "HKPOCKET" in source


class ResidueSelect(Select):
    def __init__(self, target_resname: str):
        super().__init__()
        self.target_resname = target_resname

    def accept_residue(self, residue):
        if residue.id[0] == " ":
            return 0
        return int(residue.resname.strip().upper() == self.target_resname)


def _largest_fragment_smiles(mol: Chem.Mol) -> Optional[str]:
    if mol is None:
        return None

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not fragments:
        return None

    fragment = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
    if fragment.GetNumHeavyAtoms() < 6:
        return None

    try:
        Chem.SanitizeMol(fragment)
    except Exception:
        return None
    return Chem.MolToSmiles(fragment, canonical=True)


def extract_ligand_smiles_from_pdb(pdb_path: Path, ligand_resname: str) -> Optional[str]:
    resname = (ligand_resname or "").strip().upper()
    if not resname or resname in _NON_LIGAND_RESNAMES:
        return None

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", str(pdb_path))
    except Exception:
        return None

    io_handle = io.StringIO()
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(io_handle, ResidueSelect(resname))
    ligand_block = io_handle.getvalue().strip()
    if not ligand_block:
        return None

    mol = Chem.MolFromPDBBlock(ligand_block, sanitize=False, removeHs=False)
    if mol is None:
        return None
    return _largest_fragment_smiles(mol)


def has_finite_features(smiles: str, pdb_path: str) -> Tuple[bool, str]:
    """Return (is_valid, reason) based on finite ligand and pocket features."""
    try:
        _, atom_feats, _, edge_index, edge_attr, mol_desc = featurize_ligand(smiles, use_3d=False)
    except Exception:
        return False, "ligand_featurize_exception"

    if not np.isfinite(atom_feats).all():
        return False, "ligand_atom_nonfinite"
    if edge_index is not None and edge_index.size > 0 and not np.isfinite(edge_index).all():
        return False, "ligand_edge_index_nonfinite"
    if edge_attr is not None and edge_attr.size > 0 and not np.isfinite(edge_attr).all():
        return False, "ligand_edge_attr_nonfinite"
    if not np.isfinite(mol_desc).all():
        return False, "ligand_desc_nonfinite"

    try:
        res_feats, res_coords = extract_pocket_features(pdb_path)
    except Exception:
        return False, "pocket_featurize_exception"

    if not np.isfinite(res_feats).all():
        return False, "pocket_res_nonfinite"
    if not np.isfinite(res_coords).all():
        return False, "pocket_coords_nonfinite"

    try:
        pocket_edge_attr = build_pocket_edge_attr(res_coords)
    except Exception:
        return False, "pocket_edge_exception"

    if pocket_edge_attr is not None and pocket_edge_attr.size > 0 and not np.isfinite(pocket_edge_attr).all():
        return False, "pocket_edge_attr_nonfinite"

    return True, "ok"


def load_and_filter_manifest(
    manifest_path: Path,
    max_rows: int = 0,
    feature_check: bool = True,
) -> Tuple[list[dict], Dict[str, int]]:
    """Load manifest, extract ligand SMILES, and keep only finite-feature rows."""
    rows = []
    stats: Dict[str, int] = {
        "input_rows": 0,
        "target_filtered": 0,
        "missing_label": 0,
        "missing_pdb": 0,
        "missing_resname": 0,
        "extraction_failed": 0,
        "feature_invalid": 0,
        "output_rows": 0,
    }
    cache: Dict[Tuple[str, str], Optional[str]] = {}

    try:
        with manifest_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats["input_rows"] += 1

                target_name = row.get("target_name", "")
                data_source = row.get("data_source", "")
                if not is_target_of_interest(target_name, data_source):
                    stats["target_filtered"] += 1
                    continue

                try:
                    label_val = float(row.get("label_value", ""))
                except (ValueError, TypeError):
                    stats["missing_label"] += 1
                    continue

                pdb_path = row.get("pdb_path", "").strip()
                if not pdb_path or not Path(pdb_path).exists():
                    stats["missing_pdb"] += 1
                    continue

                ligand_resname = (row.get("ligand_resname", "") or "").strip().upper()
                if not ligand_resname:
                    stats["missing_resname"] += 1
                    continue

                key = (pdb_path, ligand_resname)
                if key not in cache:
                    cache[key] = extract_ligand_smiles_from_pdb(Path(pdb_path), ligand_resname)
                smiles = cache[key]
                if not smiles:
                    stats["extraction_failed"] += 1
                    continue

                if feature_check:
                    valid, reason = has_finite_features(smiles, pdb_path)
                    if not valid:
                        stats["feature_invalid"] += 1
                        stats[reason] = stats.get(reason, 0) + 1
                        continue

                rows.append({
                    "complex_id": row.get("complex_id", ""),
                    "target_name": target_name,
                    "ligand_resname": ligand_resname,
                    "ligand_smiles": smiles,
                    "pdb_path": pdb_path,
                    "label_value": label_val,
                    "label_type": row.get("label_type", ""),
                    "data_source": data_source,
                })

                if max_rows > 0 and len(rows) >= max_rows:
                    break
    except Exception as e:
        print(f"Error loading manifest: {e}")

    stats["output_rows"] = len(rows)
    return rows, stats


def write_training_manifest(rows: list[dict], path: Path) -> None:
    """Write training manifest with required columns for LigandPocketDataset."""
    if not rows:
        raise ValueError("No training rows generated.")

    fieldnames = [
        "complex_id",
        "target_name",
        "ligand_resname",
        "ligand_smiles",
        "pdb_path",
        "label_value",
        "label_type",
        "data_source",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(stats: Dict[str, int], path: Path) -> None:
    """Write summary stats."""
    input_count = stats["input_rows"]
    output_count = stats["output_rows"]
    summary = [
        f"Input manifest rows: {input_count}",
        f"Output training rows (real extracted ligand SMILES): {output_count}",
        f"Retention rate: {100.0 * output_count / input_count:.1f}%" if input_count > 0 else "N/A",
        "",
        "Filter/extraction breakdown:",
        f"  target_filtered: {stats['target_filtered']}",
        f"  missing_label: {stats['missing_label']}",
        f"  missing_pdb: {stats['missing_pdb']}",
        f"  missing_resname: {stats['missing_resname']}",
        f"  extraction_failed: {stats['extraction_failed']}",
        f"  feature_invalid: {stats.get('feature_invalid', 0)}",
    ]

    detail_keys = sorted(
        [k for k in stats.keys() if k.endswith("_nonfinite") or k.endswith("_exception")],
    )
    if detail_keys:
        summary.append("  feature_invalid_breakdown:")
        for key in detail_keys:
            summary.append(f"    - {key}: {stats[key]}")

    summary.extend([
        "",
        "Training manifest ready for LigandPocketDataset consumption using real molecules.",
        "Columns include: ligand_smiles, pdb_path, label_value",
    ])
    path.write_text("\n".join(summary) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training manifest from real extracted ligands")
    parser.add_argument("--input-manifest", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--output-manifest", type=Path, default=OUTPUT_MANIFEST)
    parser.add_argument("--summary-path", type=Path, default=OUTPUT_SUMMARY)
    parser.add_argument("--audit-limit", type=int, default=0, help="Stop after collecting N valid rows (0 = full dataset)")
    parser.add_argument("--no-feature-check", action="store_true", help="Skip finite-feature validation")
    args = parser.parse_args()

    if not args.input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {args.input_manifest}")

    print(f"Loading and filtering manifest from {args.input_manifest}...")
    rows, stats = load_and_filter_manifest(
        args.input_manifest,
        max_rows=max(0, args.audit_limit),
        feature_check=not args.no_feature_check,
    )

    if not rows:
        raise ValueError(
            "No real ligand SMILES could be extracted from the current manifest. "
            "Check PDB paths and ligand residue names, or add an external affinity source (BindingDB/ChEMBL/PDBbind)."
        )

    write_training_manifest(rows, args.output_manifest)
    write_summary(stats, args.summary_path)

    print(f"Wrote training manifest: {args.output_manifest}")
    print(f"Rows: {len(rows)} / {stats['input_rows']}")
    print(f"Summary: {args.summary_path}")


if __name__ == "__main__":
    main()
