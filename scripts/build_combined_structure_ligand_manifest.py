"""Build a structure-ligand training manifest from KLIFS and optional HKPocket rows.

This script favors direct structure-backed examples over assay-derived labels.
It will always include KLIFS template complexes that resolve to local PDB files,
and if a HKPocket-style CSV is present it will merge those labeled complexes too.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

KLIFS_STRUCTURES = RAW_DIR / "klifs" / "structures_list.csv"
HKPOCKET_MANIFEST = RAW_DIR / "hkpocket" / "hkpocket_manifest.csv"
PDB_DIR = RAW_DIR / "pdb"
OUTPUT_MANIFEST = PROCESSED_DIR / "combined_structure_ligand_manifest.csv"
OUTPUT_MANIFEST_ALL = PROCESSED_DIR / "combined_structure_ligand_manifest_all.csv"
OUTPUT_SUMMARY = PROCESSED_DIR / "combined_structure_ligand_manifest_summary.txt"
OUTPUT_KLIFS_ONLY = PROCESSED_DIR / "klifs_structure_manifest.csv"
OUTPUT_KLIFS_ALL = PROCESSED_DIR / "klifs_structure_manifest_all.csv"

SPECIES_NORMALIZATION = {
    "HUMAN": "HOMO SAPIENS",
    "HOMO SAPIENS": "HOMO SAPIENS",
    "MOUSE": "MUS MUSCULUS",
    "MUS MUSCULUS": "MUS MUSCULUS",
    "RAT": "RATTUS NORVEGICUS",
    "RATTUS NORVEGICUS": "RATTUS NORVEGICUS",
}

REQUIRED_COLUMNS = [
    "complex_id",
    "pdb_id",
    "protein_chain",
    "ligand_resname",
    "ligand_smiles",
    "target_name",
    "species",
    "uniprot_id",
    "label_type",
    "label_value",
    "label_unit",
    "activity_type",
    "n_measurements",
    "molecule_chembl_ids",
    "template_structure_id",
    "template_quality_score",
    "template_resolution",
    "template_ligand_resname",
    "data_source",
    "pdb_path",
]

PDB_PATH_INDEX: dict[str, str] | None = None


def safe_float(value, default=float("inf")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_species_name(value: str) -> str:
    raw = str(value or "").strip().upper()
    return SPECIES_NORMALIZATION.get(raw, raw)


def build_pdb_path_index() -> dict[str, str]:
    index: dict[str, str] = {}
    if not PDB_DIR.exists():
        return index
    for path in PDB_DIR.glob("*.*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".pdb", ".ent"}:
            continue
        stem = path.stem.lower()
        if stem.startswith("pdb") and len(stem) > 3:
            code = stem[3:]
        else:
            code = stem
        if len(code) == 4:
            index.setdefault(code, str(path))
    return index


def resolve_pdb_path(pdb_code: str) -> str | None:
    global PDB_PATH_INDEX
    code = str(pdb_code or "").strip().lower()
    if not code:
        return None
    if PDB_PATH_INDEX is None:
        PDB_PATH_INDEX = build_pdb_path_index()
    return PDB_PATH_INDEX.get(code)


def load_klifs_templates() -> dict[tuple[str, str], dict]:
    rows = list(csv.DictReader(KLIFS_STRUCTURES.open()))
    best_by_target: dict[tuple[str, str], dict] = {}

    for row in rows:
        pdb_path = resolve_pdb_path(row.get("pdb"))
        if pdb_path is None:
            continue

        key = (str(row.get("kinase", "")).strip().upper(), normalize_species_name(row.get("species")))
        score = (
            -safe_float(row.get("quality_score"), default=-1.0),
            safe_float(row.get("missing_residues"), default=9999.0),
            safe_float(row.get("missing_atoms"), default=9999.0),
            safe_float(row.get("resolution"), default=9999.0),
            1 if str(row.get("allosteric_ligand", "0")) == "1" else 0,
            str(row.get("pdb", "")).lower(),
        )
        current = best_by_target.get(key)
        if current is None or score < current["_score"]:
            best_by_target[key] = {
                **row,
                "species": normalize_species_name(row.get("species")),
                "_score": score,
                "_pdb_path": pdb_path,
            }

    for row in best_by_target.values():
        row.pop("_score", None)
    return best_by_target


def load_all_klifs_structures() -> list[dict]:
    rows = []
    for row in csv.DictReader(KLIFS_STRUCTURES.open()):
        pdb_path = resolve_pdb_path(row.get("pdb"))
        if pdb_path is None:
            continue
        quality = safe_float(row.get("quality_score"), default=float("nan"))
        if quality != quality:
            continue
        rows.append(
            {
                **row,
                "species": normalize_species_name(row.get("species")),
                "_pdb_path": pdb_path,
            }
        )
    return rows


def klifs_rows_to_manifest_rows(klifs_structures: list[dict], source_label: str) -> list[dict]:
    manifest_rows = []
    for idx, template in enumerate(klifs_structures, start=1):
        quality = safe_float(template.get("quality_score"), default=float("nan"))
        if quality != quality:
            continue
        pdb_id = str(template.get("pdb", "")).lower()
        chain = template.get("chain", "")
        ligand = template.get("ligand", "")
        structure_id = str(template.get("structure_ID", "")).strip()
        manifest_rows.append(
            {
                "complex_id": f"{pdb_id}_{chain}_{ligand}_{structure_id or idx}",
                "pdb_id": pdb_id,
                "protein_chain": chain,
                "ligand_resname": ligand,
                "ligand_smiles": "",
                "target_name": template.get("kinase", ""),
                "species": normalize_species_name(template.get("species")),
                "uniprot_id": "",
                "label_type": "KLIFS_quality_score",
                "label_value": round(float(quality), 4),
                "label_unit": "unitless",
                "activity_type": "template_quality",
                "n_measurements": 1,
                "molecule_chembl_ids": "",
                "template_structure_id": template.get("structure_ID", ""),
                "template_quality_score": template.get("quality_score", ""),
                "template_resolution": template.get("resolution", ""),
                "template_ligand_resname": template.get("ligand", ""),
                "data_source": source_label,
                "pdb_path": template.get("_pdb_path", ""),
            }
        )
    return manifest_rows


def normalize_hkpocket_row(row: dict) -> dict:
    normalized = {key.strip(): value for key, value in row.items()}
    pdb_id = normalized.get("pdb_id") or normalized.get("pdb") or normalized.get("structure_pdb")
    pdb_path = normalized.get("pdb_path") or resolve_pdb_path(pdb_id)
    if pdb_path is None:
        return {}

    return {
        "complex_id": normalized.get("complex_id") or f"{str(pdb_id).lower()}_{normalized.get('protein_chain', normalized.get('chain', 'A'))}",
        "pdb_id": str(pdb_id).lower() if pdb_id else "",
        "protein_chain": normalized.get("protein_chain") or normalized.get("chain") or "",
        "ligand_resname": normalized.get("ligand_resname") or normalized.get("ligand") or "",
        "ligand_smiles": normalized.get("ligand_smiles") or normalized.get("smiles") or "",
        "target_name": normalized.get("target_name") or normalized.get("kinase") or "",
        "species": normalize_species_name(normalized.get("species") or normalized.get("organism")),
        "uniprot_id": normalized.get("uniprot_id") or normalized.get("accession") or "",
        "label_type": normalized.get("label_type") or normalized.get("assay_type") or "label",
        "label_value": normalized.get("label_value") or normalized.get("label") or normalized.get("ic50") or "",
        "label_unit": normalized.get("label_unit") or normalized.get("unit") or "",
        "activity_type": normalized.get("activity_type") or normalized.get("source") or "HKPocket",
        "n_measurements": normalized.get("n_measurements") or normalized.get("count") or 1,
        "molecule_chembl_ids": normalized.get("molecule_chembl_ids") or "",
        "template_structure_id": normalized.get("template_structure_id") or normalized.get("structure_id") or "",
        "template_quality_score": normalized.get("template_quality_score") or "",
        "template_resolution": normalized.get("template_resolution") or "",
        "template_ligand_resname": normalized.get("template_ligand_resname") or normalized.get("ligand_resname") or "",
        "data_source": normalized.get("data_source") or "HKPocket",
        "pdb_path": pdb_path,
    }


def load_hkpocket_rows() -> list[dict]:
    if not HKPOCKET_MANIFEST.exists():
        return []
    rows = []
    for row in csv.DictReader(HKPOCKET_MANIFEST.open()):
        normalized = normalize_hkpocket_row(row)
        if normalized:
            rows.append(normalized)
    return rows


def deduplicate_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for row in rows:
        key = (
            row.get("pdb_id", ""),
            row.get("protein_chain", ""),
            row.get("ligand_resname", ""),
            row.get("ligand_smiles", ""),
            row.get("label_type", ""),
            row.get("label_value", ""),
            row.get("template_structure_id", ""),
            row.get("data_source", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def write_csv(rows: list[dict], path: Path):
    if not rows:
        raise ValueError(f"No rows were generated for {path.name}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not KLIFS_STRUCTURES.exists():
        raise FileNotFoundError(f"Missing KLIFS structures CSV: {KLIFS_STRUCTURES}")

    print("Loading KLIFS templates...")
    best_structures = load_klifs_templates()
    klifs_rows = klifs_rows_to_manifest_rows(
        list(best_structures.values()),
        source_label="KLIFS template structure",
    )
    print(f"KLIFS template rows: {len(klifs_rows)}")

    print("Loading all KLIFS structure-backed rows...")
    all_klifs_structures = load_all_klifs_structures()
    klifs_rows_all = klifs_rows_to_manifest_rows(
        all_klifs_structures,
        source_label="KLIFS all structure-backed",
    )
    print(f"KLIFS all rows: {len(klifs_rows_all)}")

    print("Loading HKPocket rows if present...")
    hkpocket_rows = load_hkpocket_rows()
    print(f"HKPocket rows: {len(hkpocket_rows)}")

    combined_rows = deduplicate_rows(klifs_rows + hkpocket_rows)
    print(f"Combined rows: {len(combined_rows)}")

    combined_rows_all = deduplicate_rows(klifs_rows_all + hkpocket_rows)
    print(f"Combined all rows: {len(combined_rows_all)}")

    write_csv(klifs_rows, OUTPUT_KLIFS_ONLY)
    write_csv(klifs_rows_all, OUTPUT_KLIFS_ALL)
    write_csv(combined_rows, OUTPUT_MANIFEST)
    write_csv(combined_rows_all, OUTPUT_MANIFEST_ALL)

    summary = [
        f"KLIFS targets selected: {len(best_structures)}",
        f"KLIFS manifest rows: {len(klifs_rows)}",
        f"KLIFS all-structure rows: {len(klifs_rows_all)}",
        f"HKPocket manifest rows: {len(hkpocket_rows)}",
        f"Combined manifest rows: {len(combined_rows)}",
        f"Combined all-structure manifest rows: {len(combined_rows_all)}",
        "",
        "KLIFS rows are structure-backed template complexes using local PDB files.",
        "KLIFS all-structure rows include every resolvable KLIFS structure in structures_list.csv.",
        "HKPocket rows are merged when a HKPocket manifest exists at data/raw/hkpocket/hkpocket_manifest.csv.",
        "Prediction-time inputs remain: one pocket structure plus candidate ligand SMILES.",
    ]
    OUTPUT_SUMMARY.write_text("\n".join(summary) + "\n")

    print(f"Wrote KLIFS-only manifest to {OUTPUT_KLIFS_ONLY}")
    print(f"Wrote KLIFS all-structure manifest to {OUTPUT_KLIFS_ALL}")
    print(f"Wrote combined manifest to {OUTPUT_MANIFEST}")
    print(f"Wrote combined all-structure manifest to {OUTPUT_MANIFEST_ALL}")
    print(f"Wrote summary to {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
