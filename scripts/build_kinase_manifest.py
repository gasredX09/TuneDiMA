import csv
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

KLIFS_STRUCTURES = RAW_DIR / "klifs" / "structures_list.csv"
CHEMBL_DB = RAW_DIR / "chembl" / "chembl_36" / "chembl_36_sqlite" / "chembl_36.db"
PDB_DIR = RAW_DIR / "pdb"
MANIFEST_TEMPLATE = RAW_DIR / "manifests" / "kinase_complex_manifest.csv"
OUTPUT_MANIFEST = PROCESSED_DIR / "kinase_activity_manifest.csv"
OUTPUT_SUMMARY = PROCESSED_DIR / "kinase_activity_manifest_summary.txt"

SPECIES_NORMALIZATION = {
    "HUMAN": "HOMO SAPIENS",
    "HOMO SAPIENS": "HOMO SAPIENS",
    "MOUSE": "MUS MUSCULUS",
    "MUS MUSCULUS": "MUS MUSCULUS",
    "RAT": "RATTUS NORVEGICUS",
    "RATTUS NORVEGICUS": "RATTUS NORVEGICUS",
}


def safe_float(value, default=float("inf")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_species_name(value: str) -> str:
    raw = str(value or "").strip().upper()
    return SPECIES_NORMALIZATION.get(raw, raw)


def resolve_pdb_path(pdb_code: str) -> Path | None:
    code = pdb_code.strip().lower()
    candidates = [
        PDB_DIR / f"{code}.pdb",
        PDB_DIR / f"{code}.ent",
        PDB_DIR / f"pdb{code}.pdb",
        PDB_DIR / f"pdb{code}.ent",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_best_klifs_structures() -> dict[tuple[str, str], dict]:
    rows = list(csv.DictReader(KLIFS_STRUCTURES.open()))
    best_by_target = {}

    for row in rows:
        pdb_code = row["pdb"].lower()
        pdb_path = resolve_pdb_path(pdb_code)
        if pdb_path is None:
            continue

        key = (row["kinase"].strip().upper(), normalize_species_name(row["species"]))
        score = (
            -safe_float(row.get("quality_score"), default=-1.0),
            safe_float(row.get("missing_residues"), default=9999.0),
            safe_float(row.get("missing_atoms"), default=9999.0),
            safe_float(row.get("resolution"), default=9999.0),
            1 if str(row.get("allosteric_ligand", "0")) == "1" else 0,
            pdb_code,
        )
        current = best_by_target.get(key)
        if current is None or score < current["_score"]:
            row = dict(row)
            row["_score"] = score
            row["_pdb_path"] = str(pdb_path)
            best_by_target[key] = row

    for row in best_by_target.values():
        row.pop("_score", None)
    return best_by_target


def query_filtered_chembl_activities(best_structures: dict[tuple[str, str], dict]) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(CHEMBL_DB))
    conn.row_factory = sqlite3.Row

    symbol_rows = conn.execute(
        """
        SELECT DISTINCT
            UPPER(csyn.component_synonym) AS gene_symbol,
            td.organism AS organism,
            td.tid AS tid
        FROM target_dictionary td
        JOIN target_components tc
            ON td.tid = tc.tid
        JOIN component_synonyms csyn
            ON tc.component_id = csyn.component_id
        WHERE td.target_type = 'SINGLE PROTEIN'
          AND csyn.syn_type = 'GENE_SYMBOL'
          AND csyn.component_synonym IS NOT NULL
        """
    ).fetchall()

    symbol_to_tids = {}
    for row in symbol_rows:
        key = (str(row["gene_symbol"]).strip().upper(), normalize_species_name(row["organism"]))
        symbol_to_tids.setdefault(key, set()).add(int(row["tid"]))

    matched_tids = set()
    for kinase_name, species in best_structures.keys():
        matched_tids.update(symbol_to_tids.get((kinase_name, species), set()))

    # Fallback: if symbol mapping misses a target, try exact pref_name match.
    if not matched_tids:
        target_rows = conn.execute(
            """
            SELECT tid, pref_name, organism
            FROM target_dictionary
            WHERE target_type = 'SINGLE PROTEIN'
            """
        ).fetchall()
        target_lookup = set(best_structures.keys())
        for row in target_rows:
            key = (str(row["pref_name"]).strip().upper(), normalize_species_name(row["organism"]))
            if key in target_lookup:
                matched_tids.add(int(row["tid"]))

    query = """
    SELECT
        td.pref_name AS target_name,
        td.organism AS organism,
        cs.accession AS accession,
        cst.canonical_smiles AS canonical_smiles,
        a.standard_type AS standard_type,
        AVG(a.pchembl_value) AS mean_pchembl_value,
        COUNT(*) AS n_measurements,
        GROUP_CONCAT(DISTINCT md.chembl_id) AS molecule_chembl_ids
    FROM activities a
    JOIN assays ass
        ON a.assay_id = ass.assay_id
    JOIN target_dictionary td
        ON ass.tid = td.tid
    JOIN target_components tc
        ON td.tid = tc.tid
    JOIN component_sequences cs
        ON tc.component_id = cs.component_id
    JOIN molecule_dictionary md
        ON a.molregno = md.molregno
    JOIN compound_structures cst
        ON md.molregno = cst.molregno
    WHERE td.target_type = 'SINGLE PROTEIN'
            AND td.tid = ?
      AND ass.confidence_score >= 8
      AND a.standard_flag = 1
      AND a.standard_relation = '='
      AND a.standard_type IN ('IC50', 'Ki', 'Kd')
      AND a.standard_units = 'nM'
      AND a.standard_value IS NOT NULL
      AND a.pchembl_value IS NOT NULL
      AND cst.canonical_smiles IS NOT NULL
      AND md.structure_type = 'MOL'
      AND COALESCE(md.inorganic_flag, 0) = 0
      AND COALESCE(md.polymer_flag, 0) = 0
    GROUP BY
        td.pref_name,
        td.organism,
        cs.accession,
        cst.canonical_smiles,
        a.standard_type
    """
    rows = []
    for tid in matched_tids:
        rows.extend(conn.execute(query, (tid,)).fetchall())
    conn.close()
    return rows


def aggregate_activities(rows: list[sqlite3.Row]) -> list[dict]:
    aggregated = []
    for row in rows:
        aggregated.append(
            {
                "target_name": row["target_name"].strip().upper(),
                "species": normalize_species_name(row["organism"]),
                "uniprot_id": row["accession"],
                "ligand_smiles": row["canonical_smiles"],
                "label_type": "pChEMBL",
                "label_value": round(float(row["mean_pchembl_value"]), 4),
                "label_unit": "unitless",
                "activity_type": row["standard_type"],
                "n_measurements": int(row["n_measurements"]),
                "molecule_chembl_ids": row["molecule_chembl_ids"] or "",
            }
        )
    return aggregated


def build_manifest_rows(best_structures: dict, activity_rows: list[dict]) -> list[dict]:
    manifest_rows = []
    for idx, activity in enumerate(activity_rows, start=1):
        key = (activity["target_name"], activity["species"])
        template = best_structures.get(key)
        if template is None:
            continue

        complex_id = f"{template['pdb'].lower()}_{template['chain']}_{idx}"
        manifest_rows.append(
            {
                "complex_id": complex_id,
                "pdb_id": template["pdb"].lower(),
                "protein_chain": template["chain"],
                "ligand_resname": template["ligand"],
                "ligand_smiles": activity["ligand_smiles"],
                "target_name": template["kinase"],
                "species": template["species"],
                "uniprot_id": activity["uniprot_id"],
                "label_type": activity["label_type"],
                "label_value": activity["label_value"],
                "label_unit": activity["label_unit"],
                "activity_type": activity["activity_type"],
                "n_measurements": activity["n_measurements"],
                "molecule_chembl_ids": activity["molecule_chembl_ids"],
                "template_structure_id": template["structure_ID"],
                "template_quality_score": template["quality_score"],
                "template_resolution": template["resolution"],
                "template_ligand_resname": template["ligand"],
                "data_source": "ChEMBL activity + KLIFS template structure",
                "pdb_path": template["_pdb_path"],
            }
        )
    return manifest_rows


def build_fallback_manifest_rows(best_structures: dict) -> list[dict]:
    rows = []
    for idx, template in enumerate(best_structures.values(), start=1):
        quality = safe_float(template.get("quality_score"), default=float("nan"))
        if quality != quality:
            continue

        complex_id = f"{template['pdb'].lower()}_{template['chain']}_{idx}"
        rows.append(
            {
                "complex_id": complex_id,
                "pdb_id": template["pdb"].lower(),
                "protein_chain": template["chain"],
                "ligand_resname": template["ligand"],
                "ligand_smiles": "",
                "target_name": template["kinase"],
                "species": normalize_species_name(template["species"]),
                "uniprot_id": "",
                "label_type": "KLIFS_quality_score",
                "label_value": round(float(quality), 4),
                "label_unit": "unitless",
                "activity_type": "template_quality",
                "n_measurements": 1,
                "molecule_chembl_ids": "",
                "template_structure_id": template["structure_ID"],
                "template_quality_score": template["quality_score"],
                "template_resolution": template["resolution"],
                "template_ligand_resname": template["ligand"],
                "data_source": "KLIFS template fallback (no ChEMBL match)",
                "pdb_path": template["_pdb_path"],
            }
        )
    return rows


def write_manifest(rows: list[dict], path: Path):
    if not rows:
        raise ValueError("No manifest rows were generated.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_compatibility_template(rows: list[dict], path: Path):
    fieldnames = [
        "complex_id",
        "pdb_id",
        "protein_chain",
        "ligand_resname",
        "ligand_smiles",
        "target_name",
        "uniprot_id",
        "label_type",
        "label_value",
        "label_unit",
        "data_source",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_summary(best_structures: dict, activity_rows: list[dict], manifest_rows: list[dict], path: Path):
    covered_targets = len({(row["target_name"], row["species"]) for row in manifest_rows})
    summary = [
        f"Representative KLIFS targets: {len(best_structures)}",
        f"Filtered ChEMBL target-ligand aggregates: {len(activity_rows)}",
        f"Manifest rows written: {len(manifest_rows)}",
        f"Covered kinase/species targets: {covered_targets}",
        "",
        "Important assumption:",
        "Each row pairs a ChEMBL ligand activity with a representative KLIFS structure",
        "for the same kinase target. The ligand in the KLIFS template may differ from",
        "the ChEMBL ligand SMILES used for supervision.",
    ]
    path.write_text("\n".join(summary) + "\n")


def main():
    if not KLIFS_STRUCTURES.exists():
        raise FileNotFoundError(f"Missing KLIFS structures CSV: {KLIFS_STRUCTURES}")
    if not CHEMBL_DB.exists():
        raise FileNotFoundError(f"Missing ChEMBL SQLite DB: {CHEMBL_DB}")

    print("Selecting representative KLIFS structures...")
    best_structures = load_best_klifs_structures()
    print(f"Selected {len(best_structures)} kinase/species template structures")

    print("Querying filtered ChEMBL activities...")
    chembl_rows = query_filtered_chembl_activities(best_structures)
    print(f"Loaded {len(chembl_rows)} filtered ChEMBL activity rows")

    print("Aggregating ChEMBL activities by target and ligand...")
    activity_rows = aggregate_activities(chembl_rows)
    print(f"Aggregated to {len(activity_rows)} target-ligand rows")

    print("Joining activities to KLIFS templates...")
    manifest_rows = build_manifest_rows(best_structures, activity_rows)
    if not manifest_rows:
        print("No ChEMBL-linked rows found; building KLIFS-only fallback manifest...")
        manifest_rows = build_fallback_manifest_rows(best_structures)
    print(f"Built {len(manifest_rows)} manifest rows")

    write_manifest(manifest_rows, OUTPUT_MANIFEST)
    write_compatibility_template(manifest_rows, MANIFEST_TEMPLATE)
    write_summary(best_structures, activity_rows, manifest_rows, OUTPUT_SUMMARY)
    print(f"Wrote manifest to {OUTPUT_MANIFEST}")
    print(f"Wrote compatibility manifest to {MANIFEST_TEMPLATE}")
    print(f"Wrote summary to {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
