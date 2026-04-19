import gzip
import tarfile
from pathlib import Path

import requests
from Bio.PDB import PDBList

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CHEMBL_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_sqlite.tar.gz"
BINDINGDB_URL = "https://www.bindingdb.org/bind/downloads/BindingDB_All.tsv.gz"
DEFAULT_KINASE_PDB_IDS = ["6CMK", "5EWK", "1EWK"]


def download_file(url: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"Skipping existing file: {target}")
        return
    print(f"Downloading {url} -> {target}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(target, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def extract_tarball(source: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    if source.suffixes[-2:] == [".tar", ".gz"] or source.suffix == ".tgz":
        with tarfile.open(source, "r:gz") as tar:
            tar.extractall(path=destination)
    else:
        raise ValueError(f"Unsupported archive type: {source}")


def extract_gzip(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = destination
    with gzip.open(source, "rb") as f_in:
        with open(target, "wb") as f_out:
            f_out.write(f_in.read())


def download_chembl():
    chembl_dir = RAW_DIR / "chembl"
    chembl_archive = chembl_dir / "chembl_latest.tar.gz"
    download_file(CHEMBL_URL, chembl_archive)
    extract_tarball(chembl_archive, chembl_dir)
    print(f"ChEMBL data available in: {chembl_dir}")


def download_bindingdb():
    bindingdb_dir = RAW_DIR / "bindingdb"
    bindingdb_archive = bindingdb_dir / "BindingDB_All.tsv.gz"
    download_file(BINDINGDB_URL, bindingdb_archive)
    extracted = bindingdb_dir / "BindingDB_All.tsv"
    extract_gzip(bindingdb_archive, extracted)
    print(f"BindingDB data available in: {bindingdb_dir}")


def download_pubchem_assays():
    pubchem_dir = RAW_DIR / "pubchem"
    pubchem_dir.mkdir(parents=True, exist_ok=True)
    query = "glutamate+receptor+kinase"
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/assayid/CSV?query={query}"
    target = pubchem_dir / "pubchem_assays.csv"
    print(f"Downloading PubChem assay summary to {target}")
    try:
        download_file(url, target)
    except Exception as exc:
        print(f"Unable to download PubChem assays automatically: {exc}")
        print("Please use the PubChem PUG REST API or manual download and save to {}".format(target))
    print(f"PubChem data available in: {pubchem_dir}")


def download_pdb_structures():
    pdb_dir = RAW_DIR / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    pdbl = PDBList()
    for pdb_id in DEFAULT_KINASE_PDB_IDS:
        print(f"Downloading PDB structure {pdb_id}")
        pdbl.retrieve_pdb_file(pdb_id, pdir=str(pdb_dir), file_format="pdb")
    print(f"PDB structures downloaded to: {pdb_dir}")


def prepare_kinase_complex_layout():
    """Create a kinase-complex-oriented raw data layout and instructions.

    The current project models bound protein-ligand complexes, so broad assay dumps
    alone are not enough. This helper makes the intended structure explicit:
    KLIFS/PDB-derived bound complexes for structure, plus BindingDB/ChEMBL for labels.
    """
    klifs_dir = RAW_DIR / "klifs"
    complexes_dir = RAW_DIR / "kinase_complexes"
    manifests_dir = RAW_DIR / "manifests"
    klifs_dir.mkdir(parents=True, exist_ok=True)
    complexes_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    klifs_readme = klifs_dir / "README.txt"
    if not klifs_readme.exists():
        klifs_readme.write_text(
            "\n".join(
                [
                    "Kinase structure guidance",
                    "=========================",
                    "",
                    "Use KLIFS as the primary source of kinase bound complexes and pocket annotations.",
                    "Place exported KLIFS tables and structure metadata in this folder.",
                    "",
                    "Recommended contents:",
                    "- KLIFS structure table with kinase IDs, PDB IDs, chain IDs, and ligand names",
                    "- pocket annotation tables for kinase residues",
                    "- any structure-level metadata needed to map complexes to labels",
                    "",
                    "This repository treats ChEMBL and BindingDB as affinity-label sources, not as",
                    "a substitute for bound kinase complex structures.",
                ]
            )
            + "\n"
        )

    manifest_template = manifests_dir / "kinase_complex_manifest.csv"
    if not manifest_template.exists():
        manifest_template.write_text(
            "complex_id,pdb_id,protein_chain,ligand_resname,ligand_smiles,target_name,uniprot_id,label_type,label_value,label_unit,data_source\n"
        )

    complexes_readme = complexes_dir / "README.txt"
    if not complexes_readme.exists():
        complexes_readme.write_text(
            "\n".join(
                [
                    "Store curated kinase-bound complexes here.",
                    "",
                    "Each row in data/raw/manifests/kinase_complex_manifest.csv should correspond",
                    "to one bound kinase-ligand complex with a structure-backed pocket.",
                    "",
                    "Suggested workflow:",
                    "1. Export/select kinase structures from KLIFS or PDB.",
                    "2. Keep complexes with a bound small-molecule ligand.",
                    "3. Map those complexes to ChEMBL or BindingDB affinity labels.",
                    "4. Record the mapping in the manifest CSV.",
                ]
            )
            + "\n"
        )


def download_iuphar():
    iuphar_dir = RAW_DIR / "iuphar"
    iuphar_dir.mkdir(parents=True, exist_ok=True)
    print("IUPHAR downloads are typically provided behind a license page.")
    print("Place the IUPHAR assay CSVs into: {}".format(iuphar_dir))


def main():
    print("Starting dataset download for ChEMBL, BindingDB, PubChem, and PDB.")
    prepare_kinase_complex_layout()
    download_chembl()
    download_bindingdb()
    download_pubchem_assays()
    download_pdb_structures()
    download_iuphar()
    print("Download script completed.")


if __name__ == "__main__":
    main()
