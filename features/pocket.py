import glob
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from Bio import PDB

AMINO_ACIDS = [
    'ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU',
    'MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR',
]
AA_MAP = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']
HYDROPHOBICITY = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2,
}
CHARGE = {
    'ALA': 0, 'ARG': 1, 'ASN': 0, 'ASP': -1, 'CYS': 0,
    'GLN': 0, 'GLU': -1, 'GLY': 0, 'HIS': 0, 'ILE': 0,
    'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0, 'PRO': 0,
    'SER': 0, 'THR': 0, 'TRP': 0, 'TYR': 0, 'VAL': 0,
}
HBD_RESIDUES = {'ASN', 'GLN', 'HIS', 'LYS', 'ARG', 'SER', 'THR', 'TYR'}
HBA_RESIDUES = {'ASN', 'GLN', 'ASP', 'GLU', 'SER', 'THR', 'TYR'}


def extract_pocket_features(pdb_file: str):
    pdb_file = resolve_pdb_file(pdb_file)
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_file)
    residue_features = []
    coords = []

    for residue in structure.get_residues():
        if residue.id[0] != ' ':
            continue
        aa_name = residue.get_resname()
        if aa_name not in AA_MAP:
            continue
        backbone_coords = []
        for atom_name in BACKBONE_ATOMS:
            if atom_name in residue:
                backbone_coords.append(residue[atom_name].get_coord().astype(np.float32))
        if len(backbone_coords) == 0:
            continue
        aa_vec = np.zeros((20,), dtype=np.float32)
        aa_vec[AA_MAP[aa_name]] = 1.0
        hydrophobicity = np.array([HYDROPHOBICITY.get(aa_name, 0.0)], dtype=np.float32)
        charge = np.array([CHARGE.get(aa_name, 0.0)], dtype=np.float32)
        hbd = np.array([1.0 if aa_name in HBD_RESIDUES else 0.0], dtype=np.float32)
        hba = np.array([1.0 if aa_name in HBA_RESIDUES else 0.0], dtype=np.float32)
        residue_features.append(np.concatenate([aa_vec, hydrophobicity, charge, hbd, hba], axis=0))
        coords.append(np.mean(backbone_coords, axis=0))

    if len(residue_features) == 0:
        return np.zeros((0, 25), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    coords = np.vstack(coords)
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    local_density = np.sum((dist_mat > 0.0) & (dist_mat < 6.0), axis=1, keepdims=True).astype(np.float32)
    residue_features = [np.concatenate([feat, local_density[i]], axis=0) for i, feat in enumerate(residue_features)]
    coords = np.vstack(coords)
    centroid = coords.mean(axis=0, keepdims=True)
    coords -= centroid
    return np.vstack(residue_features), coords


def get_atom_centroid(pdb_file: str, exclude_hydrogens: bool = True):
    structure = get_structure(pdb_file)
    coords = []
    for atom in structure.get_atoms():
        if exclude_hydrogens and not is_heavy_atom(atom):
            continue
        coords.append(atom.get_coord().astype(np.float32))
    if len(coords) == 0:
        return None
    return np.mean(np.vstack(coords), axis=0)


def find_pocket_centers_by_fpocket(pdb_file: str, n_pockets: int = 1):
    fpocket_bin = shutil.which("fpocket")
    if fpocket_bin is None:
        raise FileNotFoundError("fpocket executable not found in PATH")
    pdb_path = Path(resolve_pdb_file(pdb_file))
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tmp_pdb = tmpdir_path / pdb_path.name
        shutil.copy(pdb_path, tmp_pdb)
        subprocess.run([fpocket_bin, "-f", str(tmp_pdb)], cwd=tmpdir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out_dirs = list(tmpdir_path.glob(f"{pdb_path.stem}_out*"))
        if not out_dirs:
            raise RuntimeError("fpocket did not create an output folder")
        pocket_files = sorted(out_dirs[0].glob("pocket*_atm.pdb"))
        if not pocket_files:
            raise RuntimeError("fpocket did not generate pocket atom files")

        centers = []
        for pocket_file in pocket_files[:n_pockets]:
            centroid = get_atom_centroid(str(pocket_file))
            if centroid is not None:
                centers.append(centroid)
        return centers


def find_pocket_centers_by_density(pdb_file: str, n_pockets: int = 1, radius: float = 8.0):
    structure = get_structure(pdb_file)
    residue_centroids = []
    for residue in structure.get_residues():
        if residue.id[0] != ' ':
            continue
        heavy_coords = [atom.get_coord().astype(np.float32) for atom in residue if is_heavy_atom(atom)]
        if len(heavy_coords) == 0:
            continue
        residue_centroids.append(np.mean(heavy_coords, axis=0))

    if len(residue_centroids) == 0:
        return []

    coords = np.vstack(residue_centroids)
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    density = np.sum(dist_mat < radius, axis=1)
    pocket_idx = np.argsort(density)[-n_pockets:][::-1]
    return [coords[i] for i in pocket_idx]


def find_pocket_centers(pdb_file: str, method: str = "fpocket", n_pockets: int = 1, radius: float = 8.0):
    if method == "fpocket":
        try:
            centers = find_pocket_centers_by_fpocket(pdb_file, n_pockets=n_pockets)
            if centers:
                return centers
        except Exception:
            pass
        return find_pocket_centers_by_density(pdb_file, n_pockets=n_pockets, radius=radius)
    if method == "density":
        return find_pocket_centers_by_density(pdb_file, n_pockets=n_pockets, radius=radius)
    raise ValueError(f"Unsupported pocket detection method: {method}")


def get_structure(pdb_file: str):
    pdb_file = resolve_pdb_file(pdb_file)
    parser = PDB.PDBParser(QUIET=True)
    return parser.get_structure('prot', pdb_file)


def resolve_pdb_file(pdb_file: str):
    project_root = Path(__file__).resolve().parents[1]
    path = Path(pdb_file)
    candidates = [path]

    # Legacy manifests from other clones can contain absolute paths rooted at
    # /ocean/projects/.../<user>/projects. Remap through the local project root.
    path_str = str(path)
    data_marker = "/data/"
    if data_marker in path_str:
        rel_from_data = path_str.split(data_marker, 1)[1]
        candidates.append(project_root / "data" / rel_from_data)

    if path.suffix:
        candidates.append(path.with_suffix('.pdb'))
        candidates.append(path.with_suffix('.ent'))
    else:
        candidates.append(path.with_suffix('.pdb'))
        candidates.append(path.with_suffix('.ent'))

    stem = path.stem.lower()
    parent = path.parent
    candidates.extend([
        parent / f"{stem}.pdb",
        parent / f"{stem}.ent",
        parent / f"pdb{stem}.pdb",
        parent / f"pdb{stem}.ent",
        parent / f"PDB{stem}.pdb",
        parent / f"PDB{stem}.ent",
    ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(path)


def is_heavy_atom(atom):
    name = atom.get_name().strip()
    return not name.startswith('H')


def get_ligand_centroid(pdb_file: str, exclude_resnames=('HOH',)):
    """Compute centroid of hetero/ligand atoms in a bound complex.

    This helper assumes the input PDB contains a bound non-protein ligand.
    It should not be used as the primary method for unbound prediction structures.
    For new target PDBs or AlphaFold models, define a pocket center from a cavity detector
    or active-site residue selection instead.
    """
    structure = get_structure(pdb_file)
    coords = []
    for residue in structure.get_residues():
        if residue.id[0] == ' ':
            continue
        if residue.get_resname() in exclude_resnames:
            continue
        for atom in residue:
            if not is_heavy_atom(atom):
                continue
            coords.append(atom.get_coord().astype(np.float32))
    if len(coords) == 0:
        return None
    return np.mean(np.vstack(coords), axis=0)


def select_pocket_residues_by_center(pdb_file: str, center, radius: float = 8.0):
    structure = get_structure(pdb_file)
    center = np.asarray(center, dtype=np.float32)
    selected = []
    for residue in structure.get_residues():
        if residue.id[0] != ' ':
            continue
        for atom in residue:
            if not is_heavy_atom(atom):
                continue
            if np.linalg.norm(atom.get_coord().astype(np.float32) - center) <= radius:
                selected.append(residue)
                break
    return selected


def extract_pocket_features_from_residues(residues):
    residue_features = []
    coords = []

    for residue in residues:
        aa_name = residue.get_resname()
        if aa_name not in AA_MAP:
            continue
        backbone_coords = []
        for atom_name in BACKBONE_ATOMS:
            if atom_name in residue:
                backbone_coords.append(residue[atom_name].get_coord().astype(np.float32))
        if len(backbone_coords) == 0:
            continue
        aa_vec = np.zeros((20,), dtype=np.float32)
        aa_vec[AA_MAP[aa_name]] = 1.0
        hydrophobicity = np.array([HYDROPHOBICITY.get(aa_name, 0.0)], dtype=np.float32)
        charge = np.array([CHARGE.get(aa_name, 0.0)], dtype=np.float32)
        hbd = np.array([1.0 if aa_name in HBD_RESIDUES else 0.0], dtype=np.float32)
        hba = np.array([1.0 if aa_name in HBA_RESIDUES else 0.0], dtype=np.float32)
        residue_features.append(np.concatenate([aa_vec, hydrophobicity, charge, hbd, hba], axis=0))
        coords.append(np.mean(backbone_coords, axis=0))

    if len(residue_features) == 0:
        return np.zeros((0, 25), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    coords = np.vstack(coords)
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    local_density = np.sum((dist_mat > 0.0) & (dist_mat < 6.0), axis=1, keepdims=True).astype(np.float32)
    residue_features = [np.concatenate([feat, local_density[i]], axis=0) for i, feat in enumerate(residue_features)]
    coords = np.vstack(coords)
    centroid = coords.mean(axis=0, keepdims=True)
    coords -= centroid
    return np.vstack(residue_features), coords


def extract_pocket_features_from_site(pdb_file: str, center, radius: float = 8.0):
    residues = select_pocket_residues_by_center(pdb_file, center, radius=radius)
    return extract_pocket_features_from_residues(residues)


def build_pocket_edge_index(coords: np.ndarray, threshold: float = 8.0):
    if coords.shape[0] == 0:
        return np.zeros((2, 0), dtype=np.int64)
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    edges = np.vstack(np.where((dist_mat > 0.0) & (dist_mat < threshold))).astype(np.int64)
    return edges


def build_pocket_edge_attr(coords: np.ndarray, threshold: float = 8.0):
    if coords.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32)
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    valid = (dist_mat > 0.0) & (dist_mat < threshold)
    distances = dist_mat[valid].astype(np.float32)
    return distances.reshape(-1, 1)
