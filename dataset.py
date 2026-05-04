import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
from torch.utils.data import Dataset
from torch_geometric.data import Data
from Bio.PDB import PDBParser

from .features import (
    featurize_ligand,
    extract_pocket_features,
    extract_pocket_features_from_site,
    find_pocket_centers,
    build_pocket_edge_index,
    build_pocket_edge_attr,
    resolve_pdb_file,
)


def load_manifest_records(manifest_path):
    frame = pd.read_csv(manifest_path)
    required = {"ligand_smiles", "pdb_path", "label_value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    # Drop rows with missing/non-finite core fields to avoid NaNs during training.
    frame = frame.dropna(subset=["ligand_smiles", "pdb_path", "label_value"]).copy()
    frame["label_value"] = pd.to_numeric(frame["label_value"], errors="coerce")
    frame = frame[np.isfinite(frame["label_value"].to_numpy())]

    records = []
    has_resname = "ligand_resname" in frame.columns
    for row in frame.itertuples(index=False):
        ligand_resname = str(getattr(row, "ligand_resname", "")).strip().upper() if has_resname else ""
        records.append((str(row.ligand_smiles), str(row.pdb_path), float(row.label_value), ligand_resname))
    return records


class LigandPocketDataset(Dataset):
    def __init__(self, records, use_3d=False, transform=None, cache_graphs=False):
        self.records = list(records)
        self.use_3d = use_3d
        self.transform = transform
        self.cache_graphs = cache_graphs
        self._cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if self.cache_graphs and idx in self._cache:
            lig_data, poc_data, interaction_data, label = self._cache[idx]
            return lig_data, poc_data, interaction_data, label

        record = self.records[idx]
        if len(record) >= 4:
            lig_smiles, pocket_pdb, label, ligand_resname = record[0], record[1], record[2], record[3]
        else:
            lig_smiles, pocket_pdb, label = record
            ligand_resname = ""

        lig_data = make_ligand_data(
            lig_smiles,
            use_3d=self.use_3d,
            pdb_file=pocket_pdb,
            ligand_resname=ligand_resname,
        )
        poc_data = make_pocket_data(pocket_pdb)
        interaction_data = make_interaction_data(lig_data, poc_data)
        if self.transform is not None:
            lig_data = self.transform(lig_data)
            poc_data = self.transform(poc_data)
            interaction_data = self.transform(interaction_data)
        out = (lig_data, poc_data, interaction_data, torch.tensor(float(label), dtype=torch.float))
        if self.cache_graphs:
            self._cache[idx] = out
        return out


def _compute_pocket_center_for_alignment(pdb_file: str):
    """Match pocket centering used by extract_pocket_features for coordinate alignment."""
    aa_set = {
        "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
        "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
    }
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("prot", resolve_pdb_file(pdb_file))
    except Exception:
        return None

    residue_coords = []
    for residue in structure.get_residues():
        if residue.id[0] != " ":
            continue
        if residue.get_resname() not in aa_set:
            continue
        backbone_coords = []
        for atom_name in ("N", "CA", "C", "O"):
            if atom_name in residue:
                backbone_coords.append(residue[atom_name].get_coord().astype(np.float32))
        if not backbone_coords:
            continue
        residue_coords.append(np.mean(backbone_coords, axis=0))

    if not residue_coords:
        return None
    return np.mean(np.vstack(residue_coords), axis=0)


def _extract_bound_ligand_coords(pdb_file: str, ligand_resname: str, expected_atoms: int):
    """Extract heavy-atom ligand coordinates from the bound complex PDB.

    Returns coordinates already centered into the same frame as pocket features.
    """
    resname = (ligand_resname or "").strip().upper()
    if not resname:
        return None

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", resolve_pdb_file(pdb_file))
    except Exception:
        return None

    candidates = []
    for residue in structure.get_residues():
        if residue.id[0] == " ":
            continue
        if residue.get_resname().strip().upper() != resname:
            continue
        coords = []
        for atom in residue.get_atoms():
            atom_name = atom.get_name().strip().upper()
            if atom_name.startswith("H"):
                continue
            coords.append(atom.get_coord().astype(np.float32))
        if coords:
            candidates.append(np.vstack(coords))

    if not candidates:
        return None

    # Pick the residue instance with atom count closest to ligand graph size.
    best = min(candidates, key=lambda arr: abs(arr.shape[0] - expected_atoms))
    if best.shape[0] != expected_atoms:
        return None

    center = _compute_pocket_center_for_alignment(pdb_file)
    if center is None:
        return None
    return best - center


def make_ligand_data(smiles: str, use_3d: bool = False, pdb_file: str = "", ligand_resname: str = "") -> Data:
    # Most generated manifests use placeholder ligand tags like [ANP], [8KZ], [0], etc.
    # These are not valid SMILES, so normalize them to a tiny valid ligand.
    normalized_smiles = smiles
    if re.fullmatch(r"\[[A-Za-z0-9]+\]", (normalized_smiles or "").strip()):
        normalized_smiles = "CC"

    try:
        _, atom_feats, coords, edge_index, edge_attr, mol_desc = featurize_ligand(normalized_smiles, use_3d=use_3d)
    except ValueError:
        _, atom_feats, coords, edge_index, edge_attr, mol_desc = featurize_ligand("CC", use_3d=use_3d)

    # Prefer bound ligand coordinates from the same complex when available.
    # This makes ligand-pocket interaction edges physically meaningful.
    bound_coords = _extract_bound_ligand_coords(
        pdb_file=pdb_file,
        ligand_resname=ligand_resname,
        expected_atoms=len(atom_feats),
    )
    if bound_coords is not None:
        coords = bound_coords
    elif coords is not None:
        # In screening we usually do not have a bound pose. Centering keeps generated
        # conformers in the same origin-centered frame as pocket coordinates.
        coords = coords - np.mean(coords, axis=0, keepdims=True)

    x = torch.tensor(atom_feats, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    mol_desc = torch.tensor(mol_desc, dtype=torch.float)
    pos = torch.tensor(coords, dtype=torch.float) if coords is not None else torch.empty((0, 3), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, mol_desc=mol_desc, pos=pos)


def make_pocket_data(
    pdb_file: str,
    threshold: float = 8.0,
    pocket_center=None,
    pocket_radius: float = 8.0,
    pocket_detector: str = None,
    pocket_n: int = 1,
) -> Data:
    """Create a pocket graph from a protein PDB.

    If pocket_center is provided, residues within pocket_radius of that point are used.
    If pocket_detector is provided, the center is found automatically using that method.
    This is the intended mode for unbound prediction structures where no ligand is present.
    If pocket_center and pocket_detector are both None, the current code extracts features from the full protein.
    """
    pdb_file = resolve_pdb_file(pdb_file)
    if pocket_center is None and pocket_detector is not None:
        centers = find_pocket_centers(pdb_file, method=pocket_detector, n_pockets=pocket_n, radius=pocket_radius)
        pocket_center = centers[0] if len(centers) > 0 else None
    if pocket_center is not None:
        res_feats, res_coords = extract_pocket_features_from_site(pdb_file, pocket_center, radius=pocket_radius)
    else:
        res_feats, res_coords = extract_pocket_features(pdb_file)
    x = torch.tensor(res_feats, dtype=torch.float)
    edge_index = build_pocket_edge_index(res_coords, threshold=threshold)
    edge_attr = build_pocket_edge_attr(res_coords, threshold=threshold)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    pos = torch.tensor(res_coords, dtype=torch.float) if res_coords is not None else torch.empty((0, 3), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


def make_interaction_data(lig_data: Data, poc_data: Data, cutoff: float = 6.0) -> Data:
    lig_x = lig_data.x
    poc_x = poc_data.x
    if lig_x.numel() == 0 or poc_x.numel() == 0:
        return Data(
            x=torch.empty((0, max(lig_x.size(1), poc_x.size(1))), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 3), dtype=torch.float),
            pos=torch.empty((0, 3), dtype=torch.float),
        )

    target_dim = max(lig_x.size(1), poc_x.size(1))
    lig_x = F.pad(lig_x, (0, target_dim - lig_x.size(1)))
    poc_x = F.pad(poc_x, (0, target_dim - poc_x.size(1)))
    x = torch.cat([lig_x, poc_x], dim=0)

    lig_pos = getattr(lig_data, "pos", None)
    poc_pos = getattr(poc_data, "pos", None)
    if lig_pos is None or poc_pos is None or lig_pos.numel() == 0 or poc_pos.numel() == 0:
        return Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 3), dtype=torch.float), pos=torch.empty((0, 3), dtype=torch.float))

    dist_mat = torch.cdist(lig_pos, poc_pos)
    valid = dist_mat < cutoff
    if valid.sum() == 0:
        return Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 3), dtype=torch.float), pos=torch.cat([lig_pos, poc_pos], dim=0))

    lig_idx, poc_idx = valid.nonzero(as_tuple=True)
    edge_index = torch.stack([lig_idx, poc_idx + lig_x.size(0)], dim=0)
    distances = dist_mat[lig_idx, poc_idx].unsqueeze(-1)

    lig_donor = lig_data.x[lig_idx, 7]
    lig_acceptor = lig_data.x[lig_idx, 8]
    poc_donor = poc_data.x[poc_idx, 22]
    poc_acceptor = poc_data.x[poc_idx, 23]
    hbond_possible = (((lig_donor > 0.5) & (poc_acceptor > 0.5)) | ((lig_acceptor > 0.5) & (poc_donor > 0.5))).unsqueeze(-1).float()
    hydrophobic_contact = ((lig_data.x[lig_idx, 6] > 0.5) & (poc_data.x[poc_idx, 20] > 0)).unsqueeze(-1).float()
    edge_attr = torch.cat([distances, hydrophobic_contact, hbond_possible], dim=1)
    pos = torch.cat([lig_pos, poc_pos], dim=0)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


def collate_fn(batch):
    from torch_geometric.data import Batch

    lig_graphs = [lig for lig, _, _, _ in batch]
    poc_graphs = [poc for _, poc, _, _ in batch]
    int_graphs = [interaction for _, _, interaction, _ in batch]
    labels = torch.stack([label.float() for _, _, _, label in batch])
    lig_batch = Batch.from_data_list(lig_graphs)
    poc_batch = Batch.from_data_list(poc_graphs)
    int_batch = Batch.from_data_list(int_graphs)
    lig_batch.mol_desc = torch.stack([lig.mol_desc for lig, _, _, _ in batch])
    return lig_batch, poc_batch, int_batch, labels
