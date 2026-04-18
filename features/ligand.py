import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, rdMolDescriptors
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

MORGAN_GENERATOR = AllChem.GetMorganGenerator(radius=2, fpSize=2048)

HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

HYDROPHOBIC_ATOMS = {6, 7, 8, 16}
PHARMACOPHORE_SMARTS = [
    Chem.MolFromSmarts("c1ncc(N)nc1"),      # aminopyrimidine-like
    Chem.MolFromSmarts("c1[nH]c2ccccc2[nH]1"),  # azaindole-like
    Chem.MolFromSmarts("c1ncc2ccnc(N)c2n1"),  # quinazoline-like
    Chem.MolFromSmarts("c1ccncc1"),          # pyridine-like
    Chem.MolFromSmarts("c1ncnc2[nH]cnc12"),  # purine-like
]
BASIC_GROUP = Chem.MolFromSmarts("[NX3;H2,H1,H3]")
ACIDIC_GROUP = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")

HBD_ATOMS = {7, 8}
HBA_ATOMS = {7, 8, 16}


def is_hydrophobic_atom(atom):
    return atom.GetAtomicNum() in HYDROPHOBIC_ATOMS and not atom.GetIsAromatic()


def is_hbond_donor(atom):
    return atom.GetAtomicNum() in HBD_ATOMS and atom.GetTotalNumHs() > 0 and atom.GetFormalCharge() <= 0


def is_hbond_acceptor(atom):
    return atom.GetAtomicNum() in HBA_ATOMS and atom.GetFormalCharge() <= 0


def is_protonatable_atom(atom):
    return atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 0 and atom.GetFormalCharge() <= 0


def compute_gasteiger_charges(mol):
    try:
        mol_h = Chem.AddHs(mol)
        AllChem.ComputeGasteigerCharges(mol_h)
        charges = []
        for atom in mol_h.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            if atom.HasProp('_GasteigerCharge'):
                charges.append(float(atom.GetProp('_GasteigerCharge')))
            else:
                charges.append(0.0)
        return charges
    except Exception:
        return [0.0] * mol.GetNumAtoms()


def compute_protonation_features(mol):
    basic_count = len(mol.GetSubstructMatches(BASIC_GROUP)) if BASIC_GROUP is not None else 0
    acidic_count = len(mol.GetSubstructMatches(ACIDIC_GROUP)) if ACIDIC_GROUP is not None else 0
    formal_charge = float(Chem.GetFormalCharge(mol))
    return np.array([basic_count, acidic_count, formal_charge], dtype=np.float32)


def compute_pharmacophore_flags(mol):
    flags = []
    for smarts in PHARMACOPHORE_SMARTS:
        if smarts is None:
            flags.append(0.0)
            continue
        flags.append(1.0 if mol.HasSubstructMatch(smarts) else 0.0)
    return np.array(flags, dtype=np.float32)


def compute_forcefield_energy(mol, use_3d: bool):
    if not use_3d:
        return 0.0
    mol_h = Chem.AddHs(mol)
    try:
        if AllChem.MMFFHasAllMoleculeProperties(mol_h):
            props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol_h)
        return float(ff.CalcEnergy())
    except Exception:
        return 0.0

BOND_TYPE_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, valid_values):
    return [1.0 if value == v else 0.0 for v in valid_values]


def atom_features(atom, partial_charge=0.0):
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetTotalNumHs(),
        atom.GetFormalCharge(),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(is_hydrophobic_atom(atom)),
        float(is_hbond_donor(atom)),
        float(is_hbond_acceptor(atom)),
        float(is_protonatable_atom(atom)),
        float(partial_charge),
    ] + one_hot(atom.GetHybridization(), HYBRIDIZATION_LIST)


def bond_features(bond):
    stereo = bond.GetStereo()
    stereo_one_hot = [
        float(stereo == Chem.rdchem.BondStereo.STEREONONE),
        float(stereo == Chem.rdchem.BondStereo.STEREOZ),
        float(stereo == Chem.rdchem.BondStereo.STEREOE),
    ]
    return [
        float(bond.GetIsAromatic()),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ] + one_hot(bond.GetBondType(), BOND_TYPE_LIST) + stereo_one_hot


def compute_molecular_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    pharmacophore_flags = compute_pharmacophore_flags(mol)
    protonation_flags = compute_protonation_features(mol)

    return np.concatenate([
        np.array([rdMolDescriptors.CalcExactMolWt(mol)], dtype=np.float32),
        np.array([Crippen.MolLogP(mol)], dtype=np.float32),
        np.array([rdMolDescriptors.CalcNumHBD(mol)], dtype=np.float32),
        np.array([rdMolDescriptors.CalcNumHBA(mol)], dtype=np.float32),
        np.array([rdMolDescriptors.CalcTPSA(mol)], dtype=np.float32),
        np.array([rdMolDescriptors.CalcNumRotatableBonds(mol)], dtype=np.float32),
        np.array([rdMolDescriptors.CalcNumRings(mol)], dtype=np.float32),
        np.array([rdMolDescriptors.CalcFractionCSP3(mol)], dtype=np.float32),
        protonation_flags,
        pharmacophore_flags,
    ], axis=0).astype(np.float32)


def featurize_ligand(smiles: str, use_3d: bool = False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Hydrogens are treated implicitly for the ligand graph.
    # RDKit MolFromSmiles produces a heavy-atom graph, and explicit H atoms
    # are only added temporarily for 3D coordinate generation and charge assignment.
    charges = compute_gasteiger_charges(mol)
    atom_feats = [atom_features(atom, partial_charge=charges[i]) for i, atom in enumerate(mol.GetAtoms())]
    bond_feats = []
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        bond_feat = bond_features(bond)
        bond_feats.append(bond_feat)
        bond_feats.append(bond_feat)

    fp = MORGAN_GENERATOR.GetFingerprint(mol)
    fp_bits = np.zeros((2048,), dtype=np.int32)
    ConvertToNumpyArray(fp, fp_bits)

    coords = None
    if use_3d:
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol_h)
        conf = mol_h.GetConformer()
        heavy_coords = []
        for atom in mol_h.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            heavy_coords.append(list(conf.GetAtomPosition(atom.GetIdx())))
        coords = np.array(heavy_coords, dtype=np.float32)
        atom_feats = [feat + coords[i].tolist() for i, feat in enumerate(atom_feats)]

    mol_desc = np.concatenate([
        compute_molecular_descriptors(smiles),
        np.array([compute_forcefield_energy(mol, use_3d)], dtype=np.float32),
    ], axis=0)
    return (
        fp_bits,
        np.array(atom_feats, dtype=np.float32),
        coords,
        np.array(edge_index, dtype=np.int64),
        np.array(bond_feats, dtype=np.float32),
        mol_desc,
    )
