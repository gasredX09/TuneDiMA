from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors

PAINS_SMARTS = [
    # Minimal example subset of PAINS patterns
    "[cR1]1[cR2][cR3]([#7])=c([#6])[cR4]1",
    "[#7]c1cc2ccccc2[nH]1",
]


def is_pain(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for smarts in PAINS_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(patt):
            return True
    return False


def passes_lipinski(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return (
        rdMolDescriptors.CalcMolWt(mol) <= 500
        and rdMolDescriptors.CalcNumHBD(mol) <= 5
        and rdMolDescriptors.CalcNumHBA(mol) <= 10
        and Crippen.MolLogP(mol) <= 5
    )


def filter_library(smiles_list, filter_func=None):
    if filter_func is None:
        filter_func = lambda s: passes_lipinski(s) and not is_pain(s)
    return [s for s in smiles_list if filter_func(s)]


def screen_smiles(model, smiles_list, pocket_data, device="cpu"):
    import torch
    from .dataset import make_ligand_data, make_interaction_data

    model.eval()
    predictions = []
    with torch.no_grad():
        for smiles in smiles_list:
            lig_data = make_ligand_data(smiles)
            interaction_data = make_interaction_data(lig_data, pocket_data)
            lig_batch = lig_data.to(device)
            poc_batch = pocket_data.to(device)
            int_batch = interaction_data.to(device)
            logits = model(lig_batch, poc_batch, int_batch)
            predictions.append(torch.sigmoid(logits).item())
    return predictions


if __name__ == "__main__":
    print("Library screening utilities for Lipinski and PAINS filtering.")
