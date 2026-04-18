# Chemistry-Aware Neural Network Pipeline

This folder implements the full pipeline described in the PDF:
- data collection and curation for kinase and glutamate receptor binding assays
- ligand feature extraction with RDKit
- protein pocket feature extraction with BioPython and pocket residue graphs
- PyTorch Geometric models: LigandGNN and LigandPocketNet
- training, evaluation, and screening utilities

## Setup

```bash
cd /ocean/projects/cis260039p/mjaju/projects
conda env create -f environment.yml
conda activate projects-env
python -m pip install --upgrade pip
```

If you already have an environment such as `nn`, activate it instead:

```bash
conda activate nn
```

Then install the Python dependencies inside that environment:

```bash
python -m pip install -r requirements.txt
```

## Download datasets

The repository includes a download script for the main sources.

```bash
cd /ocean/projects/cis260039p/mjaju/projects
python scripts/download_datasets.py
```

This will create `data/raw/`, fetch broad assay archives, download a few example kinase PDB structures, and create a kinase-complex scaffold:

- `data/raw/klifs/`
- `data/raw/kinase_complexes/`
- `data/raw/manifests/kinase_complex_manifest.csv`

For a real kinase bound-complex dataset, use structure sources such as KLIFS/PDB for complexes and ChEMBL/BindingDB for labels. The default downloader is a starting scaffold, not a complete kinase-complex curation pipeline.

## Data preparation

After download, prepare the raw assay and structure files for modeling. The current implementation expects raw inputs such as:

- `data/raw/chembl/`
- `data/raw/bindingdb/`
- `data/raw/pubchem/`
- `data/raw/pdb/`
- `data/raw/klifs/`
- `data/raw/manifests/kinase_complex_manifest.csv`

The code in `projects/data/curation.py` standardizes units, removes duplicates, maps to UniProt IDs, and computes p-values. For kinase complex modeling, those labels should be joined to structure-backed complexes rather than treated as standalone assay rows.

To build a first-pass kinase training manifest from KLIFS + ChEMBL, run:

```bash
cd /ocean/projects/cis260039p/mjaju/projects
python scripts/build_kinase_manifest.py
```

This writes:

- `data/processed/kinase_activity_manifest.csv`
- `data/processed/kinase_activity_manifest_summary.txt`
- `data/raw/manifests/kinase_complex_manifest.csv`

The current join uses a representative KLIFS structure per kinase target together with ChEMBL ligand activities for that same target. That matches the repo's `(SMILES, pocket PDB, label)` setup, but it does not guarantee that the ChEMBL ligand is the same ligand crystallized in the KLIFS template structure.

## Training

Run the example training script:

```bash
cd /ocean/projects/cis260039p/mjaju/projects
python -m projects.training
```

That script uses the PDF’s model architecture and training loop structure.

## Evaluation

Use the evaluation module to compute regression and classification metrics:

```bash
python -m projects.evaluate
```

## Screening

A library filtering implementation is available in `projects/screening.py`:

```bash
python -m projects.screening
```

## Project layout

- `projects/config.py`: constants and data paths
- `projects/data/curation.py`: assay standardization and schema enforcement
- `projects/features/ligand.py`: RDKit ligand featurization
- `projects/features/pocket.py`: pocket residue extraction and graph building
- `projects/dataset.py`: PyTorch Geometric dataset wrappers
- `projects/models/gnn.py`: ligand-only GCN model
- `projects/models/ligand_pocket.py`: ligand+pocket dual-branch model
- `projects/training.py`: training loop, optimizer, scheduler
- `projects/evaluate.py`: metrics calculator
- `projects/screening.py`: Lipinski and PAINS filters
- `projects/scripts/download_datasets.py`: dataset fetching scripts

## Notes

This implementation follows the PDF exactly, including:
- 3-layer GCN architecture
- BatchNorm + ReLU + Dropout
- global mean pooling
- Adam optimizer with `lr=1e-3` and `weight_decay=1e-5`
- loss examples for MSE and BCEWithLogits
- ROC-AUC, PR-AUC, RMSE, enrichment evaluation
