Store curated kinase-bound complexes here.

Each row in data/raw/manifests/kinase_complex_manifest.csv should correspond
to one bound kinase-ligand complex with a structure-backed pocket.

Suggested workflow:
1. Export/select kinase structures from KLIFS or PDB.
2. Keep complexes with a bound small-molecule ligand.
3. Map those complexes to ChEMBL or BindingDB affinity labels.
4. Record the mapping in the manifest CSV.
