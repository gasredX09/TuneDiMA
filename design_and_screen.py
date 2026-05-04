#!/usr/bin/env python3
"""Pocket-aware ligand design and screening pipeline.

This module is intended for inference-time use on new protein structures.
Given one or more target PDB files, it:

1) Detects candidate pocket centers (fpocket or density fallback)
2) Builds pocket graphs for each center
3) Assembles a ligand library from input files and/or training manifest seeds
4) Optionally generates new ligand candidates via BRICS recombination
5) Scores all candidates with a trained LigandPocketNet checkpoint
6) Writes ranked outputs per target and pocket

Important:
- The trained model consumes structure-based pocket graphs. For sequence-only inputs,
  first generate a 3D structure (e.g., AlphaFold/ESMFold) and pass the resulting PDB.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS

from projects.config import ROOT
from projects.dataset import make_interaction_data, make_ligand_data, make_pocket_data
from projects.models.ligand_pocket import LigandPocketNet
from projects.screening import filter_library


DEFAULT_PREPROCESSED_LIGANDS_PATH = "projects/data/processed/preprocessed_ligands_3d.pt"


@dataclass
class TargetInput:
    target_id: str
    pdb_path: str


def as_single_graph_batch(data):
    if not hasattr(data, "x"):
        return data
    num_nodes = int(data.x.size(0)) if data.x is not None else 0
    data.batch = torch.zeros(num_nodes, dtype=torch.long, device=data.x.device if data.x is not None else None)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design and rank ligands for new protein targets")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--target-pdb",
        action="append",
        default=[],
        help="Target PDB path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--targets-csv",
        type=str,
        default=None,
        help="CSV with columns: target_id,pdb_path",
    )
    parser.add_argument(
        "--ligand-library",
        action="append",
        default=[],
        help="Ligand library path(s): .smi/.txt (one SMILES per line) or .csv with a SMILES column.",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="smiles",
        help="Default SMILES column for CSV ligand libraries",
    )
    parser.add_argument(
        "--seed-from-manifest",
        action="store_true",
        help="Include top ligands from data/processed/training_manifest.csv as seeds",
    )
    parser.add_argument(
        "--seed-top-k",
        type=int,
        default=200,
        help="Top-K ligands by label_value to pull from training manifest",
    )
    parser.add_argument(
        "--label-direction",
        choices=["higher", "lower"],
        default="higher",
        help="Whether higher or lower label_value indicates stronger binding",
    )
    parser.add_argument(
        "--generate-brics",
        action="store_true",
        help="Generate additional candidate molecules via BRICS recombination",
    )
    parser.add_argument(
        "--max-generated",
        type=int,
        default=2000,
        help="Maximum number of BRICS-generated candidates to keep",
    )
    parser.add_argument(
        "--max-library",
        type=int,
        default=5000,
        help="Maximum final library size before scoring (after dedupe and filtering)",
    )
    parser.add_argument(
        "--save-preprocessed-ligands",
        type=str,
        nargs="?",
        const=DEFAULT_PREPROCESSED_LIGANDS_PATH,
        default=None,
        help=(
            "Write preprocessed ligand graphs to a .pt file. "
            f"If passed without a value, defaults to {DEFAULT_PREPROCESSED_LIGANDS_PATH}"
        ),
    )
    parser.add_argument(
        "--load-preprocessed-ligands",
        type=str,
        nargs="?",
        const=DEFAULT_PREPROCESSED_LIGANDS_PATH,
        default=None,
        help=(
            "Load preprocessed ligand graphs from a .pt file. "
            f"If passed without a value, defaults to {DEFAULT_PREPROCESSED_LIGANDS_PATH}"
        ),
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only build and optionally save preprocessed ligands, then exit",
    )
    parser.add_argument(
        "--diversity-preselect",
        type=int,
        default=50000,
        help="Max ligands to pre-sample before diversity picking to avoid OOM on huge libraries",
    )
    parser.add_argument(
        "--detector",
        choices=["fpocket", "density"],
        default="fpocket",
        help="Pocket detector",
    )
    parser.add_argument(
        "--n-pockets",
        type=int,
        default=3,
        help="Number of pocket centers per target",
    )
    parser.add_argument(
        "--pocket-radius",
        type=float,
        default=8.0,
        help="Radius (Angstrom) for extracting residues around pocket center",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--ligand-use-3d",
        dest="ligand_use_3d",
        action="store_true",
        help="Use 3D ligand conformers during screening",
    )
    parser.add_argument(
        "--no-ligand-use-3d",
        dest="ligand_use_3d",
        action="store_false",
        help="Disable 3D ligand conformers during screening",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Top-K ligands to report per target-pocket",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/design_screening_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.set_defaults(ligand_use_3d=True)
    return parser.parse_args()


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def load_smiles_file(path: Path, default_smiles_column: str) -> list[str]:
    if not path.exists():
        return []

    if path.suffix.lower() in {".smi", ".txt"}:
        out = []
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            out.append(stripped.split()[0])
        return out

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        for candidate_col in [default_smiles_column, "smiles", "ligand_smiles", "SMILES"]:
            if candidate_col in frame.columns:
                return [str(x).strip() for x in frame[candidate_col].tolist() if str(x).strip()]
        return []

    return []


def load_targets(target_pdb_args: list[str], targets_csv: str | None) -> list[TargetInput]:
    targets: list[TargetInput] = []

    for idx, pdb in enumerate(target_pdb_args, start=1):
        targets.append(TargetInput(target_id=f"target_{idx}", pdb_path=str(Path(pdb))))

    if targets_csv:
        frame = pd.read_csv(targets_csv)
        required = {"target_id", "pdb_path"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"targets-csv missing required columns: {sorted(missing)}")
        for row in frame.itertuples(index=False):
            targets.append(TargetInput(target_id=str(row.target_id), pdb_path=str(row.pdb_path)))

    dedup: dict[tuple[str, str], TargetInput] = {}
    for t in targets:
        key = (t.target_id, str(Path(t.pdb_path)))
        dedup[key] = t
    return list(dedup.values())


def choose_diverse(smiles_list: Iterable[str], k: int, seed: int) -> list[str]:
    """Greedy max-min diversity on Morgan fingerprints."""
    uniq = []
    seen = set()
    for s in smiles_list:
        if s and s not in seen:
            uniq.append(s)
            seen.add(s)

    if len(uniq) <= k:
        return uniq

    mols = [Chem.MolFromSmiles(s) for s in uniq]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m is not None else None for m in mols]

    valid = [(s, fp) for s, fp in zip(uniq, fps) if fp is not None]
    if len(valid) <= k:
        return [s for s, _ in valid]

    rng = random.Random(seed)
    remaining = list(range(len(valid)))
    first = rng.choice(remaining)
    picked = [first]
    remaining.remove(first)

    from rdkit import DataStructs

    while remaining and len(picked) < k:
        best_idx = None
        best_min_dist = -1.0
        for idx in remaining:
            fp_i = valid[idx][1]
            max_sim = 0.0
            for p in picked:
                sim = DataStructs.TanimotoSimilarity(fp_i, valid[p][1])
                if sim > max_sim:
                    max_sim = sim
            min_dist = 1.0 - max_sim
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        picked.append(best_idx)
        remaining.remove(best_idx)

    return [valid[i][0] for i in picked]


def load_manifest_seed_smiles(top_k: int, direction: str) -> list[str]:
    manifest_path = ROOT / "data" / "processed" / "training_manifest.csv"
    if not manifest_path.exists():
        return []

    frame = pd.read_csv(manifest_path)
    required = {"ligand_smiles", "label_value"}
    if required.difference(frame.columns):
        return []

    frame = frame.dropna(subset=["ligand_smiles", "label_value"]).copy()
    frame["label_value"] = pd.to_numeric(frame["label_value"], errors="coerce")
    frame = frame[np.isfinite(frame["label_value"].to_numpy())]
    ascending = direction == "lower"
    frame = frame.sort_values("label_value", ascending=ascending)

    raw = [str(x).strip() for x in frame["ligand_smiles"].head(top_k * 3).tolist() if str(x).strip()]
    canonical = [canonicalize_smiles(s) for s in raw]
    canonical = [s for s in canonical if s is not None]
    return choose_diverse(canonical, k=min(top_k, len(canonical)), seed=42)


def generate_brics_candidates(seed_smiles: list[str], max_generated: int, seed: int) -> list[str]:
    if not seed_smiles or max_generated <= 0:
        return []

    fragments: list[Chem.Mol] = []
    for smi in seed_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            frag_smiles_set = BRICS.BRICSDecompose(mol)
        except Exception:
            continue
        for frag in frag_smiles_set:
            frag_mol = Chem.MolFromSmiles(frag)
            if frag_mol is not None:
                fragments.append(frag_mol)

    if not fragments:
        return []

    rng = random.Random(seed)
    rng.shuffle(fragments)
    fragments = fragments[: min(len(fragments), 1500)]

    generated = []
    seen = set(seed_smiles)
    try:
        for mol in BRICS.BRICSBuild(fragments, maxDepth=3):
            smi = Chem.MolToSmiles(mol, canonical=True)
            if smi in seen:
                continue
            seen.add(smi)
            generated.append(smi)
            if len(generated) >= max_generated:
                break
    except Exception:
        return generated

    return generated


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> LigandPocketNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    arch = checkpoint.get("model_architecture", {}) if isinstance(checkpoint, dict) else {}
    state = checkpoint.get("model_state_dict", checkpoint)

    lig_in = arch.get("lig_in", 16)
    lig_weight = state.get("lig_input_proj.weight") if isinstance(state, dict) else None
    if lig_weight is not None and hasattr(lig_weight, "shape") and len(lig_weight.shape) == 2:
        lig_in = int(lig_weight.shape[1])

    model = LigandPocketNet(
        lig_in=lig_in,
        poc_in=arch.get("poc_in", 25),
        int_in=arch.get("int_in", 25),
        edge_dim=arch.get("edge_dim", 10),
        int_edge_dim=arch.get("int_edge_dim", 3),
        hidden_dim=arch.get("hidden_dim", 128),
        num_layers=arch.get("num_layers", 3),
        dropout=arch.get("dropout", 0.2),
        mol_desc_dim=arch.get("mol_desc_dim", 17),
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    return model


def build_library(args: argparse.Namespace) -> list[str]:
    raw = []

    for lib in args.ligand_library:
        print(f"Loading ligand library: {lib}")
        raw.extend(load_smiles_file(Path(lib), args.smiles_column))
        print(f"Loaded {len(raw)} raw ligand entries so far")

    manifest_seeds = []
    if args.seed_from_manifest:
        manifest_seeds = load_manifest_seed_smiles(args.seed_top_k, args.label_direction)
        raw.extend(manifest_seeds)
        print(f"Added {len(manifest_seeds)} manifest seed ligands")

    print(f"Canonicalizing {len(raw)} ligands")
    canonical = [canonicalize_smiles(s) for s in raw]
    canonical = [s for s in canonical if s is not None]
    print(f"Canonical ligands retained: {len(canonical)}")
    dedup = list(dict.fromkeys(canonical))
    print(f"Deduplicated ligands: {len(dedup)}")

    generated = []
    if args.generate_brics:
        print("Generating BRICS candidates")
        generated = generate_brics_candidates(dedup[: min(len(dedup), 600)], args.max_generated, args.seed)
        print(f"Generated BRICS candidates: {len(generated)}")

    combined = dedup + generated
    combined = list(dict.fromkeys(combined))
    print(f"Combined library before filtering: {len(combined)}")
    combined = filter_library(combined)
    print(f"Library after filtering: {len(combined)}")

    if len(combined) > args.max_library:
        preselect = min(len(combined), max(args.max_library, args.diversity_preselect))
        if len(combined) > preselect:
            print(
                f"Pre-sampling {preselect} ligands from {len(combined)} before diversity selection "
                f"(seed={args.seed})"
            )
            rng = random.Random(args.seed)
            combined = rng.sample(combined, preselect)
        print(f"Selecting diverse subset of {args.max_library} from {len(combined)} ligands")
        combined = choose_diverse(combined, k=args.max_library, seed=args.seed)
        print(f"Diverse subset selected: {len(combined)}")

    print(
        f"Library assembly complete: base={len(dedup)}, generated={len(generated)}, "
        f"filtered_final={len(combined)}, manifest_seeds={len(manifest_seeds)}"
    )
    print(f"Ligand 3D conformers enabled: {args.ligand_use_3d}")
    return combined


def preprocess_ligands(ligand_library: list[str], ligand_use_3d: bool) -> list[tuple[str, object]]:
    prepared: list[tuple[str, object]] = []
    n_failed = 0
    total = len(ligand_library)

    print(f"Preprocessing ligand graphs once: {total} ligand(s)")
    for idx, smiles in enumerate(ligand_library, start=1):
        try:
            lig = as_single_graph_batch(make_ligand_data(smiles, use_3d=ligand_use_3d))
            prepared.append((smiles, lig))
        except Exception:
            n_failed += 1

        if idx == 1 or idx % 100 == 0 or idx == total:
            print(
                f"Ligand preprocessing progress: {idx}/{total} prepared={len(prepared)} failed={n_failed}",
                flush=True,
            )

    print(f"Ligand preprocessing complete: prepared={len(prepared)} failed={n_failed}")
    return prepared


def save_preprocessed_ligands(cache_path: Path, prepared_ligands: list[tuple[str, object]], ligand_use_3d: bool) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ligand_use_3d": bool(ligand_use_3d),
        "num_ligands": len(prepared_ligands),
        "prepared_ligands": prepared_ligands,
    }
    torch.save(payload, cache_path)
    print(f"Saved preprocessed ligands: {cache_path} (count={len(prepared_ligands)})")


def load_preprocessed_ligands(cache_path: Path, ligand_use_3d: bool) -> list[tuple[str, object]]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Preprocessed ligand cache not found: {cache_path}")

    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict) or "prepared_ligands" not in payload:
        raise ValueError(f"Invalid preprocessed ligand cache format: {cache_path}")

    cache_3d = bool(payload.get("ligand_use_3d", ligand_use_3d))
    if cache_3d != bool(ligand_use_3d):
        raise ValueError(
            f"Cache ligand_use_3d={cache_3d} does not match current setting ligand_use_3d={ligand_use_3d}"
        )

    prepared = payload["prepared_ligands"]
    print(f"Loaded preprocessed ligands: {cache_path} (count={len(prepared)})")
    return prepared


def select_prepared_ligands(
    prepared_ligands: list[tuple[str, object]],
    max_library: int,
    diversity_preselect: int,
    seed: int,
) -> tuple[list[str], list[tuple[str, object]]]:
    if not prepared_ligands:
        return [], []

    smiles_all = [s for s, _ in prepared_ligands]
    if len(smiles_all) <= max_library:
        return smiles_all, prepared_ligands

    preselect = min(len(smiles_all), max(max_library, diversity_preselect))
    if len(smiles_all) > preselect:
        rng = random.Random(seed)
        sampled_idx = sorted(rng.sample(range(len(smiles_all)), preselect))
        sampled = [prepared_ligands[i] for i in sampled_idx]
    else:
        sampled = prepared_ligands

    sampled_smiles = [s for s, _ in sampled]
    chosen_smiles = choose_diverse(sampled_smiles, k=max_library, seed=seed)

    by_smiles = {}
    for s, lig in sampled:
        if s not in by_smiles:
            by_smiles[s] = lig

    selected_prepared = [(s, by_smiles[s]) for s in chosen_smiles if s in by_smiles]
    selected_smiles = [s for s, _ in selected_prepared]

    print(
        f"Selected cached diversity subset: selected={len(selected_prepared)} "
        f"from sampled={len(sampled)} total_cached={len(prepared_ligands)}"
    )
    return selected_smiles, selected_prepared


def copy_to_device(data_obj, device: torch.device):
    if hasattr(data_obj, "clone"):
        data_copy = data_obj.clone()
    else:
        data_copy = copy.deepcopy(data_obj)
    return data_copy.to(device)


def score_library_for_target(
    model: LigandPocketNet,
    device: torch.device,
    target: TargetInput,
    ligand_library: list[str],
    detector: str,
    n_pockets: int,
    pocket_radius: float,
    top_k: int,
    ligand_use_3d: bool,
    prepared_ligands: list[tuple[str, object]] | None = None,
) -> list[dict]:
    rows: list[dict] = []

    pocket_centers = []
    try:
        # Trigger pocket detection via make_pocket_data internals.
        _ = make_pocket_data(
            target.pdb_path,
            pocket_detector=detector,
            pocket_n=n_pockets,
            pocket_radius=pocket_radius,
        )
        from projects.features import find_pocket_centers

        pocket_centers = find_pocket_centers(
            target.pdb_path,
            method=detector,
            n_pockets=n_pockets,
            radius=pocket_radius,
        )
    except Exception as exc:
        print(f"Pocket detection failed for {target.target_id}: {exc}")
        pocket_centers = []

    if not pocket_centers:
        pocket_centers = [None]

    cached = prepared_ligands if prepared_ligands is not None else preprocess_ligands(ligand_library, ligand_use_3d)
    total_ligands = len(cached)
    print(
        f"Target {target.target_id}: starting scoring across {len(pocket_centers)} pocket(s) and {total_ligands} ligand(s)"
    )

    for pocket_idx, center in enumerate(pocket_centers, start=1):
        try:
            pocket_data = make_pocket_data(
                target.pdb_path,
                pocket_center=center,
                pocket_radius=pocket_radius,
            )
            pocket_data = as_single_graph_batch(pocket_data).to(device)
        except Exception as exc:
            print(f"Pocket graph failed for {target.target_id} pocket#{pocket_idx}: {exc}")
            continue

        scored = []
        n_attempted = 0
        n_failed = 0
        with torch.no_grad():
            for lig_idx, (smiles, lig_cached) in enumerate(cached, start=1):
                n_attempted += 1
                if lig_idx == 1 or lig_idx % 25 == 0 or lig_idx == total_ligands:
                    print(
                        f"Target {target.target_id} pocket#{pocket_idx}: attempted {lig_idx}/{total_ligands} ligands",
                        flush=True,
                    )
                try:
                    lig = copy_to_device(lig_cached, device)
                    inter = as_single_graph_batch(make_interaction_data(lig, pocket_data)).to(device)
                    pred = model(lig, pocket_data, inter)
                    score = float(pred.item())
                    if not math.isfinite(score):
                        continue
                    scored.append((smiles, score))
                except Exception:
                    n_failed += 1
                    continue

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        cx = float(center[0]) if center is not None else np.nan
        cy = float(center[1]) if center is not None else np.nan
        cz = float(center[2]) if center is not None else np.nan

        for rank, (smiles, score) in enumerate(top, start=1):
            rows.append(
                {
                    "target_id": target.target_id,
                    "pdb_path": target.pdb_path,
                    "detector": detector,
                    "pocket_rank": pocket_idx,
                    "pocket_center_x": cx,
                    "pocket_center_y": cy,
                    "pocket_center_z": cz,
                    "ligand_rank": rank,
                    "ligand_smiles": smiles,
                    "predicted_score": score,
                }
            )

        print(
            f"Target {target.target_id} pocket#{pocket_idx}: attempted={n_attempted}, scored={len(scored)}, failed={n_failed}, reported={len(top)}"
        )

    return rows


def write_suggested_small_sets(library: list[str], output_path: Path) -> None:
    """Write quick-to-run small candidate sets (20/50/100) from the final library."""
    if not library:
        return

    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in [20, 50, 100]:
        subset = library[: min(n, len(library))]
        out_file = out_dir / f"ligand_set_top{len(subset)}.smi"
        out_file.write_text("\n".join(subset) + "\n")


def main() -> None:
    args = parse_args()
    if args.preprocess_only and not args.save_preprocessed_ligands:
        args.save_preprocessed_ligands = DEFAULT_PREPROCESSED_LIGANDS_PATH
        print(
            f"Preprocess-only mode default save path: {args.save_preprocessed_ligands}",
            flush=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    targets = load_targets(args.target_pdb, args.targets_csv)
    if not targets:
        raise ValueError("No targets provided. Use --target-pdb and/or --targets-csv")

    model = load_checkpoint_model(checkpoint_path, device)

    if args.load_preprocessed_ligands:
        prepared_ligands = load_preprocessed_ligands(Path(args.load_preprocessed_ligands), args.ligand_use_3d)
        if not prepared_ligands:
            raise ValueError("Loaded preprocessed ligand cache is empty")
        library, prepared_ligands = select_prepared_ligands(
            prepared_ligands=prepared_ligands,
            max_library=args.max_library,
            diversity_preselect=args.diversity_preselect,
            seed=args.seed,
        )
    else:
        library = build_library(args)
        if not library:
            raise ValueError("Final ligand library is empty after parsing/filtering")

        prepared_ligands = preprocess_ligands(library, args.ligand_use_3d)
        if not prepared_ligands:
            raise ValueError("No ligands could be preprocessed for scoring")

    if args.save_preprocessed_ligands:
        save_preprocessed_ligands(
            cache_path=Path(args.save_preprocessed_ligands),
            prepared_ligands=prepared_ligands,
            ligand_use_3d=args.ligand_use_3d,
        )

    if args.preprocess_only:
        print("Preprocess-only mode: completed ligand preprocessing and exiting before scoring.")
        return

    all_rows = []
    for target in targets:
        all_rows.extend(
            score_library_for_target(
                model=model,
                device=device,
                target=target,
                ligand_library=library,
                detector=args.detector,
                n_pockets=args.n_pockets,
                pocket_radius=args.pocket_radius,
                top_k=args.top_k,
                ligand_use_3d=args.ligand_use_3d,
                prepared_ligands=prepared_ligands,
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        output_path.write_text("target_id,pdb_path,detector,pocket_rank,pocket_center_x,pocket_center_y,pocket_center_z,ligand_rank,ligand_smiles,predicted_score\n")

    write_suggested_small_sets(library, output_path)

    print(f"Wrote ranked results: {output_path}")
    print("Note: for sequence-only targets, first convert sequence to a PDB structure, then re-run this pipeline.")


if __name__ == "__main__":
    main()
