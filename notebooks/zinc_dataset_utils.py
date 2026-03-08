"""Helpers for downloading, caching, and converting ZINC molecules to NetworkX graphs."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Iterable, Optional

import networkx as nx
import pandas as pd
import requests
from rdkit import Chem


ZINC_250K_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc/250k_rndm_zinc_drugs_clean_3.csv"
)
DEFAULT_ZINC_TARGET_COLUMNS = ("logP", "qed", "SAS")


def download_zinc_dataset(
    dataset_dir: Path | str,
    url: str = ZINC_250K_URL,
    filename: str = "zinc_250k.csv",
    chunk_size: int = 1 << 20,
    force: bool = False,
) -> Path:
    """Download the ZINC CSV once and persist it to disk."""
    dataset_root = Path(dataset_dir).expanduser().resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    output_path = dataset_root / filename
    if output_path.exists() and not force:
        print(f"Using cached ZINC CSV: {output_path}")
        return output_path

    print(f"Downloading ZINC dataset from {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                handle.write(chunk)
    print(f"Saved ZINC CSV to: {output_path}")
    return output_path


def smiles_to_networkx_molecule(
    smiles: str,
    zinc_id: Optional[str] = None,
    properties: Optional[dict] = None,
) -> Optional[nx.Graph]:
    """Convert a SMILES string into a discrete-labelled molecular graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    graph = nx.Graph()
    graph.graph["smiles"] = canonical_smiles
    if zinc_id is not None:
        graph.graph["zinc_id"] = str(zinc_id)
    if properties:
        graph.graph.update(properties)

    for atom in mol.GetAtoms():
        graph.add_node(
            atom.GetIdx(),
            label=atom.GetSymbol(),
            atomic_num=int(atom.GetAtomicNum()),
            formal_charge=int(atom.GetFormalCharge()),
            aromatic=bool(atom.GetIsAromatic()),
        )

    for bond in mol.GetBonds():
        bond_label = "AROMATIC" if bond.GetIsAromatic() else str(int(round(bond.GetBondTypeAsDouble())))
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            label=bond_label,
            aromatic=bool(bond.GetIsAromatic()),
        )
    return graph


def _iter_property_columns(frame: pd.DataFrame, reserved: Iterable[str]) -> list[str]:
    reserved_names = set(reserved)
    return [column for column in frame.columns if column not in reserved_names]


def _corpus_root(dataset_dir: Path | str) -> Path:
    return Path(dataset_dir).expanduser().resolve() / "graph_corpus"


def _corpus_manifest_path(dataset_dir: Path | str) -> Path:
    return _corpus_root(dataset_dir) / "manifest.pkl"


def _bucket_filename(node_count: int) -> str:
    return f"graphs_nodes_{int(node_count):03d}.pkl"


def extract_zinc_targets(
    metadata: pd.DataFrame,
    target_columns: Iterable[str] = DEFAULT_ZINC_TARGET_COLUMNS,
) -> pd.DataFrame:
    """Extract the numeric ZINC property columns used as per-molecule targets."""
    columns = list(target_columns)
    missing_columns = [column for column in columns if column not in metadata.columns]
    if missing_columns:
        raise ValueError(
            "Metadata is missing required ZINC target columns: "
            f"{missing_columns}. Available columns: {metadata.columns.tolist()}"
        )
    return metadata.loc[:, columns].astype(float).reset_index(drop=True)


def build_zinc_graph_corpus(
    dataset_dir: Path | str,
    csv_path: Path | str,
    force: bool = False,
    smiles_column: str = "smiles",
    id_column: str = "zinc_id",
) -> dict:
    """Build the full 250K NetworkX corpus once and persist per-node-count buckets."""
    csv_file = Path(csv_path).expanduser().resolve()
    corpus_root = _corpus_root(dataset_dir)
    corpus_root.mkdir(parents=True, exist_ok=True)
    manifest_path = _corpus_manifest_path(dataset_dir)

    if manifest_path.exists() and not force:
        with manifest_path.open("rb") as handle:
            manifest = pickle.load(handle)
        print(f"Loaded cached ZINC graph corpus manifest: {manifest_path}")
        return manifest

    frame = pd.read_csv(csv_file)
    property_columns = _iter_property_columns(frame, reserved=[smiles_column, id_column])

    bucket_graphs: dict[int, list[nx.Graph]] = {}
    bucket_records: dict[int, list[dict]] = {}
    invalid_smiles_count = 0
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        graph = smiles_to_networkx_molecule(
            row_dict[smiles_column],
            zinc_id=row_dict.get(id_column),
            properties={column: row_dict[column] for column in property_columns},
        )
        if graph is None:
            invalid_smiles_count += 1
            continue
        node_count = int(graph.number_of_nodes())
        bucket_graphs.setdefault(node_count, []).append(graph)
        bucket_records.setdefault(node_count, []).append(
            {
                id_column: graph.graph.get("zinc_id"),
                smiles_column: graph.graph.get("smiles"),
                "node_count": node_count,
                "edge_count": int(graph.number_of_edges()),
                **{column: row_dict[column] for column in property_columns},
            }
        )

    metadata_frames: list[pd.DataFrame] = []
    bucket_index: dict[int, dict[str, object]] = {}
    for node_count in sorted(bucket_graphs):
        bucket_path = corpus_root / _bucket_filename(node_count)
        bucket_metadata = pd.DataFrame.from_records(bucket_records[node_count])
        payload = {
            "graphs": bucket_graphs[node_count],
            "metadata": bucket_metadata,
            "node_count": int(node_count),
        }
        with bucket_path.open("wb") as handle:
            pickle.dump(payload, handle)
        metadata_frames.append(bucket_metadata)
        bucket_index[int(node_count)] = {
            "path": bucket_path.name,
            "count": int(len(bucket_graphs[node_count])),
        }

    metadata = (
        pd.concat(metadata_frames, ignore_index=True)
        if metadata_frames
        else pd.DataFrame(columns=[id_column, smiles_column, "node_count", "edge_count", *property_columns])
    )
    manifest = {
        "csv_path": str(csv_file),
        "property_columns": property_columns,
        "bucket_index": bucket_index,
        "node_counts": sorted(bucket_index),
        "total_graphs": int(len(metadata)),
        "invalid_smiles_count": int(invalid_smiles_count),
    }
    with manifest_path.open("wb") as handle:
        pickle.dump(manifest, handle)
    print(f"Built ZINC graph corpus with {manifest['total_graphs']} molecules: {corpus_root}")
    return manifest


def _load_bucket(dataset_dir: Path | str, node_count: int) -> tuple[list[nx.Graph], pd.DataFrame]:
    bucket_path = _corpus_root(dataset_dir) / _bucket_filename(node_count)
    with bucket_path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["graphs"], payload["metadata"]


def load_zinc_graph_dataset(
    dataset_dir: Path | str,
    max_molecules: int = 100_000,
    min_node_count: Optional[int] = None,
    max_node_count: Optional[int] = 40,
    refresh_download: bool = False,
    refresh_cache: bool = False,
    url: str = ZINC_250K_URL,
) -> tuple[list[nx.Graph], pd.DataFrame]:
    """Load a node-count slice from the cached full ZINC NetworkX corpus."""
    dataset_root = Path(dataset_dir).expanduser().resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    csv_path = download_zinc_dataset(
        dataset_root,
        url=url,
        force=refresh_download,
    )
    manifest = build_zinc_graph_corpus(
        dataset_dir=dataset_root,
        csv_path=csv_path,
        force=refresh_cache,
    )

    selected_counts = [
        int(node_count)
        for node_count in manifest["node_counts"]
        if (min_node_count is None or int(node_count) >= int(min_node_count))
        and (max_node_count is None or int(node_count) <= int(max_node_count))
    ]

    graphs: list[nx.Graph] = []
    metadata_frames: list[pd.DataFrame] = []
    remaining = int(max_molecules) if max_molecules is not None else None
    for node_count in selected_counts:
        bucket_graphs, bucket_metadata = _load_bucket(dataset_root, node_count)
        if remaining is None:
            take = len(bucket_graphs)
        else:
            take = min(int(remaining), len(bucket_graphs))
        if take <= 0:
            break
        graphs.extend(bucket_graphs[:take])
        metadata_frames.append(bucket_metadata.iloc[:take].reset_index(drop=True))
        if remaining is not None:
            remaining -= take
            if remaining <= 0:
                break

    metadata = (
        pd.concat(metadata_frames, ignore_index=True)
        if metadata_frames
        else pd.DataFrame(columns=["zinc_id", "smiles", "node_count", "edge_count"])
    )
    print(
        "Loaded ZINC graph slice "
        f"(n={len(graphs)}, min_node_count={min_node_count}, max_node_count={max_node_count}) "
        f"from cached corpus: {_corpus_root(dataset_root)}"
    )
    return graphs, metadata
