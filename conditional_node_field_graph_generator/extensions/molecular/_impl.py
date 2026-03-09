"""Internal implementation for molecular conversion, drawing, and dataset helpers."""

from __future__ import annotations

import logging
from pathlib import Path
import pickle
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):  # type: ignore
        return obj


ZINC_250K_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc/250k_rndm_zinc_drugs_clean_3.csv"
)
DEFAULT_ZINC_TARGET_COLUMNS = ("logP", "qed", "SAS")
PUBCHEM_FILENAME_TEMPLATE = "AID{assay_id}_{split}.sdf"
logger = logging.getLogger(__name__)


def smiles_to_networkx_molecule(
    smiles: str,
    zinc_id: Optional[str] = None,
    properties: Optional[dict] = None,
) -> Optional[nx.Graph]:
    """Convert a SMILES string into a discrete-labelled molecular graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return molecule_to_networkx(
        mol,
        graph_metadata={
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            **({"zinc_id": str(zinc_id)} if zinc_id is not None else {}),
            **(properties or {}),
        },
    )


def molecule_to_networkx(
    mol: Chem.Mol,
    graph_metadata: Optional[dict] = None,
) -> nx.Graph:
    """Convert an RDKit molecule to a NetworkX graph with discrete labels."""
    graph = nx.Graph()
    if graph_metadata:
        graph.graph.update(graph_metadata)

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


def rdkmol_to_nx(mol: Chem.Mol) -> nx.Graph:
    """Compatibility wrapper for RDKit-to-NetworkX conversion."""
    return molecule_to_networkx(mol)


def sdf_to_nx(file: str | Path):
    """Yield NetworkX graphs from an SDF file."""
    supplier = Chem.SDMolSupplier(str(file))
    for mol in supplier:
        if mol is not None:
            yield rdkmol_to_nx(mol)


def smi_to_nx(file: str | Path):
    """Yield NetworkX graphs from a SMILES file."""
    supplier = Chem.SmilesMolSupplier(str(file))
    for mol in supplier:
        if mol is not None:
            yield rdkmol_to_nx(mol)


def networkx_to_molecule(
    graph: nx.Graph,
    sanitize: bool = True,
) -> Chem.Mol:
    """Convert a discrete-labelled molecular NetworkX graph back to an RDKit molecule."""
    rw_mol = Chem.RWMol()
    node_to_atom_idx: dict[int, int] = {}
    sorted_nodes = sorted(graph.nodes())

    for node_id in sorted_nodes:
        attrs = graph.nodes[node_id]
        atomic_num = attrs.get("atomic_num")
        label = attrs.get("label", "C")
        atom = Chem.Atom(int(atomic_num) if atomic_num is not None else str(label))
        atom.SetFormalCharge(int(attrs.get("formal_charge", 0)))
        atom.SetIsAromatic(bool(attrs.get("aromatic", False)))
        node_to_atom_idx[node_id] = rw_mol.AddAtom(atom)

    for u, v, attrs in graph.edges(data=True):
        bond_label = attrs.get("label", "1")
        if bond_label == "AROMATIC" or attrs.get("aromatic", False):
            bond_type = Chem.BondType.AROMATIC
        else:
            bond_type = {
                "1": Chem.BondType.SINGLE,
                "2": Chem.BondType.DOUBLE,
                "3": Chem.BondType.TRIPLE,
            }.get(str(bond_label), Chem.BondType.SINGLE)
        rw_mol.AddBond(node_to_atom_idx[u], node_to_atom_idx[v], bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            bond = rw_mol.GetBondBetweenAtoms(node_to_atom_idx[u], node_to_atom_idx[v])
            if bond is not None:
                bond.SetIsAromatic(True)

    mol = rw_mol.GetMol()
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("aromatic", False):
            atom = mol.GetAtomWithIdx(node_to_atom_idx[node_id])
            atom.SetIsAromatic(True)
    if sanitize:
        Chem.SanitizeMol(mol)
    return mol


def nx_to_rdkit(graph: nx.Graph) -> Chem.Mol:
    """Convert a molecular graph to an RDKit molecule without forced sanitization."""
    return networkx_to_molecule(graph, sanitize=False)


def _prepare_molecule_for_drawing(graph: nx.Graph) -> Chem.Mol:
    mol = nx_to_rdkit(graph)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)
    try:
        return Draw.rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)
    except Exception:
        return mol


def set_coordinates(compounds: Iterable[Chem.Mol]) -> None:
    """Compute 2D coordinates for RDKit molecules before drawing."""
    for mol in compounds:
        if mol is None:
            raise ValueError("set_coordinates failed on a null molecule.")
        mol.UpdatePropertyCache(strict=False)
        AllChem.Compute2DCoords(mol)


def compounds_to_image(
    compounds: Iterable[Chem.Mol],
    n_graphs_per_line: int = 5,
    size: int = 250,
    legend: Optional[Iterable[str]] = None,
):
    """Build a grid image from RDKit molecules."""
    compound_list = list(compounds)
    if not compound_list:
        return None
    set_coordinates(compound_list)
    return Draw.MolsToGridImage(
        compound_list,
        molsPerRow=max(1, int(n_graphs_per_line)),
        subImgSize=(int(size), int(size)),
        legends=list(legend) if legend is not None else None,
    )


def molecule_graphs_to_grid_image(
    graphs: Iterable[nx.Graph],
    legends: Optional[Iterable[str]] = None,
    mols_per_row: int = 4,
    sub_img_size: tuple[int, int] = (250, 200),
):
    """Render molecular graphs to an RDKit grid image."""
    graph_list = list(graphs)
    if not graph_list:
        return None
    prepared = [_prepare_molecule_for_drawing(graph) for graph in graph_list]
    if legends is None:
        legend_list = []
        for graph in graph_list:
            legend_list.append(
                str(
                    graph.graph.get("smiles")
                    or graph.graph.get("zinc_id")
                    or graph.graph.get("pubchem_cid")
                    or ""
                )
            )
    else:
        legend_list = [str(value) for value in legends]
    return Draw.MolsToGridImage(
        prepared,
        legends=legend_list,
        molsPerRow=max(1, int(mols_per_row)),
        subImgSize=sub_img_size,
        useSVG=False,
    )


def nx_to_image(
    graphs: Iterable[nx.Graph],
    n_graphs_per_line: int = 5,
    size: int = 250,
    title_key: Optional[str] = None,
    titles: Optional[Iterable[str]] = None,
):
    """Render molecular graphs to an RDKit image grid."""
    graph_list = list(graphs)
    if title_key:
        legends = [graph.graph.get(title_key, "N/A") for graph in graph_list]
    elif titles is not None:
        legends = list(titles)
    else:
        legends = [str(index) for index in range(len(graph_list))]
    compounds = [_prepare_molecule_for_drawing(graph) for graph in graph_list]
    return compounds_to_image(
        compounds,
        n_graphs_per_line=n_graphs_per_line,
        size=size,
        legend=legends,
    )


def draw_molecules(
    graphs: Iterable[nx.Graph],
    titles: Optional[Iterable[str]] = None,
    num: Optional[int] = None,
    n_graphs_per_line: int = 7,
    size: int = 7,
    legends: Optional[Iterable[str]] = None,
    mols_per_row: Optional[int] = None,
    sub_img_size: Optional[tuple[int, int]] = None,
):
    """Display a grid image for a list of molecular graphs."""
    graph_list = list(graphs)
    if num is not None:
        graph_list = graph_list[:num]
    if not graph_list:
        print("No molecules to display.")
        return None

    if titles is None and legends is None:
        legends = [str(index) for index in range(len(graph_list))]
    elif legends is None and titles is not None:
        legends = list(titles)[: len(graph_list)]

    if mols_per_row is None:
        mols_per_row = n_graphs_per_line
    if sub_img_size is None:
        pixel_size = max(150, int(size) * 35)
        sub_img_size = (pixel_size, pixel_size)

    for start in range(0, len(graph_list), 50):
        image = molecule_graphs_to_grid_image(
            graph_list[start : start + 50],
            legends=None if legends is None else list(legends)[start : start + 50],
            mols_per_row=mols_per_row,
            sub_img_size=sub_img_size,
        )
        if image is not None:
            display(image)
    return None


def resolve_pubchem_dir(pubchem_dir: Path | str | None = None) -> Path:
    """Resolve the directory that stores assay-level PubChem SDF files."""
    if pubchem_dir is not None:
        return Path(pubchem_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "notebooks" / "datasets" / "PUBCHEM"


def _pubchem_sdf_path(pubchem_dir: Path | str, assay_id: str, split: str) -> Path:
    return resolve_pubchem_dir(pubchem_dir) / PUBCHEM_FILENAME_TEMPLATE.format(assay_id=assay_id, split=split)


class SupervisedDataSetLoader(object):
    """Generic supervised dataset loader used by local chemistry workflows."""

    def __init__(
        self,
        load_func=None,
        size=None,
        use_targets_list=None,
        use_equalized=False,
        use_multiclass_to_binary=False,
        use_regression_to_binary=False,
        regression_to_binary_threshold=None,
    ):
        self.load_func = load_func
        self.size = size
        self.use_targets_list = use_targets_list
        self.use_equalized = use_equalized
        self.use_multiclass_to_binary = use_multiclass_to_binary
        self.use_regression_to_binary = use_regression_to_binary
        self.regression_to_binary_threshold = regression_to_binary_threshold

    def resize(self, data, targets, size):
        data_is_numpy = isinstance(data, np.ndarray)
        target_is_numpy = isinstance(targets, np.ndarray)
        idxs = np.random.choice(len(targets), size=size, replace=False)
        data = [data[idx] for idx in idxs]
        if data_is_numpy:
            data = np.asarray(data)
        targets = [targets[idx] for idx in idxs]
        if target_is_numpy:
            targets = np.asarray(targets)
        return data, targets

    def equalize(self, data, targets):
        data_is_numpy = isinstance(data, np.ndarray)
        target_is_numpy = isinstance(targets, np.ndarray)
        target_values = list(sorted(set(targets)))
        idxs_list = [[idx for idx in range(len(targets)) if targets[idx] == target_value] for target_value in target_values]
        min_size = min(len(idxs) for idxs in idxs_list)
        idxs_list = [np.random.choice(idxs, size=min_size, replace=False) for idxs in idxs_list]
        data = sum([[data[idx] for idx in idxs] for idxs in idxs_list], [])
        if data_is_numpy:
            data = np.asarray(data)
        targets = sum([[targets[idx] for idx in idxs] for idxs in idxs_list], [])
        if target_is_numpy:
            targets = np.asarray(targets)
        return data, targets

    def binarize_multiclass(self, targets):
        target_is_numpy = isinstance(targets, np.ndarray)
        targets = [target % 2 for target in targets]
        if target_is_numpy:
            targets = np.asarray(targets)
        return targets

    def binarize_regression(self, targets):
        target_is_numpy = isinstance(targets, np.ndarray)
        targets = [target < self.regression_to_binary_threshold for target in targets]
        if target_is_numpy:
            targets = np.asarray(targets)
        return targets

    def keep_target(self, data, targets):
        data_is_numpy = isinstance(data, np.ndarray)
        target_is_numpy = isinstance(targets, np.ndarray)
        target_values = set(self.use_targets_list)
        idxs = [idx for idx, target in enumerate(targets) if target in target_values]
        data = [data[idx] for idx in idxs]
        if data_is_numpy:
            data = np.asarray(data)
        targets = [targets[idx] for idx in idxs]
        if target_is_numpy:
            targets = np.asarray(targets)
        return data, targets

    def load(self):
        data, targets = self.load_func()
        if self.use_targets_list is not None:
            data, targets = self.keep_target(data, targets)
        if self.use_equalized:
            data, targets = self.equalize(data, targets)
        if self.size is not None and self.size < len(targets):
            data, targets = self.resize(data, targets, self.size)
        if self.use_multiclass_to_binary:
            targets = self.binarize_multiclass(targets)
        if self.use_regression_to_binary:
            targets = self.binarize_regression(targets)
        return data, targets


class RDKitMolFileLoader(object):
    """Load RDKit molecules from file and convert them to graphs."""

    def __init__(self):
        pass

    def read(self, filename):
        ext = str(filename).split(".")[-1]
        if ext == "sdf":
            return sdf_to_nx(filename)
        if ext == "smi":
            return smi_to_nx(filename)
        raise Exception(f"Unknown extension: {ext!r}")

    def load(self, filename):
        return list(self.read(filename))


class PubChemLoader(RDKitMolFileLoader):
    """Load PubChem assay SDF files into graphs with binary targets."""

    def __init__(self):
        self.pubchem_dir = str(resolve_pubchem_dir())

    def get_assay_description(self, assay_id):
        import urllib.request

        req = urllib.request.Request(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{assay_id}/description/JSON",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req) as url:  # nosec B310 - intentional remote fetch helper
            data = eval(url.read())
        return data["PC_AssayContainer"][0]["assay"]["descr"]["name"]

    def download(self, assay_id, active=True, stepsize=50):
        split = "active" if active else "inactive"
        path = _pubchem_sdf_path(self.pubchem_dir, assay_id, split)
        if path.exists():
            return str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sid = 1 if active else 2
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/"
            f"{assay_id}/CSV?sid={sid}&record_type=3d"
        )
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
        return str(path)

    def load(self, assay_id, dirname="PUBCHEM", format_type="sdf"):
        pubchem_dir = resolve_pubchem_dir(dirname if dirname != "PUBCHEM" else self.pubchem_dir)
        active_path = _pubchem_sdf_path(pubchem_dir, assay_id, "active")
        inactive_path = _pubchem_sdf_path(pubchem_dir, assay_id, "inactive")
        active_graphs = super().load(active_path)
        inactive_graphs = super().load(inactive_path)
        data = active_graphs + inactive_graphs
        targets = [1] * len(active_graphs) + [0] * len(inactive_graphs)
        return data, targets


def load_pubchem_graph_dataset(
    pubchem_dir: Path | str | None = None,
    assay_id: str = "651610",
    dataset_size: Optional[int] = None,
    max_node_count: Optional[int] = None,
    use_equalized: bool = False,
    random_state: int = 0,
):
    """Load a PubChem assay dataset and return graphs, targets, and metadata."""
    loader = PubChemLoader()
    loader.pubchem_dir = str(resolve_pubchem_dir(pubchem_dir))

    def load_func():
        return loader.load(assay_id, dirname=loader.pubchem_dir)

    graphs, targets = SupervisedDataSetLoader(
        load_func=load_func,
        size=dataset_size,
        use_equalized=use_equalized,
    ).load()
    graphs = np.asarray(graphs, dtype=object)
    targets = np.asarray(targets)

    metadata = pd.DataFrame(
        {
            "target": targets,
            "node_count": [graph.number_of_nodes() for graph in graphs],
            "edge_count": [graph.number_of_edges() for graph in graphs],
        }
    )
    if max_node_count is not None:
        keep = metadata["node_count"].to_numpy() <= int(max_node_count)
        graphs = graphs[keep]
        targets = targets[keep]
        metadata = metadata.loc[keep].reset_index(drop=True)
    return graphs.tolist(), targets, metadata


def download_zinc_dataset(
    dataset_dir: Path | str,
    url: str = ZINC_250K_URL,
    filename: str = "zinc_250k.csv",
    force: bool = False,
):
    """Download the ZINC CSV file if needed."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / filename
    if csv_path.exists() and not force:
        return csv_path
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    csv_path.write_bytes(response.content)
    return csv_path


def _zinc_graph_bucket_path(dataset_dir: Path, node_count: int) -> Path:
    return dataset_dir / "graph_corpus" / f"graphs_nodes_{int(node_count):03d}.pkl"


def _zinc_graph_bucket_relative_path(node_count: int) -> Path:
    return Path("graph_corpus") / f"graphs_nodes_{int(node_count):03d}.pkl"


def _normalize_zinc_cached_path(dataset_dir: Path, cached_path: Path | str) -> str:
    path = Path(cached_path).expanduser()
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(dataset_dir))
    except ValueError:
        return str(path)


def _resolve_zinc_bucket_path(dataset_dir: Path, cached_path: Path | str, node_count: int) -> Path:
    path = Path(cached_path).expanduser()
    if not path.is_absolute():
        return (dataset_dir / path).resolve()
    if path.exists():
        return path
    # Older manifests stored absolute paths, which break when the synced
    # dataset moves to a different filesystem root. Fall back to the
    # canonical bucket location for the current dataset directory.
    return _zinc_graph_bucket_path(dataset_dir, node_count)


def _normalize_zinc_corpus_manifest(dataset_dir: Path, manifest: dict) -> tuple[dict, bool]:
    """Backfill derived fields for older cached manifests."""
    normalized = dict(manifest)
    changed = False

    node_counts = [int(node_count) for node_count in normalized.get("node_counts", [])]
    if normalized.get("node_counts") != node_counts:
        normalized["node_counts"] = node_counts
        changed = True

    bucket_files = normalized.get("bucket_files")
    if bucket_files is None:
        normalized["bucket_files"] = {
            node_count: str(_zinc_graph_bucket_relative_path(node_count))
            for node_count in node_counts
        }
        changed = True
    else:
        normalized_bucket_files = {
            int(node_count): _normalize_zinc_cached_path(dataset_dir, bucket_path)
            for node_count, bucket_path in bucket_files.items()
        }
        if normalized_bucket_files != bucket_files:
            normalized["bucket_files"] = normalized_bucket_files
            changed = True

    csv_path = normalized.get("csv_path")
    if csv_path is not None:
        normalized_csv_path = _normalize_zinc_cached_path(dataset_dir, csv_path)
        if normalized_csv_path != csv_path:
            normalized["csv_path"] = normalized_csv_path
            changed = True

    return normalized, changed


def _normalize_zinc_bucket_items(items: object) -> tuple[list[tuple[nx.Graph, dict]], bool]:
    """Backfill older bucket payloads into the current pair representation."""
    if isinstance(items, list):
        if all(isinstance(item, tuple) and len(item) == 2 for item in items):
            return items, False
        raise ValueError("Unsupported ZINC bucket list format.")

    if isinstance(items, dict) and "graphs" in items and "metadata" in items:
        graphs = list(items["graphs"])
        metadata = items["metadata"]
        if isinstance(metadata, pd.DataFrame):
            rows = metadata.to_dict(orient="records")
        elif isinstance(metadata, list):
            rows = [dict(row) for row in metadata]
        else:
            raise ValueError("Unsupported legacy ZINC bucket metadata format.")
        if len(graphs) != len(rows):
            raise ValueError("Legacy ZINC bucket graphs and metadata lengths differ.")
        return list(zip(graphs, rows)), True

    raise ValueError("Unsupported ZINC bucket format.")


def build_zinc_graph_corpus(
    dataset_dir: Path | str,
    csv_path: Path | str,
    force: bool = False,
):
    """Convert a ZINC CSV table into cached graph buckets grouped by node count."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    corpus_dir = dataset_dir / "graph_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus_dir / "manifest.pkl"
    if manifest_path.exists() and not force:
        with open(manifest_path, "rb") as handle:
            manifest = pickle.load(handle)
        manifest, changed = _normalize_zinc_corpus_manifest(dataset_dir, manifest)
        if changed:
            with open(manifest_path, "wb") as handle:
                pickle.dump(manifest, handle)
        return manifest

    frame = pd.read_csv(csv_path)
    buckets: dict[int, list[tuple[nx.Graph, dict]]] = {}
    invalid_smiles_count = 0
    for row in frame.to_dict(orient="records"):
        graph = smiles_to_networkx_molecule(
            row["smiles"],
            zinc_id=row.get("zinc_id"),
            properties={key: row[key] for key in row if key not in {"smiles"}},
        )
        if graph is None:
            invalid_smiles_count += 1
            continue
        node_count = graph.number_of_nodes()
        buckets.setdefault(node_count, []).append((graph, row))

    total_graphs = 0
    node_counts = sorted(buckets)
    for node_count, items in buckets.items():
        bucket_path = _zinc_graph_bucket_path(dataset_dir, node_count)
        with open(bucket_path, "wb") as handle:
            pickle.dump(items, handle)
        total_graphs += len(items)

    manifest = {
        "csv_path": _normalize_zinc_cached_path(dataset_dir, csv_path),
        "total_graphs": total_graphs,
        "invalid_smiles_count": invalid_smiles_count,
        "node_counts": node_counts,
        "bucket_files": {
            node_count: str(_zinc_graph_bucket_relative_path(node_count))
            for node_count in node_counts
        },
    }
    with open(manifest_path, "wb") as handle:
        pickle.dump(manifest, handle)
    return manifest


def load_zinc_graph_dataset(
    dataset_dir: Path | str,
    max_molecules: int = 100_000,
    min_node_count: Optional[int] = None,
    max_node_count: Optional[int] = 40,
):
    """Load cached ZINC graphs and their metadata."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    manifest_path = dataset_dir / "graph_corpus" / "manifest.pkl"
    with open(manifest_path, "rb") as handle:
        manifest = pickle.load(handle)
    manifest, changed = _normalize_zinc_corpus_manifest(dataset_dir, manifest)
    if changed:
        with open(manifest_path, "wb") as handle:
            pickle.dump(manifest, handle)

    selected_node_counts = [
        node_count
        for node_count in manifest["node_counts"]
        if (min_node_count is None or node_count >= min_node_count)
        and (max_node_count is None or node_count <= max_node_count)
    ]

    graphs = []
    metadata_rows = []
    for node_count in selected_node_counts:
        bucket_path = _resolve_zinc_bucket_path(
            dataset_dir,
            manifest["bucket_files"][node_count],
            node_count,
        )
        with open(bucket_path, "rb") as handle:
            items = pickle.load(handle)
        normalized_items, changed = _normalize_zinc_bucket_items(items)
        if changed:
            with open(bucket_path, "wb") as handle:
                pickle.dump(normalized_items, handle)
        for graph, row in normalized_items:
            graphs.append(graph)
            metadata_rows.append(row)
            if len(graphs) >= max_molecules:
                metadata = pd.DataFrame(metadata_rows)
                return graphs, metadata

    metadata = pd.DataFrame(metadata_rows)
    return graphs, metadata


def extract_zinc_targets(
    metadata: pd.DataFrame,
    target_columns: Iterable[str] = DEFAULT_ZINC_TARGET_COLUMNS,
) -> pd.DataFrame:
    """Extract requested target columns from ZINC metadata."""
    return metadata.loc[:, list(target_columns)].copy()
