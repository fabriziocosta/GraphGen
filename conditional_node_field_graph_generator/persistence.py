"""General persistence helpers for fitted NodeField objects."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import uuid
from typing import Optional

import dill as pickle
import pandas as pd

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):  # type: ignore
        print(obj)


def resolve_saved_generator_dir(model_dir=None):
    if model_dir is not None:
        root = Path(model_dir).expanduser().resolve()
    else:
        root = next(
            candidate.resolve()
            for candidate in [Path.cwd(), Path.cwd().parent]
            if (candidate / "conditional_node_field_graph_generator").exists()
        ) / ".artifacts" / "saved_generators"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sanitize_model_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return token or "gg"


def _infer_training_set_size(graph_generator) -> Optional[int]:
    conditioning = getattr(graph_generator, "training_graph_conditioning_", None)
    if conditioning is None:
        return None
    try:
        size = len(conditioning)
    except Exception:
        return None
    return int(size) if size >= 0 else None


def save_graph_generator(graph_generator, model_name=None, model_dir=None):
    model_root = resolve_saved_generator_dir(model_dir=model_dir)
    stem = _sanitize_model_token(model_name or "gg")
    training_set_size = _infer_training_set_size(graph_generator)
    if training_set_size is not None:
        stem = f"{stem}-n{training_set_size}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    short_id = uuid.uuid4().hex[:6]
    filename = f"{stem}-{timestamp}-{short_id}.pkl"
    path = model_root / filename
    with open(path, "wb") as handle:
        pickle.dump(graph_generator, handle)
    print(f"Saved graph generator as: {filename}")
    print(path)
    return filename


def list_saved_graph_generators(model_dir=None):
    model_root = resolve_saved_generator_dir(model_dir=model_dir)
    files = sorted(model_root.glob("*.pkl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        print(f"No saved graph generators found in {model_root}")
        return []
    rows = [
        {
            "name": path.name,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
        }
        for path in files
    ]
    frame = pd.DataFrame(rows)
    display(frame)
    return [path.name for path in files]


def load_graph_generator(model_name, model_dir=None):
    model_root = resolve_saved_generator_dir(model_dir=model_dir)
    requested = str(model_name).strip()
    candidates = []
    direct_path = Path(requested).expanduser()
    if direct_path.is_file():
        candidates = [direct_path.resolve()]
    else:
        names_to_try = {requested}
        if not requested.endswith(".pkl"):
            names_to_try.add(f"{requested}.pkl")
        for candidate_name in names_to_try:
            candidate_path = model_root / candidate_name
            if candidate_path.is_file():
                candidates.append(candidate_path.resolve())
        if not candidates:
            pattern = requested[:-4] if requested.endswith(".pkl") else requested
            matches = sorted(model_root.glob(f"{pattern}*.pkl"))
            candidates = [path.resolve() for path in matches]
    if not candidates:
        raise FileNotFoundError(f"Could not find a saved graph generator matching {requested!r} in {model_root}.")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple saved graph generators match {requested!r}: "
            + ", ".join(path.name for path in candidates)
        )
    path = candidates[0]
    with open(path, "rb") as handle:
        graph_generator = pickle.load(handle)
    print(f"Loaded graph generator: {path.name}")
    print(path)
    return graph_generator
