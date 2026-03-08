"""Checkpoint discovery helpers used by notebooks and demos."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):  # type: ignore
        print(obj)


def _resolve_checkpoint_root(checkpoint_root=None):
    if checkpoint_root is not None:
        root = Path(checkpoint_root).expanduser().resolve()
    else:
        root = next(
            candidate.resolve()
            for candidate in [Path.cwd(), Path.cwd().parent]
            if (candidate / "conditional_node_field_graph_generator").exists()
        ) / ".artifacts" / "checkpoints" / "node_field"
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_training_checkpoints(checkpoint_root=None):
    checkpoint_root = _resolve_checkpoint_root(checkpoint_root=checkpoint_root)
    checkpoint_files = sorted(checkpoint_root.glob("*/*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoint_files:
        print(f"No training checkpoints found in {checkpoint_root}")
        return []
    rows = []
    for path in checkpoint_files:
        rows.append(
            {
                "run_dir": path.parent.name,
                "checkpoint": path.name,
                "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
            }
        )
    frame = pd.DataFrame(rows)
    display(frame)
    return [str(path.resolve()) for path in checkpoint_files]


def find_latest_checkpoint(
    checkpoint_root=None,
    prefer_last: bool = True,
) -> Optional[str]:
    checkpoint_root = _resolve_checkpoint_root(checkpoint_root=checkpoint_root)
    run_dirs = sorted(
        [path for path in checkpoint_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        candidates: list[Path] = []
        if prefer_last:
            last_path = run_dir / "last.ckpt"
            if last_path.is_file():
                candidates.append(last_path)
        candidates.extend(
            sorted(
                [path for path in run_dir.glob("*.ckpt") if path.name != "last.ckpt"],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
        if candidates:
            return str(candidates[0].resolve())
    return None


def describe_resume_checkpoint(ckpt_path: Optional[str]) -> None:
    if ckpt_path is None:
        print("No checkpoint selected for resume; training will start from scratch.")
        return
    path = Path(ckpt_path).expanduser().resolve()
    print(f"Resuming training from checkpoint: {path.name}")
    print(path)

