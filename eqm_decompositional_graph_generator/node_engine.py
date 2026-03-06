"""Node-level EqM model engine (merged module)."""

from dataclasses import dataclass
from typing import List, Any, Optional, Tuple, Sequence, Union
import numpy as np

@dataclass
class GraphConditioningBatch:
    """Explicit graph-level conditioning signals passed into the generator."""

    graph_embeddings: np.ndarray
    node_counts: np.ndarray
    edge_counts: np.ndarray
    node_label_histograms: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(len(self.graph_embeddings))


@dataclass
class NodeGenerationBatch:
    """Explicit node-level training inputs and semantic supervision signals."""

    node_embeddings_list: List[np.ndarray]
    node_presence_mask: np.ndarray
    node_degree_targets: np.ndarray
    node_label_targets: Optional[List[np.ndarray]] = None
    edge_pairs: Optional[List[Tuple[int, int, int]]] = None
    edge_targets: Optional[np.ndarray] = None
    edge_label_pairs: Optional[List[Tuple[int, int, int]]] = None
    edge_label_targets: Optional[np.ndarray] = None
    auxiliary_edge_pairs: Optional[List[Tuple[int, int, int]]] = None
    auxiliary_edge_targets: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(len(self.node_embeddings_list))


@dataclass
class GeneratedNodeBatch:
    """Generator outputs separated into embeddings and explicit semantic predictions."""

    node_embeddings_list: List[np.ndarray]
    node_presence_mask: Optional[np.ndarray] = None
    node_degree_predictions: Optional[np.ndarray] = None
    node_labels: Optional[List[np.ndarray]] = None
    edge_probability_matrices: Optional[List[np.ndarray]] = None
    edge_label_matrices: Optional[List[np.ndarray]] = None

    def __len__(self) -> int:
        return int(len(self.node_embeddings_list))


class ConditionalNodeGeneratorBase:
    """
    Abstract base class for conditional node generators.

    All conditional node generators should inherit from this class and implement
    the fit and predict methods.

    Methods
    -------
    fit(
        node_batch: NodeGenerationBatch,
        graph_conditioning: GraphConditioningBatch,
    )
        Fit the model to training data.

    predict(graph_conditioning)
        Generate samples conditioned on explicit graph-level signals.
    """

    def fit(
        self,
        node_batch: NodeGenerationBatch,
        graph_conditioning: GraphConditioningBatch,
        targets: Optional[Sequence[Any]] = None,
    ):
        raise NotImplementedError("fit() must be implemented by subclasses.")

    def predict(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
    ):
        raise NotImplementedError("predict() must be implemented by subclasses.")


import contextlib
import os
import sys
import time
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
class CrossTransformerEncoderLayer(nn.Module):
    """Pre-norm transformer block with self-attention followed by cross-attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x1 = self.norm1(x)
        x = x + self.dropout1(
            self.self_attn(
                x1,
                x1,
                x1,
                key_padding_mask=self_key_padding_mask,
            )[0]
        )
        if query_mask is not None:
            x = x * query_mask.unsqueeze(-1).to(dtype=x.dtype)

        x2 = self.norm2(x)
        x = x + self.dropout2(
            self.cross_attn(
                x2,
                k,
                v,
                key_padding_mask=cross_key_padding_mask,
            )[0]
        )
        if query_mask is not None:
            x = x * query_mask.unsqueeze(-1).to(dtype=x.dtype)

        x3 = self.norm3(x)
        x = x + self.dropout3(self.ff(x3))
        if query_mask is not None:
            x = x * query_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x


def plot_metrics(
    train_metrics: Dict[str, Sequence[float]],
    val_metrics: Dict[str, Sequence[float]],
    window: int = 10,
    alpha: float = 0.3,
) -> None:
    """Visualise train/validation metrics with a geometric moving average.

    Args:
        train_metrics (Dict[str, Sequence[float]]): Input value.
        val_metrics (Dict[str, Sequence[float]]): Input value.
        window (int): Optional input value.
        alpha (float): Optional input value.
    """

    def _moving_average(data: Sequence[float], window_size: int) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if len(arr) < window_size:
            return np.array([])
        arr_clamped = np.where(arr <= 0, np.finfo(float).tiny, arr)
        log_arr = np.log(arr_clamped)
        kernel = np.ones(window_size, dtype=float) / window_size
        smoothed_log = np.convolve(log_arr, kernel, mode="valid")
        return np.exp(smoothed_log)

    metrics = [
        name
        for name in train_metrics.keys()
        if len(train_metrics.get(name, [])) > 0 and len(val_metrics.get(name, [])) > 0
    ]
    if not metrics:
        return

    fig, ax0 = plt.subplots(figsize=(15, 8))
    axes = [ax0] + [ax0.twinx() for _ in range(len(metrics) - 1)]
    for i, ax in enumerate(axes[1:], start=1):
        ax.spines["right"].set_position(("outward", 56 * i))
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.patch.set_visible(False)

    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    default_colors = color_cycle.by_key()["color"] if color_cycle is not None else ["blue", "red", "green", "purple", "orange"]
    colors = [default_colors[i % len(default_colors)] for i in range(len(metrics))]
    lines, labels = [], []
    for name, ax, color in zip(metrics, axes, colors):
        train_vals = train_metrics[name]
        val_vals = val_metrics[name]
        count = min(len(train_vals), len(val_vals))
        train = train_vals[:count]
        val = val_vals[:count]
        epochs = np.arange(1, count + 1)
        ax.plot(epochs, train, color=color, alpha=alpha)
        ax.plot(epochs, val, color=color, linestyle="--", alpha=alpha)
        sm_train = _moving_average(train, window)
        sm_val = _moving_average(val, window)
        if sm_train.size:
            sm_epochs = np.arange(window, window + len(sm_train))
            line_train, = ax.plot(sm_epochs, sm_train, color=color, linewidth=2, label=f"Train {name} (MA{window})")
            line_val, = ax.plot(sm_epochs, sm_val, color=color, linewidth=2, linestyle="--", label=f"Val {name} (MA{window})")
            lines += [line_train, line_val]
            labels += [f"Train {name} (MA{window})", f"Val {name} (MA{window})"]
        ax.set_ylabel(name, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.set_yscale("log")

    fig.legend(lines, labels, loc="upper center", ncol=max(len(lines) // 2, 1), fontsize="small")
    ax0.set_xlabel("Epoch")
    ax0.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


class EdgeMLP(nn.Module):
    """Pairwise MLP used for edge presence or edge-label scoring."""

    def __init__(self, latent_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1, output_dim: int = 1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * latent_dim
        self.output_dim = int(output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(4 * latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(h_i - h_j)
        prod = h_i * h_j
        x = torch.cat([h_i, h_j, diff, prod], dim=-1)
        out = self.mlp(x)
        return out.squeeze(-1) if self.output_dim == 1 else out


class MetricsLogger(pl.callbacks.Callback):
    """Collect end-of-epoch metrics into the module's history lists."""

    def on_fit_start(self, trainer, pl_module):
        pl_module._fit_start_time = time.time()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0, int(round(float(seconds))))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"

    @staticmethod
    def _component_summary(pl_module, metrics: Dict[str, torch.Tensor], prefix: str):
        """Collect weighted loss components so differently scaled terms are comparable.

        Args:
            pl_module (Any): Input value.
            metrics (Dict[str, torch.Tensor]): Input value.
            prefix (str): Input value.

        Returns:
            Any: Computed result.
        """
        component_specs = [
            ("eqm", "recon", 1.0),
            ("deg", "deg_ce", float(getattr(pl_module, "lambda_degree_importance", 1.0))),
            ("exist", "exist", float(getattr(pl_module, "lambda_node_exist_importance", 1.0))),
            ("node_label", "node_label_ce", float(getattr(pl_module, "lambda_node_label_importance", 1.0))),
            ("edge_label", "edge_label_ce", float(getattr(pl_module, "lambda_edge_label_importance", 1.0))),
            ("edge", "edge_ce", float(getattr(pl_module, "lambda_locality_importance", 1.0))),
            ("aux", "aux_locality_ce", float(getattr(pl_module, "lambda_auxiliary_locality_importance", 1.0))),
        ]

        raw_total = float(metrics.get(f"{prefix}_total", torch.tensor(0.0)).item())
        components = []
        weighted_sum = 0.0
        for label, metric_name, scale in component_specs:
            key = f"{prefix}_{metric_name}"
            if key not in metrics:
                continue
            raw_value = float(metrics[key].item())
            weighted_value = raw_value * scale
            weighted_sum += weighted_value
            components.append((label, raw_value, weighted_value))

        if not components:
            return raw_total, [], None, 0.0

        denominator = weighted_sum if weighted_sum > 0 else 1.0
        dominant_label, _, dominant_weighted = max(components, key=lambda item: item[2])
        normalized_components = [
            (label, raw, weighted, weighted / denominator)
            for label, raw, weighted in components
        ]
        return raw_total, normalized_components, dominant_label, dominant_weighted / denominator

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.train_losses.append(m.get("train_total", torch.tensor(0.0)).item())
        pl_module.train_deg_ce.append(m.get("train_deg_ce", torch.tensor(0.0)).item())
        pl_module.train_recon.append(m.get("train_recon", torch.tensor(0.0)).item())
        if hasattr(pl_module, "train_exist"):
            pl_module.train_exist.append(m.get("train_exist", torch.tensor(0.0)).item())
        if hasattr(pl_module, "train_node_label_ce"):
            pl_module.train_node_label_ce.append(m.get("train_node_label_ce", m.get("train_label_ce", torch.tensor(0.0))).item())
        elif hasattr(pl_module, "train_label_ce"):
            pl_module.train_label_ce.append(m.get("train_label_ce", m.get("train_node_label_ce", torch.tensor(0.0))).item())
        if hasattr(pl_module, "train_edge_label_ce"):
            pl_module.train_edge_label_ce.append(m.get("train_edge_label_ce", torch.tensor(0.0)).item())
        if getattr(pl_module, "use_locality_supervision", False):
            pl_module.train_edge_loss.append(m.get("train_edge_ce", m.get("train_edge_loss", torch.tensor(0.0))).item())
            pl_module.train_edge_acc.append(m.get("train_edge_acc", torch.tensor(0.0)).item())
        if getattr(pl_module, "use_auxiliary_locality_supervision", False):
            pl_module.train_aux_edge_loss.append(m.get("train_aux_locality_ce", m.get("train_aux_edge_loss", torch.tensor(0.0))).item())
            pl_module.train_aux_edge_acc.append(m.get("train_aux_edge_acc", torch.tensor(0.0)).item())

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.val_losses.append(m.get("val_total", torch.tensor(0.0)).item())
        pl_module.val_deg_ce.append(m.get("val_deg_ce", torch.tensor(0.0)).item())
        pl_module.val_recon.append(m.get("val_recon", torch.tensor(0.0)).item())
        if hasattr(pl_module, "val_exist"):
            pl_module.val_exist.append(m.get("val_exist", torch.tensor(0.0)).item())
        if hasattr(pl_module, "val_node_label_ce"):
            pl_module.val_node_label_ce.append(m.get("val_node_label_ce", m.get("val_label_ce", torch.tensor(0.0))).item())
        elif hasattr(pl_module, "val_label_ce"):
            pl_module.val_label_ce.append(m.get("val_label_ce", m.get("val_node_label_ce", torch.tensor(0.0))).item())
        if hasattr(pl_module, "val_edge_label_ce"):
            pl_module.val_edge_label_ce.append(m.get("val_edge_label_ce", torch.tensor(0.0)).item())
        if getattr(pl_module, "use_locality_supervision", False):
            pl_module.val_edge_loss.append(m.get("val_edge_ce", m.get("val_edge_loss", torch.tensor(0.0))).item())
            pl_module.val_edge_acc.append(m.get("val_edge_acc", torch.tensor(0.0)).item())
        if getattr(pl_module, "use_auxiliary_locality_supervision", False):
            pl_module.val_aux_edge_loss.append(m.get("val_aux_locality_ce", m.get("val_aux_edge_loss", torch.tensor(0.0))).item())
            pl_module.val_aux_edge_acc.append(m.get("val_aux_edge_acc", torch.tensor(0.0)).item())

        verbose_level = 0
        try:
            verbose_level = int(getattr(pl_module, "verbose", 0))
        except (TypeError, ValueError):
            verbose_level = 1 if getattr(pl_module, "verbose", False) else 0
        if verbose_level >= 2:
            interval = int(getattr(pl_module, "verbose_epoch_interval", 10))
            current_epoch = int(getattr(trainer, "current_epoch", -1)) + 1
            if interval > 0 and (current_epoch % interval == 0):
                train_total, train_components, train_dominant, train_dominant_share = self._component_summary(pl_module, m, "train")
                val_total, val_components, val_dominant, val_dominant_share = self._component_summary(pl_module, m, "val")
                max_epochs = getattr(trainer, "max_epochs", None)
                fit_start_time = getattr(pl_module, "_fit_start_time", None)
                eta_label = None
                if (
                    isinstance(max_epochs, int)
                    and max_epochs > 0
                    and fit_start_time is not None
                    and current_epoch > 0
                ):
                    elapsed_seconds = max(0.0, time.time() - float(fit_start_time))
                    average_epoch_seconds = elapsed_seconds / float(current_epoch)
                    remaining_epochs = max(0, max_epochs - current_epoch)
                    eta_seconds = remaining_epochs * average_epoch_seconds
                    eta_label = self._format_duration(eta_seconds)
                epoch_label = (
                    f"Epoch {current_epoch}/{max_epochs}"
                    if isinstance(max_epochs, int) and max_epochs > 0
                    else f"Epoch {current_epoch}"
                )
                if eta_label is not None:
                    epoch_label += f" | ETA {eta_label}"
                ordered_labels = []
                for label, *_ in train_components + val_components:
                    if label not in ordered_labels:
                        ordered_labels.append(label)

                def _components_to_map(components):
                    return {label: (raw, weighted, share) for label, raw, weighted, share in components}

                train_map = _components_to_map(train_components)
                val_map = _components_to_map(val_components)

                def _format_row(prefix_label, total_value, component_map, dominant_label, dominant_share):
                    def _format_share(value: float) -> str:
                        if value <= 0:
                            return "  0%"
                        if value < 0.001:
                            return "<0.1%"
                        return f"{value:>5.1%}"

                    row = f"{prefix_label:<5} total={total_value:>9.1f}"
                    for label in ordered_labels:
                        if label in component_map:
                            _, weighted, share = component_map[label]
                            row += f" | {label:>10} {weighted:>9.1f} {_format_share(share)}"
                        else:
                            row += f" | {label:>10} {'-':>9} {'-':>5}"
                    if dominant_label is not None:
                        row += f" | dominant={dominant_label} ({_format_share(dominant_share).strip()})"
                    return row

                print(f"{epoch_label}:")
                print("  " + _format_row("train", train_total, train_map, train_dominant, train_dominant_share))
                print("  " + _format_row("val", val_total, val_map, val_dominant, val_dominant_share))


import contextlib
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from .support import run_trainer_fit


class EqMGraphDataset(Dataset):
    """Dataset carrying graph node features, conditioning, masks, and label targets."""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        node_presence_mask: np.ndarray,
        node_degree_targets: np.ndarray,
        node_label_targets: Optional[np.ndarray] = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.node_presence_mask = torch.tensor(node_presence_mask, dtype=torch.bool)
        self.node_degree_targets = torch.tensor(node_degree_targets, dtype=torch.long)
        self.node_label_targets = None
        if node_label_targets is not None:
            self.node_label_targets = torch.tensor(node_label_targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.node_label_targets is None:
            return self.X[idx], self.Y[idx], self.node_presence_mask[idx], self.node_degree_targets[idx]
        return self.X[idx], self.Y[idx], self.node_presence_mask[idx], self.node_degree_targets[idx], self.node_label_targets[idx]


class EqMGraphWithEdgesDataset(Dataset):
    """Dataset carrying edge supervision, auxiliary locality, and optional node labels."""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        edge_pairs: List[Tuple[int, int, int]],
        edge_targets: np.ndarray,
        edge_label_pairs: Optional[List[Tuple[int, int, int]]],
        edge_label_targets: Optional[np.ndarray],
        auxiliary_edge_pairs: Optional[List[Tuple[int, int, int]]],
        auxiliary_edge_targets: Optional[np.ndarray],
        node_presence_mask: np.ndarray,
        node_degree_targets: np.ndarray,
        node_label_targets: Optional[np.ndarray] = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.node_presence_mask = torch.tensor(node_presence_mask, dtype=torch.bool)
        self.node_degree_targets = torch.tensor(node_degree_targets, dtype=torch.long)
        self.node_label_targets = None
        if node_label_targets is not None:
            self.node_label_targets = torch.tensor(node_label_targets, dtype=torch.long)
        self.edge_idx_by_graph: Dict[int, List[Tuple[int, int]]] = {b: [] for b in range(len(X))}
        self.edge_lbl_by_graph: Dict[int, List[float]] = {b: [] for b in range(len(X))}
        if edge_pairs is not None and edge_targets is not None:
            for (b, i, j), lbl in zip(edge_pairs, edge_targets):
                self.edge_idx_by_graph[b].append((i, j))
                self.edge_lbl_by_graph[b].append(lbl)
        self.aux_edge_idx_by_graph: Dict[int, List[Tuple[int, int]]] = {b: [] for b in range(len(X))}
        self.aux_edge_lbl_by_graph: Dict[int, List[float]] = {b: [] for b in range(len(X))}
        if auxiliary_edge_pairs is not None and auxiliary_edge_targets is not None:
            for (b, i, j), lbl in zip(auxiliary_edge_pairs, auxiliary_edge_targets):
                self.aux_edge_idx_by_graph[b].append((i, j))
                self.aux_edge_lbl_by_graph[b].append(lbl)
        self.edge_label_idx_by_graph: Dict[int, List[Tuple[int, int]]] = {b: [] for b in range(len(X))}
        self.edge_label_tgt_by_graph: Dict[int, List[int]] = {b: [] for b in range(len(X))}
        if edge_label_pairs is not None and edge_label_targets is not None:
            for (b, i, j), lbl in zip(edge_label_pairs, edge_label_targets):
                self.edge_label_idx_by_graph[b].append((i, j))
                self.edge_label_tgt_by_graph[b].append(lbl)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        edge_idxs = (
            torch.tensor(self.edge_idx_by_graph[idx], dtype=torch.long)
            if self.edge_idx_by_graph[idx]
            else torch.empty((0, 2), dtype=torch.long)
        )
        edge_lbls = (
            torch.tensor(self.edge_lbl_by_graph[idx], dtype=torch.float32)
            if self.edge_lbl_by_graph[idx]
            else torch.empty((0,), dtype=torch.float32)
        )
        aux_edge_idxs = (
            torch.tensor(self.aux_edge_idx_by_graph[idx], dtype=torch.long)
            if self.aux_edge_idx_by_graph[idx]
            else torch.empty((0, 2), dtype=torch.long)
        )
        aux_edge_lbls = (
            torch.tensor(self.aux_edge_lbl_by_graph[idx], dtype=torch.float32)
            if self.aux_edge_lbl_by_graph[idx]
            else torch.empty((0,), dtype=torch.float32)
        )
        edge_label_idxs = (
            torch.tensor(self.edge_label_idx_by_graph[idx], dtype=torch.long)
            if self.edge_label_idx_by_graph[idx]
            else torch.empty((0, 2), dtype=torch.long)
        )
        edge_label_tgts = (
            torch.tensor(self.edge_label_tgt_by_graph[idx], dtype=torch.long)
            if self.edge_label_tgt_by_graph[idx]
            else torch.empty((0,), dtype=torch.long)
        )
        if self.node_label_targets is None:
            return self.X[idx], self.Y[idx], edge_idxs, edge_lbls, edge_label_idxs, edge_label_tgts, aux_edge_idxs, aux_edge_lbls, self.node_presence_mask[idx], self.node_degree_targets[idx]
        return self.X[idx], self.Y[idx], edge_idxs, edge_lbls, edge_label_idxs, edge_label_tgts, aux_edge_idxs, aux_edge_lbls, self.node_presence_mask[idx], self.node_degree_targets[idx], self.node_label_targets[idx]


def collate_eqm_graph_with_edges(batch):
    """Batch EqMGraphWithEdgesDataset items into tensors with optional label targets.

    Args:
        batch (Any): Input value.

    Returns:
        Any: Computed result.
    """
    xs, ys, masks, degree_targets = [], [], [], []
    local_edge_idxs, local_edge_lbls = [], []
    edge_label_idxs_list, edge_label_targets_list = [], []
    aux_local_edge_idxs, aux_local_edge_lbls = [], []
    label_targets = []
    has_labels = len(batch[0]) == 11
    for sample in batch:
        if has_labels:
            x, y, ei, el, edge_label_ei, edge_label_el, aux_ei, aux_el, mask, deg, node_labels = sample
            label_targets.append(node_labels)
        else:
            x, y, ei, el, edge_label_ei, edge_label_el, aux_ei, aux_el, mask, deg = sample
        xs.append(x)
        ys.append(y)
        masks.append(mask)
        degree_targets.append(deg)
        local_edge_idxs.append(ei)
        local_edge_lbls.append(el)
        edge_label_idxs_list.append(edge_label_ei)
        edge_label_targets_list.append(edge_label_el)
        aux_local_edge_idxs.append(aux_ei)
        aux_local_edge_lbls.append(aux_el)
    X = torch.stack(xs)
    Y = torch.stack(ys)
    M = torch.stack(masks)
    D = torch.stack(degree_targets)
    def _pack_pairs(per_graph_idxs, per_graph_lbls):
        all_edge_idxs = []
        all_edge_lbls = []
        for b, (ei, el) in enumerate(zip(per_graph_idxs, per_graph_lbls)):
            if ei.numel() == 0:
                continue
            b_col = torch.full((ei.size(0), 1), b, dtype=torch.long)
            all_edge_idxs.append(torch.cat([b_col, ei], dim=1))
            all_edge_lbls.append(el)
        if all_edge_idxs:
            return torch.cat(all_edge_idxs, dim=0), torch.cat(all_edge_lbls, dim=0)
        return torch.empty((0, 3), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

    edge_idx, edge_lbl = _pack_pairs(local_edge_idxs, local_edge_lbls)
    edge_label_idx, edge_label_tgt = _pack_pairs(edge_label_idxs_list, edge_label_targets_list)
    aux_edge_idx, aux_edge_lbl = _pack_pairs(aux_local_edge_idxs, aux_local_edge_lbls)
    if has_labels:
        label_tensor = torch.stack(label_targets)
        return X, Y, edge_idx, edge_lbl, edge_label_idx, edge_label_tgt, aux_edge_idx, aux_edge_lbl, M, D, label_tensor
    return X, Y, edge_idx, edge_lbl, edge_label_idx, edge_label_tgt, aux_edge_idx, aux_edge_lbl, M, D


class EqMDecompositionalNodeGeneratorModule(pl.LightningModule):
    """Conditional EqM model with an explicit scalar energy and score via autograd."""

    def __init__(
        self,
        number_of_rows_per_example: int,
        input_feature_dimension: int,
        condition_feature_dimension: int,
        latent_embedding_dimension: int,
        number_of_transformer_layers: int,
        transformer_attention_head_count: int,
        transformer_dropout: float = 0.1,
        learning_rate: float = 1e-3,
        verbose: bool = False,
        verbose_epoch_interval: int = 10,
        enable_early_stopping: bool = True,
        early_stopping_monitor: str = "val_eqm",
        early_stopping_mode: str = "min",
        early_stopping_patience: int = 30,
        early_stopping_min_delta: float = 0.0,
        restore_best_checkpoint: bool = True,
        artifact_root_dir: Optional[str] = None,
        checkpoint_root_dir: Optional[str] = None,
        important_feature_index: int = 1,
        max_degree: Optional[int] = None,
        lambda_degree_importance: float = 1.0,
        degree_temperature: Optional[float] = None,
        degree_min_val: float = 0.0,
        degree_range_val: float = 1.0,
        lambda_node_exist_importance: float = 1.0,
        lambda_node_label_importance: float = 1.0,
        lambda_edge_label_importance: float = 1.0,
        use_locality_supervision: bool = False,
        lambda_locality_importance: float = 1.0,
        use_auxiliary_locality_supervision: bool = False,
        lambda_auxiliary_locality_importance: float = 1.0,
        exist_pos_weight: Union[torch.Tensor, float] = 1.0,
        edge_pos_weight: Union[torch.Tensor, float] = 1.0,
        auxiliary_edge_pos_weight: Union[torch.Tensor, float] = 1.0,
        num_node_label_classes: int = 0,
        use_node_label_head: bool = False,
        num_edge_label_classes: int = 0,
        use_edge_label_head: bool = False,
        eqm_sigma: float = 0.2,
        sampling_step_size: float = 0.05,
        sampling_steps: int = 100,
        langevin_noise_scale: float = 0.0,
        pool_condition_tokens: bool = False,
        guidance_enabled: bool = False,
        target_condition_start_index: int = 0,
        target_condition_feature_count: int = 0,
        cfg_condition_dropout_prob: float = 0.1,
        cfg_null_target_strategy: str = "zero",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["verbose"])

        if max_degree is None:
            raise ValueError("max_degree must be provided when initializing the EqM model.")
        if eqm_sigma <= 0:
            raise ValueError(f"eqm_sigma must be positive (got {eqm_sigma}).")
        if sampling_step_size <= 0:
            raise ValueError(f"sampling_step_size must be positive (got {sampling_step_size}).")
        if sampling_steps <= 0:
            raise ValueError(f"sampling_steps must be positive (got {sampling_steps}).")
        if not 0.0 <= cfg_condition_dropout_prob <= 1.0:
            raise ValueError(
                f"cfg_condition_dropout_prob must be in [0, 1] (got {cfg_condition_dropout_prob})."
            )
        if cfg_null_target_strategy not in {"zero"}:
            raise ValueError(
                f"cfg_null_target_strategy must be one of ['zero'] (got {cfg_null_target_strategy!r})."
            )

        self.number_of_rows_per_example = number_of_rows_per_example
        self.input_feature_dimension = input_feature_dimension
        self.condition_feature_dimension = condition_feature_dimension
        self.latent_embedding_dimension = latent_embedding_dimension
        self.number_of_transformer_layers = number_of_transformer_layers
        self.transformer_attention_head_count = transformer_attention_head_count
        self.transformer_dropout = transformer_dropout
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.verbose_epoch_interval = int(verbose_epoch_interval)
        self.enable_early_stopping = bool(enable_early_stopping)
        self.early_stopping_monitor = str(early_stopping_monitor)
        self.early_stopping_mode = str(early_stopping_mode)
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.restore_best_checkpoint = bool(restore_best_checkpoint)
        if artifact_root_dir is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            artifact_root_dir = os.path.join(repo_root, ".artifacts")
        self.artifact_root_dir = str(artifact_root_dir)
        if checkpoint_root_dir is None:
            checkpoint_root_dir = os.path.join(self.artifact_root_dir, "checkpoints", "eqm")
        self.checkpoint_root_dir = str(checkpoint_root_dir)
        self.important_feature_index = important_feature_index
        self.max_degree = int(max_degree)
        self.lambda_degree_importance = lambda_degree_importance
        self.degree_temperature = degree_temperature
        self.lambda_node_exist_importance = lambda_node_exist_importance
        self.lambda_node_label_importance = lambda_node_label_importance
        self.lambda_edge_label_importance = lambda_edge_label_importance
        self.use_locality_supervision = bool(use_locality_supervision)
        self.lambda_locality_importance = lambda_locality_importance
        self.use_auxiliary_locality_supervision = bool(use_auxiliary_locality_supervision)
        self.lambda_auxiliary_locality_importance = lambda_auxiliary_locality_importance
        self.num_node_label_classes = int(num_node_label_classes)
        self.use_node_label_head = bool(use_node_label_head and num_node_label_classes > 0)
        self.num_edge_label_classes = int(num_edge_label_classes)
        self.use_edge_label_head = bool(use_edge_label_head and num_edge_label_classes > 0)
        self.eqm_sigma = float(eqm_sigma)
        self.sampling_step_size = float(sampling_step_size)
        self.sampling_steps = int(sampling_steps)
        self.langevin_noise_scale = float(langevin_noise_scale)
        self.pool_condition_tokens = bool(pool_condition_tokens)
        self.guidance_enabled = bool(guidance_enabled)
        self.target_condition_start_index = int(
            0 if target_condition_start_index is None else target_condition_start_index
        )
        self.target_condition_feature_count = int(
            0 if target_condition_feature_count is None else target_condition_feature_count
        )
        self.cfg_condition_dropout_prob = float(cfg_condition_dropout_prob)
        self.cfg_null_target_strategy = str(cfg_null_target_strategy)
        self.use_guidance = False
        self.use_existence_head = True
        self.constant_existence_value = 1.0

        self.register_buffer(
            "exist_pos_weight",
            torch.as_tensor(exist_pos_weight, dtype=torch.float32),
        )
        self.register_buffer(
            "edge_pos_weight",
            torch.as_tensor(edge_pos_weight, dtype=torch.float32),
        )
        self.register_buffer(
            "auxiliary_edge_pos_weight",
            torch.as_tensor(auxiliary_edge_pos_weight, dtype=torch.float32),
        )
        self.register_buffer("deg_min_val", torch.tensor(degree_min_val, dtype=torch.float32))
        self.register_buffer("deg_range_val", torch.tensor(max(degree_range_val, 1e-8), dtype=torch.float32))

        self.train_losses = []
        self.val_losses = []
        self.train_deg_ce = []
        self.val_deg_ce = []
        self.train_loss_all = []
        self.val_loss_all = []
        self.train_exist = []
        self.val_exist = []
        self.train_node_label_ce = []
        self.val_node_label_ce = []
        self.train_edge_label_ce = []
        self.val_edge_label_ce = []
        self.train_recon = []
        self.val_recon = []
        if self.use_locality_supervision:
            self.train_edge_loss = []
            self.val_edge_loss = []
            self.train_edge_acc = []
            self.val_edge_acc = []
        if self.use_auxiliary_locality_supervision:
            self.train_aux_edge_loss = []
            self.val_aux_edge_loss = []
            self.train_aux_edge_acc = []
            self.val_aux_edge_acc = []

        self.layernorm_in = nn.LayerNorm(input_feature_dimension, elementwise_affine=True)
        self.linear_encoder_input_to_latent = nn.Linear(input_feature_dimension, latent_embedding_dimension)
        self.linear_encoder_condition_to_latent = nn.Linear(condition_feature_dimension, latent_embedding_dimension)
        self.shared_transformer = nn.ModuleList(
            [
                CrossTransformerEncoderLayer(
                    embed_dim=latent_embedding_dimension,
                    num_heads=transformer_attention_head_count,
                    dropout=transformer_dropout,
                )
                for _ in range(number_of_transformer_layers)
            ]
        )
        self.potential_head = nn.Sequential(
            nn.LayerNorm(latent_embedding_dimension),
            nn.Linear(latent_embedding_dimension, latent_embedding_dimension),
            nn.GELU(),
            nn.Linear(latent_embedding_dimension, 1),
        )
        self.degree_head = nn.Linear(latent_embedding_dimension, self.max_degree + 1)
        self.exist_head = nn.Linear(latent_embedding_dimension, 1)
        self.node_label_head = (
            nn.Linear(latent_embedding_dimension, self.num_node_label_classes)
            if self.use_node_label_head
            else None
        )
        self.edge_label_head = None
        if self.use_edge_label_head:
            self.edge_label_head = EdgeMLP(
                latent_dim=latent_embedding_dimension,
                hidden_dim=2 * latent_embedding_dimension,
                dropout=transformer_dropout,
                output_dim=self.num_edge_label_classes,
            )
        if self.use_locality_supervision:
            self.edge_head = EdgeMLP(
                latent_dim=latent_embedding_dimension,
                hidden_dim=2 * latent_embedding_dimension,
                dropout=transformer_dropout,
            )
        if self.use_auxiliary_locality_supervision:
            self.auxiliary_edge_head = EdgeMLP(
                latent_dim=latent_embedding_dimension,
                hidden_dim=2 * latent_embedding_dimension,
                dropout=transformer_dropout,
            )

    def _compute_edge_probability_matrices(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        """Evaluate the edge head on every ordered node pair.

        Args:
            latent_tokens (torch.Tensor): Input value.

        Returns:
            torch.Tensor: Computed result.
        """
        batch_size, node_count, latent_dim = latent_tokens.shape
        src = latent_tokens.unsqueeze(2).expand(batch_size, node_count, node_count, latent_dim)
        dst = latent_tokens.unsqueeze(1).expand(batch_size, node_count, node_count, latent_dim)
        edge_logits = self.edge_head(
            src.reshape(-1, latent_dim),
            dst.reshape(-1, latent_dim),
        ).reshape(batch_size, node_count, node_count)
        edge_probs = torch.sigmoid(edge_logits)
        diagonal_mask = torch.eye(node_count, dtype=torch.bool, device=latent_tokens.device).unsqueeze(0)
        edge_probs = edge_probs.masked_fill(diagonal_mask, 0.0)
        return edge_probs

    def _compute_edge_label_logits(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        """Evaluate the edge-label head on every ordered node pair.

        Args:
            latent_tokens (torch.Tensor): Input value.

        Returns:
            torch.Tensor: Computed result.
        """
        batch_size, node_count, latent_dim = latent_tokens.shape
        src = latent_tokens.unsqueeze(2).expand(batch_size, node_count, node_count, latent_dim)
        dst = latent_tokens.unsqueeze(1).expand(batch_size, node_count, node_count, latent_dim)
        edge_label_logits = self.edge_label_head(
            src.reshape(-1, latent_dim),
            dst.reshape(-1, latent_dim),
        ).reshape(batch_size, node_count, node_count, self.num_edge_label_classes)
        return edge_label_logits

    def _encode_with_condition(
        self,
        input_rows: torch.Tensor,
        global_condition_vector: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_mask is not None:
            input_rows = input_rows * node_mask.unsqueeze(-1).to(dtype=input_rows.dtype)

        x_norm = self.layernorm_in(input_rows)
        latent_tokens = self.linear_encoder_input_to_latent(x_norm)

        if global_condition_vector.dim() == 2:
            cond_tokens = global_condition_vector.unsqueeze(1)
        elif global_condition_vector.dim() == 3:
            cond_tokens = global_condition_vector
        else:
            raise ValueError(
                "global_condition_vector must have shape (B, C) or (B, M, C); "
                f"received {tuple(global_condition_vector.shape)}"
            )

        if self.pool_condition_tokens and cond_tokens.size(1) > 1:
            cond_tokens = cond_tokens.mean(dim=1, keepdim=True)

        weight = self.linear_encoder_condition_to_latent.weight.transpose(0, 1)
        cond_proj = torch.matmul(cond_tokens, weight)
        bias = self.linear_encoder_condition_to_latent.bias
        if bias is not None:
            cond_proj = cond_proj + bias

        if node_mask is not None:
            latent_tokens = latent_tokens * node_mask.unsqueeze(-1).to(dtype=latent_tokens.dtype)

        self_padding_mask = None if node_mask is None else ~node_mask
        for layer in self.shared_transformer:
            latent_tokens = layer(
                latent_tokens,
                k=cond_proj,
                v=cond_proj,
                self_key_padding_mask=self_padding_mask,
                query_mask=node_mask,
            )
            if node_mask is not None:
                latent_tokens = latent_tokens * node_mask.unsqueeze(-1).to(dtype=latent_tokens.dtype)
        return latent_tokens

    def _build_noise_scale(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.eqm_sigma)

    def _has_target_conditioning(self) -> bool:
        return self.guidance_enabled and self.target_condition_feature_count > 0

    def _target_condition_slice(self) -> slice:
        start = self.target_condition_start_index
        end = start + self.target_condition_feature_count
        return slice(start, end)

    def _null_target_conditioning(self, global_condition: torch.Tensor) -> torch.Tensor:
        if not self._has_target_conditioning():
            return global_condition
        cond = global_condition.clone()
        cond[:, self._target_condition_slice()] = 0.0
        return cond

    def _apply_cfg_dropout(self, global_condition: torch.Tensor) -> torch.Tensor:
        if not self._has_target_conditioning():
            return global_condition
        if self.cfg_condition_dropout_prob <= 0.0:
            return global_condition
        batch_size = global_condition.shape[0]
        drop_mask = (torch.rand(batch_size, device=global_condition.device) < self.cfg_condition_dropout_prob)
        if not torch.any(drop_mask):
            return global_condition
        cond = global_condition.clone()
        cond[drop_mask, self._target_condition_slice()] = 0.0
        return cond

    def _compute_score_field(
        self,
        noisy_input: torch.Tensor,
        global_condition_vector: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        *,
        create_graph: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_tokens = self._encode_with_condition(noisy_input, global_condition_vector, node_mask=node_mask)
        phi_per_node = self.potential_head(latent_tokens).squeeze(-1)
        if node_mask is not None:
            phi_per_node = phi_per_node * node_mask.to(dtype=phi_per_node.dtype)
        phi = phi_per_node.sum(dim=1)
        score = -torch.autograd.grad(
            phi.sum(),
            noisy_input,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        return score, phi, latent_tokens

    def _eqm_loss(
        self,
        input_examples: torch.Tensor,
        global_condition: torch.Tensor,
        node_presence_mask: Optional[torch.Tensor] = None,
        node_degree_targets: Optional[torch.Tensor] = None,
        node_label_targets: Optional[torch.Tensor] = None,
        *,
        create_graph: bool,
    ) -> Tuple[dict, torch.Tensor]:
        eps = torch.randn_like(input_examples)
        noise_scale = self._build_noise_scale(input_examples)
        noisy_input = (input_examples + noise_scale * eps).detach().requires_grad_(True)
        score_mask = (
            node_presence_mask.unsqueeze(-1).to(dtype=input_examples.dtype)
            if node_presence_mask is not None
            else torch.ones_like(input_examples[..., :1])
        )

        score, _, latent_noisy = self._compute_score_field(
            noisy_input,
            global_condition,
            node_mask=node_presence_mask,
            create_graph=create_graph,
        )
        target_score = -eps / noise_scale
        eqm_error = (score - target_score).pow(2) * score_mask
        loss_eqm = eqm_error.sum() / score_mask.sum().clamp_min(1.0)

        denoised = noisy_input + noise_scale.pow(2) * score
        latent_clean = self._encode_with_condition(denoised, global_condition, node_mask=node_presence_mask)
        logits_deg = self.degree_head(latent_clean)
        logits_exist = self.exist_head(latent_clean).squeeze(-1) if self.use_existence_head else None
        logits_label = self.node_label_head(latent_clean) if self.use_node_label_head else None

        if self.use_existence_head:
            if node_presence_mask is None:
                raise RuntimeError("Existence supervision requires explicit node_presence_mask targets.")
            target_exist = node_presence_mask.float()
            exist_loss_map = F.binary_cross_entropy_with_logits(
                logits_exist,
                target_exist,
                pos_weight=self.exist_pos_weight,
                reduction="none",
            )
            # Train existence on all slots so padded nodes contribute true negatives.
            loss_exist = exist_loss_map.mean()
            presence_mask = node_presence_mask.to(dtype=exist_loss_map.dtype)
        else:
            presence_mask = (
                node_presence_mask.to(dtype=input_examples.dtype)
                if node_presence_mask is not None
                else torch.ones_like(input_examples[..., 0])
            )
            loss_exist = input_examples.new_zeros(())

        if node_degree_targets is None:
            raise RuntimeError("Degree supervision requires explicit node_degree_targets.")
        true_deg_class = torch.clamp(node_degree_targets.long(), 0, self.max_degree)

        if self.degree_temperature is not None and self.degree_temperature > 0:
            logits_for_loss = logits_deg / self.degree_temperature
        else:
            logits_for_loss = logits_deg

        deg_loss_map = F.cross_entropy(
            logits_for_loss.reshape(-1, self.max_degree + 1),
            true_deg_class.reshape(-1),
            reduction="none",
        )
        deg_mask = presence_mask.reshape(-1)
        loss_deg_ce = (deg_loss_map * deg_mask).sum() / deg_mask.sum().clamp_min(1.0)
        if self.use_node_label_head and node_label_targets is not None:
            label_valid_mask = node_presence_mask if node_presence_mask is not None else torch.ones_like(node_label_targets, dtype=torch.bool)
            safe_targets = torch.where(label_valid_mask, node_label_targets, torch.zeros_like(node_label_targets))
            label_loss_map = F.cross_entropy(
                logits_label.reshape(-1, self.num_node_label_classes),
                safe_targets.reshape(-1),
                reduction="none",
            ).reshape_as(safe_targets)
            label_mask = label_valid_mask.to(dtype=label_loss_map.dtype)
            loss_label_ce = (label_loss_map * label_mask).sum() / label_mask.sum().clamp_min(1.0)
        else:
            loss_label_ce = input_examples.new_zeros(())

        total_loss = (
            loss_eqm
            + self.lambda_degree_importance * loss_deg_ce
        )
        if self.use_existence_head:
            total_loss = total_loss + self.lambda_node_exist_importance * loss_exist
        if self.use_node_label_head:
            total_loss = total_loss + self.lambda_node_label_importance * loss_label_ce

        return (
            {
                "total": total_loss,
                "eqm": loss_eqm,
                "exist": loss_exist,
                "deg_ce": loss_deg_ce,
                "label_ce": loss_label_ce,
            },
            latent_clean if self.use_locality_supervision else latent_noisy,
        )

    def training_step(self, batch, batch_idx):
        uses_pairwise_supervision = (
            self.use_locality_supervision
            or self.use_edge_label_head
            or self.use_auxiliary_locality_supervision
        )
        if uses_pairwise_supervision:
            if self.use_node_label_head:
                input_examples, global_condition, edge_idx, edge_labels, edge_label_idx, edge_label_targets, aux_edge_idx, aux_edge_labels, node_presence_mask, node_degree_targets, node_label_targets = batch
            else:
                input_examples, global_condition, edge_idx, edge_labels, edge_label_idx, edge_label_targets, aux_edge_idx, aux_edge_labels, node_presence_mask, node_degree_targets = batch
                node_label_targets = None
        else:
            if self.use_node_label_head:
                input_examples, global_condition, node_presence_mask, node_degree_targets, node_label_targets = batch
            else:
                input_examples, global_condition, node_presence_mask, node_degree_targets = batch
                node_label_targets = None
            edge_label_idx = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
            edge_label_targets = torch.empty((0,), dtype=torch.long, device=input_examples.device)
            aux_edge_idx = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
            aux_edge_labels = torch.empty((0,), dtype=torch.float32, device=input_examples.device)

        batch_size = int(node_presence_mask.shape[0])
        conditioned_global = self._apply_cfg_dropout(global_condition)
        losses, latent_tokens = self._eqm_loss(
            input_examples,
            conditioned_global,
            node_presence_mask=node_presence_mask,
            node_degree_targets=node_degree_targets,
            node_label_targets=node_label_targets,
            create_graph=True,
        )
        total_loss = losses["total"]

        if self.use_locality_supervision and edge_idx.numel() > 0:
            b, i, j = edge_idx.unbind(1)
            h_i = latent_tokens[b, i]
            h_j = latent_tokens[b, j]
            edge_logits = self.edge_head(h_i, h_j)
            edge_loss = F.binary_cross_entropy_with_logits(
                edge_logits,
                edge_labels,
                pos_weight=self.edge_pos_weight,
            )
            total_loss = total_loss + self.lambda_locality_importance * edge_loss

            with torch.no_grad():
                edge_pred = (torch.sigmoid(edge_logits) > 0.5).float()
                edge_acc = (edge_pred == edge_labels).float().mean()

            self.log("train_edge_ce", edge_loss, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("train_edge_acc", edge_acc, on_step=False, on_epoch=True, batch_size=batch_size)

        if self.use_edge_label_head and edge_label_idx.numel() > 0:
            b, i, j = edge_label_idx.unbind(1)
            h_i = latent_tokens[b, i]
            h_j = latent_tokens[b, j]
            edge_label_logits = self.edge_label_head(h_i, h_j)
            edge_label_loss = F.cross_entropy(edge_label_logits, edge_label_targets)
            total_loss = total_loss + self.lambda_edge_label_importance * edge_label_loss
            self.log("train_edge_label_ce", edge_label_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        if self.use_auxiliary_locality_supervision and aux_edge_idx.numel() > 0:
            b, i, j = aux_edge_idx.unbind(1)
            h_i = latent_tokens[b, i]
            h_j = latent_tokens[b, j]
            aux_edge_logits = self.auxiliary_edge_head(h_i, h_j)
            aux_edge_loss = F.binary_cross_entropy_with_logits(
                aux_edge_logits,
                aux_edge_labels,
                pos_weight=self.auxiliary_edge_pos_weight,
            )
            total_loss = total_loss + self.lambda_auxiliary_locality_importance * aux_edge_loss

            with torch.no_grad():
                aux_edge_pred = (torch.sigmoid(aux_edge_logits) > 0.5).float()
                aux_edge_acc = (aux_edge_pred == aux_edge_labels).float().mean()

            self.log("train_aux_locality_ce", aux_edge_loss, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("train_aux_edge_acc", aux_edge_acc, on_step=False, on_epoch=True, batch_size=batch_size)

        self.log("train_total", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train_recon", losses["eqm"], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train_eqm", losses["eqm"], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train_deg_ce", losses["deg_ce"], on_step=False, on_epoch=True, batch_size=batch_size)
        if self.use_existence_head:
            self.log("train_exist", losses["exist"], on_step=False, on_epoch=True, batch_size=batch_size)
        if self.use_node_label_head:
            self.log("train_node_label_ce", losses["label_ce"], on_step=False, on_epoch=True, batch_size=batch_size)
        return total_loss

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            uses_pairwise_supervision = (
                self.use_locality_supervision
                or self.use_edge_label_head
                or self.use_auxiliary_locality_supervision
            )
            if uses_pairwise_supervision:
                if self.use_node_label_head:
                    input_examples, global_condition, edge_idx, edge_labels, edge_label_idx, edge_label_targets, aux_edge_idx, aux_edge_labels, node_presence_mask, node_degree_targets, node_label_targets = batch
                else:
                    input_examples, global_condition, edge_idx, edge_labels, edge_label_idx, edge_label_targets, aux_edge_idx, aux_edge_labels, node_presence_mask, node_degree_targets = batch
                    node_label_targets = None
            else:
                if self.use_node_label_head:
                    input_examples, global_condition, node_presence_mask, node_degree_targets, node_label_targets = batch
                else:
                    input_examples, global_condition, node_presence_mask, node_degree_targets = batch
                    node_label_targets = None
                edge_label_idx = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
                edge_label_targets = torch.empty((0,), dtype=torch.long, device=input_examples.device)
                aux_edge_idx = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
                aux_edge_labels = torch.empty((0,), dtype=torch.float32, device=input_examples.device)

            batch_size = int(node_presence_mask.shape[0])
            losses, latent_tokens = self._eqm_loss(
                input_examples,
                global_condition,
                node_presence_mask=node_presence_mask,
                node_degree_targets=node_degree_targets,
                node_label_targets=node_label_targets,
                create_graph=False,
            )
            total_loss = losses["total"]

            if self.use_locality_supervision and edge_idx.numel() > 0:
                b, i, j = edge_idx.unbind(1)
                h_i = latent_tokens[b, i]
                h_j = latent_tokens[b, j]
                edge_logits = self.edge_head(h_i, h_j)
                edge_loss = F.binary_cross_entropy_with_logits(
                    edge_logits,
                    edge_labels,
                    pos_weight=self.edge_pos_weight,
                )
                total_loss = total_loss + self.lambda_locality_importance * edge_loss

                edge_pred = (torch.sigmoid(edge_logits) > 0.5).float()
                edge_acc = (edge_pred == edge_labels).float().mean()

                self.log("val_edge_ce", edge_loss, on_step=False, on_epoch=True, batch_size=batch_size)
                self.log("val_edge_acc", edge_acc, on_step=False, on_epoch=True, batch_size=batch_size)

            if self.use_edge_label_head and edge_label_idx.numel() > 0:
                b, i, j = edge_label_idx.unbind(1)
                h_i = latent_tokens[b, i]
                h_j = latent_tokens[b, j]
                edge_label_logits = self.edge_label_head(h_i, h_j)
                edge_label_loss = F.cross_entropy(edge_label_logits, edge_label_targets)
                total_loss = total_loss + self.lambda_edge_label_importance * edge_label_loss
                self.log("val_edge_label_ce", edge_label_loss, on_step=False, on_epoch=True, batch_size=batch_size)

            if self.use_auxiliary_locality_supervision and aux_edge_idx.numel() > 0:
                b, i, j = aux_edge_idx.unbind(1)
                h_i = latent_tokens[b, i]
                h_j = latent_tokens[b, j]
                aux_edge_logits = self.auxiliary_edge_head(h_i, h_j)
                aux_edge_loss = F.binary_cross_entropy_with_logits(
                    aux_edge_logits,
                    aux_edge_labels,
                    pos_weight=self.auxiliary_edge_pos_weight,
                )
                total_loss = total_loss + self.lambda_auxiliary_locality_importance * aux_edge_loss

                aux_edge_pred = (torch.sigmoid(aux_edge_logits) > 0.5).float()
                aux_edge_acc = (aux_edge_pred == aux_edge_labels).float().mean()

                self.log("val_aux_locality_ce", aux_edge_loss, on_step=False, on_epoch=True, batch_size=batch_size)
                self.log("val_aux_edge_acc", aux_edge_acc, on_step=False, on_epoch=True, batch_size=batch_size)

            self.log("val_total", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log("val_recon", losses["eqm"], on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("val_eqm", losses["eqm"], on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("val_deg_ce", losses["deg_ce"], on_step=False, on_epoch=True, batch_size=batch_size)
            if self.use_existence_head:
                self.log("val_exist", losses["exist"], on_step=False, on_epoch=True, batch_size=batch_size)
            if self.use_node_label_head:
                self.log("val_node_label_ce", losses["label_ce"], on_step=False, on_epoch=True, batch_size=batch_size)
        return total_loss.detach()

    def on_train_end(self):
        if not self.verbose:
            return
        plot_metrics(
            train_metrics={
                "total": self.train_losses,
                "deg_ce": self.train_deg_ce,
                "eqm": self.train_recon,
                **({"exist": self.train_exist} if self.use_existence_head else {}),
                **({"node_label_ce": self.train_node_label_ce} if self.use_node_label_head else {}),
                **({"edge_label_ce": self.train_edge_label_ce} if self.use_edge_label_head else {}),
                **({"edge_ce": self.train_edge_loss} if self.use_locality_supervision else {}),
                **({"aux_locality": self.train_aux_edge_loss} if self.use_auxiliary_locality_supervision else {}),
            },
            val_metrics={
                "total": self.val_losses,
                "deg_ce": self.val_deg_ce,
                "eqm": self.val_recon,
                **({"exist": self.val_exist} if self.use_existence_head else {}),
                **({"node_label_ce": self.val_node_label_ce} if self.use_node_label_head else {}),
                **({"edge_label_ce": self.val_edge_label_ce} if self.use_edge_label_head else {}),
                **({"edge_ce": self.val_edge_loss} if self.use_locality_supervision else {}),
                **({"aux_locality": self.val_aux_edge_loss} if self.use_auxiliary_locality_supervision else {}),
            },
            window=10,
            alpha=0.1,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def set_guidance_classifier(self, num_classes: int) -> None:
        raise NotImplementedError("Classifier guidance is not implemented for EqMDecompositionalNodeGenerator.")

    def train_guidance_classifier(self, *args, **kwargs):
        raise NotImplementedError("Classifier guidance is not implemented for EqMDecompositionalNodeGenerator.")

    def generate(
        self,
        global_condition: torch.Tensor,
        total_steps: Optional[int] = None,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        global_condition_unconditional: Optional[torch.Tensor] = None,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
        use_heads_projection: bool = False,
        exist_threshold: float = 0.5,
    ) -> torch.Tensor:
        del desired_target, desired_class
        if guidance_scale < 0:
            raise ValueError(f"guidance_scale must be >= 0 (got {guidance_scale}).")
        self.eval()
        steps = int(total_steps if total_steps is not None else self.sampling_steps)
        eta = self.sampling_step_size
        batch_size = global_condition.size(0)
        device = global_condition.device
        use_cfg = global_condition_unconditional is not None
        if use_cfg and global_condition_unconditional.shape != global_condition.shape:
            raise ValueError(
                "global_condition_unconditional must have the same shape as global_condition "
                f"(got {tuple(global_condition_unconditional.shape)} vs {tuple(global_condition.shape)})."
            )
        self._last_edge_probability_matrices = None
        self._last_edge_label_matrices = None
        self._last_node_presence_mask = None
        self._last_deg_classes = None
        self._last_node_label_classes = None

        node_mask = None
        x = torch.randn(
            batch_size,
            self.number_of_rows_per_example,
            self.input_feature_dimension,
            device=device,
        )

        for _ in range(steps):
            x = x.detach().requires_grad_(True)
            with torch.enable_grad():
                score_cond, _, latent_tokens = self._compute_score_field(
                    x,
                    global_condition,
                    node_mask=node_mask,
                    create_graph=False,
                )
                if use_cfg:
                    score_uncond, _, _ = self._compute_score_field(
                        x,
                        global_condition_unconditional,
                        node_mask=node_mask,
                        create_graph=False,
                    )
                    score = score_uncond + guidance_scale * (score_cond - score_uncond)
                else:
                    score = score_cond
            x = (x + eta * score).detach()
            if self.langevin_noise_scale > 0:
                x = x + np.sqrt(2.0 * eta) * self.langevin_noise_scale * torch.randn_like(x)

        if use_heads_projection:
            with torch.enable_grad():
                x_for_heads = x.detach().requires_grad_(True)
                _, _, latent_tokens = self._compute_score_field(
                    x_for_heads,
                    global_condition,
                    node_mask=node_mask,
                    create_graph=False,
                )
            with torch.no_grad():
                logits_deg = self.degree_head(latent_tokens)
                logits_exist = self.exist_head(latent_tokens).squeeze(-1) if self.use_existence_head else None
                logits_label = self.node_label_head(latent_tokens) if self.use_node_label_head else None

            if self.use_existence_head:
                exist_probs = torch.sigmoid(logits_exist)
                self._last_node_presence_mask = (exist_probs >= exist_threshold).detach().cpu()
            else:
                self._last_node_presence_mask = torch.ones(
                    batch_size,
                    self.number_of_rows_per_example,
                    dtype=torch.bool,
                    device=x.device,
                ).cpu()
            self._last_deg_classes = torch.argmax(logits_deg, dim=-1).detach().cpu()
            if logits_label is not None:
                self._last_node_label_classes = torch.argmax(logits_label, dim=-1).detach().cpu()
            if self.use_locality_supervision:
                edge_probs = self._compute_edge_probability_matrices(latent_tokens)
                self._last_edge_probability_matrices = edge_probs.detach().cpu()
            if self.use_edge_label_head:
                edge_label_logits = self._compute_edge_label_logits(latent_tokens)
                edge_label_classes = torch.argmax(edge_label_logits, dim=-1)
                self._last_edge_label_matrices = edge_label_classes.detach().cpu()

        return x.detach()


class EqMDecompositionalNodeGenerator(ConditionalNodeGeneratorBase):
    """Scikit-learn friendly facade for a conditional EqM node generator."""

    def __init__(
        self,
        latent_embedding_dimension: int = 128,
        number_of_transformer_layers: int = 4,
        transformer_attention_head_count: int = 4,
        transformer_dropout: float = 0.1,
        learning_rate: float = 1e-3,
        maximum_epochs: int = 10,
        batch_size: int = 32,
        total_steps: int = 100,
        verbose: bool = False,
        verbose_epoch_interval: int = 10,
        enable_early_stopping: bool = True,
        early_stopping_monitor: str = "val_eqm",
        early_stopping_mode: str = "min",
        early_stopping_patience: int = 30,
        early_stopping_min_delta: float = 0.0,
        restore_best_checkpoint: bool = True,
        artifact_root_dir: Optional[str] = None,
        checkpoint_root_dir: Optional[str] = None,
        important_feature_index: int = 1,
        lambda_degree_importance: float = 1.0,
        degree_temperature: Optional[float] = None,
        lambda_node_exist_importance: float = 1.0,
        lambda_node_label_importance: float = 1.0,
        default_exist_pos_weight: float = 1.0,
        lambda_locality_importance: float = 1.0,
        lambda_auxiliary_locality_importance: float = 1.0,
        lambda_edge_label_importance: float = 1.0,
        pool_condition_tokens: bool = False,
        eqm_sigma: float = 0.2,
        sampling_step_size: float = 0.05,
        sampling_steps: Optional[int] = None,
        langevin_noise_scale: float = 0.0,
        require_embedded_node_label_histogram: bool = False,
        cfg_condition_dropout_prob: float = 0.1,
        cfg_null_target_strategy: str = "zero",
        target_classification_max_distinct: int = 20,
    ):
        self.latent_embedding_dimension = latent_embedding_dimension
        self.number_of_transformer_layers = number_of_transformer_layers
        self.transformer_attention_head_count = transformer_attention_head_count
        self.transformer_dropout = transformer_dropout
        self.learning_rate = learning_rate
        self.maximum_epochs = maximum_epochs
        self.batch_size = batch_size
        self.total_steps = int(total_steps)
        self.verbose = verbose
        self.verbose_epoch_interval = int(verbose_epoch_interval)
        self.enable_early_stopping = bool(enable_early_stopping)
        self.early_stopping_monitor = str(early_stopping_monitor)
        self.early_stopping_mode = str(early_stopping_mode)
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.restore_best_checkpoint = bool(restore_best_checkpoint)
        if artifact_root_dir is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            artifact_root_dir = os.path.join(repo_root, ".artifacts")
        self.artifact_root_dir = str(artifact_root_dir)
        if checkpoint_root_dir is None:
            checkpoint_root_dir = os.path.join(self.artifact_root_dir, "checkpoints", "eqm")
        self.checkpoint_root_dir = str(checkpoint_root_dir)
        self.important_feature_index = important_feature_index
        self.lambda_degree_importance = lambda_degree_importance
        self.degree_temperature = degree_temperature
        self.lambda_node_exist_importance = lambda_node_exist_importance
        self.lambda_node_label_importance = lambda_node_label_importance
        self.lambda_edge_label_importance = lambda_edge_label_importance
        self.default_exist_pos_weight = default_exist_pos_weight
        self.lambda_locality_importance = lambda_locality_importance
        self.lambda_auxiliary_locality_importance = lambda_auxiliary_locality_importance
        self.pool_condition_tokens = bool(pool_condition_tokens)
        self.use_guidance = False
        self.use_locality_supervision = False
        self.eqm_sigma = float(eqm_sigma)
        self.sampling_step_size = float(sampling_step_size)
        self.sampling_steps = int(sampling_steps if sampling_steps is not None else total_steps)
        self.langevin_noise_scale = float(langevin_noise_scale)
        self.cfg_condition_dropout_prob = float(cfg_condition_dropout_prob)
        self.cfg_null_target_strategy = str(cfg_null_target_strategy)
        self.target_classification_max_distinct = int(target_classification_max_distinct)
        if not 0.0 <= self.cfg_condition_dropout_prob <= 1.0:
            raise ValueError(
                f"cfg_condition_dropout_prob must be in [0, 1] (got {self.cfg_condition_dropout_prob})."
            )
        if self.cfg_null_target_strategy not in {"zero"}:
            raise ValueError(
                f"cfg_null_target_strategy must be one of ['zero'] (got {self.cfg_null_target_strategy!r})."
            )
        if self.target_classification_max_distinct < 1:
            raise ValueError(
                "target_classification_max_distinct must be >= 1 "
                f"(got {self.target_classification_max_distinct})."
            )
        self.require_embedded_node_label_histogram = False
        self.use_existence_head = True
        self.constant_existence_value = 1.0
        self.use_node_label_head = False
        self.node_label_classes_ = None
        self.node_label_to_index_ = None
        self.edge_label_classes_ = None
        self.edge_label_to_index_ = None
        self.base_condition_feature_dimension = None
        self.node_label_histogram_dimension = 0
        self.last_predicted_node_label_classes_ = None
        self.last_predicted_edge_probability_matrices_ = None
        self.last_predicted_edge_label_matrices_ = None
        self.target_mode_ = None
        self.target_classes_ = None
        self.target_to_index_ = None
        self.target_scaler_ = None
        self.target_condition_dim_ = 0
        self.target_condition_start_ = None
        self.guidance_enabled_ = False

        self.number_of_rows_per_example = None
        self.input_feature_dimension = None
        self.model = None
        self.x_scaler = None
        self.y_scaler = None
        self.D_max = None
        self.condition_token_count = 1
        self.condition_feature_dimension = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_checkpoint_path_ = None
        self.best_checkpoint_score_ = None
        self.best_checkpoint_epoch_ = None

    def _plan_channel(self, channel_name: str):
        """Return the named channel from the orchestration supervision plan when available.

        Args:
            channel_name (str): Input value.

        Returns:
            Any: Computed result.
        """
        plan = getattr(self, "supervision_plan_", None)
        if plan is None:
            return None
        return getattr(plan, channel_name, None)

    def _planned_enabled(self, channel_name: str, fallback: bool) -> bool:
        """Resolve whether a supervision channel is intended to be active.

        Args:
            channel_name (str): Input value.
            fallback (bool): Input value.

        Returns:
            bool: Computed result.
        """
        channel = self._plan_channel(channel_name)
        if channel is None:
            return fallback
        return bool(getattr(channel, "enabled", fallback))

    def _effective_supervision_flags(
        self,
        edge_pairs: Optional[List[Tuple[int, int, int]]],
        edge_targets: Optional[np.ndarray],
        edge_label_pairs: Optional[List[Tuple[int, int, int]]],
        edge_label_targets: Optional[np.ndarray],
        auxiliary_edge_pairs: Optional[List[Tuple[int, int, int]]],
        auxiliary_edge_targets: Optional[np.ndarray],
    ) -> Tuple[bool, bool, bool]:
        """Combine the supervision plan with the actually supplied arrays.

        Args:
            edge_pairs (Optional[List[Tuple[int, int, int]]]): Input value.
            edge_targets (Optional[np.ndarray]): Input value.
            edge_label_pairs (Optional[List[Tuple[int, int, int]]]): Input value.
            edge_label_targets (Optional[np.ndarray]): Input value.
            auxiliary_edge_pairs (Optional[List[Tuple[int, int, int]]]): Input value.
            auxiliary_edge_targets (Optional[np.ndarray]): Input value.

        Returns:
            Tuple[bool, bool, bool]: Computed result.
        """
        planned_direct_edges = self._planned_enabled("direct_edges", False)
        planned_aux_locality = self._planned_enabled("auxiliary_locality", False)
        planned_edge_labels = self._planned_enabled(
            "edge_labels",
            edge_label_pairs is not None and edge_label_targets is not None,
        )

        effective_locality = planned_direct_edges and edge_pairs is not None and edge_targets is not None
        effective_auxiliary_locality = (
            planned_aux_locality and auxiliary_edge_pairs is not None and auxiliary_edge_targets is not None
        )
        effective_edge_labels = planned_edge_labels and edge_label_pairs is not None and edge_label_targets is not None
        return effective_locality, effective_auxiliary_locality, effective_edge_labels

    def _fit_scalers(self, X_array: np.ndarray, y_array: np.ndarray) -> None:
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.x_scaler.fit(X_array.reshape(-1, X_array.shape[2]))
        if y_array.ndim == 3:
            self.y_scaler.fit(y_array.reshape(-1, y_array.shape[-1]))
        else:
            self.y_scaler.fit(y_array)

    def _transform_data(self, X_array: np.ndarray, y_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size, row_count, feature_count = X_array.shape
        X_scaled = self.x_scaler.transform(X_array.reshape(-1, feature_count)).reshape(batch_size, row_count, feature_count)
        if y_array.ndim == 3:
            cond_batch, token_count, cond_dim = y_array.shape
            y_scaled = self.y_scaler.transform(y_array.reshape(-1, cond_dim)).reshape(cond_batch, token_count, cond_dim)
        else:
            y_scaled = self.y_scaler.transform(y_array)
        return X_scaled, y_scaled

    def _inverse_transform_input(self, X_array: np.ndarray) -> np.ndarray:
        batch_size, row_count, feature_count = X_array.shape
        return self.x_scaler.inverse_transform(X_array.reshape(-1, feature_count)).reshape(batch_size, row_count, feature_count)

    def _fit_node_label_vocab(self, node_label_targets: List[np.ndarray]) -> None:
        flat_labels = [label for labels in node_label_targets for label in np.asarray(labels, dtype=object).tolist()]
        if len(flat_labels) == 0:
            self.node_label_classes_ = np.asarray([], dtype=object)
            self.node_label_to_index_ = {}
            self.use_node_label_head = False
            self.node_label_histogram_dimension = 0
            return
        self.node_label_classes_ = np.unique(np.asarray(flat_labels, dtype=object))
        self.node_label_to_index_ = {label: idx for idx, label in enumerate(self.node_label_classes_)}
        self.node_label_histogram_dimension = int(len(self.node_label_classes_))
        self.use_node_label_head = self.node_label_histogram_dimension > 1

    def _fit_edge_label_vocab(self, edge_label_targets: np.ndarray) -> None:
        flat_labels = np.asarray(edge_label_targets, dtype=object).tolist()
        if len(flat_labels) == 0:
            self.edge_label_classes_ = np.asarray([], dtype=object)
            self.edge_label_to_index_ = {}
            self.use_edge_label_head = False
            return
        self.edge_label_classes_ = np.unique(np.asarray(flat_labels, dtype=object))
        self.edge_label_to_index_ = {label: idx for idx, label in enumerate(self.edge_label_classes_)}
        self.use_edge_label_head = len(self.edge_label_classes_) > 1

    def _encode_edge_label_targets(
        self,
        edge_label_targets: np.ndarray,
    ) -> np.ndarray:
        return np.asarray([self.edge_label_to_index_[label] for label in np.asarray(edge_label_targets, dtype=object)], dtype=np.int64)

    def _encode_node_label_targets(
        self,
        node_label_targets: List[np.ndarray],
        max_num_rows: int,
    ) -> np.ndarray:
        encoded = np.zeros((len(node_label_targets), max_num_rows), dtype=np.int64)
        for graph_idx, labels in enumerate(node_label_targets):
            labels = np.asarray(labels, dtype=object)
            for node_idx, label in enumerate(labels[:max_num_rows]):
                encoded[graph_idx, node_idx] = self.node_label_to_index_[label]
        return encoded

    def _compute_node_label_histograms(self, encoded_targets: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        histograms = np.zeros((encoded_targets.shape[0], self.node_label_histogram_dimension), dtype=float)
        if self.node_label_histogram_dimension == 0:
            return histograms
        for graph_idx in range(encoded_targets.shape[0]):
            valid_indices = encoded_targets[graph_idx][valid_mask[graph_idx]]
            if valid_indices.size == 0:
                continue
            counts = np.bincount(valid_indices, minlength=self.node_label_histogram_dimension).astype(float)
            total = counts.sum()
            if total > 0:
                histograms[graph_idx] = counts / total
        return histograms

    def _compose_condition_array(self, graph_conditioning: GraphConditioningBatch) -> np.ndarray:
        """Compose a concrete NN conditioning matrix from explicit semantic fields.

        Args:
            graph_conditioning (GraphConditioningBatch): Input value.

        Returns:
            np.ndarray: Computed result.
        """
        graph_embeddings = np.asarray(graph_conditioning.graph_embeddings, dtype=float)
        if graph_embeddings.ndim == 1:
            graph_embeddings = graph_embeddings[:, None]
        node_counts = np.asarray(graph_conditioning.node_counts, dtype=float).reshape(-1, 1)
        edge_counts = np.asarray(graph_conditioning.edge_counts, dtype=float).reshape(-1, 1)
        parts = [graph_embeddings, node_counts, edge_counts]
        if graph_conditioning.node_label_histograms is not None:
            parts.append(np.asarray(graph_conditioning.node_label_histograms, dtype=float))
        return np.concatenate(parts, axis=1)

    def _fit_target_encoder(self, targets: Sequence[Any]) -> None:
        targets_array = np.asarray(targets, dtype=object)
        unique_targets = np.unique(targets_array)
        if unique_targets.size <= self.target_classification_max_distinct:
            self.target_mode_ = "classification"
            self.target_classes_ = unique_targets
            self.target_to_index_ = {
                target_value: int(index)
                for index, target_value in enumerate(self.target_classes_.tolist())
            }
            self.target_scaler_ = None
            self.target_condition_dim_ = int(len(self.target_classes_))
        else:
            self.target_mode_ = "regression"
            self.target_classes_ = None
            self.target_to_index_ = None
            self.target_scaler_ = MinMaxScaler()
            numeric_targets = np.asarray(targets, dtype=float).reshape(-1, 1)
            self.target_scaler_.fit(numeric_targets)
            self.target_condition_dim_ = 1
        self.guidance_enabled_ = True

    def _reset_target_encoder(self) -> None:
        self.target_mode_ = None
        self.target_classes_ = None
        self.target_to_index_ = None
        self.target_scaler_ = None
        self.target_condition_dim_ = 0
        self.target_condition_start_ = None
        self.guidance_enabled_ = False

    def _encode_targets(self, targets: Sequence[Any]) -> np.ndarray:
        if not self.guidance_enabled_ or self.target_mode_ is None:
            return np.zeros((len(targets), 0), dtype=float)
        if self.target_mode_ == "classification":
            encoded = np.zeros((len(targets), self.target_condition_dim_), dtype=float)
            for row, target in enumerate(targets):
                if target not in self.target_to_index_:
                    raise ValueError(f"Unknown classification target value: {target!r}")
                encoded[row, self.target_to_index_[target]] = 1.0
            return encoded
        numeric_targets = np.asarray(targets, dtype=float).reshape(-1, 1)
        return self.target_scaler_.transform(numeric_targets)

    def _normalize_desired_target(
        self,
        desired_target: Optional[Union[int, float, Sequence[Any]]],
        batch_size: int,
    ) -> Optional[List[Any]]:
        if desired_target is None:
            return None
        if isinstance(desired_target, (list, tuple, np.ndarray)):
            values = list(desired_target)
            if len(values) != batch_size:
                raise ValueError(
                    "desired_target sequence length must match the batch size "
                    f"(got {len(values)} values for batch size {batch_size})."
                )
            return values
        return [desired_target] * batch_size

    def _compute_binary_pos_weight(self, targets: Optional[np.ndarray], default: float = 1.0) -> float:
        """Compute a BCE positive-class weight from binary targets.

        Args:
            targets (Optional[np.ndarray]): Input value.
            default (float): Optional input value.

        Returns:
            float: Computed result.
        """
        if targets is None:
            return float(default)
        targets_array = np.asarray(targets, dtype=float).reshape(-1)
        if targets_array.size == 0:
            return float(default)
        positive = int(np.sum(targets_array >= 0.5))
        negative = int(targets_array.size - positive)
        if positive == 0:
            return float(default)
        if negative > positive:
            return float(negative) / float(positive)
        return 1.0

    def setup(
        self,
        node_batch: NodeGenerationBatch,
        graph_conditioning: GraphConditioningBatch,
        targets: Optional[Sequence[Any]] = None,
    ):
        node_encodings_list = node_batch.node_embeddings_list
        edge_pairs = node_batch.edge_pairs
        edge_targets = node_batch.edge_targets
        edge_label_pairs = node_batch.edge_label_pairs
        edge_label_targets = node_batch.edge_label_targets
        auxiliary_edge_pairs = node_batch.auxiliary_edge_pairs
        auxiliary_edge_targets = node_batch.auxiliary_edge_targets
        node_label_targets = node_batch.node_label_targets
        edge_label_plan = self._plan_channel("edge_labels")
        planned_direct_edges = self._planned_enabled("direct_edges", False)
        planned_aux_locality = self._planned_enabled("auxiliary_locality", False)
        planned_learned_edge_labels = (
            getattr(edge_label_plan, "mode", None) == "learned"
            if edge_label_plan is not None
            else edge_label_pairs is not None and edge_label_targets is not None
        )
        effective_locality, effective_auxiliary_locality, effective_edge_labels = self._effective_supervision_flags(
            edge_pairs,
            edge_targets,
            edge_label_pairs,
            edge_label_targets,
            auxiliary_edge_pairs,
            auxiliary_edge_targets,
        )
        if planned_direct_edges and not effective_locality and self.verbose:
            print("Direct-edge channel planned as learned, but usable edge_pairs/edge_targets were not supplied.")
        if planned_aux_locality and not effective_auxiliary_locality and self.verbose:
            print("Auxiliary-locality channel planned as learned, but usable auxiliary_edge_pairs/auxiliary_edge_targets were not supplied.")
        if planned_learned_edge_labels and not effective_edge_labels and self.verbose:
            print("Edge-label channel planned as learned, but usable edge_label_pairs/edge_label_targets were not supplied.")
        if not effective_locality:
            edge_pairs = None
            edge_targets = None
        if not effective_auxiliary_locality:
            auxiliary_edge_pairs = None
            auxiliary_edge_targets = None
        if not effective_edge_labels:
            edge_label_pairs = None
            edge_label_targets = None

        max_num_rows = max(x.shape[0] for x in node_encodings_list)
        self.number_of_rows_per_example = max_num_rows

        padded_examples = []
        for x in node_encodings_list:
            if x.shape[0] < max_num_rows:
                x = np.pad(x, ((0, max_num_rows - x.shape[0]), (0, 0)), mode="constant", constant_values=0)
            padded_examples.append(x)

        X_array = np.stack(padded_examples, axis=0)
        base_condition_array = self._compose_condition_array(graph_conditioning)
        self.target_condition_start_ = int(base_condition_array.shape[1])
        if targets is not None:
            if len(targets) != len(node_encodings_list):
                raise ValueError(
                    "targets length must match node batch size "
                    f"(got {len(targets)} targets for {len(node_encodings_list)} graphs)."
                )
            self._fit_target_encoder(targets)
            target_condition_array = self._encode_targets(targets)
            y_array = np.concatenate([base_condition_array, target_condition_array], axis=1)
        else:
            self._reset_target_encoder()
            y_array = base_condition_array
        self.condition_token_count = 1
        valid_mask = np.asarray(node_batch.node_presence_mask, dtype=bool)

        encoded_node_label_targets = None
        if node_label_targets is not None:
            self._fit_node_label_vocab(node_label_targets)
            encoded_node_label_targets = self._encode_node_label_targets(node_label_targets, max_num_rows)
            self.base_condition_feature_dimension = base_condition_array.shape[1]
        else:
            self.base_condition_feature_dimension = base_condition_array.shape[1]
            self.use_node_label_head = False
            self.node_label_histogram_dimension = 0
            self.node_label_classes_ = None
            self.node_label_to_index_ = None

        if edge_label_targets is not None:
            self._fit_edge_label_vocab(edge_label_targets)
        else:
            self.use_edge_label_head = False
            self.edge_label_classes_ = None
            self.edge_label_to_index_ = None

        self.condition_feature_dimension = y_array.shape[1]

        self._fit_scalers(X_array, y_array)

        ones = int(valid_mask.sum())
        zeros = int(valid_mask.size) - ones
        if ones == 0:
            exist_pos_weight = float(self.default_exist_pos_weight)
        elif zeros > ones:
            exist_pos_weight = float(zeros) / float(ones)
        else:
            exist_pos_weight = 1.0
        edge_pos_weight = self._compute_binary_pos_weight(edge_targets, default=1.0)
        auxiliary_edge_pos_weight = self._compute_binary_pos_weight(auxiliary_edge_targets, default=1.0)

        deg_column = np.asarray(node_batch.node_degree_targets, dtype=float).reshape(-1, 1)
        degree_scaler = MinMaxScaler().fit(deg_column)
        deg_min_val = degree_scaler.data_min_[0]
        deg_range_val = degree_scaler.data_range_[0]
        if deg_range_val == 0:
            deg_range_val = 1e-8

        X_scaled, y_scaled = self._transform_data(X_array, y_array)
        self.input_feature_dimension = X_scaled.shape[2]
        self.D_max = int(np.asarray(node_batch.node_degree_targets, dtype=np.int64).max())
        valid_exist_targets = valid_mask[valid_mask]
        all_same_node_size = np.all(valid_mask.sum(axis=1) == valid_mask.sum(axis=1)[0])
        disable_existence = (
            all_same_node_size
            and valid_exist_targets.size > 0
            and np.all(valid_exist_targets == valid_exist_targets.flat[0])
        )
        self.use_existence_head = not disable_existence
        if disable_existence:
            self.constant_existence_value = float(valid_exist_targets.flat[0])
            if self.verbose:
                print(
                    "Existence supervision disabled: all training graphs have the same node count "
                    "and the valid existence target is constant."
                )
        else:
            self.constant_existence_value = 1.0

        if self.verbose:
            if effective_locality:
                print("Direct edge supervision enabled: horizon-1 edge presence will be learned and used by the decoder.")
            elif planned_direct_edges:
                print("Direct edge supervision disabled at setup time: the plan requested it but usable training pairs were not supplied.")
            else:
                print("Direct edge supervision disabled: the supervision plan does not request horizon-1 edge prediction.")

            if effective_auxiliary_locality:
                print("Auxiliary locality supervision enabled: higher-horizon locality pairs will be used only as an encoding regularizer.")
            elif effective_locality:
                print("Auxiliary locality supervision disabled: only direct edge supervision will be used; no extra locality head is trained.")
            elif planned_aux_locality:
                print("Auxiliary locality supervision disabled at setup time: the plan requested it but usable higher-horizon pairs were not supplied.")
            else:
                print("Auxiliary locality supervision disabled: the supervision plan does not request higher-horizon locality regularization.")

            if self.use_edge_label_head and effective_edge_labels:
                print(
                    "Edge-label supervision enabled: discrete edge labels will be predicted by the generator."
                )
            elif planned_learned_edge_labels:
                print("Edge-label supervision disabled at setup time: the plan requested it but usable labelled edges were not supplied.")
            elif edge_label_plan is not None and edge_label_plan.mode == "constant":
                print(
                    f"Edge-label supervision collapsed to a constant label: {edge_label_plan.constant_value}."
                )
            elif edge_label_targets is not None:
                print(
                    "Edge-label supervision collapsed to a constant label: no edge-label head will be trained."
                )
            else:
                print("Edge-label supervision disabled: the supervision plan does not request learned edge-label prediction.")
            if effective_locality:
                print(f"Direct-edge BCE positive weight: {edge_pos_weight:.3f}.")
            if effective_auxiliary_locality:
                print(f"Auxiliary-locality BCE positive weight: {auxiliary_edge_pos_weight:.3f}.")

        self.model = EqMDecompositionalNodeGeneratorModule(
            number_of_rows_per_example=self.number_of_rows_per_example,
            input_feature_dimension=self.input_feature_dimension,
            condition_feature_dimension=self.condition_feature_dimension,
            latent_embedding_dimension=self.latent_embedding_dimension,
            number_of_transformer_layers=self.number_of_transformer_layers,
            transformer_attention_head_count=self.transformer_attention_head_count,
            transformer_dropout=self.transformer_dropout,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
            verbose_epoch_interval=self.verbose_epoch_interval,
            important_feature_index=self.important_feature_index,
            max_degree=self.D_max,
            lambda_degree_importance=self.lambda_degree_importance,
            degree_temperature=self.degree_temperature,
            degree_min_val=deg_min_val,
            degree_range_val=deg_range_val,
            lambda_node_exist_importance=self.lambda_node_exist_importance,
            lambda_node_label_importance=self.lambda_node_label_importance,
            lambda_edge_label_importance=self.lambda_edge_label_importance,
            use_locality_supervision=effective_locality,
            lambda_locality_importance=self.lambda_locality_importance,
            use_auxiliary_locality_supervision=effective_auxiliary_locality,
            lambda_auxiliary_locality_importance=self.lambda_auxiliary_locality_importance,
            exist_pos_weight=exist_pos_weight,
            edge_pos_weight=edge_pos_weight,
            auxiliary_edge_pos_weight=auxiliary_edge_pos_weight,
            num_node_label_classes=self.node_label_histogram_dimension,
            use_node_label_head=self.use_node_label_head,
            num_edge_label_classes=0 if self.edge_label_classes_ is None else len(self.edge_label_classes_),
            use_edge_label_head=self.use_edge_label_head,
            eqm_sigma=self.eqm_sigma,
            sampling_step_size=self.sampling_step_size,
            sampling_steps=self.sampling_steps,
            langevin_noise_scale=self.langevin_noise_scale,
            pool_condition_tokens=self.pool_condition_tokens,
            guidance_enabled=self.guidance_enabled_,
            target_condition_start_index=self.target_condition_start_,
            target_condition_feature_count=self.target_condition_dim_,
            cfg_condition_dropout_prob=self.cfg_condition_dropout_prob,
            cfg_null_target_strategy=self.cfg_null_target_strategy,
        )
        self.model.use_existence_head = self.use_existence_head
        self.model.constant_existence_value = self.constant_existence_value
        self.edge_pos_weight_ = float(edge_pos_weight)
        self.auxiliary_edge_pos_weight_ = float(auxiliary_edge_pos_weight)
        if int(self.verbose) >= 1:
            parameter_count = sum(param.numel() for param in self.model.parameters())
            trainable_parameter_count = sum(
                param.numel() for param in self.model.parameters() if param.requires_grad
            )
            print(
                "ANN size: "
                f"parameters={parameter_count:,}, "
                f"trainable={trainable_parameter_count:,}."
            )

    def fit(
        self,
        node_batch: NodeGenerationBatch,
        graph_conditioning: GraphConditioningBatch,
        targets: Optional[Sequence[Any]] = None,
    ):
        node_encodings_list = node_batch.node_embeddings_list
        edge_pairs = node_batch.edge_pairs
        edge_targets = node_batch.edge_targets
        edge_label_pairs = node_batch.edge_label_pairs
        edge_label_targets = node_batch.edge_label_targets
        auxiliary_edge_pairs = node_batch.auxiliary_edge_pairs
        auxiliary_edge_targets = node_batch.auxiliary_edge_targets
        node_label_targets = node_batch.node_label_targets
        X_array = np.stack(
            [
                np.pad(x, ((0, self.number_of_rows_per_example - x.shape[0]), (0, 0)), mode="constant", constant_values=0)
                for x in node_encodings_list
            ],
            axis=0,
        )
        base_condition_array = self._compose_condition_array(graph_conditioning)
        if self.guidance_enabled_:
            if targets is None:
                if int(self.verbose) >= 1:
                    print("Guidance targets were not provided at fit time; using unconditional (null) targets.")
                target_condition_array = np.zeros((len(base_condition_array), self.target_condition_dim_), dtype=float)
            else:
                if len(targets) != len(base_condition_array):
                    raise ValueError(
                        "targets length must match node batch size "
                        f"(got {len(targets)} targets for {len(base_condition_array)} graphs)."
                    )
                target_condition_array = self._encode_targets(targets)
            y_array = np.concatenate([base_condition_array, target_condition_array], axis=1)
        else:
            y_array = base_condition_array
        mask_array = np.asarray(node_batch.node_presence_mask, dtype=bool)
        degree_target_array = np.asarray(node_batch.node_degree_targets, dtype=np.int64)
        encoded_node_label_targets = None
        encoded_edge_label_targets = None
        if node_label_targets is not None and self.node_label_histogram_dimension > 0:
            encoded_node_label_targets = self._encode_node_label_targets(node_label_targets, self.number_of_rows_per_example)
        if edge_label_targets is not None and self.edge_label_to_index_ is not None:
            encoded_edge_label_targets = self._encode_edge_label_targets(edge_label_targets)
        X_scaled, y_scaled = self._transform_data(X_array, y_array)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_array, dtype=torch.bool)
        degree_tensor = torch.tensor(degree_target_array, dtype=torch.long)

        effective_locality, effective_auxiliary_locality, effective_edge_labels = self._effective_supervision_flags(
            edge_pairs,
            edge_targets,
            edge_label_pairs,
            encoded_edge_label_targets,
            auxiliary_edge_pairs,
            auxiliary_edge_targets,
        )
        if effective_locality or effective_auxiliary_locality or effective_edge_labels:
            dataset = EqMGraphWithEdgesDataset(
                X_scaled,
                y_scaled,
                edge_pairs,
                edge_targets,
                edge_label_pairs,
                encoded_edge_label_targets,
                auxiliary_edge_pairs,
                auxiliary_edge_targets,
                mask_array,
                degree_target_array,
                encoded_node_label_targets,
            )
            dataset_size = len(node_encodings_list)
            train_size = int(0.9 * dataset_size)
            val_size = dataset_size - train_size
            indices = torch.randperm(dataset_size).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_eqm_graph_with_edges,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_eqm_graph_with_edges,
            )
        else:
            if encoded_node_label_targets is None:
                dataset = TensorDataset(X_tensor, y_tensor, mask_tensor, degree_tensor)
            else:
                label_tensor = torch.tensor(encoded_node_label_targets, dtype=torch.long)
                dataset = TensorDataset(X_tensor, y_tensor, mask_tensor, degree_tensor, label_tensor)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        callbacks = [MetricsLogger()]
        checkpoint_dir = os.path.join(
            self.checkpoint_root_dir,
            f"{self.__class__.__name__}_{uuid.uuid4().hex}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:03d}-{val_total:.4f}",
            monitor=self.early_stopping_monitor,
            mode=self.early_stopping_mode,
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
            save_weights_only=False,
        )
        callbacks.append(checkpoint_callback)
        if int(self.verbose) >= 1:
            print(f"Writing Lightning logs to {os.path.join(self.artifact_root_dir, 'lightning_logs')}")
            print(f"Writing checkpoints to {checkpoint_dir}")
        if self.enable_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=self.early_stopping_monitor,
                    mode=self.early_stopping_mode,
                    patience=self.early_stopping_patience,
                    min_delta=self.early_stopping_min_delta,
                )
            )

        trainer = pl.Trainer(
            max_epochs=self.maximum_epochs,
            callbacks=callbacks,
            logger=True,
            default_root_dir=self.artifact_root_dir,
            enable_checkpointing=True,
            enable_progress_bar=False,
            log_every_n_steps=max(1, min(10, len(train_loader))),
        )
        if not self.verbose:
            with suppress_output():
                run_trainer_fit(
                    trainer,
                    self.model,
                    train_loader,
                    val_loader,
                    context=f"{self.__class__.__name__}.fit",
                )
        else:
            run_trainer_fit(
                trainer,
                self.model,
                train_loader,
                val_loader,
                context=f"{self.__class__.__name__}.fit",
            )
        self.best_checkpoint_path_ = checkpoint_callback.best_model_path or None
        best_score = checkpoint_callback.best_model_score
        self.best_checkpoint_score_ = float(best_score.item()) if best_score is not None else None
        if self.restore_best_checkpoint and self.best_checkpoint_path_:
            checkpoint = torch.load(self.best_checkpoint_path_, map_location=self.device)
            best_epoch = checkpoint.get("epoch")
            self.best_checkpoint_epoch_ = int(best_epoch) if best_epoch is not None else None
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            if int(self.verbose) >= 1:
                stopped_epoch = int(getattr(trainer, "current_epoch", -1)) + 1
                epoch_msg = (
                    f"best_epoch={self.best_checkpoint_epoch_ + 1}, "
                    if self.best_checkpoint_epoch_ is not None
                    else ""
                )
                score_msg = (
                    f"{self.early_stopping_monitor}={self.best_checkpoint_score_:.4f}"
                    if self.best_checkpoint_score_ is not None
                    else f"{self.early_stopping_monitor}=unknown"
                )
                print(
                    f"Restored best checkpoint from {self.best_checkpoint_path_} "
                    f"({epoch_msg}{score_msg}, stopped_epoch={stopped_epoch})."
                )

    def predict(
        self,
        graph_conditioning: GraphConditioningBatch,
        desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
        guidance_scale: float = 1.0,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
    ) -> GeneratedNodeBatch:
        if guidance_scale < 0:
            raise ValueError(f"guidance_scale must be >= 0 (got {guidance_scale}).")
        if desired_target is None and desired_class is not None:
            desired_target = desired_class

        self.device = next(self.model.parameters()).device
        base_condition_array = self._compose_condition_array(graph_conditioning)
        desired_targets = self._normalize_desired_target(desired_target, len(base_condition_array))
        use_cfg_guidance = self.guidance_enabled_ and (desired_targets is not None)
        if self.guidance_enabled_:
            if desired_targets is None:
                target_condition_array = np.zeros((len(base_condition_array), self.target_condition_dim_), dtype=float)
            else:
                target_condition_array = self._encode_targets(desired_targets)
            cond_array = np.concatenate([base_condition_array, target_condition_array], axis=1)
        else:
            if desired_targets is not None and int(self.verbose) >= 1:
                print("desired_target was provided, but guidance conditioning is not available; falling back to unguided generation.")
            cond_array = base_condition_array
        cond_scaled = self.y_scaler.transform(cond_array)
        cond_tensor = torch.tensor(cond_scaled, dtype=torch.float32, device=self.device)
        cond_uncond_tensor = None
        if use_cfg_guidance:
            cond_uncond_scaled = np.asarray(cond_scaled, dtype=float).copy()
            cond_uncond_scaled[:, self.target_condition_start_:self.target_condition_start_ + self.target_condition_dim_] = 0.0
            cond_uncond_tensor = torch.tensor(cond_uncond_scaled, dtype=torch.float32, device=self.device)

        generated = self.model.generate(
            cond_tensor,
            total_steps=self.sampling_steps,
            desired_target=desired_target,
            guidance_scale=guidance_scale,
            global_condition_unconditional=cond_uncond_tensor,
            desired_class=desired_class,
            use_heads_projection=True,
        )

        gen_np = generated.detach().cpu().numpy()
        gen_orig = self._inverse_transform_input(gen_np)
        node_presence_mask = getattr(self.model, "_last_node_presence_mask", None)
        if node_presence_mask is None:
            node_presence_mask = np.ones(
                (len(gen_orig), gen_orig.shape[1]),
                dtype=bool,
            )
        else:
            node_presence_mask = node_presence_mask.numpy()

        deg_classes = getattr(self.model, "_last_deg_classes", None)
        node_degree_predictions = None
        if deg_classes is not None:
            node_degree_predictions = deg_classes.cpu().numpy()
        label_classes = getattr(self.model, "_last_node_label_classes", None)
        predicted_node_labels = None
        self.last_predicted_node_label_classes_ = None
        if label_classes is not None and self.node_label_classes_ is not None:
            label_classes = label_classes.cpu().numpy()
            predicted_node_labels = [
                self.node_label_classes_[label_classes[index]]
                for index in range(label_classes.shape[0])
            ]
            self.last_predicted_node_label_classes_ = predicted_node_labels
        edge_probability_matrices = getattr(self.model, "_last_edge_probability_matrices", None)
        predicted_edge_probability_matrices = None
        self.last_predicted_edge_probability_matrices_ = None
        if edge_probability_matrices is not None:
            edge_probability_matrices = edge_probability_matrices.cpu().numpy()
            predicted_edge_probability_matrices = [
                edge_probability_matrices[index]
                for index in range(edge_probability_matrices.shape[0])
            ]
            self.last_predicted_edge_probability_matrices_ = predicted_edge_probability_matrices
        edge_label_matrices = getattr(self.model, "_last_edge_label_matrices", None)
        predicted_edge_label_matrices = None
        self.last_predicted_edge_label_matrices_ = None
        if edge_label_matrices is not None and self.edge_label_classes_ is not None:
            edge_label_matrices = edge_label_matrices.cpu().numpy()
            predicted_edge_label_matrices = [
                self.edge_label_classes_[edge_label_matrices[index]]
                for index in range(edge_label_matrices.shape[0])
            ]
            self.last_predicted_edge_label_matrices_ = predicted_edge_label_matrices

        return GeneratedNodeBatch(
            node_embeddings_list=[gen_orig[index] for index in range(gen_orig.shape[0])],
            node_presence_mask=node_presence_mask,
            node_degree_predictions=node_degree_predictions,
            node_labels=predicted_node_labels,
            edge_probability_matrices=predicted_edge_probability_matrices,
            edge_label_matrices=predicted_edge_label_matrices,
        )

    def plot_metrics(self, window: int = 10, alpha: float = 0.3):
        if self.model is None:
            print("Model is not fitted yet.")
            return
        plot_metrics(
            train_metrics={
                "total": self.model.train_losses,
                "deg_ce": self.model.train_deg_ce,
                "eqm": self.model.train_recon,
                **({"exist": self.model.train_exist} if self.model.use_existence_head else {}),
                **({"node_label_ce": self.model.train_node_label_ce} if self.model.use_node_label_head else {}),
                **({"edge_label_ce": self.model.train_edge_label_ce} if self.model.use_edge_label_head else {}),
                **({"edge_ce": self.model.train_edge_loss} if self.model.use_locality_supervision else {}),
                **({"aux_locality": self.model.train_aux_edge_loss} if self.model.use_auxiliary_locality_supervision else {}),
            },
            val_metrics={
                "total": self.model.val_losses,
                "deg_ce": self.model.val_deg_ce,
                "eqm": self.model.val_recon,
                **({"exist": self.model.val_exist} if self.model.use_existence_head else {}),
                **({"node_label_ce": self.model.val_node_label_ce} if self.model.use_node_label_head else {}),
                **({"edge_label_ce": self.model.val_edge_label_ce} if self.model.use_edge_label_head else {}),
                **({"edge_ce": self.model.val_edge_loss} if self.model.use_locality_supervision else {}),
                **({"aux_locality": self.model.val_aux_edge_loss} if self.model.use_auxiliary_locality_supervision else {}),
            },
            window=window,
            alpha=alpha,
        )
