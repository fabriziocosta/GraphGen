import contextlib
import math
import os
import sys
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


def get_sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Encode scalar time values into sinusoidal embeddings."""
    half_dim = dim // 2
    inv_freq = torch.exp(
        torch.arange(0, half_dim, device=t.device).float() * (-math.log(10000) / (half_dim - 1))
    )
    angles = t * inv_freq.view(1, -1)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


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
    """Visualise train/validation metrics with a geometric moving average."""

    def _moving_average(data: Sequence[float], window_size: int) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if len(arr) < window_size:
            return np.array([])
        arr_clamped = np.where(arr <= 0, np.finfo(float).tiny, arr)
        log_arr = np.log(arr_clamped)
        kernel = np.ones(window_size, dtype=float) / window_size
        smoothed_log = np.convolve(log_arr, kernel, mode="valid")
        return np.exp(smoothed_log)

    fig, ax0 = plt.subplots(figsize=(15, 8))
    metrics = list(train_metrics.keys())
    axes = [ax0] + [ax0.twinx() for _ in range(len(metrics) - 1)]
    for i, ax in enumerate(axes[1:], start=1):
        ax.spines["right"].set_position(("outward", 60 * i))
    colors = ["blue", "red", "green", "purple", "orange"]
    lines, labels = [], []
    for name, ax, color in zip(metrics, axes, colors):
        train_vals = train_metrics[name]
        val_vals = val_metrics[name]
        if len(train_vals) < 1 or len(val_vals) < 1:
            continue
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
