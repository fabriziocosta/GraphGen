import contextlib
import math
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
        """Collect weighted loss components so differently scaled terms are comparable."""
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
