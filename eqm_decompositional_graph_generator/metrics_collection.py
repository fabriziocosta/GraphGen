"""Metric collection callbacks for EqM training."""

from typing import Dict
import time

import pytorch_lightning as pl
import torch


class MetricsLogger(pl.callbacks.Callback):
    """Collect end-of-epoch metrics into the module's history lists."""

    def on_fit_start(self, trainer, pl_module):
        pl_module._fit_start_time = time.time()
        pl_module._ema_metrics = {}

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0, int(round(float(seconds))))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"

    @staticmethod
    def _component_summary(pl_module, metrics: Dict[str, torch.Tensor], prefix: str):
        component_specs = [
            ("eqm", "recon", 1.0),
            ("deg", "deg_ce", float(getattr(pl_module, "lambda_degree_importance", 1.0))),
            ("exist", "exist", float(getattr(pl_module, "lambda_node_exist_importance", 1.0))),
            ("node_count", "node_count_loss", float(getattr(pl_module, "lambda_node_count_importance", 0.0))),
            ("node_label", "node_label_ce", float(getattr(pl_module, "lambda_node_label_importance", 1.0))),
            ("edge_label", "edge_label_ce", float(getattr(pl_module, "lambda_edge_label_importance", 1.0))),
            (
                "edge",
                "edge_ce",
                float(
                    getattr(
                        pl_module,
                        "lambda_direct_edge_importance",
                        getattr(pl_module, "lambda_locality_importance", 1.0),
                    )
                ),
            ),
            ("edge_count", "edge_count_loss", float(getattr(pl_module, "lambda_edge_count_importance", 0.0))),
            (
                "deg_edge_consistency",
                "degree_edge_consistency_loss",
                float(getattr(pl_module, "lambda_degree_edge_consistency_importance", 0.0)),
            ),
            (
                "aux",
                "aux_locality_ce",
                float(
                    getattr(
                        pl_module,
                        "lambda_auxiliary_edge_importance",
                        getattr(pl_module, "lambda_auxiliary_locality_importance", 1.0),
                    )
                ),
            ),
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

    @staticmethod
    def _update_ema_metric(trainer, pl_module, metric_name: str, metric_value: float) -> float:
        alpha = float(getattr(pl_module, "early_stopping_ema_alpha", 0.3))
        if not 0.0 < alpha <= 1.0:
            alpha = 0.3
        previous = pl_module._ema_metrics.get(metric_name)
        ema_value = metric_value if previous is None else alpha * metric_value + (1.0 - alpha) * previous
        pl_module._ema_metrics[metric_name] = float(ema_value)
        ema_key = f"{metric_name}_ema"
        ema_tensor = torch.tensor(float(ema_value), dtype=torch.float32)
        trainer.callback_metrics[ema_key] = ema_tensor
        if hasattr(trainer, "logged_metrics") and isinstance(trainer.logged_metrics, dict):
            trainer.logged_metrics[ema_key] = ema_tensor
        return float(ema_value)

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
        self._update_ema_metric(trainer, pl_module, "val_total", pl_module.val_losses[-1])
        self._update_ema_metric(trainer, pl_module, "val_eqm", pl_module.val_recon[-1])

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
