"""Metric plotting helpers for EqM training."""

from typing import Dict, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def _loess_smooth(data: Sequence[float], window_size: int) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    n = arr.size
    if n < 3:
        return arr.copy()

    use_log_domain = bool(np.all(arr > 0))
    working = np.log(arr) if use_log_domain else arr.copy()
    x = np.arange(n, dtype=float)
    span = max(3, min(n, int(round(window_size))))
    smoothed = np.empty(n, dtype=float)

    for idx in range(n):
        distances = np.abs(x - x[idx])
        nearest = np.argpartition(distances, span - 1)[:span]
        local_x = x[nearest]
        local_y = working[nearest]

        max_distance = float(np.max(np.abs(local_x - x[idx])))
        if max_distance <= 0.0:
            smoothed[idx] = working[idx]
            continue

        u = np.abs(local_x - x[idx]) / max_distance
        weights = np.power(1.0 - np.power(np.clip(u, 0.0, 1.0), 3.0), 3.0)
        X = np.column_stack([np.ones_like(local_x), local_x - x[idx]])
        weighted_design = X * weights[:, None]
        xtwx = X.T @ weighted_design
        xtwy = weighted_design.T @ local_y
        try:
            beta = np.linalg.solve(xtwx, xtwy)
            smoothed[idx] = beta[0]
        except np.linalg.LinAlgError:
            smoothed[idx] = np.average(local_y, weights=weights)

    return np.exp(smoothed) if use_log_domain else smoothed


def _format_log_tick(value: float, _: float) -> str:
    if not np.isfinite(value) or value <= 0:
        return ""
    if value >= 1e4 or value < 1e-2:
        return f"{value:.1e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    if value >= 10:
        return f"{value:.0f}" if np.isclose(value, round(value)) else f"{value:.1f}"
    if value >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _style_log_axis(axis: plt.Axes) -> None:
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=5))
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(_format_log_tick))
    axis.yaxis.set_minor_locator(mticker.NullLocator())
    axis.yaxis.set_minor_formatter(mticker.NullFormatter())
    axis.yaxis.get_offset_text().set_visible(False)


def plot_metrics(
    train_metrics: Dict[str, Sequence[float]],
    val_metrics: Dict[str, Sequence[float]],
    window: int = 10,
    alpha: float = 0.55,
) -> None:
    """Visualise train/validation metrics with LOESS-smoothed overlays."""
    metrics = [
        name
        for name in train_metrics.keys()
        if len(train_metrics.get(name, [])) > 0 and len(val_metrics.get(name, [])) > 0
    ]
    if not metrics:
        return

    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    default_colors = (
        color_cycle.by_key()["color"]
        if color_cycle is not None
        else ["blue", "red", "green", "purple", "orange"]
    )
    panel_specs = [
        ("Structural Losses", ["total", "eqm", "deg_ce", "exist"]),
        ("Semantic And Pairwise Losses", ["node_label_ce", "edge_label_ce", "edge_ce", "aux_locality"]),
    ]
    active_panels = [
        (title, [name for name in panel_metrics if name in metrics])
        for title, panel_metrics in panel_specs
    ]
    active_panels = [(title, panel_metrics) for title, panel_metrics in active_panels if panel_metrics]
    if not active_panels:
        active_panels = [("Metrics", metrics)]

    fig_height = 4.8 * len(active_panels) + 1.2
    fig, axes = plt.subplots(
        len(active_panels),
        1,
        figsize=(16, fig_height),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes[:, 0]
    color_by_metric = {
        metric_name: default_colors[idx % len(default_colors)]
        for idx, metric_name in enumerate(metrics)
    }
    lines, labels = [], []

    for ax, (panel_title, panel_metrics) in zip(flat_axes, active_panels):
        for metric_idx, name in enumerate(panel_metrics):
            metric_ax = ax if metric_idx == 0 else ax.twinx()
            if metric_idx > 0:
                metric_ax.spines["right"].set_position(("outward", 50 * (metric_idx - 1)))
                metric_ax.spines["right"].set_visible(True)

            color = color_by_metric[name]
            train_vals = train_metrics[name]
            val_vals = val_metrics[name]
            count = min(len(train_vals), len(val_vals))
            train = train_vals[:count]
            val = val_vals[:count]
            epochs = np.arange(1, count + 1)
            metric_ax.plot(epochs, train, color=color, alpha=alpha, linewidth=1.35)
            metric_ax.plot(epochs, val, color=color, linestyle="--", alpha=alpha, linewidth=1.35)
            sm_train = _loess_smooth(train, window)
            sm_val = _loess_smooth(val, window)
            line_train, = metric_ax.plot(
                epochs,
                sm_train,
                color=color,
                linewidth=2.2,
                label=f"Train {name} (LOESS)",
            )
            line_val, = metric_ax.plot(
                epochs,
                sm_val,
                color=color,
                linewidth=2.2,
                linestyle="--",
                label=f"Val {name} (LOESS)",
            )
            _style_log_axis(metric_ax)
            metric_ax.set_ylabel(name, color=color, rotation=90)
            metric_ax.tick_params(axis="y", colors=color)
            if metric_idx == 0:
                metric_ax.yaxis.set_label_position("left")
                metric_ax.yaxis.tick_left()
            else:
                metric_ax.yaxis.set_label_position("right")
                metric_ax.yaxis.tick_right()
            lines += [line_train, line_val]
            labels += [f"Train {name} (LOESS)", f"Val {name} (LOESS)"]

        ax.set_title(panel_title)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    flat_axes[-1].set_xlabel("Epoch")
    fig.legend(lines, labels, loc="upper center", ncol=max(min(len(lines), 6), 1), fontsize="small")
    fig.subplots_adjust(left=0.08, right=0.74, top=0.90, hspace=0.28)
    plt.show()
