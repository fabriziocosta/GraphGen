"""Utility helpers for lean notebook execution cells."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _median_iqr(values):
    """Return (q1, median, q3) for a numeric sequence."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return q1, med, q3


def plot_similarity_distribution_with_iqr(sim_high, sim_low, target_high, target_low):
    """Plot median + IQR whiskers and print concise summary stats."""
    q1_high, med_high, q3_high = _median_iqr(sim_high)
    q1_low, med_low, q3_low = _median_iqr(sim_low)

    labels = [f"desired_target={target_high}", f"desired_target={target_low}"]
    medians = np.array([med_high, med_low], dtype=float)
    lower = medians - np.array([q1_high, q1_low], dtype=float)
    upper = np.array([q3_high, q3_low], dtype=float) - medians

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, medians, color=["#4C78A8", "#F58518"], alpha=0.85)
    ax.errorbar(
        x,
        medians,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="black",
        capsize=8,
        linewidth=1.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cosine Similarity To Hidden Target")
    ax.set_title("Generated Similarity Distributions (Median With IQR Whiskers)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()

    print(
        f"{labels[0]} -> n={len(sim_high)}, median={med_high:.4f}, "
        f"q1={q1_high:.4f}, q3={q3_high:.4f}"
    )
    print(
        f"{labels[1]} -> n={len(sim_low)}, median={med_low:.4f}, "
        f"q1={q1_low:.4f}, q3={q3_low:.4f}"
    )
    if len(sim_high) and len(sim_low):
        print(f"median(high) - median(low) = {med_high - med_low:.4f}")

    return {
        "high": {"n": len(sim_high), "q1": q1_high, "median": med_high, "q3": q3_high},
        "low": {"n": len(sim_low), "q1": q1_low, "median": med_low, "q3": q3_low},
        "median_delta": med_high - med_low if len(sim_high) and len(sim_low) else np.nan,
    }
