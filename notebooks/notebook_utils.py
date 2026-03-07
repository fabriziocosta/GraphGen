"""Utility helpers for lean notebook execution cells."""

from __future__ import annotations

import contextlib
import hashlib
import math
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):  # type: ignore
        print(obj)


def sample_positive_endpoint_pair(graphs, targets):
    positive_indices = np.flatnonzero(np.asarray(targets) != 0)
    if positive_indices.size < 2:
        raise RuntimeError("Need at least two positive training graphs for interpolation.")
    selected_indices = np.random.choice(positive_indices, size=2, replace=False)
    selected_targets = [targets[int(idx)] for idx in selected_indices]
    return (
        selected_indices.tolist(),
        selected_targets,
        graphs[int(selected_indices[0])],
        graphs[int(selected_indices[1])],
    )


def offset_neg_graphs(graphs, targets, offset=10):
    out_graphs = []
    for graph, target in zip(graphs, targets):
        if target == 0:
            for u in graph.nodes():
                graph.nodes[u]["label"] += offset
        out_graphs.append(graph.copy())
    return out_graphs, targets


def select_pos_neg(sampled_graphs, sampled_targets, n_lines=3, n_graphs_per_line=12):
    cap = n_graphs_per_line * n_lines
    pos_graphs = [g for g, t in zip(sampled_graphs, sampled_targets) if t == 1][:cap]
    neg_graphs = [g for g, t in zip(sampled_graphs, sampled_targets) if t != 1][:cap]
    return pos_graphs, neg_graphs


def plot_sample(
    sampled_graphs,
    sampled_targets,
    haystack=None,
    n_lines=3,
    n_graphs_per_line=12,
    compute_is_valid_fn=None,
):
    from coco_grape.visualizer.display import draw_graphs

    pos_graphs, neg_graphs = select_pos_neg(
        sampled_graphs,
        sampled_targets,
        n_lines=n_lines,
        n_graphs_per_line=n_graphs_per_line,
    )
    gs = pos_graphs + neg_graphs
    if haystack is not None:
        if compute_is_valid_fn is None:
            raise ValueError("compute_is_valid_fn is required when haystack is provided.")
        ts = list(compute_is_valid_fn(pos_graphs, haystack)) + list(compute_is_valid_fn(neg_graphs, haystack))
        draw_graphs(gs, ts, n_graphs_per_line=n_graphs_per_line)
    else:
        draw_graphs(gs, n_graphs_per_line=n_graphs_per_line)


def infer_display_mode(graphs):
    if not graphs:
        return "not_molecule"

    # Common atom symbols used by molecular datasets.
    atom_symbols = {
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Br", "I",
    }

    for graph in graphs[:20]:
        graph_meta = getattr(graph, "graph", {})
        if any(key in graph_meta for key in ("smiles", "mol", "molecule", "inchi")):
            return "molecule"
        for _, attrs in graph.nodes(data=True):
            if "symbol" in attrs or "atomic_num" in attrs or "atom" in attrs:
                return "molecule"
            label = attrs.get("label")
            if isinstance(label, str) and label in atom_symbols:
                return "molecule"
    return "not_molecule"


def plot_networkx_graphs(
    graphs,
    cmap="tab20",
    light=0.4,
    size=4,
    n_cols=None,
    show_label=True,
    color_offset=200,
    mode="not_molecule",
    titles=None,
):
    if mode == "molecule":
        from coco_grape.visualizer.mol_display import draw_molecules

        draw_molecules(graphs)
        return

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    n_graphs = len(graphs)
    if n_graphs == 0:
        print("No graphs to display.")
        return
    if titles is not None and len(titles) != n_graphs:
        raise ValueError(f"titles must align with graphs (got {len(titles)} titles for {n_graphs} graphs).")
    if n_cols is None:
        n_cols = n_graphs
    n_rows = math.ceil(n_graphs / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(size * n_cols, size * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = list(axes.flatten())
    else:
        axes = axes.flatten()

    def get_color_for_label(label):
        hash_val = hashlib.md5(str(label).encode("utf-8")).hexdigest()
        numeric_hash = int(hash_val, 16) + color_offset
        normalized = (numeric_hash % 1000) / 999.0
        base_color = cmap(normalized)
        lightened = tuple((1 - light) * base_color[i] + light for i in range(3))
        if len(base_color) == 4:
            lightened += (base_color[3],)
        return lightened

    for i, graph in enumerate(graphs):
        ax = axes[i]
        ax.axis("off")
        if titles is not None:
            ax.set_title(str(titles[i]))
        pos = nx.kamada_kawai_layout(graph)
        node_colors = []
        labels = {}
        for node in graph.nodes():
            label = graph.nodes[node].get("label", "")
            node_colors.append(get_color_for_label(label))
            labels[node] = str(label)
        nx.draw_networkx_edges(graph, pos, width=2, ax=ax)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, edgecolors="black", linewidths=2)
        if show_label:
            nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()


@contextlib.contextmanager
def _temporary_decoder_n_jobs(graph_generator, decoder_n_jobs):
    decoder = getattr(graph_generator, "graph_decoder", None)
    if decoder is None or decoder_n_jobs is None:
        yield
        return

    original_n_jobs = getattr(decoder, "n_jobs", 1)
    if int(decoder_n_jobs) < 0:
        cpu_count = os.cpu_count() or 1
        resolved_n_jobs = max(1, cpu_count + 1 + int(decoder_n_jobs))
    else:
        resolved_n_jobs = max(1, int(decoder_n_jobs))

    decoder.n_jobs = resolved_n_jobs
    try:
        yield
    finally:
        decoder.n_jobs = original_n_jobs


def show_samples(n_samples, graphs, targets, graph_generator, mode=None, decoder_n_jobs=-1):
    display_mode = mode or infer_display_mode(graphs)

    def _show_group(title, seed_graphs):
        print(f"{title} Graphs:")
        plot_networkx_graphs(seed_graphs[:n_samples], n_cols=n_samples, mode=display_mode)

        with _temporary_decoder_n_jobs(graph_generator, decoder_n_jobs):
            unfiltered_graphs = graph_generator.sample_conditioned_on_random(
                seed_graphs,
                n_samples,
                apply_feasibility_filtering=False,
            )
        print(f"Sampled {title} Graphs Without Feasibility Filtering ({len(unfiltered_graphs)}/{n_samples}):")
        plot_networkx_graphs(unfiltered_graphs, n_cols=max(1, len(unfiltered_graphs)), mode=display_mode)

        with _temporary_decoder_n_jobs(graph_generator, decoder_n_jobs):
            filtered_graphs = graph_generator.sample_conditioned_on_random(
                seed_graphs,
                n_samples,
                apply_feasibility_filtering=True,
            )
        print(f"Sampled {title} Graphs With Feasibility Filtering ({len(filtered_graphs)}/{n_samples}):")
        plot_networkx_graphs(filtered_graphs, n_cols=max(1, len(filtered_graphs)), mode=display_mode)

    neg_graphs = [graph for graph, target in zip(graphs, targets) if target == 0]
    pos_graphs = [graph for graph, target in zip(graphs, targets) if target != 0]
    _show_group("Positive", pos_graphs)
    _show_group("Negative", neg_graphs)


def graph_label_histogram(graph, label_classes):
    hist = np.zeros(len(label_classes), dtype=float)
    class_to_idx = {label: idx for idx, label in enumerate(label_classes)}
    for _, data in graph.nodes(data=True):
        label = data.get("label")
        if label in class_to_idx:
            hist[class_to_idx[label]] += 1.0
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def summarize_label_histogram_alignment(graph_generator, graphs, n_compare=16):
    equilibrium_matching_generator = graph_generator.conditional_node_generator_model
    label_classes = [int(label) if isinstance(label, np.integer) else label for label in equilibrium_matching_generator.node_label_classes_]

    input_graphs = graphs[:n_compare]
    graph_conditioning = graph_generator.graph_encode(input_graphs)
    conditioning_histograms = np.vstack([graph_label_histogram(graph, label_classes) for graph in input_graphs])

    if getattr(graph_generator, "feasibility_estimator", None) is not None:
        decoded_slots = graph_generator._decode_with_feasibility_slots(graph_conditioning)
        matched_pairs = [
            (conditioning_histograms[idx], graph)
            for idx, graph in enumerate(decoded_slots)
            if graph is not None
        ]
        if not matched_pairs:
            raise RuntimeError("No feasible generated graphs were available for histogram comparison.")
        conditioning_histograms = np.vstack([conditioning_hist for conditioning_hist, _ in matched_pairs])
        generated_graphs = [graph for _, graph in matched_pairs]
    else:
        generated_graphs = graph_generator.decode(graph_conditioning)

    generated_histograms = np.vstack([graph_label_histogram(graph, label_classes) for graph in generated_graphs])
    histogram_diff = generated_histograms - conditioning_histograms
    abs_diff = np.abs(histogram_diff)

    summary = {
        "label_classes": label_classes,
        "mean_conditioning": conditioning_histograms.mean(axis=0),
        "mean_generated": generated_histograms.mean(axis=0),
        "mean_diff": histogram_diff.mean(axis=0),
        "mean_abs_diff": abs_diff.mean(axis=0),
        "average_l1_error": abs_diff.sum(axis=1).mean(),
    }
    summary["mistake_order"] = np.argsort(-summary["mean_abs_diff"])
    return summary


def plot_label_histogram_alignment(summary):
    label_classes = summary["label_classes"]
    mean_conditioning = summary["mean_conditioning"]
    mean_generated = summary["mean_generated"]
    mean_abs_diff = summary["mean_abs_diff"]
    mean_diff = summary["mean_diff"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    x = np.arange(len(label_classes))
    width = 0.38

    axes[0].bar(x - width / 2, mean_conditioning, width, label="conditioning")
    axes[0].bar(x + width / 2, mean_generated, width, label="generated")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(label_classes, rotation=45, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Average label histogram")
    axes[0].legend()

    axes[1].bar(x, mean_abs_diff, color="tomato")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(label_classes, rotation=45, ha="right")
    axes[1].set_ylim(0, max(0.05, mean_abs_diff.max() * 1.2))
    axes[1].set_title("Average absolute histogram difference")

    axes[2].bar(x, mean_diff, color=["seagreen" if v >= 0 else "firebrick" for v in mean_diff])
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(label_classes, rotation=45, ha="right")
    axes[2].set_title("Signed histogram bias")

    plt.tight_layout()
    plt.show()


def run_label_histogram_analysis(graph_generator, graphs, n_compare=5):
    summary = summarize_label_histogram_alignment(graph_generator, graphs, n_compare=n_compare)
    plot_label_histogram_alignment(summary)
    return summary


def fit_graph_generator(graph_generator, train_graphs):
    graph_generator.fit(train_graphs)
    return graph_generator


def _normalized_counter(values):
    series = pd.Series(list(values), dtype=object)
    if len(series) == 0:
        return pd.Series(dtype=float)
    counts = series.value_counts(dropna=False).sort_index(key=lambda idx: idx.map(str))
    return counts / counts.sum()


def _collect_graph_statistics(graphs):
    node_counts = [graph.number_of_nodes() for graph in graphs]
    edge_counts = [graph.number_of_edges() for graph in graphs]
    node_labels = [attrs.get("label") for graph in graphs for _, attrs in graph.nodes(data=True)]
    edge_labels = [attrs.get("label") for graph in graphs for _, _, attrs in graph.edges(data=True)]
    return {
        "node_count": _normalized_counter(node_counts),
        "edge_count": _normalized_counter(edge_counts),
        "atom_label": _normalized_counter(node_labels),
        "bond_label": _normalized_counter(edge_labels),
    }


def _compare_distribution(real_dist, generated_dist, metric_name):
    support = real_dist.index.union(generated_dist.index)
    real_aligned = real_dist.reindex(support, fill_value=0.0)
    generated_aligned = generated_dist.reindex(support, fill_value=0.0)
    comparison = pd.DataFrame({"real": real_aligned, "generated": generated_aligned})
    comparison["abs_diff"] = (comparison["generated"] - comparison["real"]).abs()
    tv_distance = 0.5 * comparison["abs_diff"].sum()
    print(f"{metric_name}: TV distance = {tv_distance:.4f}")
    return comparison.sort_index(key=lambda idx: idx.map(str)), tv_distance


def compare_real_vs_generated(graph_generator, reference_graphs, apply_feasibility_filtering=True):
    reference_graphs = list(reference_graphs)
    reference_conditioning = graph_generator.graph_encode(reference_graphs)
    decoded_slots = graph_generator._decode_with_feasibility_slots(
        reference_conditioning,
        apply_feasibility_filtering=apply_feasibility_filtering,
    )
    paired_graphs = [
        (real_graph, generated_graph)
        for real_graph, generated_graph in zip(reference_graphs, decoded_slots)
        if generated_graph is not None
    ]
    if not paired_graphs:
        raise RuntimeError("No feasible generated graphs were returned for the comparison set.")
    real_eval_graphs = [real_graph for real_graph, _ in paired_graphs]
    generated_eval_graphs = [generated_graph for _, generated_graph in paired_graphs]
    print(f"Compared {len(generated_eval_graphs)}/{len(reference_graphs)} feasible generated graphs.")
    real_stats = _collect_graph_statistics(real_eval_graphs)
    generated_stats = _collect_graph_statistics(generated_eval_graphs)

    comparison_tables = {}
    summary_rows = []
    metrics = [
        ("node_count", "Node count"),
        ("edge_count", "Edge count"),
        ("atom_label", "Atom label"),
        ("bond_label", "Bond label"),
    ]
    for metric_name, title in metrics:
        comparison, tv_distance = _compare_distribution(real_stats[metric_name], generated_stats[metric_name], title)
        comparison_tables[metric_name] = comparison
        summary_rows.append({"metric": title, "tv_distance": tv_distance})

    distribution_summary = pd.DataFrame(summary_rows).sort_values("tv_distance")
    display(distribution_summary)
    for metric_name, title in metrics:
        print(title)
        display(comparison_tables[metric_name])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (metric_name, title) in zip(axes.flatten(), metrics):
        table = comparison_tables[metric_name]
        x = np.arange(len(table))
        width = 0.38
        ax.bar(x - width / 2, table["real"], width, label="real")
        ax.bar(x + width / 2, table["generated"], width, label="generated")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(idx) for idx in table.index], rotation=45, ha="right")
        ax.legend()
    plt.tight_layout()
    plt.show()

    return {
        "summary": distribution_summary,
        "comparison_tables": comparison_tables,
        "real_graphs": real_eval_graphs,
        "generated_graphs": generated_eval_graphs,
    }


def _median_iqr(values):
    """Return (q1, median, q3) for a numeric sequence.

    Args:
        values (Any): Input value.

    Returns:
        Any: Computed result.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return q1, med, q3


def plot_similarity_distribution_with_iqr(sim_high, sim_low, target_high, target_low):
    """Plot median + IQR whiskers and print concise summary stats.

    Args:
        sim_high (Any): Input value.
        sim_low (Any): Input value.
        target_high (Any): Input value.
        target_low (Any): Input value.

    Returns:
        Any: Computed result.
    """
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
