import numpy as np
import pandas as pd

from eqm_decompositional_graph_generator.conditional_node_generator_base import GraphConditioningBatch


def _interpolate_integer_series(start, end, ts, minimum):
    values = np.rint([(1.0 - t) * start + t * end for t in ts]).astype(np.int64)
    return np.maximum(values, np.int64(minimum))


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


def interpolate(graph_generator, graph_a, graph_b, k=7, apply_feasibility_filtering=True):
    cond_a = graph_generator.graph_encode([graph_a])
    cond_b = graph_generator.graph_encode([graph_b])
    ts = np.linspace(0.0, 1.0, k + 2)[1:-1]

    interpolated_graph_embeddings = np.stack([
        (1.0 - t) * cond_a.graph_embeddings[0] + t * cond_b.graph_embeddings[0]
        for t in ts
    ], axis=0)
    interpolated_node_counts = _interpolate_integer_series(
        cond_a.node_counts[0],
        cond_b.node_counts[0],
        ts,
        minimum=1,
    )
    interpolated_edge_counts = _interpolate_integer_series(
        cond_a.edge_counts[0],
        cond_b.edge_counts[0],
        ts,
        minimum=0,
    )

    interpolated_histograms = None
    if cond_a.node_label_histograms is not None and cond_b.node_label_histograms is not None:
        interpolated_histograms = np.stack([
            (1.0 - t) * cond_a.node_label_histograms[0] + t * cond_b.node_label_histograms[0]
            for t in ts
        ], axis=0)

    interpolated_conditioning = GraphConditioningBatch(
        graph_embeddings=interpolated_graph_embeddings,
        node_counts=interpolated_node_counts,
        edge_counts=interpolated_edge_counts,
        node_label_histograms=interpolated_histograms,
    )
    decoded_slots = graph_generator._decode_with_feasibility_slots(
        interpolated_conditioning,
        apply_feasibility_filtering=apply_feasibility_filtering,
    )
    step_summary = pd.DataFrame({
        "step": np.arange(1, len(ts) + 1),
        "t": np.round(ts, 3),
        "target_nodes": interpolated_node_counts,
        "target_edges": interpolated_edge_counts,
        "decoded": [graph is not None for graph in decoded_slots],
    })
    return {
        "ts": ts,
        "conditioning": interpolated_conditioning,
        "decoded_slots": decoded_slots,
        "generated_graphs": [graph for graph in decoded_slots if graph is not None],
        "summary": step_summary,
    }
