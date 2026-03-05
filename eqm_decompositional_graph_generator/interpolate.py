import numpy as np
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
    """Compatibility wrapper. Prefer graph_generator.interpolate(...)."""
    return graph_generator.interpolate(
        graph_a,
        graph_b,
        k=k,
        apply_feasibility_filtering=apply_feasibility_filtering,
    )
