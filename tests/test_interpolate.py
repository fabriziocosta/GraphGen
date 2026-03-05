import numpy as np
import pandas as pd
import pytest

from eqm_decompositional_graph_generator.interpolate import (
    _interpolate_integer_series,
    interpolate,
    sample_positive_endpoint_pair,
)
from eqm_decompositional_graph_generator.conditional_node_generator_base import GraphConditioningBatch


class _FakeGraphGenerator:
    def __init__(self):
        self.last_decode_args = None

    def graph_encode(self, graphs):
        graph = graphs[0]
        return GraphConditioningBatch(
            graph_embeddings=np.asarray([graph["embedding"]], dtype=float),
            node_counts=np.asarray([graph["node_count"]], dtype=np.int64),
            edge_counts=np.asarray([graph["edge_count"]], dtype=np.int64),
            node_label_histograms=np.asarray([graph["hist"]], dtype=float),
        )

    def _decode_with_feasibility_slots(self, conditioning, apply_feasibility_filtering=True):
        self.last_decode_args = (conditioning, apply_feasibility_filtering)
        out = []
        for idx in range(len(conditioning)):
            out.append(None if idx % 2 else {"decoded_index": idx})
        return out

    def interpolate(self, graph_a, graph_b, k=7, apply_feasibility_filtering=True):
        cond_a = self.graph_encode([graph_a])
        cond_b = self.graph_encode([graph_b])
        ts = np.linspace(0.0, 1.0, k + 2)[1:-1]

        interpolated_graph_embeddings = np.stack(
            [(1.0 - t) * cond_a.graph_embeddings[0] + t * cond_b.graph_embeddings[0] for t in ts],
            axis=0,
        )
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
        interpolated_histograms = np.stack(
            [(1.0 - t) * cond_a.node_label_histograms[0] + t * cond_b.node_label_histograms[0] for t in ts],
            axis=0,
        )
        interpolated_conditioning = GraphConditioningBatch(
            graph_embeddings=interpolated_graph_embeddings,
            node_counts=interpolated_node_counts,
            edge_counts=interpolated_edge_counts,
            node_label_histograms=interpolated_histograms,
        )
        decoded_slots = self._decode_with_feasibility_slots(
            interpolated_conditioning,
            apply_feasibility_filtering=apply_feasibility_filtering,
        )
        step_summary = pd.DataFrame(
            {
                "step": np.arange(1, len(ts) + 1),
                "t": np.round(ts, 3),
                "target_nodes": interpolated_node_counts,
                "target_edges": interpolated_edge_counts,
                "decoded": [graph is not None for graph in decoded_slots],
            }
        )
        return {
            "ts": ts,
            "conditioning": interpolated_conditioning,
            "decoded_slots": decoded_slots,
            "generated_graphs": [graph for graph in decoded_slots if graph is not None],
            "summary": step_summary,
        }


def test_interpolate_integer_series_rounds_and_respects_minimum():
    ts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    values = _interpolate_integer_series(2, 6, ts, minimum=3)
    assert values.dtype == np.int64
    np.testing.assert_array_equal(values, np.array([3, 3, 4, 5, 6], dtype=np.int64))


def test_sample_positive_endpoint_pair_returns_positive_graphs():
    np.random.seed(7)
    graphs = ["g0", "g1", "g2", "g3"]
    targets = np.array([0, 1, 1, 0])

    selected_indices, selected_targets, graph_a, graph_b = sample_positive_endpoint_pair(graphs, targets)

    assert len(selected_indices) == 2
    assert graph_a in ("g1", "g2")
    assert graph_b in ("g1", "g2")
    assert selected_targets == [1, 1]
    assert selected_indices[0] != selected_indices[1]


def test_sample_positive_endpoint_pair_raises_with_insufficient_positives():
    with pytest.raises(RuntimeError, match="Need at least two positive"):
        sample_positive_endpoint_pair(["a", "b"], np.array([0, 1]))


def test_interpolate_returns_conditioning_and_summary():
    graph_a = {
        "embedding": np.array([1.0, 0.0, 2.0]),
        "node_count": 3,
        "edge_count": 2,
        "hist": np.array([0.7, 0.3]),
    }
    graph_b = {
        "embedding": np.array([0.0, 2.0, 1.0]),
        "node_count": 5,
        "edge_count": 8,
        "hist": np.array([0.2, 0.8]),
    }
    gen = _FakeGraphGenerator()

    result = interpolate(gen, graph_a, graph_b, k=3, apply_feasibility_filtering=False)

    assert set(result.keys()) == {"ts", "conditioning", "decoded_slots", "generated_graphs", "summary"}
    assert isinstance(result["summary"], pd.DataFrame)
    assert len(result["ts"]) == 3
    assert len(result["conditioning"]) == 3
    np.testing.assert_array_equal(result["conditioning"].node_counts, np.array([4, 4, 4], dtype=np.int64))
    np.testing.assert_array_equal(result["conditioning"].edge_counts, np.array([4, 5, 6], dtype=np.int64))
    assert result["summary"]["decoded"].tolist() == [True, False, True]
    assert len(result["generated_graphs"]) == 2
    assert gen.last_decode_args[1] is False
