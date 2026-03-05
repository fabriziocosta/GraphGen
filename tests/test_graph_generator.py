import numpy as np
import networkx as nx
import pytest

from eqm_decompositional_graph_generator.decompositional_encoder_decoder import (
    EqMDecompositionalGraphGenerator,
)


class _GraphVectorizer:
    def __init__(self):
        self.fitted_graph_count = None

    def fit(self, graphs):
        self.fitted_graph_count = len(graphs)
        return self

    def transform(self, graphs):
        rows = []
        for graph in graphs:
            rows.append([graph.number_of_nodes(), graph.number_of_edges()])
        return np.asarray(rows, dtype=float)


class _NodeVectorizer:
    def fit(self, graphs):
        return self

    def transform(self, graphs):
        output = []
        for graph in graphs:
            emb = np.asarray(
                [[float(graph.degree(node)), float(node)] for node in graph.nodes()],
                dtype=float,
            )
            output.append(emb)
        return output


class _Component:
    def __init__(self, verbose=False):
        self.verbose = verbose


def _labeled_graph():
    graph = nx.Graph()
    graph.add_node(0, label="C")
    graph.add_node(1, label="O")
    graph.add_edge(0, 1, label="-")
    return graph


def _unlabeled_edge_graph():
    graph = nx.Graph()
    graph.add_node(0, label="C")
    graph.add_node(1, label="N")
    graph.add_edge(0, 1)
    return graph


def test_graph_generator_init_validates_inputs():
    with pytest.raises(ValueError, match="locality_sample_fraction"):
        EqMDecompositionalGraphGenerator(locality_sample_fraction=0.0)
    with pytest.raises(ValueError, match="locality_horizon must be >= 1"):
        EqMDecompositionalGraphGenerator(locality_horizon=0)
    with pytest.raises(ValueError, match="locality_sampling_strategy"):
        EqMDecompositionalGraphGenerator(locality_sampling_strategy="bad")
    with pytest.raises(ValueError, match="locality_target_positive_ratio"):
        EqMDecompositionalGraphGenerator(locality_target_positive_ratio=1.1)
    with pytest.raises(ValueError, match="max_feasibility_attempts"):
        EqMDecompositionalGraphGenerator(max_feasibility_attempts=0)
    with pytest.raises(ValueError, match="feasibility_candidates_per_attempt"):
        EqMDecompositionalGraphGenerator(feasibility_candidates_per_attempt=0)
    with pytest.raises(ValueError, match="feasibility_failure_mode"):
        EqMDecompositionalGraphGenerator(feasibility_failure_mode="drop")


def test_toggle_verbose_updates_nested_components():
    node_model = _Component(verbose=False)
    decoder = _Component(verbose=False)
    generator = EqMDecompositionalGraphGenerator(
        conditional_node_generator_model=node_model,
        graph_decoder=decoder,
        verbose=False,
    )

    generator.toggle_verbose()

    assert generator.verbose is True
    assert node_model.verbose is True
    assert decoder.verbose is True


def test_build_supervision_plan_modes_depend_on_labels_and_horizon():
    generator = EqMDecompositionalGraphGenerator(locality_horizon=2, verbose=False)
    node_label_targets = [np.asarray(["C", "C"], dtype=object), np.asarray(["C"], dtype=object)]
    edge_label_targets = np.asarray(["-"], dtype=object)

    plan = generator._build_supervision_plan(
        graphs=[],
        node_label_targets=node_label_targets,
        edge_label_targets=edge_label_targets,
    )

    assert plan.node_labels.mode == "constant"
    assert plan.node_labels.constant_value == "C"
    assert plan.edge_labels.mode == "constant"
    assert plan.direct_edges.enabled is True
    assert plan.auxiliary_locality.enabled is True
    assert plan.auxiliary_locality.horizon == 2


def test_graphs_to_edge_label_targets_disables_channel_if_any_edge_missing_label():
    generator = EqMDecompositionalGraphGenerator(verbose=False)
    graph = _unlabeled_edge_graph()

    edge_targets, edge_pairs = generator.graphs_to_edge_label_targets([graph])

    assert edge_targets is None
    assert edge_pairs is None


def test_graphs_to_edge_label_targets_returns_ordered_pairs_for_labeled_edges():
    generator = EqMDecompositionalGraphGenerator(verbose=False)
    graph = _labeled_graph()

    edge_targets, edge_pairs = generator.graphs_to_edge_label_targets([graph])

    assert edge_targets is not None
    assert edge_pairs is not None
    # Undirected edge appears twice because pairs are enumerated over ordered (i, j).
    assert edge_pairs == [(0, 0, 1), (0, 1, 0)]
    assert edge_targets.tolist() == ["-", "-"]


def test_encode_paths_return_expected_shapes_and_counts():
    g1 = _labeled_graph()
    g2 = _labeled_graph()
    g2.add_node(2, label="N")
    g2.add_edge(1, 2, label="=")
    graphs = [g1, g2]

    generator = EqMDecompositionalGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        verbose=False,
    )

    node_embeddings = generator.node_encode(graphs)
    conditioning = generator.graph_encode(graphs)
    node_embeddings_2, conditioning_2 = generator.encode(graphs)

    assert len(node_embeddings) == 2
    assert node_embeddings[0].shape == (2, 2)
    assert node_embeddings[1].shape == (3, 2)
    np.testing.assert_array_equal(conditioning.node_counts, np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(conditioning.edge_counts, np.array([1, 2], dtype=np.int64))
    assert conditioning.graph_embeddings.shape == (2, 2)
    assert len(node_embeddings_2) == 2
    np.testing.assert_array_equal(conditioning_2.graph_embeddings, conditioning.graph_embeddings)


def test_build_node_batch_masks_presence_and_degrees():
    graph = _labeled_graph()
    node_embeddings = [np.zeros((2, 3), dtype=float)]
    generator = EqMDecompositionalGraphGenerator(verbose=False)

    batch = generator._build_node_batch(
        graphs=[graph],
        node_embeddings_list=node_embeddings,
        node_label_targets=[np.asarray(["C", "O"], dtype=object)],
    )

    assert batch.node_presence_mask.shape == (1, 2)
    assert batch.node_presence_mask.tolist() == [[True, True]]
    assert batch.node_degree_targets.tolist() == [[1, 1]]
