import numpy as np
import networkx as nx
import pytest
import pulp

from eqm_decompositional_graph_generator.graph_engine import (
    DEFAULT_DUMMY_NODE_LABEL,
    EqMDecompositionalGraphDecoder,
    EqMDecompositionalGraphGenerator,
)
from eqm_decompositional_graph_generator.node_engine import GeneratedNodeBatch


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


def _unlabeled_node_graph():
    graph = nx.Graph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, label="-")
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


def test_graphs_to_node_label_targets_uses_dummy_label_when_all_nodes_are_unlabelled():
    generator = EqMDecompositionalGraphGenerator(verbose=False)

    node_targets = generator.graphs_to_node_label_targets([_unlabeled_node_graph()])

    assert len(node_targets) == 1
    assert node_targets[0].tolist() == [DEFAULT_DUMMY_NODE_LABEL, DEFAULT_DUMMY_NODE_LABEL]


def test_graphs_to_node_label_targets_rejects_mixed_labelled_and_unlabelled_nodes():
    generator = EqMDecompositionalGraphGenerator(verbose=False)
    graph = _labeled_graph()
    del graph.nodes[1]["label"]

    with pytest.raises(ValueError, match="either present for every node"):
        generator.graphs_to_node_label_targets([graph])


def test_build_supervision_plan_uses_dummy_label_as_constant_when_nodes_are_unlabelled():
    generator = EqMDecompositionalGraphGenerator(verbose=False)
    node_label_targets = generator.graphs_to_node_label_targets([_unlabeled_node_graph()])

    plan = generator._build_supervision_plan(
        graphs=[],
        node_label_targets=node_label_targets,
        edge_label_targets=np.asarray(["-"], dtype=object),
    )

    assert plan.node_labels.mode == "constant"
    assert plan.node_labels.constant_value == DEFAULT_DUMMY_NODE_LABEL


def test_decode_node_labels_assigns_dummy_constant_label_for_unlabelled_training_setup():
    decoder = EqMDecompositionalGraphDecoder(verbose=False)
    decoder.supervision_plan_ = type(
        "_Plan",
        (),
        {
            "node_labels": type(
                "_Channel",
                (),
                {"mode": "constant", "constant_value": DEFAULT_DUMMY_NODE_LABEL},
            )()
        },
    )()

    labels = decoder.decode_node_labels(
        GeneratedNodeBatch(node_embeddings_list=[np.zeros((2, 3), dtype=float)])
    )

    assert labels[0].tolist() == [DEFAULT_DUMMY_NODE_LABEL, DEFAULT_DUMMY_NODE_LABEL]


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


def test_optimize_adjacency_matrix_raises_when_solver_status_is_not_optimal(monkeypatch):
    decoder = EqMDecompositionalGraphDecoder(verbose=False)

    def _fake_solve(self, solver):
        del solver
        self.status = pulp.LpStatusInfeasible
        return self.status

    monkeypatch.setattr(pulp.LpProblem, "solve", _fake_solve)

    with pytest.raises(RuntimeError, match="did not produce an optimal solution"):
        decoder.optimize_adjacency_matrix(
            prob_matrix=np.array([[0.0, 0.9], [0.9, 0.0]], dtype=float),
            target_degrees=[1, 1],
        )


def test_optimize_adjacency_matrix_raises_when_variable_value_is_missing(monkeypatch):
    decoder = EqMDecompositionalGraphDecoder(verbose=False)

    def _fake_solve(self, solver):
        del solver
        self.status = pulp.LpStatusOptimal
        return self.status

    monkeypatch.setattr(pulp.LpProblem, "solve", _fake_solve)

    with pytest.raises(RuntimeError, match="without assigning all decision variables"):
        decoder.optimize_adjacency_matrix(
            prob_matrix=np.array([[0.0, 0.9], [0.9, 0.0]], dtype=float),
            target_degrees=[1, 1],
        )
