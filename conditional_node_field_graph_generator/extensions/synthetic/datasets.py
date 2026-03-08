"""Synthetic dataset builders."""

import networkx as nx
import numpy as np
from toolz import curry

from .primitives import make_graph_generator


def _make_duplicate_detection_estimator():
    try:
        from AbstractGraph.hash_graph import GraphHashDeduper
    except ImportError as exc:
        raise ImportError(
            "Graph duplicate filtering requires optional dependency 'AbstractGraph'. "
            "Install it to use make_graphs_classification_dataset(), "
            "make_two_types_graphs_classification_dataset(), or ArtificialGraphDatasetConstructor.sample()."
        ) from exc
    return GraphHashDeduper()


class AttributeGenerator(object):
    def __init__(self, data_mtx, targets):
        self.target_classes = sorted(list(set(targets)))
        self.num_classes = len(self.target_classes)
        self.attributes = [data_mtx[[i for i, y in enumerate(targets) if y == t]] for t in self.target_classes]

    def transform(self, class_seq):
        attribute_list = []
        for y in class_seq:
            attributes = self.attributes[y]
            idx = np.random.randint(len(attributes))
            attribute_list.append(attributes[idx].flatten())
        return attribute_list


@curry
def make_graph(graph_generator, alphabet_size, attribute_generator):
    graph = graph_generator
    nx.set_edge_attributes(graph, "-", "label")

    if attribute_generator is not None:
        num_classes = attribute_generator.num_classes
    else:
        num_classes = alphabet_size

    labels = np.random.randint(num_classes, size=nx.number_of_nodes(graph))
    labels_dict = {node_idx: label for node_idx, label in enumerate(labels)}
    nx.set_node_attributes(graph, labels_dict, "true_label")

    labels_dict = {node_idx: label % alphabet_size for node_idx, label in enumerate(labels)}
    nx.set_node_attributes(graph, labels_dict, "label")

    if attribute_generator is not None:
        attributes = attribute_generator.transform(labels)
        attributes_dict = {node_idx: attribute for node_idx, attribute in enumerate(attributes)}
        nx.set_node_attributes(graph, attributes_dict, "vec")
    return graph.copy()


class ArtificialGraphConstructor(object):
    def __init__(self, graph_type="cycle", instance_size=4, alphabet_size=3, attribute_generator=None):
        self.graph_type = graph_type
        self.instance_size = instance_size
        self.alphabet_size = alphabet_size
        self.attribute_generator = attribute_generator
        self.graph_generator = make_graph_generator(graph_type, instance_size)

    def sample(self, n_samples=1):
        samples = [
            make_graph(self.graph_generator, self.alphabet_size, self.attribute_generator)
            for _ in range(n_samples)
        ]
        if n_samples == 1:
            return samples[0]
        return samples


def link_graphs(graph_source, graph_target, n_link_edges=0):
    n = nx.number_of_nodes(graph_source)
    graph_source_endpoints = np.random.randint(nx.number_of_nodes(graph_source), size=n_link_edges)
    graph_target_endpoints = np.random.randint(nx.number_of_nodes(graph_target), size=n_link_edges)
    graph = nx.disjoint_union(graph_source, graph_target)
    for u, v in graph_source.edges():
        graph.edges[u, v]["true_label"] = "source"
    for u, v in graph_target.edges():
        graph.edges[u + n, v + n]["true_label"] = "destination"
    for s, t in zip(graph_source_endpoints, graph_target_endpoints):
        graph.add_edge(s, t + n, label="-", true_label="joint")
    return graph


def make_graphs(
    graph_generator_target_type,
    graph_generator_context_type,
    target_size,
    context_size,
    alphabet_size,
    attribute_generator,
    n_link_edges,
    num_graphs,
    use_single_target=True,
):
    context_graphs = []
    for _ in range(num_graphs):
        graph_generator = make_graph_generator(graph_generator_context_type, context_size)
        context_graph = make_graph(graph_generator, alphabet_size, attribute_generator)
        context_graphs.append(context_graph.copy())

    if use_single_target:
        graph_generator = make_graph_generator(graph_generator_target_type, target_size)
        target_graph = make_graph(graph_generator, alphabet_size, attribute_generator)
        target_graphs = [target_graph.copy()] * num_graphs
    else:
        target_graphs = []
        for _ in range(num_graphs):
            graph_generator = make_graph_generator(graph_generator_target_type, target_size)
            target_graph = make_graph(graph_generator, alphabet_size, attribute_generator)
            target_graphs.append(target_graph.copy())

    graphs = [
        link_graphs(graph_source=target_graph, graph_target=context_graph, n_link_edges=n_link_edges)
        for target_graph, context_graph in zip(target_graphs, context_graphs)
    ]
    return graphs


def make_graphs_classification_dataset(
    graph_generator_target_type,
    graph_generator_context_type,
    target_size,
    context_size,
    alphabet_size,
    n_link_edges,
    num_graphs,
    attribute_generator=None,
):
    pos_graphs = make_graphs(
        graph_generator_target_type,
        graph_generator_context_type,
        target_size,
        context_size,
        alphabet_size,
        attribute_generator,
        n_link_edges,
        num_graphs,
        use_single_target=True,
    )
    neg_graphs = make_graphs(
        graph_generator_target_type,
        graph_generator_context_type,
        target_size,
        context_size,
        alphabet_size,
        attribute_generator,
        n_link_edges,
        num_graphs,
        use_single_target=False,
    )
    gdde = _make_duplicate_detection_estimator()
    pos_graphs = gdde.fit_filter(pos_graphs)
    neg_graphs = gdde.filter(neg_graphs)
    targets = np.array([1] * len(pos_graphs) + [0] * len(neg_graphs))
    graphs = pos_graphs + neg_graphs
    return graphs, targets, pos_graphs, neg_graphs


def make_two_types_graphs_classification_dataset(
    graph_generator_target_type_pos,
    graph_generator_context_type_pos,
    graph_generator_target_type_neg,
    graph_generator_context_type_neg,
    target_size,
    context_size,
    alphabet_size,
    n_link_edges,
    num_graphs,
    attribute_generator=None,
):
    pos_graphs = make_graphs(
        graph_generator_target_type_pos,
        graph_generator_context_type_pos,
        target_size,
        context_size,
        alphabet_size,
        attribute_generator,
        n_link_edges,
        num_graphs,
        use_single_target=True,
    )
    neg_graphs = make_graphs(
        graph_generator_target_type_neg,
        graph_generator_context_type_neg,
        target_size,
        context_size,
        alphabet_size,
        attribute_generator,
        n_link_edges,
        num_graphs,
        use_single_target=True,
    )
    gdde = _make_duplicate_detection_estimator()
    pos_graphs = gdde.fit_filter(pos_graphs)
    neg_graphs = gdde.filter(neg_graphs)
    targets = np.array([1] * len(pos_graphs) + [0] * len(neg_graphs))
    graphs = pos_graphs + neg_graphs
    return graphs, targets, pos_graphs, neg_graphs


class ArtificialGraphDatasetConstructor(object):
    def __init__(
        self,
        graph_generator_target_type_pos,
        graph_generator_context_type_pos,
        graph_generator_target_type_neg,
        graph_generator_context_type_neg,
        target_size_pos,
        context_size_pos,
        alphabet_size_pos,
        n_link_edges_pos,
        target_size_neg,
        context_size_neg,
        alphabet_size_neg,
        n_link_edges_neg,
        attribute_generator=None,
    ):
        self.graph_generator_target_type_pos = graph_generator_target_type_pos
        self.graph_generator_context_type_pos = graph_generator_context_type_pos
        self.graph_generator_target_type_neg = graph_generator_target_type_neg
        self.graph_generator_context_type_neg = graph_generator_context_type_neg
        self.target_size_pos = target_size_pos
        self.context_size_pos = context_size_pos
        self.alphabet_size_pos = alphabet_size_pos
        self.n_link_edges_pos = n_link_edges_pos
        self.target_size_neg = target_size_neg
        self.context_size_neg = context_size_neg
        self.alphabet_size_neg = alphabet_size_neg
        self.n_link_edges_neg = n_link_edges_neg
        self.attribute_generator = attribute_generator

    def get_graph_types(self):
        return ["path", "tree", "cycle", "degree", "regular", "dense"]

    def sample(self, n_samples, return_separate_classes=False):
        pos_graphs = make_graphs(
            self.graph_generator_target_type_pos,
            self.graph_generator_context_type_pos,
            self.target_size_pos,
            self.context_size_pos,
            self.alphabet_size_pos,
            self.attribute_generator,
            self.n_link_edges_pos,
            n_samples,
            use_single_target=True,
        )
        neg_graphs = make_graphs(
            self.graph_generator_target_type_neg,
            self.graph_generator_context_type_neg,
            self.target_size_neg,
            self.context_size_neg,
            self.alphabet_size_neg,
            self.attribute_generator,
            self.n_link_edges_neg,
            n_samples,
            use_single_target=True,
        )
        gdde = _make_duplicate_detection_estimator()
        pos_graphs = gdde.fit_filter(pos_graphs)
        neg_graphs = gdde.filter(neg_graphs)
        targets = np.array([1] * len(pos_graphs) + [0] * len(neg_graphs))
        graphs = pos_graphs + neg_graphs
        if return_separate_classes:
            return pos_graphs, neg_graphs
        return graphs, targets


__all__ = [
    "ArtificialGraphConstructor",
    "ArtificialGraphDatasetConstructor",
    "AttributeGenerator",
    "link_graphs",
    "make_graph",
    "make_graphs",
    "make_graphs_classification_dataset",
    "make_two_types_graphs_classification_dataset",
]
