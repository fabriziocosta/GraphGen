"""Primitive synthetic graph samplers."""

import random

import networkx as nx
import numpy as np
from toolz import curry


def _safe_random_tree(n: int) -> nx.Graph:
    """Return a random tree on n nodes."""
    if hasattr(nx.generators.trees, "random_tree"):
        return nx.generators.trees.random_tree(n)

    if n <= 1:
        g = nx.Graph()
        g.add_nodes_from(range(n))
        return g
    prufer = np.random.randint(0, n, size=n - 2)
    g = nx.Graph()
    degree = np.ones(n, dtype=int)
    degree += np.bincount(prufer, minlength=n)
    for node in prufer:
        leaf = np.where(degree == 1)[0][0]
        g.add_edge(leaf, node)
        degree[leaf] -= 1
        degree[node] -= 1
    u, v = np.where(degree == 1)[0]
    g.add_edge(u, v)
    return g


def _ensure_connected_graph(g: nx.Graph) -> nx.Graph:
    """Connect graph components by adding bridge edges between them."""
    if g.number_of_nodes() <= 1:
        return g
    if nx.is_connected(g):
        return g
    components = [list(component) for component in nx.connected_components(g)]
    for idx in range(len(components) - 1):
        left = components[idx]
        right = components[idx + 1]
        u = random.choice(left)
        v = random.choice(right)
        g.add_edge(u, v)
    return g


class RandomGraphConstructor(object):
    def __init__(self, integers_range=18, instance_size=4, alphabet_size=3):
        self.integers_range = integers_range
        self.instance_size = instance_size
        self.alphabet_size = alphabet_size

    def __sample_single(self):
        i_idxs = np.random.randint(self.integers_range, size=self.instance_size)
        j_idxs = np.random.randint(self.integers_range, size=self.instance_size)
        edges = []
        for i, j in zip(i_idxs, j_idxs):
            if i < j:
                edges.append((i, j))
        g = nx.Graph()
        for edge in edges:
            g.add_edge(edge[0], edge[1])
        g = nx.convert_node_labels_to_integers(g)
        labels = np.random.randint(self.alphabet_size, size=nx.number_of_nodes(g))
        labels = {i: label for i, label in enumerate(labels)}
        nx.set_node_attributes(g, labels, "label")
        nx.set_edge_attributes(g, "-", "label")
        return g

    def sample(self, n_samples=1):
        samples = [self.__sample_single() for _ in range(n_samples)]
        if n_samples == 1:
            return samples[0]
        return samples


@curry
def random_path_graph(n):
    g = nx.path_graph(n)
    g = nx.convert_node_labels_to_integers(g)
    return g


@curry
def random_tree_graph(n):
    g = _safe_random_tree(n)
    g = nx.convert_node_labels_to_integers(g)
    return g


@curry
def random_cycle_graph(n):
    g = _safe_random_tree(n)
    terminals = [u for u in g.nodes() if g.degree(u) == 1]
    random.shuffle(terminals)
    for i in range(0, len(terminals), 2):
        e_start = terminals[i]
        if i + 1 < len(terminals):
            e_end = terminals[i + 1]
            g.add_edge(e_start, e_end)
    g = nx.convert_node_labels_to_integers(g)
    return g


@curry
def random_regular_graph(d, n):
    if n <= 0:
        raise ValueError(f"n must be positive (got {n}).")
    if d < 0 or d >= n:
        raise ValueError(f"d must satisfy 0 <= d < n (got d={d}, n={n}).")
    if (n * d) % 2 != 0:
        raise ValueError(f"n*d must be even for a d-regular graph (got d={d}, n={n}).")
    g = nx.random_regular_graph(d, n)
    g = nx.convert_node_labels_to_integers(g)
    return g


@curry
def random_degree_seq(n, dmax):
    sequence = np.linspace(1, dmax, n).astype(int)
    g = nx.expected_degree_graph(sequence)
    g = nx.convert_node_labels_to_integers(g)
    return g


@curry
def random_dense_graph(n, m):
    g = nx.dense_gnm_random_graph(n, m)
    max_cc = max(nx.connected_components(g), key=lambda x: len(x))
    g = nx.subgraph(g, max_cc)
    g = nx.convert_node_labels_to_integers(g)
    return g


def make_graph_generator(graph_type, instance_size):
    graph_generator = None
    if graph_type == "path":
        graph_generator = random_path_graph(n=instance_size)
    if graph_type == "tree":
        graph_generator = random_tree_graph(n=instance_size)
    if graph_type == "cycle":
        graph_generator = random_cycle_graph(n=instance_size)
    if graph_type == "degree":
        n = instance_size
        dmax = 4
        max_attempts = 128
        last_candidate = None
        for _ in range(max_attempts):
            candidate = random_degree_seq(n, dmax)
            last_candidate = candidate
            is_connected = False
            try:
                is_connected = nx.is_connected(candidate)
            except nx.NetworkXPointlessConcept:
                is_connected = False
            if is_connected:
                graph_generator = candidate
                break
        if graph_generator is None:
            graph_generator = _ensure_connected_graph(last_candidate if last_candidate is not None else nx.Graph())
    if graph_type == "regular":
        if instance_size == 1:
            graph_generator = random_regular_graph(d=0, n=instance_size)
        else:
            default_degree = 3
            regular_degree = default_degree if (instance_size * default_degree) % 2 == 0 else 2
            graph_generator = random_regular_graph(d=regular_degree, n=instance_size)
    if graph_type == "dense":
        graph_generator = random_dense_graph(n=instance_size, m=instance_size + instance_size // 2)

    if graph_generator is None:
        raise ValueError(f"Unknown graph generator type: {graph_type}")
    return graph_generator


__all__ = [
    "RandomGraphConstructor",
    "make_graph_generator",
    "random_cycle_graph",
    "random_degree_seq",
    "random_dense_graph",
    "random_path_graph",
    "random_regular_graph",
    "random_tree_graph",
]
