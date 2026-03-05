"""Runtime and data utilities (merged module)."""

import time
from functools import wraps


def _verbosity_level(instance) -> int:
    """Interpret legacy boolean verbosity as levels, defaulting to 0.

    Args:
        instance (Any): Input value.

    Returns:
        int: Computed result.
    """
    if instance is None or not hasattr(instance, "verbose"):
        return 0
    value = getattr(instance, "verbose")
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1 if value else 0

def timeit(func):
    """A decorator to time member functions, printing the function's name,.

    Args:
        func (Any): Input value.

    Returns:
        Any: Computed result.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        elapsed_time = end_time - start_time
        elapsed_minutes = elapsed_time / 60
        elapsed_hours = elapsed_minutes / 60

        instance = args[0] if args else None
        if _verbosity_level(instance) >= 3:
            class_name = instance.__class__.__name__ if instance else "UnknownClass"
            print(f"Class '{class_name}', Function '{func.__name__}' executed in {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes, {elapsed_hours:.2f} hours).")

        return result

    return wrapper


import sys


def run_trainer_fit(trainer, model, train_loader, val_loader, context: str) -> None:
    """Run Lightning training while surfacing notebook-hostile SystemExit failures.

    Args:
        trainer (Any): Input value.
        model (Any): Input value.
        train_loader (Any): Input value.
        val_loader (Any): Input value.
        context (str): Input value.
    """
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except SystemExit as exc:
        code = exc.code if exc.code is not None else "None"
        argv_preview = " ".join(sys.argv[:5])
        raise RuntimeError(
            f"{context} aborted with SystemExit({code}). "
            "This usually means some code inside the training stack called "
            "CLI-style argument parsing or sys.exit(). "
            f"Current sys.argv starts with: {argv_preview!r}"
        ) from exc


import numpy as np
import networkx as nx
from toolz import curry
import random
from collections import defaultdict


def _make_duplicate_detection_estimator():
    try:
        from coco_grape.module.graph_duplicate_detection_estimator import GraphDuplicateDetectionEstimator
    except ImportError as exc:
        raise ImportError(
            "Graph duplicate filtering requires optional dependency 'coco-grape'. "
            "Install it to use make_graphs_classification_dataset(), "
            "make_two_types_graphs_classification_dataset(), or ArtificialGraphDatasetConstructor.sample()."
        ) from exc
    return GraphDuplicateDetectionEstimator()

def _safe_random_tree(n: int) -> nx.Graph:
    """Return a random tree on n nodes.

    Args:
        n (int): Input value.

    Returns:
        nx.Graph: Computed result.
    """
    if hasattr(nx.generators.trees, "random_tree"):
        # modern NetworkX versions
        return nx.generators.trees.random_tree(n)
    else:
        # fallback: generate random Prüfer sequence
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
        for i,j in zip(i_idxs, j_idxs): 
            if i < j: edges.append((i,j))
        g = nx.Graph()
        for e in edges: g.add_edge(e[0], e[1])
        g = nx.convert_node_labels_to_integers(g)
        labels = np.random.randint(self.alphabet_size, size=nx.number_of_nodes(g))
        labels = {i:label for i,label in enumerate(labels)}
        nx.set_node_attributes(g, labels, 'label')
        nx.set_edge_attributes(g, '-', 'label')
        return g

    def sample(self, n_samples=1):
        samples = [self.__sample_single() for i in range(n_samples)]
        if n_samples == 1: return samples[0]
        return samples


@curry
def random_path_graph(n):
    g = nx.path_graph(n)
    g = nx.convert_node_labels_to_integers(g)
    return g


@curry
def random_tree_graph(n):
    """Return a random tree with n nodes.

    Args:
        n (Any): Input value.

    Returns:
        Any: Computed result.
    """
    g = _safe_random_tree(n)
    g = nx.convert_node_labels_to_integers(g)
    return g

@curry
def random_cycle_graph(n):
    g = _safe_random_tree(n)
    terminals = [u for u in g.nodes()if g.degree(u) == 1]
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
    # a graph is chosen uniformly at random from the set of all graphs with n nodes and m edges
    g = nx.dense_gnm_random_graph(n, m)
    max_cc = max(nx.connected_components(g), key=lambda x: len(x))
    g = nx.subgraph(g, max_cc)
    g = nx.convert_node_labels_to_integers(g)
    return g

def make_graph_generator(graph_type, instance_size):
    graph_generator = None
    if graph_type == 'path':
        graph_generator = random_path_graph(n=instance_size)

    if graph_type == 'tree':
        graph_generator = random_tree_graph(n=instance_size)

    if graph_type == 'cycle':
        graph_generator = random_cycle_graph(n=instance_size)

    if graph_type == 'degree':
        n = instance_size
        dmax = 4
        max_attempts = 128
        graph_generator = None
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

    if graph_type == 'regular':
        if instance_size == 1:
            graph_generator = random_regular_graph(d=0, n=instance_size)
        else:
            default_degree = 3
            regular_degree = default_degree if (instance_size * default_degree) % 2 == 0 else 2
            graph_generator = random_regular_graph(d=regular_degree, n=instance_size)

    if graph_type == 'dense':
        graph_generator = random_dense_graph(n=instance_size, m=instance_size + instance_size // 2)
        
    assert graph_generator is not None, 'Unknown graph generator type:%s'%graph_type
    return graph_generator

class AttributeGenerator(object):
    def __init__(self, data_mtx, targets):
        self.target_classes = sorted(list(set(targets)))
        self.num_classes = len(self.target_classes)
        self.attributes = [data_mtx[[i for i,y in enumerate(targets) if y==t]] for t in self.target_classes]
    
    def transform(self, class_seq):
        attribute_list = []
        for y in class_seq:
            attributes = self.attributes[y]
            idx = np.random.randint(len(attributes))
            attribute_list.append(attributes[idx].flatten())
        return attribute_list
    
@curry
def make_graph(graph_generator, alphabet_size, attribute_generator):
    G = graph_generator
    nx.set_edge_attributes(G, '-', 'label')
    
    if attribute_generator is not None: num_classes = attribute_generator.num_classes
    else: num_classes = alphabet_size

    labels = np.random.randint(num_classes, size=nx.number_of_nodes(G))
    labels_dict = {node_idx:label for node_idx,label in enumerate(labels)}
    nx.set_node_attributes(G, labels_dict, 'true_label')
    
    labels_dict = {node_idx:label%alphabet_size for node_idx,label in enumerate(labels)}
    nx.set_node_attributes(G, labels_dict, 'label')
    
    if attribute_generator is not None:
        attributes = attribute_generator.transform(labels)
        attributes_dict = {node_idx:attribute for node_idx,attribute in enumerate(attributes)}
        nx.set_node_attributes(G, attributes_dict, 'vec')
    return G.copy()


class ArtificialGraphConstructor(object):
    def __init__(self, graph_type='cycle', instance_size=4, alphabet_size=3, attribute_generator=None):
        self.graph_type = graph_type
        self.instance_size = instance_size
        self.alphabet_size = alphabet_size
        self.attribute_generator = attribute_generator
        self.graph_generator = make_graph_generator(graph_type, instance_size)
    
    def sample(self, n_samples=1):
        samples = [make_graph(self.graph_generator, self.alphabet_size, self.attribute_generator) for i in range(n_samples)]
        if n_samples == 1: return samples[0]
        return samples


def link_graphs(graph_source, graph_target, n_link_edges=0):
    n = nx.number_of_nodes(graph_source)
    graph_source_endpoints = np.random.randint(nx.number_of_nodes(graph_source), size=n_link_edges)
    graph_target_endpoints = np.random.randint(nx.number_of_nodes(graph_target), size=n_link_edges)
    graph = nx.disjoint_union(graph_source, graph_target)
    for u,v in graph_source.edges():
        graph.edges[u,v]['true_label'] = 'source'
    for u,v in graph_target.edges():
        graph.edges[u+n,v+n]['true_label'] = 'destination'
    for s,t in zip(graph_source_endpoints, graph_target_endpoints):
        graph.add_edge(s,t+n, label='-', true_label='joint')
    return graph

def make_graphs(graph_generator_target_type, graph_generator_context_type, target_size, context_size, alphabet_size, attribute_generator, n_link_edges, num_graphs, use_single_target=True):
    context_graphs = []
    for i in range(num_graphs):
        graph_generator = make_graph_generator(graph_generator_context_type, context_size)
        G_context = make_graph(graph_generator, alphabet_size, attribute_generator)
        context_graphs.append(G_context.copy())

    if use_single_target:
        graph_generator = make_graph_generator(graph_generator_target_type, target_size)
        G_target = make_graph(graph_generator, alphabet_size, attribute_generator)
        target_graphs = [G_target.copy()]*num_graphs
    else:
        target_graphs = []
        for i in range(num_graphs):
            graph_generator = make_graph_generator(graph_generator_target_type, target_size)
            G_target = make_graph(graph_generator, alphabet_size, attribute_generator)
            target_graphs.append(G_target.copy())
    
    graphs = [link_graphs(graph_source=G_target, graph_target=G_context, n_link_edges=n_link_edges) for G_target, G_context in zip(target_graphs, context_graphs)]
    return graphs 

def make_graphs_classification_dataset(graph_generator_target_type, graph_generator_context_type, target_size, context_size, alphabet_size, n_link_edges, num_graphs, attribute_generator=None):
    pos_graphs = make_graphs(graph_generator_target_type, graph_generator_context_type, target_size, context_size, alphabet_size, attribute_generator, n_link_edges, num_graphs, use_single_target=True)
    neg_graphs = make_graphs(graph_generator_target_type, graph_generator_context_type, target_size, context_size, alphabet_size, attribute_generator, n_link_edges, num_graphs, use_single_target=False)
    gdde = _make_duplicate_detection_estimator()
    pos_graphs = gdde.fit_filter(pos_graphs)
    neg_graphs = gdde.filter(neg_graphs)
    targets = np.array([1]*len(pos_graphs)+[0]*len(neg_graphs))
    graphs = pos_graphs + neg_graphs
    return graphs, targets, pos_graphs, neg_graphs

def make_two_types_graphs_classification_dataset(graph_generator_target_type_pos, graph_generator_context_type_pos, graph_generator_target_type_neg, graph_generator_context_type_neg, target_size, context_size, alphabet_size, n_link_edges, num_graphs, attribute_generator=None):
    pos_graphs = make_graphs(graph_generator_target_type_pos, graph_generator_context_type_pos, target_size, context_size, alphabet_size, attribute_generator, n_link_edges, num_graphs, use_single_target=True)
    neg_graphs = make_graphs(graph_generator_target_type_neg, graph_generator_context_type_neg, target_size, context_size, alphabet_size, attribute_generator, n_link_edges, num_graphs, use_single_target=True)
    gdde = _make_duplicate_detection_estimator()
    pos_graphs = gdde.fit_filter(pos_graphs)
    neg_graphs = gdde.filter(neg_graphs)
    targets = np.array([1]*len(pos_graphs)+[0]*len(neg_graphs))
    graphs = pos_graphs + neg_graphs
    return graphs, targets, pos_graphs, neg_graphs


class ArtificialGraphDatasetConstructor(object):
    def __init__(self, 
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
                 attribute_generator=None):
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
        graph_types = ['path', 'tree', 'cycle', 'degree', 'regular', 'dense']
        return graph_types

    def sample(self, n_samples, return_separate_classes=False):
        pos_graphs = make_graphs(self.graph_generator_target_type_pos, 
                                 self.graph_generator_context_type_pos, 
                                 self.target_size_pos, 
                                 self.context_size_pos, 
                                 self.alphabet_size_pos, 
                                 self.attribute_generator, 
                                 self.n_link_edges_pos, 
                                 n_samples, 
                                 use_single_target=True)
        neg_graphs = make_graphs(self.graph_generator_target_type_neg, 
                                 self.graph_generator_context_type_neg, 
                                 self.target_size_neg, 
                                 self.context_size_neg, 
                                 self.alphabet_size_neg, 
                                 self.attribute_generator, 
                                 self.n_link_edges_neg, 
                                 n_samples, 
                                 use_single_target=True)
        gdde = _make_duplicate_detection_estimator()
        pos_graphs = gdde.fit_filter(pos_graphs)
        neg_graphs = gdde.filter(neg_graphs)
        targets = np.array([1]*len(pos_graphs)+[0]*len(neg_graphs))
        graphs = pos_graphs + neg_graphs
        if return_separate_classes: return pos_graphs, neg_graphs
        return graphs, targets

def make_combined_graphs(graphs1, targets1, graphs2=None, targets2=None, number_of_graphs=1, number_of_edges=1):
    """Combines pairs of graphs from two lists with matching targets, adds edges between them,.

    Args:
        graphs1 (Any): Input value.
        targets1 (Any): Input value.
        graphs2 (Any): Optional input value.
        targets2 (Any): Optional input value.
        number_of_graphs (Any): Optional input value.
        number_of_edges (Any): Optional input value.

    Returns:
        Any: Computed result.
    """

    # If graphs2 and targets2 are None, use graphs1 and targets1
    if graphs2 is None or targets2 is None:
        graphs2 = graphs1
        targets2 = targets1

    # Map targets to indices in the graphs lists
    target_to_indices1 = defaultdict(list)
    for idx, target in enumerate(targets1):
        target_to_indices1[target].append(idx)

    target_to_indices2 = defaultdict(list)
    for idx, target in enumerate(targets2):
        target_to_indices2[target].append(idx)

    # Find common targets between both lists
    common_targets = set(target_to_indices1.keys()).intersection(target_to_indices2.keys())
    if not common_targets:
        raise ValueError("No matching targets found between the two graph lists.")

    combined_graphs = []
    combined_targets = []

    for _ in range(number_of_graphs):
        # Randomly select a common target
        target = random.choice(list(common_targets))

        # Randomly select one graph from each list with the matching target
        idx1 = random.choice(target_to_indices1[target])
        idx2 = random.choice(target_to_indices2[target])

        # Avoid pairing the same graph with itself if both lists are the same
        if graphs1 is graphs2 and idx1 == idx2:
            if len(target_to_indices1[target]) > 1:
                while idx2 == idx1:
                    idx2 = random.choice(target_to_indices2[target])
            else:
                # Only one graph with this target, can't avoid pairing it with itself
                pass

        graph1 = graphs1[idx1]
        graph2 = graphs2[idx2]

        # Relabel nodes to avoid conflicts
        mapping1 = {node: f'g1_{idx1}_{node}' for node in graph1.nodes()}
        mapping2 = {node: f'g2_{idx2}_{node}' for node in graph2.nodes()}
        graph1_relabelled = nx.relabel_nodes(graph1, mapping1)
        graph2_relabelled = nx.relabel_nodes(graph2, mapping2)

        # Combine the two graphs
        combined_graph = nx.compose(graph1_relabelled, graph2_relabelled)

        # Get node lists from each graph
        nodes1 = list(graph1_relabelled.nodes())
        nodes2 = list(graph2_relabelled.nodes())

        # Add edges between nodes from graph1 and graph2
        for _ in range(number_of_edges):
            u = random.choice(nodes1)
            v = random.choice(nodes2)
            combined_graph.add_edge(u, v, label='-')

        # Renumber all nodes from 0 to number of nodes consecutively
        combined_graph = nx.convert_node_labels_to_integers(combined_graph, label_attribute='old_idx')

        combined_graphs.append(combined_graph)
        combined_targets.append(target)

    return combined_graphs, combined_targets
