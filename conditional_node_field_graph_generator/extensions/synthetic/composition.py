"""Synthetic graph composition helpers."""

from collections import defaultdict
import random

import networkx as nx


def make_combined_graphs(graphs1, targets1, graphs2=None, targets2=None, number_of_graphs=1, number_of_edges=1):
    """Combine pairs of graphs with matching targets and add cross-graph edges."""
    if graphs2 is None or targets2 is None:
        graphs2 = graphs1
        targets2 = targets1

    target_to_indices1 = defaultdict(list)
    for idx, target in enumerate(targets1):
        target_to_indices1[target].append(idx)

    target_to_indices2 = defaultdict(list)
    for idx, target in enumerate(targets2):
        target_to_indices2[target].append(idx)

    common_targets = set(target_to_indices1.keys()).intersection(target_to_indices2.keys())
    if not common_targets:
        raise ValueError("No matching targets found between the two graph lists.")

    combined_graphs = []
    combined_targets = []

    for _ in range(number_of_graphs):
        target = random.choice(list(common_targets))
        idx1 = random.choice(target_to_indices1[target])
        idx2 = random.choice(target_to_indices2[target])

        if graphs1 is graphs2 and idx1 == idx2 and len(target_to_indices1[target]) > 1:
            while idx2 == idx1:
                idx2 = random.choice(target_to_indices2[target])

        graph1 = graphs1[idx1]
        graph2 = graphs2[idx2]

        mapping1 = {node: f"g1_{idx1}_{node}" for node in graph1.nodes()}
        mapping2 = {node: f"g2_{idx2}_{node}" for node in graph2.nodes()}
        graph1_relabelled = nx.relabel_nodes(graph1, mapping1)
        graph2_relabelled = nx.relabel_nodes(graph2, mapping2)

        combined_graph = nx.compose(graph1_relabelled, graph2_relabelled)
        nodes1 = list(graph1_relabelled.nodes())
        nodes2 = list(graph2_relabelled.nodes())

        for _ in range(number_of_edges):
            u = random.choice(nodes1)
            v = random.choice(nodes2)
            combined_graph.add_edge(u, v, label="-")

        combined_graph = nx.convert_node_labels_to_integers(combined_graph, label_attribute="old_idx")
        combined_graphs.append(combined_graph)
        combined_targets.append(target)

    return combined_graphs, combined_targets


__all__ = ["make_combined_graphs"]
