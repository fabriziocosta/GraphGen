"""Compatibility wrapper for synthetic primitive graph samplers."""

from .extensions.synthetic.primitives import (
    RandomGraphConstructor,
    make_graph_generator,
    random_cycle_graph,
    random_degree_seq,
    random_dense_graph,
    random_path_graph,
    random_regular_graph,
    random_tree_graph,
)

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
