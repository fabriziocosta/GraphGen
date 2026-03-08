"""Compatibility wrapper for synthetic dataset builders."""

from .extensions.synthetic.datasets import (
    ArtificialGraphConstructor,
    ArtificialGraphDatasetConstructor,
    AttributeGenerator,
    link_graphs,
    make_graph,
    make_graphs,
    make_graphs_classification_dataset,
    make_two_types_graphs_classification_dataset,
)

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
