## Artificial Graph Utilities

This document describes the artificial-graph and synthetic-dataset helpers that support demo workflows, synthetic experiments, and some tests.

These utilities are not part of the core NodeField model. New code should access them through the synthetic extension namespace:
- [`conditional_node_field_graph_generator/extensions/synthetic/__init__.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/synthetic/__init__.py)

The maintained implementation now lives in:
- [`conditional_node_field_graph_generator/extensions/synthetic/primitives.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/synthetic/primitives.py)
- [`conditional_node_field_graph_generator/extensions/synthetic/datasets.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/synthetic/datasets.py)
- [`conditional_node_field_graph_generator/extensions/synthetic/composition.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/synthetic/composition.py)

The root-level files remain available only as compatibility wrappers:
- [`synthetic_graph_primitives.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/synthetic_graph_primitives.py)
- [`synthetic_graph_datasets.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/synthetic_graph_datasets.py)
- [`graph_composition.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/graph_composition.py)

## Main Responsibilities

### 1. Primitive synthetic graph samplers

The primitive samplers generate unlabeled graph structures that are later decorated with node and edge labels.

Main functions in [`extensions/synthetic/primitives.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/synthetic/primitives.py):

- `random_path_graph(n)`
  Build a path graph with `n` nodes.

- `random_tree_graph(n)`
  Build a random tree with `n` nodes.

- `random_cycle_graph(n)`
  Build a cycle-like graph by augmenting a random tree with extra edges between terminal nodes.

- `random_regular_graph(d, n)`
  Build a `d`-regular graph with `n` nodes.

- `random_degree_seq(n, dmax)`
  Build an expected-degree graph from a simple degree sequence.

- `random_dense_graph(n, m)`
  Build a dense random graph with `n` nodes and `m` edges, then keep the largest connected component.

- `make_graph_generator(graph_type, instance_size)`
  Dispatch helper that selects one of the primitive samplers from a graph-type string.

Supported `graph_type` values:
- `path`
- `tree`
- `cycle`
- `degree`
- `regular`
- `dense`

Utility classes:
- `RandomGraphConstructor`
  Samples lightweight random graphs based on random integer edge endpoints rather than the graph-type dispatcher above.

## 2. Labeling and attribute decoration

The dataset helpers can decorate graph nodes with labels and optional feature vectors.

Main pieces in [`extensions/synthetic/datasets.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/synthetic/datasets.py):

- `AttributeGenerator`
  Samples node-level auxiliary attribute vectors conditioned on sampled class labels.

- `make_graph(graph_generator, alphabet_size, attribute_generator)`
  Takes a primitive graph and assigns:
  - node `true_label`
  - node `label`
  - optional node `vec`
  - edge label `"-"`

Label behavior:
- `true_label` is sampled in the full class space
- `label` is the class modulo `alphabet_size`
- if `attribute_generator` is present, each node also receives a sampled vector attribute

## 3. Linked source/context graph construction

The synthetic datasets are based on joining a target graph with a context graph.

Key functions:
- `link_graphs(graph_source, graph_target, n_link_edges=0)`
  Build a disjoint union of two graphs and add `n_link_edges` random cross-graph edges.

- `make_graphs(...)`
  Generate batches of linked source/context graphs using a target graph family and a context graph family.

Important behavior:
- `use_single_target=True`
  Reuses one target graph across all samples.

- `use_single_target=False`
  Samples a fresh target graph for each example.

This distinction is used to create positive and negative synthetic classes.

## 4. Synthetic classification datasets

The main binary dataset builders are:

- `make_graphs_classification_dataset(...)`
  Positive class:
  - same target graph reused across samples
  Negative class:
  - fresh target graph per sample

- `make_two_types_graphs_classification_dataset(...)`
  Positive and negative classes are built from different target/context graph families.

Both functions:
- deduplicate graphs using `AbstractGraph.hash_graph.GraphHashDeduper`
- return:
  - `graphs`
  - `targets`
  - `pos_graphs`
  - `neg_graphs`

## 5. Dataset constructor classes

Class wrappers:

- `ArtificialGraphConstructor`
  Small wrapper around a single primitive graph family plus optional attribute generation.

- `ArtificialGraphDatasetConstructor`
  Higher-level wrapper for binary synthetic datasets with separate positive and negative graph-family settings.

Important methods:
- `get_graph_types()`
  Returns the supported graph-family names.

- `sample(n_samples, return_separate_classes=False)`
  Returns either:
  - `(graphs, targets)`, or
  - `(pos_graphs, neg_graphs)` when `return_separate_classes=True`

## 6. Graph composition utilities

[`extensions/synthetic/composition.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/synthetic/composition.py) contains:

- `make_combined_graphs(graphs1, targets1, graphs2=None, targets2=None, number_of_graphs=1, number_of_edges=1)`

This helper:
- samples pairs of graphs with matching targets
- relabels nodes to avoid collisions
- composes the graphs
- adds random cross-graph edges
- renumbers the final nodes consecutively

Typical use:
- making larger synthetic graphs from smaller target-matched components
- augmentation for demo experiments

## Dependency note

The synthetic dataset builders depend on `AbstractGraph` for graph deduplication.

That dependency enters through:
- `GraphHashDeduper`

This is why the synthetic extension is considered auxiliary rather than core.

## Intended Scope

Use these utilities for:
- synthetic demos
- artificial benchmark construction
- quick graph-family experiments
- notebook support

Do not treat them as required for:
- the core Conditional Node Field model
- molecule workflows
- generic production graph ingestion
