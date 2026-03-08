# TODO

This file tracks the work that is still genuinely open after the cleanup passes.

## Remaining Refactors

- Decide whether to keep the root-level synthetic compatibility wrappers or remove them after one more migration pass:
  - [`synthetic_graph_primitives.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/synthetic_graph_primitives.py)
  - [`synthetic_graph_datasets.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/synthetic_graph_datasets.py)
  - [`graph_composition.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/graph_composition.py)

## Notebook / Demo Follow-Up

- Clean notebook prose/comments and saved outputs that still refer to historical helper locations or outdated traceback paths.

## Tests

- Add direct tests for the demo extension modules beyond the current checkpoint-discovery and interpolation coverage.

- Simplify the test suite so it targets only:
  - core package modules
  - extension package modules

## Open Design Decisions

- Decide whether the demo extension should stay as one package or be reorganized further across:
  - `extensions/demo/pipeline.py`
  - `extensions/demo/visualization.py`
  - `extensions/demo/storage.py`

- Review whether `runtime_utils.py` should remain in core or move into a clearer internal helpers namespace.

## Model / Feature Work

- Add full tokenized-conditioning support at the graph-generator level so [`ConditionalNodeFieldGraphGenerator`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/conditional_node_field_graph_generator.py) can work directly with structured condition memories, not only flat graph-level conditioning vectors plus node and edge count channels.

  This should cover:
  - tokenized conditioning emitted directly by graph encoders
  - graph-to-graph conditioning based on node embeddings from a previous graph
  - abstract-graph conditioning where high-level motif or scaffold tokens drive concrete graph generation
