# Demo Extension

This section documents the demo-oriented extension layer for NodeField.

The demo extension is useful for:
- notebook-facing dataset preparation
- reusable plotting and lightweight analysis helpers
- saved-generator persistence helpers for interactive workflows

Primary entry points live under:
- [`conditional_node_field_graph_generator/extensions/demo/__init__.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/demo/__init__.py)
- [`conditional_node_field_graph_generator/extensions/demo/pipeline.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/demo/pipeline.py)
- [`conditional_node_field_graph_generator/extensions/demo/visualization.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/demo/visualization.py)
- [`conditional_node_field_graph_generator/extensions/demo/storage.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/extensions/demo/storage.py)

Boundary:
- this extension is not required for the core NodeField model
- it exists to support notebooks, demos, and interactive experiment flows

Transition note:
- new code should import demo helpers from `conditional_node_field_graph_generator.extensions.demo`
- the older `notebooks/*.py` helper wrappers are transitional only
