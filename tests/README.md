# Test Suite

This directory contains the focused `pytest` suite for NodeField.

The tests are mostly unit and small integration checks. They are designed to protect:
- core API contracts
- supervision and batching logic
- guidance-path behavior
- decoder invariants
- lightweight notebook/helper utilities
- local molecular helper behavior

The suite is intentionally biased toward fast, deterministic checks rather than long end-to-end training runs.

## Files

[`tests/test_graph_generator.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/tests/test_graph_generator.py)

This is the largest test module. It covers the graph-level orchestration layer, including:
- constructor validation
- supervision-plan decisions
- label handling
- encode/decode path behavior
- cached graph-conditioning sampling
- interpolation-based condition sampling
- adjacency optimization failure handling
- serial vs parallel decode consistency

[`tests/test_cfg_guidance.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/tests/test_cfg_guidance.py)

This module covers the distinction between classifier-free guidance and the separate post-hoc guidance predictor path. It checks:
- explicit CFG target-mode behavior
- routing of CFG arguments through graph-level decode/sample APIs
- routing of classifier-guided and regression-guided calls
- score-mixing behavior
- invalid negative guidance scales
- predictor mode inference for the separate guidance path

[`tests/test_batches_and_lightning_utils.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/tests/test_batches_and_lightning_utils.py)

This module focuses on lower-level contracts and utilities:
- batch dataclass length behavior
- trainer-fit wrapper behavior
- train/validation subset splitting
- EMA metric tracking
- restored-checkpoint summary formatting
- selected loss helpers
- plotting-key compatibility
- package export expectations for the main public classes

[`tests/test_timeit.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/tests/test_timeit.py)

This module checks the verbose timing helpers:
- verbosity-level normalization
- conditional printing behavior for the timing decorator

[`tests/test_interpolate.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/tests/test_interpolate.py)

This module covers lightweight interpolation and notebook-helper behavior:
- integer interpolation utilities
- positive-endpoint sampling for interpolation demos
- summary object structure for interpolation results

[`tests/test_molecular_graph_utils.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/tests/test_molecular_graph_utils.py)

This module covers the local chemistry helper layer:
- SMILES to graph conversion
- graph-to-RDKit round-trip behavior
- local PubChem assay loading from bundled SDF files
- supervised dataset equalization/resizing
- tiny ZINC corpus construction from a CSV fixture

## What The Tests Prioritize

The suite is strongest at:
- interface validation
- behavior around edge cases
- routing and separation between guidance modes
- deterministic helper logic
- small local integration checks

The suite is weaker at:
- long end-to-end training runs
- realistic large-scale dataset runs
- notebook execution as full documents
- performance/regression benchmarking

That tradeoff is intentional. The goal is fast feedback on code changes, not exhaustive experiment replay.

## Running The Tests

Run the full suite:

```bash
pytest -q
```

Run a single module:

```bash
pytest tests/test_graph_generator.py -q
```

Run a focused subset by keyword:

```bash
pytest -q -k guidance
pytest -q -k molecular
pytest -q -k interpolate
```

## Practical Notes

- Some tests rely on bundled local assay files under `notebooks/datasets/PUBCHEM/`.
- The molecular tests are still lightweight, but they can be slower than the pure unit tests because they parse local SDF files.
- Notebook behavior is tested only through selected helper functions, not by replaying full notebooks.
- The remaining tests increasingly target the extension and core modules directly rather than transitional compatibility layers.

## Intended Role Of This Directory

Use this test suite to validate:
- refactors of the core model/orchestration code
- changes to CFG or separate guidance behavior
- changes to extension helper modules and their public utility functions
- changes to the molecular extension helpers

Do not treat it as proof that:
- a full training pipeline is numerically optimal
- all notebooks run unchanged on every environment
- all optional dependencies are installed and configured
