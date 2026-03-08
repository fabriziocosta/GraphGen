# Conditional Node Field for Graph Generation

NodeField is a conditional graph generation framework based on decompositional encoding and decoding, coupled with stationary node-field dynamics. Its central premise is to use an explicit graph kernel to derive node embeddings without end-to-end training while incorporating user-defined priors, thereby enabling the rapid injection of structured prior knowledge independently of the available data. The framework supports both classifier-free guidance (CFG) for target-conditioned sampling and separate post-hoc guidance through an auxiliary classifier or regressor.

The framework uses two distinct vectorization processes: one to derive node embeddings, and another to construct a graph-level context vector used as conditioning information. These representations need not coincide and may capture substantially different aspects of the graph. Conditioned on this graph-level context, which acts as an explicit latent representation, the model employs a conditional energy-based generator trained through denoising score matching under Gaussian corruption and sampled via Langevin-style dynamics. Unlike diffusion-based methods, this formulation does not rely on an explicit time variable or a reverse diffusion schedule. The explicit latent space further supports operations such as interpolation, which can be translated into meaningful graph interpolations.

Training is supplemented by auxiliary objectives, including node-degree prediction, node-label prediction, edge-label prediction, and edge-existence prediction. At sampling time, the model can either use CFG on the target-conditioning path or use a separately trained post-hoc guidance predictor, depending on the workflow. The resulting structural and semantic predictions are passed to a decoder that reconstructs the final graph through constrained combinatorial optimization, formulated as an integer programming problem that reconciles predicted degrees and edge probabilities in a globally coherent manner.

## Documentation

The main technical documentation lives under [`docs/`](docs/). The documents are split by responsibility so that the modeling details, orchestration layer, decoder logic, and API surface can each be read independently.

[`docs/1_CONDITIONAL_NODE_FIELD_GRAPH_GENERATOR_README.md`](docs/1_CONDITIONAL_NODE_FIELD_GRAPH_GENERATOR_README.md)

This document focuses on the graph-generator orchestration layer. It explains how raw graphs are vectorized, how supervision channels are assembled, how the node generator and decoder are coordinated, how graph-level sampling and interpolation work, and how feasibility filtering and graph-level guidance are exposed.

[`docs/2_CONDITIONAL_NODE_FIELD_README.md`](docs/2_CONDITIONAL_NODE_FIELD_README.md)

This is the main conceptual and modeling document. It explains the Conditional Node Field formulation itself, including the stationary energy-based interpretation, the conditioning pathway, the vector-versus-token conditioning interface used by cross-attention, and the architectural design choices.

[`docs/2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md`](docs/2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md)

This companion document covers the operational behavior of the node model. It explains the auxiliary losses, the full training objective, sampling updates, inference-time projection, main hyperparameters, and the interpretation of logged training metrics.

[`docs/3_CONDITIONAL_NODE_FIELD_GRAPH_DECODER_README.md`](docs/3_CONDITIONAL_NODE_FIELD_GRAPH_DECODER_README.md)

This document covers the decoder and constraint-solving stage. It explains how node-level predictions are converted into final `networkx` graphs, how edge probabilities and predicted degrees are reconciled, how connectivity constraints are enforced, and how the ILP-based adjacency projection behaves.

[`docs/4_TARGET_GUIDANCE_README.md`](docs/4_TARGET_GUIDANCE_README.md)

This document is dedicated to target guidance. It explains the two supported approaches, classifier-free guidance (CFG) and separate post-hoc guidance through an auxiliary classifier or regressor, and makes the API split between them explicit.

[`docs/5_MAIN_CLASS_INTERFACES_README.md`](docs/5_MAIN_CLASS_INTERFACES_README.md)

This is the interface reference for the main public classes. It summarizes the constructor and workflow methods for the batch dataclasses, the node generator, the graph decoder, and the graph generator, and it explains what the main parameters mean together with the practical effect of increasing or decreasing them.

[`docs/PREFERENCES.md`](docs/PREFERENCES.md)

This is a local development conventions file. It covers documentation and notebook preferences rather than the model itself.

[`docs/extensions/molecular/README.md`](docs/extensions/molecular/README.md)

This extension document covers the chemistry-specific support layer. It points to the molecular conversion, dataset, and visualization utilities used for PubChem, ZINC, and notebook molecule workflows.

[`docs/extensions/synthetic/README.md`](docs/extensions/synthetic/README.md)

This extension document covers the synthetic-graph support layer. It points to the artificial graph primitives, synthetic dataset builders, and graph-composition helpers used mainly in demos and tests.

[`docs/extensions/demo/README.md`](docs/extensions/demo/README.md)

This extension document covers the demo-oriented helper layer. It points to the reusable notebook pipeline helpers, visualization utilities, and checkpoint helpers used in the maintained example notebooks. Saved-generator serialization now lives in the core [`conditional_node_field_graph_generator/persistence.py`](/Users/fabriziocosta/Resilio%20Sync/Sync/Projects/GraphGen/conditional_node_field_graph_generator/persistence.py) module.

The repository includes:
- A trainable conditional node generator.
- A graph-level generator that handles encoding, supervision construction, and decoding.
- Notebook workflows for experiments and analysis.
- Unit tests for core utility and generation behavior.

## Project Layout

```text
NodeField/
├── conditional_node_field_graph_generator/
│   ├── conditional_node_field_generator.py
│   ├── conditional_node_field_graph_generator.py
│   ├── extensions/
│   ├── metrics_collection.py
│   ├── metrics_visualization.py
│   └── training_policy.py
├── docs/
│   ├── 1_CONDITIONAL_NODE_FIELD_GRAPH_GENERATOR_README.md
│   ├── 2_CONDITIONAL_NODE_FIELD_README.md
│   ├── 2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md
│   ├── 3_CONDITIONAL_NODE_FIELD_GRAPH_DECODER_README.md
│   ├── 4_TARGET_GUIDANCE_README.md
│   ├── 5_MAIN_CLASS_INTERFACES_README.md
│   ├── extensions/
│   │   ├── demo/
│   │   │   └── README.md
│   │   ├── molecular/
│   │   │   └── README.md
│   │   └── synthetic/
│   │       └── README.md
│   └── PREFERENCES.md
├── notebooks/
│   ├── demo.ipynb
│   ├── demo_chem.ipynb
│   ├── demo_optimization.ipynb
│   └── demo_zinc.ipynb
├── tests/
├── .artifacts/
└── README.md
```

Key paths:

- `conditional_node_field_graph_generator/`
  Core package with the Conditional Node Field model, graph-generator orchestration, decoder support, metrics helpers, and training utilities.

- `conditional_node_field_graph_generator/conditional_node_field_generator.py`
  Node-level generator implementation, batch dataclasses, sampling logic, and support for CFG and separate post-hoc guidance.

- `conditional_node_field_graph_generator/conditional_node_field_graph_generator.py`
  High-level graph generator, supervision assembly, decode orchestration, and graph-level sampling helpers.

- `conditional_node_field_graph_generator/extensions/`
  Optional extension layers for demo workflows, molecular graph tooling, and synthetic/artificial graph utilities.

- `docs/`
  Technical documentation for the model, public interfaces, graph generator, decoder, extension layers, and local development conventions.

- `notebooks/`
  Demo and experiment notebooks. Reusable notebook support logic is being moved into `extensions/demo`.

- `tests/`
  Pytest suite for generator behavior and supporting modules.

- `.artifacts/`
  Local checkpoints and generated artifacts. This directory is ignored by git.

## Installation

1. Create a Python environment (Python 3.10+ recommended).
2. Install core dependencies:

```bash
pip install "numpy<2" torch pytorch-lightning scipy pandas scikit-learn networkx matplotlib pulp dill
```

3. Install optional extras as needed:
- `jupyterlab` or `notebook` to run notebooks.
- `NSPPK` and `AbstractGraph`.

## Quick Start

```python
from conditional_node_field_graph_generator import (
    ConditionalNodeFieldGenerator,
    ConditionalNodeFieldGraphDecoder,
    ConditionalNodeFieldGraphGenerator,
)
```

Typical high-level workflow:
1. Prepare graphs (`networkx.Graph`) with node/edge labels as needed.
2. Build vectorizers for graph-level and node-level embeddings.
3. Instantiate `ConditionalNodeFieldGenerator`.
4. Wrap it in `ConditionalNodeFieldGraphGenerator` (optionally with a decoder).
5. Train with `.fit(...)`.
6. Generate with `.sample(...)` or `.sample_conditioned_on_random(...)`.

If training is interrupted, you can resume the training state by passing `ckpt_path=...` to `.fit(...)`, provided you point to one of the Lightning checkpoints written under the configured checkpoint root.

By default, `.sample(...)` reuses cached graph-level conditioning rows from the training set. It can also be configured to stochastically interpolate between pairs of cached training embeddings in graph-conditioning space, with the same interpolation coefficient applied to graph embeddings, node counts, and edge counts.

When guidance targets are available, sampling can also use classifier-free conditioning through
`desired_target` and `guidance_scale`. The detailed mechanics are documented in
[`docs/4_TARGET_GUIDANCE_README.md`](docs/4_TARGET_GUIDANCE_README.md).

Notebook examples:
- `notebooks/demo.ipynb`
- `notebooks/demo_chem.ipynb`
- `notebooks/demo_optimization.ipynb`

## Running Tests

```bash
pytest -q
```

Targeted run example:

```bash
pytest tests/test_graph_generator.py -q
```

## Data and Artifacts

Large datasets and training artifacts are intentionally excluded from version control.

Ignored locations include:
- `.artifacts/`
- `notebooks/datasets/`

Keep experimental outputs in ignored paths to avoid inflating repository history.

## Notes for Notebook Development

Notebook execution flow is kept lean by design:
- Prefer assigning variables and calling functions from `.py` modules.
- Place reusable notebook logic in extension modules, especially `conditional_node_field_graph_generator/extensions/demo/`.
- Clear notebook outputs before committing.
