# Conditional Node Field for Graph Generation

GraphGen is a conditional graph generation framework based on decompositional encoding and decoding, coupled with stationary node-field dynamics. Its central premise is to use an explicit graph kernel to derive node embeddings without end-to-end training while incorporating user-defined priors, thereby enabling the rapid injection of structured prior knowledge independently of the available data.

The framework uses two distinct vectorization processes: one to derive node embeddings, and another to construct a graph-level context vector used as conditioning information. These representations need not coincide and may capture substantially different aspects of the graph. Conditioned on this graph-level context, which acts as an explicit latent representation, the model employs a conditional energy-based generator trained through denoising score matching under Gaussian corruption and sampled via Langevin-style dynamics. Unlike diffusion-based methods, this formulation does not rely on an explicit time variable or a reverse diffusion schedule. The explicit latent space further supports operations such as interpolation, which can be translated into meaningful graph interpolations.

Training is supplemented by auxiliary objectives, including node-degree prediction, node-label prediction, edge-label prediction, and edge-existence prediction. The resulting structural and semantic predictions are passed to a decoder that reconstructs the final graph through constrained combinatorial optimization, formulated as an integer programming problem that reconciles predicted degrees and edge probabilities in a globally coherent manner.

## Documentation

The main technical documentation lives under [`docs/`](docs/). The documents are split by responsibility so that the modeling details, orchestration layer, decoder logic, and API surface can each be read independently.

[`docs/CONDITIONAL_NODE_FIELD_README.md`](docs/CONDITIONAL_NODE_FIELD_README.md)

This is the main conceptual and modeling document. It explains the Conditional Node Field formulation itself, including the score-matching objective, the stationary energy-based interpretation, the conditioning pathway, the iterative sampling dynamics, and the distinction between classifier-free guidance and separate post-hoc guidance.

[`docs/MAIN_CLASS_INTERFACES_README.md`](docs/MAIN_CLASS_INTERFACES_README.md)

This is the interface reference for the main public classes. It summarizes the constructor and workflow methods for the batch dataclasses, the node generator, the graph decoder, and the graph generator, and it explains what the main parameters mean together with the practical effect of increasing or decreasing them.

[`docs/CONDITIONAL_NODE_FIELD_GRAPH_GENERATOR_README.md`](docs/CONDITIONAL_NODE_FIELD_GRAPH_GENERATOR_README.md)

This document focuses on the graph-generator orchestration layer. It explains how raw graphs are vectorized, how supervision channels are assembled, how the node generator and decoder are coordinated, how graph-level sampling and interpolation work, and how feasibility filtering and graph-level guidance are exposed.

[`docs/CONDITIONAL_NODE_FIELD_GRAPH_DECODER_README.md`](docs/CONDITIONAL_NODE_FIELD_GRAPH_DECODER_README.md)

This document covers the decoder and constraint-solving stage. It explains how node-level predictions are converted into final `networkx` graphs, how edge probabilities and predicted degrees are reconciled, how connectivity constraints are enforced, and how the ILP-based adjacency projection behaves.

[`docs/PREFERENCES.md`](docs/PREFERENCES.md)

This is a local development conventions file. It covers documentation and notebook preferences rather than the model itself.

The repository includes:
- A trainable conditional node generator.
- A graph-level generator that handles encoding, supervision construction, and decoding.
- Notebook workflows for experiments and analysis.
- Unit tests for core utility and generation behavior.

## Project Layout

- `conditional_node_field_graph_generator/`
  Core package:
  - `conditional_node_field_generator.py`: Conditional Node Field model, batch dataclasses, and shared NN/callback blocks.
  - `conditional_node_field_graph_generator.py`: graph generator/decoder orchestration and interpolation helpers.
  - `support.py`: runtime decorators/helpers plus artificial graph dataset constructors.
- `notebooks/`
  Experiment and demo notebooks, plus notebook-specific helpers in `notebook_utils.py`.
- `tests/`
  Pytest suite for generator behavior and helper modules.
- `docs/`
  Architecture notes, decoder details, Conditional Node Field internals, and local development preferences.
- `.artifacts/`
  Local artifacts (checkpoints/models); ignored by git.

## Installation

1. Create a Python environment (Python 3.10+ recommended).
2. Install core dependencies:

```bash
pip install "numpy<2" torch pytorch-lightning scipy pandas scikit-learn networkx matplotlib pulp dill
```

3. Install optional extras as needed:
- `jupyterlab` or `notebook` to run notebooks.
- `coco-grape`, `NSPPK`, `AbstractGraph` if you run notebook flows that import them.

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

By default, `.sample(...)` reuses cached graph-level conditioning rows from the training set. It can also be configured to stochastically interpolate between pairs of cached training embeddings in graph-conditioning space, with the same interpolation coefficient applied to graph embeddings, node counts, and edge counts.

When guidance targets are available, sampling can also use classifier-free conditioning through
`desired_target` and `guidance_scale`. The detailed mechanics are documented in
[`docs/CONDITIONAL_NODE_FIELD_README.md`](docs/CONDITIONAL_NODE_FIELD_README.md).

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
- Place reusable notebook logic in helper modules (for example `notebooks/notebook_utils.py`).
- Clear notebook outputs before committing.
