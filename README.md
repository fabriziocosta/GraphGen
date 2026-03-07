# GraphGen

GraphGen is a conditional graph generation framework based on decompositional encoding and decoding, coupled with equilibrium-style node-generation dynamics. Its central premise is to use an explicit graph kernel to derive node embeddings without end-to-end training while incorporating user-defined priors, thereby enabling the rapid injection of structured prior knowledge independently of the available data.

The framework uses two distinct vectorization processes: one to derive node embeddings, and another to construct a graph-level context vector used as conditioning information. These representations need not coincide and may capture substantially different aspects of the graph. Conditioned on this graph-level context, which acts as an explicit latent representation, the model employs a conditional energy-based generator trained through denoising score matching under Gaussian corruption and sampled via Langevin-style dynamics. Unlike diffusion-based methods, this formulation does not rely on an explicit time variable or a reverse diffusion schedule. The explicit latent space further supports operations such as interpolation, which can be translated into meaningful graph interpolations.

Training is supplemented by auxiliary objectives, including node-degree prediction, node-label prediction, edge-label prediction, and edge-existence prediction. The resulting structural and semantic predictions are passed to a decoder that reconstructs the final graph through constrained combinatorial optimization, formulated as an integer programming problem that reconciles predicted degrees and edge probabilities in a globally coherent manner.

Technical documentation lives under [`docs/`](docs/):
- [`docs/EQUILIBRIUM_MATCHING_README.md`](docs/EQUILIBRIUM_MATCHING_README.md): Equilibrium Matching node-generator internals and training/sampling behavior.
- [`docs/GRAPH_GENERATOR_README.md`](docs/GRAPH_GENERATOR_README.md): graph-generator orchestration architecture.
- [`docs/DECODER_README.md`](docs/DECODER_README.md): decoder and constraint-solver details.
- [`docs/PREFERENCES.md`](docs/PREFERENCES.md): local documentation and notebook conventions.

The repository includes:
- A trainable conditional node generator.
- A graph-level generator that handles encoding, supervision construction, and decoding.
- Notebook workflows for experiments and analysis.
- Unit tests for core utility and generation behavior.

## Project Layout

- `equilibrium_matching_decompositional_graph_generator/`
  Core package:
  - `node_engine.py`: Equilibrium Matching node model, batch dataclasses, and shared NN/callback blocks.
  - `graph_engine.py`: graph generator/decoder orchestration and interpolation helpers.
  - `support.py`: runtime decorators/helpers plus artificial graph dataset constructors.
- `notebooks/`
  Experiment and demo notebooks, plus notebook-specific helpers in `notebook_utils.py`.
- `tests/`
  Pytest suite for generator behavior and helper modules.
- `docs/`
  Architecture notes, decoder details, Equilibrium Matching internals, and local development preferences.
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
from equilibrium_matching_decompositional_graph_generator import (
    EquilibriumMatchingDecompositionalNodeGenerator,
    EquilibriumMatchingDecompositionalGraphDecoder,
    EquilibriumMatchingDecompositionalGraphGenerator,
)
```

Typical high-level workflow:
1. Prepare graphs (`networkx.Graph`) with node/edge labels as needed.
2. Build vectorizers for graph-level and node-level embeddings.
3. Instantiate `EquilibriumMatchingDecompositionalNodeGenerator`.
4. Wrap it in `EquilibriumMatchingDecompositionalGraphGenerator` (optionally with a decoder).
5. Train with `.fit(...)`.
6. Generate with `.sample(...)` or `.sample_conditioned_on_random(...)`.

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
- `PUBCHEM/`
- `notebooks/PUBCHEM/`

Keep experimental outputs in ignored paths to avoid inflating repository history.

## Notes for Notebook Development

Notebook execution flow is kept lean by design:
- Prefer assigning variables and calling functions from `.py` modules.
- Place reusable notebook logic in helper modules (for example `notebooks/notebook_utils.py`).
- Clear notebook outputs before committing.
