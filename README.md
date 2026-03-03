# GraphGen

GraphGen now maintains the Equilibrium Matching generator as the supported path for conditional graph generation.

## Maintained Surface
- `node_diffusion/eqm_conditional_node_generator.py` – supported conditional generator backend.
- `node_diffusion/decompositional_encoder_decoder.py` – orchestration, supervision planning, graph encode/decode, and optimization-based final decoding.
- `node_diffusion/generator_shared.py` – shared transformer, edge-head, plotting, and metric utilities used by the maintained generator.
- `notebooks/demo_eqm.ipynb` – maintained end-to-end notebook.
- `EqM_README.md` – detailed notes on the EqM formulation and implementation.

## Archived Legacy
The diffusion-era implementation and related notebooks were archived under `v_0_1/`.

- `v_0_1/node_diffusion/conditional_denoising_node_generator.py`
- `v_0_1/notebooks/demo.ipynb`
- `v_0_1/notebooks/demo_chem.ipynb`
- `v_0_1/notebooks/Denoising Conditional Node Graph Generator.ipynb`

These files are kept for historical reference and older experiments. They are no longer the maintained path.

## Installation
1. Create and activate a Python environment.
2. Install the core dependencies:
   ```bash
   pip install "numpy<2" torch pytorch-lightning scipy pandas scikit-learn networkx matplotlib pulp dill
   ```
3. Install project-specific extras as needed:
   - `coco-grape` for the vectorization helpers used by the decompositional pipeline
   - `jupyterlab` or `notebook` if you want to run the notebook

## Quick Start
```python
from node_diffusion.eqm_conditional_node_generator import EqMConditionalNodeGenerator
from node_diffusion.decompositional_encoder_decoder import (
    ConditionalNodeGeneratorModel,
    DecompositionalEncoderDecoder,
    DecompositionalNodeEncoderDecoder,
)
```

The maintained workflow is demonstrated in `notebooks/demo_eqm.ipynb`.

## Status
This codebase is still research-oriented, but the supported generator path is now EqM-only. New work should target the maintained files listed above rather than the archived `v_0_1/` implementation.
