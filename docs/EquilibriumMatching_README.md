# Equilibrium Matching Decompositional Node Generator

This document explains the `EquilibriumMatchingDecompositionalNodeGenerator` implemented in this repository, what equations it uses during training and sampling, and how the surrounding graph-generation pipeline fits together.

The implementation lives primarily in
[`../equilibrium_matching_decompositional_graph_generator/node_engine.py`](../equilibrium_matching_decompositional_graph_generator/node_engine.py).

## Overview

The maintained node generator in this repository is:

- `EquilibriumMatchingDecompositionalNodeGenerator`
  A stationary, energy-based conditional generator inspired by Equilibrium Matching.

The Equilibrium Matching model replaces diffusion time-conditioning and reverse-time denoising with:

- a scalar conditional energy or potential,
- a score field obtained as the gradient of that scalar,
- a stationary score-matching objective,
- iterative relaxation or Langevin-style sampling in feature space.

In this repository, the Equilibrium Matching generator is used as the `conditioning -> node embeddings` stage inside the broader decompositional pipeline:

1. encode each training graph into node feature matrices and graph-level conditioning vectors,
2. train the Equilibrium Matching conditional generator to map graph-level conditions to node-level embeddings,
3. use the model heads and graph decoder to map generated node-level predictions back into graphs.

## High-Level Idea

Diffusion models learn a time-dependent denoising field:

$$
\epsilon_\theta(x_t, c, t)
$$

or equivalently a time-dependent score field.

The Equilibrium Matching model implemented here learns a stationary conditional score field:

$$
g_\theta(x, c)
$$

where:

- $x$ is a padded node-feature tensor for one graph,
- $c$ is the graph-level conditioning input,
- $g_\theta$ is constrained to be integrable because it comes from a scalar potential.

Instead of directly outputting a vector field, this implementation defines a scalar conditional potential:

$$
\phi_\theta(x, c)
$$

and obtains the score field by differentiation:

$$
g_\theta(x, c) = - \nabla_x \phi_\theta(x, c)
$$

This means the learned field is conservative by construction.

## Energy-Based View

The conceptual target is a conditional energy-based model:

$$
p_\theta(x \mid c) \propto \exp(-E_\theta(x, c))
$$

with:

$$
E_\theta(x, c) \equiv \phi_\theta(x, c)
$$

Then:

$$
\nabla_x \log p_\theta(x \mid c) = -\nabla_x E_\theta(x, c)
$$

and because this code uses $E_\theta = \phi_\theta$, the implemented score field is:

$$
g_\theta(x, c) = -\nabla_x \phi_\theta(x, c)
$$

This is the field that drives both training and generation.

## What the Model Predicts

The Equilibrium Matching module does not directly output node features. It outputs:

1. a scalar conditional potential through `potential_head`,
2. a score field through autograd,
3. auxiliary head predictions for:
   - node existence,
   - node degree,
   - optional locality supervision.

So the primary generative object is the score:

$$
g_\theta(x, c)
$$

not a reconstruction vector and not a diffusion residual.

Node existence should be interpreted as a learned occupancy process over node slots. In this repository, the global node count is conditioned explicitly, but the model still learns which specific slots should materialize. That allows generation to stay gradual: several candidate node slots can remain plausible early in sampling and only later coalesce into a committed support set.

## Inputs and Outputs

### Inputs

For a batch of graphs, the Equilibrium Matching module consumes:

- `input_examples`
  Shape $(B, N, D)$, padded node features.
- `global_condition`
  Shape $(B, C)$ or $(B, M, C)$, graph-level conditioning vectors or tokens.
- explicit graph-size channels inside the conditioning input, including node count and edge count.
- `node_mask`
  Shape $(B, N)$, boolean mask indicating valid node slots.

The important distinction is:

- the conditioned node count is a global cardinality target,
- `node_mask` is the per-slot realization of that target.

Those are not equivalent. A scalar count does not identify which latent node positions are active, and the Equilibrium Matching dynamics are free to explore alternative support sets before settling on a final one.

At training time, if node-label supervision is enabled, the model receives explicit
per-node categorical targets rather than a graph-level label-composition summary.

### Outputs

At inference time, the wrapper returns:

- a list of generated node-feature matrices in original feature scale,
- optionally with degree channel overwritten by the auxiliary degree head,
- and node existence channel snapped using the existence head.

Additionally, if node-label supervision is enabled, the wrapper stores the per-node
predicted categorical labels from the final latent state in:

```python
equilibrium_matching_generator.last_predicted_node_label_classes_
```

These labels are not written directly into the node-feature tensor because node labels
are categorical metadata, not continuous feature channels in the Equilibrium Matching state.

## Data Preprocessing

The wrapper preserves the established preprocessing behavior:

- node-feature tensors are padded to a common maximum node count,
- node features are scaled with `MinMaxScaler`,
- conditional features are scaled separately,
- degree scaling statistics are stored so discrete degree labels can be recovered,
- padded positions are masked during Equilibrium Matching training and attention.

This matters because the Equilibrium Matching model operates in scaled feature space, while the downstream decoder expects outputs back in the original feature space.

It also matters generatively: padding is not just a batching convenience. It provides a larger latent support over which the model can express tentative node occupancy decisions before the existence process sharpens into a final graph.

## Model Architecture

The Equilibrium Matching generator uses a conditional transformer backbone:

1. node features are layer-normalized,
2. node features are projected into a latent dimension,
3. graph-level conditions are projected into the same latent space,
4. node tokens attend to condition tokens through stacked cross-transformer layers,
5. latent node tokens are converted into a scalar potential and auxiliary predictions.

### Backbone

Let:

- $x \in \mathbb{R}^{B \times N \times D}$ be node features,
- $c \in \mathbb{R}^{B \times M \times C}$ be condition tokens,
- $h \in \mathbb{R}^{B \times N \times H}$ be latent node tokens.

The encoder computes:

$$
h = f_\theta(x, c)
$$

using learned linear projections and repeated cross-attention blocks.

### Potential

Each latent node token contributes a scalar:

$$
\phi_i = \mathrm{MLP}(h_i)
$$

and the graph-level scalar potential is the masked sum:

$$
\phi_\theta(x, c) = \sum_{i=1}^{N} m_i \, \phi_i
$$

where $m_i \in \{0, 1\}$ is the node mask.

### Score

The score is computed by differentiating the scalar potential with respect to the input node features:

$$
g_\theta(x, c) = -\nabla_x \phi_\theta(x, c)
$$

This is done with PyTorch autograd in the implementation.

## Core Equilibrium Matching Training Objective

The basic training construction is Gaussian corruption of clean data.

Given clean node features:

$$
x
$$

sample Gaussian noise:

$$
\varepsilon \sim \mathcal{N}(0, I)
$$

and define noisy inputs:

$$
\tilde{x} = x + \sigma \varepsilon
$$

### Target Score

Using denoising score-matching logic, the target field for Gaussian corruption is:

$$
g^*(\tilde{x}) = - \frac{\varepsilon}{\sigma}
$$

This is also the implementation target score.

### Equilibrium Matching Loss Used Here

The implementation minimizes masked mean squared error between learned score and target score:

$$\mathcal{L}_{\text{equilibrium_matching}} = \mathbb{E}_{x,\varepsilon}\left[\left\|g_\theta(\tilde{x}, c) + \frac{\varepsilon}{\sigma}\right\|^2\right]$$

with masking applied to padded node positions.

Expanded with mask $m$:

$$\mathcal{L}_{\text{equilibrium_matching}} = \frac{\sum_{b,i,d} m_{b,i}\left(g_\theta(\tilde{x}, c)_{b,i,d} + \frac{\varepsilon_{b,i,d}}{\sigma}\right)^2}{\sum_{b,i,d} m_{b,i}}$$

This is the primary generative loss in the current code.

## Denoised Estimate

After learning the score on noisy inputs, the implementation forms a denoised estimate:

$$
\hat{x} = \tilde{x} + \sigma^2 g_\theta(\tilde{x}, c)
$$

This is a feature-wise version of the usual denoising correction.

That estimate is then reused for auxiliary supervised heads.

## Auxiliary And Structural Losses

The Equilibrium Matching generator is not only trained to learn a conditional energy landscape. It also predicts graph-structural properties through supervised heads and soft global consistency terms.

The complete loss is easiest to understand as a sum of three groups:

1. generative score matching,
2. local supervised heads,
3. graph-level soft consistency penalties.

### 1. Equilibrium Matching Score Loss

This is the main generative term:

$$
\mathcal{L}_{\text{equilibrium_matching}}
$$

It teaches the model’s score field to match the denoising score implied by Gaussian corruption.

Operationally:

- acts on every valid feature dimension of every valid node,
- is masked over padded rows,
- is always present.

This term is logged as:

- `equilibrium_matching`

### 2. Node Existence Loss

The existence target is the true node-support mask:

$$
y^{\text{exist}}_{b,i} \in \{0, 1\}
$$

The existence head predicts logits:

$$
\ell^{\text{exist}}_{b,i}
$$

and the implementation applies binary cross-entropy with logits:

$$
\mathcal{L}_{\text{exist}} = \mathrm{BCEWithLogits}(\ell^{\text{exist}}, y^{\text{exist}})
$$

with positive-class reweighting through `exist_pos_weight`.

Important detail:

- this loss is evaluated on all padded slots, not only valid nodes,
- real nodes act as positives,
- padded rows act as true negatives.

That means this term does two jobs:

- teaches occupancy,
- teaches the model not to materialize padded slots.

This term is logged as:

- `exist`

If all training graphs have the same node count and the existence target is constant, the implementation disables the existence head and drops this term.

### 3. Node Count Loss

This is a soft global consistency term built on top of the existence head.

The conditioning vector contains an explicit desired node count:

$$
n^{\text{target}}_b
$$

The model’s existence logits imply an expected number of materialized nodes:

$$
\hat{n}_b = \sum_i \sigma(\ell^{\text{exist}}_{b,i})
$$

The implementation penalizes disagreement with a Huber loss:

$$
\mathcal{L}_{\text{node-count}} =
\mathrm{Huber}(\hat{n}_b, n^{\text{target}}_b)
$$

This term is useful because the per-slot BCE loss does not by itself guarantee that the total occupancy mass matches the desired graph size.

This term is logged as:

- `node_count_loss`

and weighted by:

- `lambda_node_count_importance`

### 4. Degree Classification Loss

The degree target is the true node degree clipped to the supported class range:

$$
y^{\text{deg}}_{b,i} \in \{0, \dots, D_{\max}\}
$$

The degree head predicts logits:

$$
\ell^{\text{deg}}_{b,i} \in \mathbb{R}^{D_{\max}+1}
$$

and the implementation applies masked cross-entropy:

$$
\mathcal{L}_{\text{deg}} =
\mathrm{MaskedCrossEntropy}(\ell^{\text{deg}}, y^{\text{deg}})
$$

Masking means:

- only materialized nodes contribute,
- padded rows do not contribute.

This term is logged as:

- `deg_ce`

and weighted by:

- `lambda_degree_importance`

### 5. Node Label Loss

If node labels are supervised and not collapsed to a constant, the node-label head predicts categorical logits:

$$
\ell^{\text{label}}_{b,i} \in \mathbb{R}^{K}
$$

for encoded node-label targets:

$$
y^{\text{label}}_{b,i} \in \{0, \dots, K-1\}
$$

The loss is masked cross-entropy:

$$
\mathcal{L}_{\text{label}} =
\mathrm{MaskedCrossEntropy}(\ell^{\text{label}}, y^{\text{label}})
$$

Only valid nodes contribute.

This term is logged as:

- `node_label_ce`

and weighted by:

- `lambda_node_label_importance`

If node labels are constant or disabled by the supervision plan, this term is absent.

### 6. Direct Edge Locality Loss

If direct edge supervision is enabled, the model scores selected node pairs with an edge MLP:

$$
\ell^{\text{edge}}_{(i,j)} = f_{\text{edge}}(h_i, h_j)
$$

and applies BCE with logits against direct edge-presence targets:

$$
\mathcal{L}_{\text{edge}} =
\mathrm{BCEWithLogits}(\ell^{\text{edge}}, y^{\text{edge}})
$$

This is the main structural pairwise supervision term used by the decoder path.

This term is logged as:

- `edge_ce`

and weighted by:

- `lambda_direct_edge_importance`

### 7. Edge Count Loss

This is a soft global consistency term on the full soft adjacency field.

From the dense edge-probability matrix:

$$
P_{b,ij}
$$

the implementation forms a symmetrized expected undirected edge count:

$$
\hat{m}_b = \sum_{i < j} \frac{P_{b,ij} + P_{b,ji}}{2}
$$

restricted to currently materialized node slots.

The conditioning vector contains a desired edge count:

$$
m^{\text{target}}_b
$$

and the loss is:

$$
\mathcal{L}_{\text{edge-count}} =
\mathrm{Huber}(\hat{m}_b, m^{\text{target}}_b)
$$

This term encourages the soft edge field to match the requested graph density before the decoder’s discrete optimization stage.

This term is logged as:

- `edge_count_loss`

and weighted by:

- `lambda_edge_count_importance`

### 8. Degree/Edge Handshake Consistency Loss

For any undirected graph, the handshake identity says:

$$
\sum_i \deg(i) = 2 |E|
$$

The implementation turns the degree logits into expected degrees:

$$
\hat{d}_{b,i} = \sum_{k=0}^{D_{\max}} k \cdot \mathrm{softmax}(\ell^{\text{deg}}_{b,i})_k
$$

and forms the expected total degree:

$$
\hat{D}_b = \sum_i \hat{d}_{b,i}
$$

over materialized nodes.

It then compares that to twice the desired edge count:

$$
\mathcal{L}_{\text{deg-edge}} =
\mathrm{Huber}(\hat{D}_b, 2 m^{\text{target}}_b)
$$

This term is not a replacement for degree supervision or edge supervision. It is a soft graph-level compatibility penalty tying the degree head and the edge-count target together.

This term is logged as:

- `degree_edge_consistency_loss`

and weighted by:

- `lambda_degree_edge_consistency_importance`

### 9. Edge Label Loss

If edge labels are supervised and not collapsed to a constant, the edge-label head predicts categorical logits for supervised node pairs:

$$
\ell^{\text{edge-label}}_{(i,j)} \in \mathbb{R}^{C}
$$

and the loss is standard cross-entropy:

$$
\mathcal{L}_{\text{edge-label}} =
\mathrm{CrossEntropy}(\ell^{\text{edge-label}}, y^{\text{edge-label}})
$$

This term is logged as:

- `edge_label_ce`

and weighted by:

- `lambda_edge_label_importance`

### 10. Auxiliary Locality Loss

If higher-horizon locality supervision is enabled, the model uses a second edge MLP to predict auxiliary locality targets for node pairs that are not necessarily direct edges.

The loss is again BCE with logits:

$$
\mathcal{L}_{\text{aux}} =
\mathrm{BCEWithLogits}(\ell^{\text{aux}}, y^{\text{aux}})
$$

This term is intended as representation regularization rather than as the primary decoder-facing edge signal.

This term is logged as:

- `aux_locality_ce`

and weighted by:

- `lambda_auxiliary_edge_importance`

## Total Training Objective

The implementation builds `total_loss` additively from whichever terms are active for the current dataset and supervision plan:

$$
\mathcal{L}_{\text{total}} =
\mathcal{L}_{\text{equilibrium_matching}}
+ \lambda_{\text{deg}} \mathcal{L}_{\text{deg}}
+ \lambda_{\text{exist}} \mathcal{L}_{\text{exist}}
+ \lambda_{\text{node-count}} \mathcal{L}_{\text{node-count}}
+ \lambda_{\text{node-label}} \mathcal{L}_{\text{label}}
+ \lambda_{\text{edge}} \mathcal{L}_{\text{edge}}
+ \lambda_{\text{edge-count}} \mathcal{L}_{\text{edge-count}}
+ \lambda_{\text{deg-edge}} \mathcal{L}_{\text{deg-edge}}
+ \lambda_{\text{edge-label}} \mathcal{L}_{\text{edge-label}}
+ \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}
$$

subject to these activation rules:

- `equilibrium_matching` and `deg` are always present,
- `exist` is present only if the existence head is enabled,
- `node-count` is present only if the existence head is enabled and `lambda_node_count_importance > 0`,
- `node-label` is present only if the node-label head is enabled,
- `edge` is present only if direct locality supervision is enabled,
- `edge-count` is present only if the direct edge head is enabled and `lambda_edge_count_importance > 0`,
- `deg-edge` is present only if `lambda_degree_edge_consistency_importance > 0`,
- `edge-label` is present only if the edge-label head is enabled,
- `aux` is present only if auxiliary locality supervision is enabled.

So the actual total objective for any one run is a dataset- and configuration-dependent subset of the expression above.

## Sampling

Generation does not use diffusion reverse steps. It uses iterative relaxation in the learned score field.

Starting from Gaussian noise:

$$
x_0 \sim \mathcal{N}(0, I)
$$

the model repeatedly updates:

$$
x_{k+1} = x_k + \eta \, g_\theta(x_k, c)
$$

where:

- $\eta$ is `sampling_step_size`,
- $g_\theta(x_k, c) = -\nabla_x \phi_\theta(x_k, c)$.

Because:

$$
g_\theta(x, c) = \nabla_x \log p_\theta(x \mid c)
$$

this moves samples toward higher conditional probability, or equivalently lower energy.

### Optional Langevin Noise

If enabled, the implementation adds stochasticity:

$$x_{k+1} = x_k + \eta g_\theta(x_k, c) + \sqrt{2\eta} \, \alpha \, \xi_k$$

where:

- $\xi_k \sim \mathcal{N}(0, I)$,
- $\alpha$ is `langevin_noise_scale`.

This can improve diversity at the cost of noisier trajectories.

## Final Projection at Inference

After the iterative Equilibrium Matching updates, the model runs one final pass on the final sample and applies auxiliary heads:

- node existence logits are thresholded and overwrite channel 0,
- degree logits are converted to class indices and used to overwrite the degree channel after inverse scaling.
- node-label logits are converted to categorical predictions and stored separately.

This is a practical post-processing step that helps enforce discrete structure.

## Node-Label Supervision Behavior

In the maintained Equilibrium Matching path, node labels are supervised locally through the per-node
categorical head. The graph-level conditioning vector does not include a separate
node-label histogram.

The graph-level conditioning features are:

$$
[n_{\mathrm{nodes}}, 2 \cdot n_{\mathrm{edges}}]
$$

in addition to the learned graph embedding channels.

If the original graph encoding is:

$$
c \in \mathbb{R}^{C}
$$

then the Equilibrium Matching model receives the same width-$C$ graph-level condition tensor, optionally
augmented only by downstream target-guidance channels when classifier-free guidance is enabled.

This keeps the roles separate:

- graph-level conditioning carries global graph context and explicit size channels,
- node-label supervision teaches the model which label each node slot should predict.

if the conditioning input is tokenized.

## Padding and Masking

This is important for correctness.

The Equilibrium Matching implementation explicitly masks padded node positions in several places:

- the latent encoder input,
- energy aggregation,
- Equilibrium Matching score loss,
- existence loss,
- degree loss,
- self-attention through key padding masks,
- query outputs after transformer blocks.

Without this masking, padded rows would act like fake training examples and distort the learned energy landscape.

## Equilibrium Matching Design Characteristics

The most important characteristics of the maintained Equilibrium Matching implementation are:

### 1. No time variable

Diffusion uses:

$$
t
$$

or:

$$
\sigma(t)
$$

and learns a time-dependent denoiser.

Equilibrium Matching here is stationary:

$$
g_\theta(x, c)
$$

with no explicit diffusion time embedding.

### 2. Scalar potential instead of direct denoising head

Diffusion predicts:

$$
\epsilon_\theta(x_t, c, t)
$$

Equilibrium Matching predicts:

$$
\phi_\theta(x, c)
$$

and differentiates it to obtain the score.

### 3. Relaxation sampling instead of reverse diffusion

Diffusion sampling traverses a schedule from noisy to clean.

Equilibrium Matching sampling repeatedly applies:

$$
x \leftarrow x + \eta g_\theta(x, c)
$$

in the stationary score field.

## Relationship to the Paper

This repository implements an explicit-energy conditional Equilibrium Matching-style generator rather than a full paper-faithful reproduction of every Equilibrium Matching formulation detail.

The important design commitments are:

- conservative field by construction,
- score obtained as the gradient of a scalar,
- stationary score matching objective,
- iterative energy-relaxation sampling.

So this implementation is best thought of as:

- Equilibrium Matching-inspired,
- explicit-energy,
- conditional,
- adapted to padded node-feature tensors and downstream graph decoding.

## Main Hyperparameters

### `equilibrium_matching_sigma`

Base noise scale for corruption during training.

Larger values:

- make the model learn broader score behavior,
- can stabilize or oversmooth depending on data.

Smaller values:

- make training more local,
- can lead to sharper but less robust score estimates.

Practical effect:

- increase `equilibrium_matching_sigma` if training is unstable or the learned score field looks too noisy,
- decrease `equilibrium_matching_sigma` if training is stable but generations look overly smooth or generic.

Conceptually, `equilibrium_matching_sigma` changes what score field the model learns, not how sampling uses that field.

### `sampling_step_size`

Controls the magnitude of each Equilibrium Matching update at generation time.

Too small:

- slow movement,
- under-relaxed samples.

Too large:

- unstable trajectories,
- divergence or oscillation.

Practical effect:

- reduce `sampling_step_size` if generations are chaotic, unstable, or overshoot,
- increase it slightly if sampling is too slow or the latent state changes too little from step to step.

Conceptually, `sampling_step_size` changes how aggressively the learned score field is followed at inference time.

### `sampling_steps`

Number of Equilibrium Matching sampling iterations.

### `langevin_noise_scale`

Amount of stochastic noise added during sampling.

- `0.0` gives deterministic relaxation updates for a fixed random seed.
- positive values inject extra exploration and sample diversity.

If it is too small:

- samples may collapse to a narrow mode family,
- diversity may be limited.

If it is too large:

- sampling becomes noisy,
- feasibility and structural fidelity may degrade.

Practical effect:

- keep it at `0.0` when you want stable, reproducible generation,
- raise it slightly when samples are too similar and you want more diversity,
- reduce it again if graph quality or feasibility starts to degrade.

In short:

- `equilibrium_matching_sigma` changes what the model learns,
- `sampling_step_size` changes how hard sampling follows that learned field,
- `langevin_noise_scale` changes how much randomness is injected while following it.

### `lambda_degree_importance`

Weight on degree supervision.

### `lambda_node_exist_importance`

Weight on node existence supervision.

### `lambda_node_count_importance`

Weight on the soft node-count consistency loss.

### `lambda_direct_edge_importance`

Weight on locality supervision.

### `lambda_edge_count_importance`

Weight on the soft edge-count consistency loss.

### `lambda_degree_edge_consistency_importance`

Weight on the soft handshake-consistency loss tying total degree mass to twice the desired edge count.

### `lambda_node_label_importance`

Weight on node-label supervision.

### `lambda_edge_label_importance`

Weight on edge-label supervision.

### `lambda_auxiliary_edge_importance`

Weight on auxiliary higher-horizon locality supervision.

## Training Metrics

The Equilibrium Matching generator records and plots:

- `total`
  Full training objective.
- `equilibrium_matching`
  Core Equilibrium Matching score-matching loss.
- `deg_ce`
  Degree classification loss.
- `exist`
  Node existence BCE loss.
- `locality`
  Optional locality supervision loss.

Additional loss terms may be logged even if they are not plotted by default:

- `node_count_loss`
- `edge_count_loss`
- `degree_edge_consistency_loss`
- `edge_label_ce`
- `aux_locality_ce`

With `verbose=True`, the model plots these at the end of training through `on_train_end()`.

## Typical Training Flow

At a conceptual level, one training step is:

1. start from padded, scaled node features $x$,
2. sample Gaussian noise $\varepsilon$,
3. build noisy features $\tilde{x} = x + s \odot \varepsilon$,
4. encode $(\tilde{x}, c)$ into latent tokens,
5. compute scalar potential $\phi_\theta(\tilde{x}, c)$,
6. differentiate to get score $g_\theta(\tilde{x}, c)$,
7. minimize score-matching loss against $-\varepsilon / s$,
8. form denoised estimate $\hat{x}$,
9. compute degree and existence losses on $\hat{x}$,
10. add optional locality supervision.

## Typical Sampling Flow

1. sample initial node features from Gaussian noise,
2. repeatedly compute the conditional score field,
3. update the sample in the score direction,
4. optionally add Langevin noise,
5. run a final structural projection using existence and degree heads,
6. inverse-transform back to original feature scale.

## Practical Notes

### 1. The model still depends on good preprocessing

The Equilibrium Matching model is sensitive to scaling because it operates directly in feature space and uses gradients with respect to inputs.

### 2. Sampling hyperparameters matter a lot

If generations are poor, `sampling_step_size`, `sampling_steps`, and `equilibrium_matching_sigma` are the first places to look.

### 3. The auxiliary heads matter

Without the existence and degree heads, the Equilibrium Matching field alone may produce softer outputs that are harder for the downstream decoder to interpret structurally.

### 4. Guidance is not implemented in phase 1

Classifier guidance is not part of the maintained Equilibrium Matching decompositional API.

## Source Notes

The core intuition follows the Equilibrium Matching direction described by:

- Wang and Du, *Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models*

The denoising score-matching identity behind the Gaussian corruption target is closely related to:

- Vincent, *A Connection Between Score Matching and Denoising Autoencoders*

This repository adapts those ideas into a conditional node-feature generator for graph generation.

## Glossary

### Auxiliary head

A supervised prediction layer attached to the latent representation, used here for node existence, degree, and locality.

### Condition or conditioning vector

Graph-level information supplied to the Equilibrium Matching generator so it can produce node features for a desired graph context.

### Conservative field

A vector field that can be written as the gradient of a scalar potential.

### Degree channel

The feature dimension designated to represent node degree. In this repo the default index is `1`.

### Energy

A scalar function $E(x, c)$ whose negative gradient defines the score or force field driving samples toward higher probability regions.

### Equilibrium Matching

Equilibrium Matching. In this repo it refers to a stationary, energy-based score-learning approach used instead of diffusion.

### Existence channel

The feature dimension used to indicate whether a node slot is active. In this repo the default signal is feature channel `0`.

### Graph-level condition token

A projected conditioning vector used by the transformer backbone as cross-attention memory.

### Integrable field

A vector field that is the gradient of some scalar function. Here this is enforced by differentiating a scalar potential.

### Langevin noise

Gaussian noise injected during sampling updates to encourage exploration and approximate stochastic sampling rather than purely deterministic relaxation.

### Locality supervision

Optional training labels on node pairs indicating whether they should be considered locally connected or related.

### Mask

A boolean tensor indicating which node positions are real and which are padding.

### Potential

The scalar function $\phi_\theta(x, c)$ implemented by the Equilibrium Matching module. In this code it acts as the conditional energy.

### Score

The gradient of log-density with respect to input features. In this implementation:

$$
g_\theta(x, c) = -\nabla_x \phi_\theta(x, c)
$$

### Stationary field

A field without an explicit diffusion-time variable. The same field is used throughout training and sampling.

### Transformer backbone

The stack of attention blocks that combines node features with graph-level conditioning information before potential and auxiliary heads are applied.
