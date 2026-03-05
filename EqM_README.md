# EqM Decompositional Node Generator

This document explains the `EqMDecompositionalNodeGenerator` implemented in this repository, how it differs from the diffusion-based generator, what equations it uses during training and sampling, and how the surrounding graph-generation pipeline fits together.

The implementation lives primarily in
[`node_engine.py`](eqm_decompositional_graph_generator/node_engine.py).

## Overview

The repo now contains an archived diffusion generator and one maintained EqM generator:

- `ConditionalNodeGenerator`
  A diffusion-style conditional denoising model.
- `EqMDecompositionalNodeGenerator`
  A stationary, energy-based conditional generator inspired by Equilibrium Matching (EqM).

The EqM model replaces diffusion time-conditioning and reverse-time denoising with:

- a scalar conditional energy or potential,
- a score field obtained as the gradient of that scalar,
- a stationary score-matching objective,
- iterative relaxation or Langevin-style sampling in feature space.

In this repository, the EqM generator is used as the `conditioning -> node embeddings` stage inside the broader decompositional pipeline:

1. encode each training graph into node feature matrices and graph-level conditioning vectors,
2. train the EqM conditional generator to map graph-level conditions to node-level embeddings,
3. train downstream classifiers and decoders that map node embeddings back into graphs.

## High-Level Idea

Diffusion models learn a time-dependent denoising field:

$$
\epsilon_\theta(x_t, c, t)
$$

or equivalently a time-dependent score field.

The EqM model implemented here learns a stationary conditional score field:

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

The EqM module does not directly output node features. It outputs:

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

## Inputs and Outputs

### Inputs

For a batch of graphs, the EqM module consumes:

- `input_examples`
  Shape $(B, N, D)$, padded node features.
- `global_condition`
  Shape $(B, C)$ or $(B, M, C)$, graph-level conditioning vectors or tokens.
- `node_label_histogram`
  Shape $(B, K)$, a normalized histogram over the $K$ categorical node-label classes.
  In this repository, the histogram is concatenated to the graph-level conditioning vector
  before scaling, so the effective conditioning input becomes:

$$
c' = [c ; h]
$$

  where:

$$
h_k = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}[y_i^{\text{label}} = k]
$$

  and $n$ is the number of valid nodes in the graph.
- `node_mask`
  Shape $(B, N)$, boolean mask indicating valid node slots.

At training time, the histogram is computed from the true graph node labels.
Once the label vocabulary is fitted, calling:

```python
decompositional_encoder_decoder.graph_encode(graphs)
```

returns conditioning vectors that already include the concatenated node-label histogram.
The intended generation path is therefore to reuse conditioning vectors extracted from
real graphs, with the histogram already embedded in them.

### Outputs

At inference time, the wrapper returns:

- a list of generated node-feature matrices in original feature scale,
- optionally with degree channel overwritten by the auxiliary degree head,
- and node existence channel snapped using the existence head.

Additionally, if node-label supervision is enabled, the wrapper stores the per-node
predicted categorical labels from the final latent state in:

```python
eqm_generator.last_predicted_node_label_classes_
```

These labels are not written directly into the node-feature tensor because node labels
are categorical metadata, not continuous feature channels in the EqM state.

## Data Preprocessing

The wrapper preserves the diffusion API and preprocessing behavior:

- node-feature tensors are padded to a common maximum node count,
- node features are scaled with `MinMaxScaler`,
- conditional features are scaled separately,
- degree scaling statistics are stored so discrete degree labels can be recovered,
- padded positions are masked during EqM training and attention.

This matters because the EqM model operates in scaled feature space, while the downstream decoder expects outputs back in the original feature space.

## Model Architecture

The EqM generator reuses the same general conditional transformer backbone style as the diffusion model:

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

## Core EqM Training Objective

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

More precisely in this implementation, the corruption is feature-wise:

$$
\tilde{x} = x + s \odot \varepsilon
$$

where $s$ is a per-feature noise scale tensor. The degree channel uses reduced noise:

$$
s_d = \frac{\sigma}{f_{\mathrm{deg}}}
$$

where $f_{\mathrm{deg}}$ is `noise_degree_factor`.

while other channels use:

$$
s = \sigma
$$

### Target Score

Using denoising score-matching logic, the target field for Gaussian corruption is:

$$
g^*(\tilde{x}) = - \frac{\varepsilon}{\sigma}
$$

In the feature-wise version used here:

$$
g^*(\tilde{x}) = - \frac{\varepsilon}{s}
$$

where the division is elementwise.

### EqM Loss Used Here

The implementation minimizes masked mean squared error between learned score and target score:

$$\mathcal{L}_{\text{eqm}} = \mathbb{E}_{x,\varepsilon}\left[\left\|g_\theta(\tilde{x}, c) + \frac{\varepsilon}{s}\right\|^2\right]$$

with masking applied to padded node positions.

Expanded with mask $m$:

$$\mathcal{L}_{\text{eqm}} = \frac{\sum_{b,i,d} m_{b,i}\left(g_\theta(\tilde{x}, c)_{b,i,d} + \frac{\varepsilon_{b,i,d}}{s_{b,i,d}}\right)^2}{\sum_{b,i,d} m_{b,i}}$$

This is the primary generative loss in the current code.

## Denoised Estimate

After learning the score on noisy inputs, the implementation forms a denoised estimate:

$$
\hat{x} = \tilde{x} + s^2 \odot g_\theta(\tilde{x}, c)
$$

This is a feature-wise version of the usual denoising correction.

That estimate is then reused for auxiliary supervised heads.

## Auxiliary Losses

The EqM generator is not only trained to learn a conditional energy landscape. It also predicts graph-structural properties through supervised heads.

### Node Existence Loss

The existence target is derived from feature channel 0:

$$
y^{\text{exist}}_{b,i} = \mathbf{1}[x_{b,i,0} \ge 0.5]
$$

The model predicts logits:

$$
\ell^{\text{exist}}_{b,i}
$$

and the masked BCE loss is:

$$
\mathcal{L}_{\text{exist}}
=
\mathrm{MaskedBCEWithLogits}(\ell^{\text{exist}}, y^{\text{exist}})
$$

with class weighting via `exist_pos_weight`.

### Degree Loss

The degree target comes from the designated degree channel, inverse-scaled back to original degree space, rounded to a class index, and clipped:

$$
y^{\text{deg}} = \mathrm{clip}(\mathrm{round}(\mathrm{unscale}(x_{\text{deg}})), 0, D_{\max})
$$

The degree head produces:

$$
\ell^{\text{deg}} \in \mathbb{R}^{D_{\max}+1}
$$

and the masked classification loss is:

$$
\mathcal{L}_{\text{deg}}
=
\mathrm{MaskedCrossEntropy}(\ell^{\text{deg}}, y^{\text{deg}})
$$

### Node Label Loss

Node labels are categorical and are supervised directly from the graph node attribute
`label`.

Let:

$$
y^{\text{label}}_{b,i} \in \{0, \dots, K-1\}
$$

be the encoded node-label class for node $i$ in graph $b$, and let the node-label
head produce logits:

$$
\ell^{\text{label}}_{b,i} \in \mathbb{R}^{K}
$$

The masked classification loss is:

$$
\mathcal{L}_{\text{label}}
=
\mathrm{MaskedCrossEntropy}(\ell^{\text{label}}, y^{\text{label}})
$$

This loss is applied only on valid node positions, using the same node mask used by
the EqM loss.

The implementation logs this term as `label_ce`.

### Locality Supervision Loss

If edge supervision is enabled, pairs of node latents are scored by an edge MLP:

$$
\ell^{\text{edge}}_{(i,j)} = f_{\text{edge}}(h_i, h_j)
$$

and trained with binary cross-entropy against supplied locality labels:

$$
\mathcal{L}_{\text{edge}}
=
\mathrm{BCEWithLogits}(\ell^{\text{edge}}, y^{\text{edge}})
$$

## Total Training Objective

The full objective used in the implementation is:

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{eqm}}
+
\lambda_{\text{exist}} \mathcal{L}_{\text{exist}}
+
\lambda_{\text{deg}} \mathcal{L}_{\text{deg}}
+
\lambda_{\text{label}} \mathcal{L}_{\text{label}}
+
\lambda_{\text{local}} \mathcal{L}_{\text{edge}}
$$

when locality supervision is enabled.

Otherwise:

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{eqm}}
+
\lambda_{\text{exist}} \mathcal{L}_{\text{exist}}
+
\lambda_{\text{deg}} \mathcal{L}_{\text{deg}}
+
\lambda_{\text{label}} \mathcal{L}_{\text{label}}
$$

If the dataset has a degenerate fixed-size existence target, the implementation disables
the existence head automatically and drops $\mathcal{L}_{\text{exist}}$ from the
effective objective.

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

$$
x_{k+1}
=
x_k + \eta g_\theta(x_k, c)
+
\sqrt{2\eta} \, \alpha \, \xi_k
$$

where:

- $\xi_k \sim \mathcal{N}(0, I)$,
- $\alpha$ is `langevin_noise_scale`.

This can improve diversity at the cost of noisier trajectories.

## Final Projection at Inference

After the iterative EqM updates, the model runs one final pass on the final sample and applies auxiliary heads:

- node existence logits are thresholded and overwrite channel 0,
- degree logits are converted to class indices and used to overwrite the degree channel after inverse scaling.
- node-label logits are converted to categorical predictions and stored separately.

This is a practical post-processing step that helps enforce discrete structure.

## Histogram Conditioning Behavior

The important modeling idea is that node-label composition is exposed twice:

1. globally, through the label histogram in the conditioning vector,
2. locally, through the per-node categorical label head.

So the EqM stage is trained to do both:

- respond to a requested label mixture at graph level,
- organize node latents so each valid node also carries a predictable discrete label.

In the current code, the first two graph-level conditioning features are already:

$$
[n_{\mathrm{nodes}}, 2 \cdot n_{\mathrm{edges}}]
$$

and the label histogram is appended after those existing graph features, not used as a
replacement for them.

If the original graph encoding is:

$$
c \in \mathbb{R}^{C}
$$

and the histogram has width $K$, then the EqM model actually receives:

$$
c' \in \mathbb{R}^{C + K}
$$

or tokenwise:

$$
c'_{b,m} = [c_{b,m} ; h_b]
$$

if the conditioning input is tokenized.

In other words, the supported path in the current code is:

1. `graph_encode(graphs)`
   This extracts graph features, node counts, and edge counts from a real graph.

`EqMDecompositionalNodeGenerator` no longer relies on an embedded node-label histogram
inside the conditioning vector. Label histograms are evaluated after generation rather
than injected directly as conditioning.

## Padding and Masking

This is important for correctness.

The EqM implementation explicitly masks padded node positions in several places:

- the latent encoder input,
- energy aggregation,
- EqM score loss,
- existence loss,
- degree loss,
- self-attention through key padding masks,
- query outputs after transformer blocks.

Without this masking, padded rows would act like fake training examples and distort the learned energy landscape.

## Difference from the Diffusion Generator

The most important differences from `ConditionalNodeGenerator` are:

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

EqM here is stationary:

$$
g_\theta(x, c)
$$

with no explicit diffusion time embedding.

### 2. Scalar potential instead of direct denoising head

Diffusion predicts:

$$
\epsilon_\theta(x_t, c, t)
$$

EqM predicts:

$$
\phi_\theta(x, c)
$$

and differentiates it to obtain the score.

### 3. Relaxation sampling instead of reverse diffusion

Diffusion sampling traverses a schedule from noisy to clean.

EqM sampling repeatedly applies:

$$
x \leftarrow x + \eta g_\theta(x, c)
$$

in the stationary score field.

## Relationship to the Paper

This repository implements an explicit-energy conditional EqM-style generator rather than a full paper-faithful reproduction of every EqM formulation detail.

The important design commitments are:

- conservative field by construction,
- score obtained as the gradient of a scalar,
- stationary score matching objective,
- iterative energy-relaxation sampling.

So this implementation is best thought of as:

- EqM-inspired,
- explicit-energy,
- conditional,
- adapted to padded node-feature tensors and downstream graph decoding.

## Main Hyperparameters

### `eqm_sigma`

Base noise scale for corruption during training.

Larger values:

- make the model learn broader score behavior,
- can stabilize or oversmooth depending on data.

Smaller values:

- make training more local,
- can lead to sharper but less robust score estimates.

Practical effect:

- increase `eqm_sigma` if training is unstable or the learned score field looks too noisy,
- decrease `eqm_sigma` if training is stable but generations look overly smooth or generic.

Conceptually, `eqm_sigma` changes what score field the model learns, not how sampling uses that field.

### `noise_degree_factor`

Reduces corruption on the degree feature channel.

This is useful because degree is often more discrete and semantically fragile than continuous latent channels.

### `sampling_step_size`

Controls the magnitude of each EqM update at generation time.

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

Number of EqM sampling iterations.

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

- `eqm_sigma` changes what the model learns,
- `sampling_step_size` changes how hard sampling follows that learned field,
- `langevin_noise_scale` changes how much randomness is injected while following it.

### `lambda_degree_importance`

Weight on degree supervision.

### `lambda_node_exist_importance`

Weight on node existence supervision.

### `lambda_locality_importance`

Weight on locality supervision.

## Training Metrics

The EqM generator records and plots:

- `total`
  Full training objective.
- `eqm`
  Core EqM score-matching loss.
- `deg_ce`
  Degree classification loss.
- `exist`
  Node existence BCE loss.
- `locality`
  Optional locality supervision loss.

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

The EqM model is sensitive to scaling because it operates directly in feature space and uses gradients with respect to inputs.

### 2. Sampling hyperparameters matter a lot

If generations are poor, `sampling_step_size`, `sampling_steps`, and `eqm_sigma` are the first places to look.

### 3. The auxiliary heads matter

Without the existence and degree heads, the EqM field alone may produce softer outputs that are harder for the downstream decoder to interpret structurally.

### 4. Guidance is not implemented in phase 1

Classifier guidance is not part of the maintained EqM decompositional API.

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

Graph-level information supplied to the EqM generator so it can produce node features for a desired graph context.

### Conservative field

A vector field that can be written as the gradient of a scalar potential.

### Degree channel

The feature dimension designated to represent node degree. In this repo the default index is `1`.

### Energy

A scalar function $E(x, c)$ whose negative gradient defines the score or force field driving samples toward higher probability regions.

### EqM

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

The scalar function $\phi_\theta(x, c)$ implemented by the EqM module. In this code it acts as the conditional energy.

### Score

The gradient of log-density with respect to input features. In this implementation:

$$
g_\theta(x, c) = -\nabla_x \phi_\theta(x, c)
$$

### Stationary field

A field without an explicit diffusion-time variable. The same field is used throughout training and sampling.

### Transformer backbone

The stack of attention blocks that combines node features with graph-level conditioning information before potential and auxiliary heads are applied.
