# Conditional Node Field Optimization, Hyperparameters, and Metrics

This companion document collects the practical optimization-facing details of the Conditional Node Field model:

- main training and sampling hyperparameters
- lambda interpretation after semantic loss normalization
- recorded metrics
- verbose epoch-summary semantics
- worked examples of raw weighted loss contributions

The conceptual model remains in
[`2_CONDITIONAL_NODE_FIELD_README.md`](2_CONDITIONAL_NODE_FIELD_README.md), and the training-loss
definitions remain in
[`2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md`](2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md).

## Documentation Map

[`2_CONDITIONAL_NODE_FIELD_README.md`](2_CONDITIONAL_NODE_FIELD_README.md)

This is the conceptual reference for the Conditional Node Field model itself: energy-based interpretation, conditioning layouts, architecture, and design rationale.

[`2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md`](2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md)

This is the training-loss reference: auxiliary and structural losses, the total objective, sampling updates, inference-time projection, and padding/masking behavior.

[`2D_TARGET_GUIDANCE_README.md`](2D_TARGET_GUIDANCE_README.md)

This is the dedicated reference for the two supported target-guidance routes: classifier-free guidance (CFG) and separate post-hoc guidance through an auxiliary classifier or regressor.

[`4_MAIN_CLASS_INTERFACES_README.md`](4_MAIN_CLASS_INTERFACES_README.md)

This is the API reference for the main public classes and their parameters.

## Main Hyperparameters

### `node_field_sigma`

Base noise scale for corruption during training.

Larger values:

- make the model learn broader score behavior,
- can stabilize or oversmooth depending on data.

Smaller values:

- make training more local,
- can lead to sharper but less robust score estimates.

Practical effect:

- increase `node_field_sigma` if training is unstable or the learned score field looks too noisy,
- decrease it if training is stable but generations look overly smooth or generic.

Conceptually, `node_field_sigma` changes what score field the model learns, not how sampling uses that field.

### `sampling_step_size`

Controls the magnitude of each Conditional Node Field update at generation time.

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

Number of Conditional Node Field sampling iterations.

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

- `node_field_sigma` changes what the model learns,
- `sampling_step_size` changes how hard sampling follows that learned field,
- `langevin_noise_scale` changes how much randomness is injected while following it.

### `lambda_degree_importance`

Weight on degree supervision. With the current normalization, `1.0` is a reasonable default starting point.

### `lambda_node_exist_importance`

Weight on node existence supervision. With the current normalization, `1.0` is a reasonable default starting point.

### `lambda_node_count_importance`

Weight on the soft node-count consistency loss. Because this is now a relative graph-level correction term, `0.1` to `0.5` is a reasonable starting range.

### `lambda_direct_edge_importance`

Weight on locality supervision. With the current normalization, `1.0` is a reasonable default starting point.

### `lambda_edge_count_importance`

Weight on the soft edge-count consistency loss. Because this is now a relative graph-level correction term, `0.1` to `0.5` is a reasonable starting range.

### `lambda_degree_edge_consistency_importance`

Weight on the soft handshake-consistency loss tying total degree mass to twice the desired edge count. `0.1` to `0.5` is a reasonable starting range.

### `lambda_node_label_importance`

Weight on node-label supervision. With the current normalization, `1.0` is a reasonable default starting point.

### `lambda_edge_label_importance`

Weight on edge-label supervision. With the current normalization, `1.0` is a reasonable default starting point.

### `lambda_auxiliary_edge_importance`

Weight on auxiliary higher-horizon locality supervision. As a regularizer, it often starts well around `0.25` to `0.5`.

## Training Metrics

The Conditional Node Field generator records and plots:

- `total`
  Full training objective.
- `node_field`
  Core Conditional Node Field score-matching loss.
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

## Semantically Normalized Optimization Losses and Epoch-Summary Percentages

There is an important distinction between:

- the raw losses that are actually optimized and logged internally, and
- the fact that the implementation now normalizes the main loss families to more stable semantic units, but some terms can still differ in scale because they supervise different concepts.

The optimization itself is unchanged. `total_loss` is still built from the raw terms:

```math
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{node\_field}}
+ \lambda_{\mathrm{deg}} \mathcal{L}_{\mathrm{deg}}
+ \lambda_{\mathrm{exist}} \mathcal{L}_{\mathrm{exist}}
+ \cdots
```

Those are the values that drive backpropagation, checkpoint selection, early stopping, and the
verbose epoch-summary percentages.

### Why the lambdas are now more interpretable

The node-field loss is computed as:

```math
\mathcal{L}_{\mathrm{node\_field}}
=
\frac{
\sum_{b,n,d}
\mathbf{1}_{\mathrm{active}}(b,n)
\left(g_\theta(\tilde{x}_{bnd}, c_b) + \varepsilon_{bnd}/s_{bnd}\right)^2
}{
D \sum_{b,n}
\mathbf{1}_{\mathrm{active}}(b,n)
}
```

So `node_field` is now a per-active-node, per-feature loss.
Likewise:

- `deg_ce` is a per-active-node classification loss,
- `exist` is a per-slot BCE loss,
- `node_label_ce` is a per-labeled-node loss,
- `edge_ce` and `edge_label_ce` are per-supervised-pair losses,
- `node_count_loss`, `edge_count_loss`, and `degree_edge_consistency_loss` are normalized graph-level relative consistency losses.

That means the lambdas now behave much more like conceptual priorities and much less like hidden compensations for graph size or feature dimension.

So a printed line such as:

```text
train total= 9535.4 | node_field 25.0 [0.3%] | deg 1132.5 [11.9%] | edge 3716.6 [39.0%] | ...
```

is numerically correct as a decomposition of the raw weighted objective. The percentages and the
`dominant=...` label in the verbose epoch summary are computed from these same raw weighted terms.

### Explicit worked example

Suppose the training code has these raw weighted values at one epoch:

- `node_field = 25.0`
- `deg_ce = 1132.5`
- `exist = 422.4`
- `node_count_loss = 505.6`
- `node_label_ce = 867.7`
- `edge_label_ce = 1308.7`
- `edge_ce = 3716.6`
- `edge_count_loss = 1202.2`
- `degree_edge_consistency_loss = 354.7`

Raw total:

```math
25.0 + 1132.5 + 422.4 + 505.6 + 867.7 + 1308.7 + 3716.6 + 1202.2 + 354.7 \approx 9535.4
```

If you formed raw percentages from that total, you would get:

- node field: `25.0 / 9535.4 ~ 0.3%`
- deg: `1132.5 / 9535.4 ~ 11.9%`
- exist: `422.4 / 9535.4 ~ 4.4%`
- node count: `505.6 / 9535.4 ~ 5.3%`
- node label: `867.7 / 9535.4 ~ 9.1%`
- edge label: `1308.7 / 9535.4 ~ 13.7%`
- edge: `3716.6 / 9535.4 ~ 39.0%`
- edge count: `1202.2 / 9535.4 ~ 12.6%`
- degree/edge consistency: `354.7 / 9535.4 ~ 3.7%`

Those raw percentages are mathematically correct and they do indicate the actual contribution of
each weighted term to the optimized `total_loss`. Because the losses are now normalized closer to
their semantic units, those percentages are much more interpretable than before.

### How to interpret the metrics now

- `train_total` and `val_total`
  Still refer to the raw optimized objective.

- `train_node_field` and `val_node_field`
  Refer to the raw score-matching loss.

- the verbose epoch line
  Uses raw weighted values.

- the percentages in that line
  Are raw optimization shares.

So the safest interpretation is:

- use `train_total` / `val_total` for optimization and checkpointing meaning,
- use the printed component percentages as the actual weighted contribution of each displayed term to the raw optimized objective,
- use the lambdas as conceptual weights, with `1.0` as the natural first thing to try for per-node and per-pair terms.
