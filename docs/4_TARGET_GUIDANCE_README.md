# Target Guidance

This document explains the two target-guidance routes supported by NodeField:

- classifier-free guidance (CFG) over explicit target-conditioning channels
- separate post-hoc guidance through an auxiliary classifier or regressor

These are both first-class features, but they are intentionally exposed through different APIs because they operate differently during training and sampling.

It complements:

- [`2_CONDITIONAL_NODE_FIELD_README.md`](2_CONDITIONAL_NODE_FIELD_README.md)
- [`2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md`](2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md)
- [`1_CONDITIONAL_NODE_FIELD_GRAPH_GENERATOR_README.md`](1_CONDITIONAL_NODE_FIELD_GRAPH_GENERATOR_README.md)
- [`5_MAIN_CLASS_INTERFACES_README.md`](5_MAIN_CLASS_INTERFACES_README.md)

## Two Supported Approaches

### 1. Classifier-Free Guidance

CFG trains the generator itself to operate with and without the requested target condition, then mixes those two score evaluations at sampling time.

Conceptually:

- conditional branch: the graph condition plus the requested target
- unconditional branch: the same graph condition, but with the target-conditioning slice nulled out

If the ordinary graph condition is:

$$
c
$$

and the optional target-conditioning channels are:

$$
t
$$

then the model is trained on:

$$
[\; c, t \;]
$$

and on the null-target branch:

$$
[\; c, 0 \;]
$$

The important design choice is that the graph-level condition remains intact. Only the optional target slice is dropped.

### 2. Separate Post-Hoc Guidance

Post-hoc guidance leaves the generator sampling model unchanged and instead adds a gradient from a separately trained predictor during sampling.

For classification, the guided score can be viewed as:

```math
g_{\mathrm{guided}}(x, c, y)
\approx
g_\theta(x, c)
+
\lambda \nabla_x \log p_\psi(y \mid x, c)
```

For regression, the same idea becomes:

```math
g_{\mathrm{guided}}(x, c, y^\star)
\approx
g_\theta(x, c)
- \lambda \nabla_x \ell(r_\psi(x, c), y^\star)
```

where:

- $g_\theta$ is the generator score
- $p_\psi$ is a separate classifier
- $r_\psi$ is a separate regressor
- $\lambda$ is the guidance strength

The generator is pretrained normally, and the extra predictor is attached afterward.

## Why Both Exist

CFG is useful when:

- the target should be part of the generator's native conditioning interface
- the conditional/unconditional branches should be learned jointly by the generator
- the target semantics are part of the main generation problem

Separate post-hoc guidance is useful when:

- a pretrained generator should be reused for multiple downstream targets
- you want to add new property objectives later without retraining the generator
- the guidance objective is task-specific and modular

## CFG Training Behavior

CFG is activated only when:

- explicit guidance targets are passed during `fit(...)`
- `cfg_target_mode` was configured explicitly as `"classification"` or `"regression"`

At training time, the implementation:

- determines the target-conditioning width from the encoded targets
- appends that target encoding to the graph-level condition vector
- randomly drops the target slice on some examples using `cfg_condition_dropout_prob`
- uses `cfg_null_target_strategy` for the null-target branch

This teaches the generator to evaluate both:

$$
g_\theta(x, [c, t])
$$

and:

$$
g_\theta(x, [c, 0])
$$

with one shared model.

## CFG Sampling Behavior

At inference time, CFG is used only when:

- the generator was trained with CFG support
- `desired_target` is passed
- `guidance_scale` is nonnegative

The implementation combines branches as:

```math
g_{\mathrm{cfg}}(x, c, t)
=
g_\theta(x, [c, 0])
+
s \left(
g_\theta(x, [c, t]) - g_\theta(x, [c, 0])
\right)
```

where:

- $s$ is `guidance_scale`

Operational interpretation:

- `guidance_scale = 0` gives the unconditional branch
- `guidance_scale = 1` gives the ordinary conditional branch
- `guidance_scale > 1` amplifies the target-driven component

## Post-Hoc Guidance Training Behavior

Post-hoc guidance does not change generator training.

Instead, after the generator is already fitted:

- configure a guidance predictor
- train it as either a classifier or regressor
- use its gradient only at sampling time

This is why post-hoc guidance is more modular but also depends on the quality of an extra predictor model.

## Public API Split

The implementation keeps the two guidance routes separate in the public API.

### CFG API

CFG uses the ordinary target-conditioning path:

- configure with `cfg_target_mode="classification"` or `cfg_target_mode="regression"`
- node generator:
  - `fit(..., targets=...)`
  - `predict(..., desired_target=..., guidance_scale=...)`
- graph generator:
  - `fit(..., targets=...)`
  - `decode(..., desired_target=..., guidance_scale=...)`
  - `sample(..., desired_target=..., guidance_scale=...)`
  - `conditional_sample(..., desired_target=..., guidance_scale=...)`

### Post-Hoc Guidance API

Post-hoc guidance uses a separate predictor path:

- generic predictor setup:
  - `set_guidance_predictor(...)`
  - `train_guidance_predictor(...)`
- classifier aliases:
  - `set_guidance_classifier(...)`
  - `train_guidance_classifier(...)`
- node generator:
  - `predict_classifier_guided(..., desired_class=..., classifier_scale=...)`
  - `predict_regression_guided(..., desired_target=..., predictor_scale=...)`
- graph generator:
  - `decode_classifier_guided(...)`
  - `sample_classifier_guided(...)`
  - `conditional_sample_classifier_guided(...)`
  - `decode_regression_guided(...)`
  - `sample_regression_guided(...)`
  - `conditional_sample_regression_guided(...)`

## Important Constraint

The implementation supports both guidance strategies, but not both in the same sampling call.

Choose either:

- CFG through `desired_target` plus `guidance_scale`
- post-hoc classifier/regression guidance through the predictor-specific guided methods

They are intentionally not blended in one public call because that would blur semantics and make behavior harder to reason about.

## Practical Summary

Use CFG when:

- the target belongs inside the generator’s core conditioning interface
- you want joint conditional/unconditional training

Use post-hoc guidance when:

- the generator should stay generic
- downstream steering objectives should remain modular

For exact constructor and method signatures, see:

- [`5_MAIN_CLASS_INTERFACES_README.md`](5_MAIN_CLASS_INTERFACES_README.md)
