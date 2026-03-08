# Main Class Interfaces

This document is the explicit API reference for the main public classes in the Conditional Node Field graph-generation stack.

It complements the architecture documents:

- [`2_CONDITIONAL_NODE_FIELD_README.md`](2_CONDITIONAL_NODE_FIELD_README.md)
- [`2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md`](2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md)
- [`1_CONDITIONAL_NODE_FIELD_GRAPH_GENERATOR_README.md`](1_CONDITIONAL_NODE_FIELD_GRAPH_GENERATOR_README.md)
- [`3_CONDITIONAL_NODE_FIELD_GRAPH_DECODER_README.md`](3_CONDITIONAL_NODE_FIELD_GRAPH_DECODER_README.md)
- [`4_TARGET_GUIDANCE_README.md`](4_TARGET_GUIDANCE_README.md)

The conceptual discussion of the conditioning interface itself, including single-vector conditioning
versus token-sequence conditioning under cross-attention, lives in
[`2_CONDITIONAL_NODE_FIELD_README.md`](2_CONDITIONAL_NODE_FIELD_README.md).

The detailed training, loss, and metrics discussion lives in
[`2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md`](2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md).

The goal here is practical:

- show the main public interfaces in one place,
- explain what each parameter means,
- explain what usually happens when you increase or decrease that parameter.

## Scope

The main public classes are:

- `GraphConditioningBatch`
- `NodeGenerationBatch`
- `GeneratedNodeBatch`
- `ConditionalNodeFieldGenerator`
- `ConditionalNodeFieldGraphDecoder`
- `ConditionalNodeFieldGraphGenerator`

This document focuses on public constructors and public workflow methods. Internal helper methods are intentionally omitted.

## Batch Containers

These dataclasses are the explicit data contracts between the orchestration layer, the node generator, and the graph decoder.

### `GraphConditioningBatch`

Used for node generation and graph decoding.

```python
GraphConditioningBatch(
    graph_embeddings: np.ndarray,
    node_counts: np.ndarray,
    edge_counts: np.ndarray,
)
```

Parameters:

- `graph_embeddings`
  Graph-level conditioning embeddings, usually shape `(B, G)` where `B` is batch size.
  Increase embedding dimension: more capacity to encode graph context, but also more model capacity and more risk of overfitting.
  Decrease embedding dimension: simpler conditioning, lower memory, but less descriptive graph context.

- `node_counts`
  Integer node-count targets per graph, usually shape `(B,)`.
  Larger values request larger decoded graphs.
  Smaller values request smaller decoded graphs.

- `edge_counts`
  Integer edge-count targets per graph, usually shape `(B,)`.
  Larger values bias generation toward denser graphs.
  Smaller values bias generation toward sparser graphs.

### `NodeGenerationBatch`

Used for training `ConditionalNodeFieldGenerator`.

```python
NodeGenerationBatch(
    node_embeddings_list: List[np.ndarray],
    node_presence_mask: np.ndarray,
    node_degree_targets: np.ndarray,
    node_label_targets: Optional[List[np.ndarray]] = None,
    edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
    edge_targets: Optional[np.ndarray] = None,
    edge_label_pairs: Optional[List[Tuple[int, int, int]]] = None,
    edge_label_targets: Optional[np.ndarray] = None,
    auxiliary_edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
    auxiliary_edge_targets: Optional[np.ndarray] = None,
)
```

Parameters:

- `node_embeddings_list`
  Per-graph node-feature matrices before padding.
  Higher feature dimension can carry richer node information but increases model size and training difficulty.
  Lower feature dimension is easier to fit but may lose structural or semantic information.

- `node_presence_mask`
  Boolean mask of active node slots after padding, usually shape `(B, N)`.
  More `True` entries correspond to larger effective graphs.
  More `False` entries correspond to more padding and less active structure.

- `node_degree_targets`
  Integer degree targets per node slot, usually shape `(B, N)`.
  Higher degree targets encourage denser local connectivity.
  Lower degree targets encourage sparser local connectivity.

- `node_label_targets`
  Optional categorical node labels for learned node-label prediction.
  More label diversity makes the task richer but harder.
  Less label diversity can make the label head trivial or unnecessary.

- `edge_pairs`
  Optional `(graph_index, i, j)` training pairs for direct edge supervision.
  More pairs give stronger edge supervision but increase training cost.
  Fewer pairs reduce cost but weaken edge-learning signals.

- `edge_targets`
  Optional binary labels aligned with `edge_pairs`.
  More positive edges strengthen direct adjacency learning.
  More negatives push the model toward conservative edge prediction.

- `edge_label_pairs`
  Optional `(graph_index, i, j)` pairs for edge-label supervision.
  More pairs give stronger edge-label learning signal.
  Fewer pairs reduce supervision strength.

- `edge_label_targets`
  Optional categorical edge labels aligned with `edge_label_pairs`.
  More label classes increase semantic coverage but also difficulty.
  Fewer classes simplify the task.

- `auxiliary_edge_pairs`
  Optional extra supervision pairs, typically for longer-horizon locality.
  More auxiliary pairs can stabilize structural learning.
  Too many can distract from the main direct-edge objective.

- `auxiliary_edge_targets`
  Optional labels aligned with `auxiliary_edge_pairs`.
  Higher positive rate tends to encourage broader connectivity.
  Lower positive rate tends to encourage stricter locality.

### `GeneratedNodeBatch`

Returned by `ConditionalNodeFieldGenerator.predict*`.

```python
GeneratedNodeBatch(
    node_presence_mask: Optional[np.ndarray] = None,
    node_degree_predictions: Optional[np.ndarray] = None,
    node_labels: Optional[List[np.ndarray]] = None,
    edge_probability_matrices: Optional[List[np.ndarray]] = None,
    edge_label_matrices: Optional[List[np.ndarray]] = None,
)
```

Fields:

- `node_presence_mask`
  Predicted active node slots.
  More `True` entries mean larger generated graphs.
  Fewer `True` entries mean smaller generated graphs.

- `node_degree_predictions`
  Predicted per-node degree classes or values.
  Larger predicted degrees usually produce denser decoded graphs.
  Smaller predicted degrees usually produce sparser decoded graphs.

- `node_labels`
  Optional predicted node labels.

- `edge_probability_matrices`
  Optional dense per-graph edge-probability matrices.
  Higher probabilities give the decoder more confidence in including edges.
  Lower probabilities make the decoder rely more on constraint tradeoffs.

- `edge_label_matrices`
  Optional dense per-graph edge-label predictions.

## `ConditionalNodeFieldGenerator`

This is the node-level generator. It learns to map graph-level conditions to node-level structural and semantic outputs.

Primary implementation:

- [`../conditional_node_field_graph_generator/conditional_node_field_generator.py`](../conditional_node_field_graph_generator/conditional_node_field_generator.py)

### Constructor

```python
ConditionalNodeFieldGenerator(
    latent_embedding_dimension: int = 128,
    number_of_transformer_layers: int = 4,
    transformer_attention_head_count: int = 4,
    transformer_dropout: float = 0.1,
    learning_rate: float = 1e-3,
    maximum_epochs: int = 10,
    batch_size: int = 32,
    total_steps: int = 100,
    verbose: bool = False,
    verbose_epoch_interval: int = 10,
    enable_early_stopping: bool = True,
    early_stopping_monitor: str = "val_total",
    early_stopping_mode: str = "min",
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 0.0,
    early_stopping_ema_alpha: float = 0.3,
    restore_best_checkpoint: bool = True,
    artifact_root_dir: Optional[str] = None,
    checkpoint_root_dir: Optional[str] = None,
    important_feature_index: int = 1,
    lambda_degree_importance: float = 1.0,
    degree_temperature: Optional[float] = None,
    lambda_node_exist_importance: float = 1.0,
    lambda_node_count_importance: float = 0.0,
    lambda_node_label_importance: float = 1.0,
    default_exist_pos_weight: float = 1.0,
    lambda_direct_edge_importance: float = 1.0,
    lambda_edge_count_importance: float = 0.0,
    lambda_degree_edge_consistency_importance: float = 0.0,
    lambda_auxiliary_edge_importance: float = 1.0,
    lambda_edge_label_importance: float = 1.0,
    pool_condition_tokens: bool = False,
    node_field_sigma: float = 0.2,
    sampling_step_size: float = 0.05,
    sampling_steps: Optional[int] = None,
    langevin_noise_scale: float = 0.0,
    cfg_target_mode: Optional[str] = None,
    cfg_condition_dropout_prob: float = 0.1,
    cfg_null_target_strategy: str = "zero",
    target_classification_max_distinct: int = 20,
)
```

#### Model Capacity And Optimization

- `latent_embedding_dimension`
  Hidden size of the node-field backbone.
  Increase: more expressiveness, more memory, slower training, higher overfitting risk.
  Decrease: faster and cheaper, but easier to underfit.

- `number_of_transformer_layers`
  Depth of the conditional transformer stack.
  Increase: stronger modeling of condition-node interactions, but slower and harder to optimize.
  Decrease: simpler model, but may miss higher-order dependencies.

- `transformer_attention_head_count`
  Number of attention heads.
  Increase: more subspaces for interaction patterns, but higher compute and possible instability if the latent size is small.
  Decrease: cheaper and simpler, but less expressive attention.

- `transformer_dropout`
  Dropout used in the transformer and guidance predictor.
  Increase: stronger regularization, often better generalization, but can underfit if too high.
  Decrease: better training fit, but higher overfitting risk.

- `learning_rate`
  Optimizer step size.
  Increase: faster learning, but more instability and divergence risk.
  Decrease: more stable, but slower convergence and higher risk of getting stuck.

- `maximum_epochs`
  Upper bound on training epochs.
  Increase: more fitting time, better chance to converge, but more overfitting and more runtime.
  Decrease: faster experiments, but higher underfitting risk.

- `batch_size`
  Minibatch size for training.
  Increase: smoother gradients, usually faster throughput, but higher memory use and sometimes worse generalization.
  Decrease: noisier gradients, lower memory, sometimes better generalization, but slower wall-clock training.

#### Training Loop And Checkpointing

- `total_steps`
  Default number of iterative sampling steps if `sampling_steps` is not set separately.
  Increase: longer generation trajectories and potentially better refinement, but slower sampling.
  Decrease: faster sampling, but rougher or less converged generations.

- `verbose`
  Controls logging verbosity.
  Enable: more diagnostics.
  Disable: quieter runs.

- `verbose_epoch_interval`
  Logging interval during training.
  Increase: less log noise.
  Decrease: more detailed training traces.
  Important: see the `Training Metrics` section in [`2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md`](2B_CONDITIONAL_NODE_FIELD_TRAINING_README.md).

- `enable_early_stopping`
  Whether to stop based on validation behavior.
  Enable: safer default for long runs.
  Disable: always trains to `maximum_epochs`.

- `early_stopping_monitor`
  Validation metric name to monitor.
  Choosing a more task-aligned metric usually improves checkpoint quality.
  Choosing the wrong metric can stop on the wrong behavior.

- `early_stopping_mode`
  `"min"` or `"max"` depending on the monitored metric.
  Wrong choice will invert the stopping criterion.

- `early_stopping_patience`
  Number of stale validation checks tolerated.
  Increase: more tolerance to noisy validation curves, slower stopping.
  Decrease: earlier stopping, but higher risk of cutting training too soon.

- `early_stopping_min_delta`
  Minimum improvement required to count as progress.
  Increase: stricter improvement criterion.
  Decrease: easier to register small improvements.

- `early_stopping_ema_alpha`
  Smoothing factor for monitored validation behavior.
  Increase toward `1.0`: less smoothing, faster reaction to recent changes.
  Decrease toward `0.0`: more smoothing, slower reaction, less sensitivity to noise.

- `restore_best_checkpoint`
  Whether to restore the best saved validation checkpoint after training.
  Enable: usually safer for final inference quality.
  Disable: keeps the last epoch state instead.

- `artifact_root_dir`
  Root directory for run artifacts.
  No model-quality effect, only output organization.

- `checkpoint_root_dir`
  Root directory for checkpoints.
  No model-quality effect, only checkpoint location.

#### Loss Weights And Structural Bias

- `important_feature_index`
  Index of the node feature treated as the main scalar feature of interest.
  Change this only if the node feature layout changes.
  Wrong value misaligns supervision semantics.

- `lambda_degree_importance`
  Weight of node-degree supervision.
  The loss is normalized per active node, so `1.0` is now a reasonable default starting point.
  Increase: stronger pressure to match degrees, often improving decoded structure.
  Decrease: weaker degree supervision, more freedom in latent generation.

- `degree_temperature`
  Temperature used in degree prediction behavior.
  Increase: softer degree distributions.
  Decrease: sharper degree preferences.

- `lambda_node_exist_importance`
  Weight of node-existence supervision.
  The loss is normalized per slot, so `1.0` is now a reasonable default starting point.
  Increase: stronger pressure on node-slot occupancy accuracy.
  Decrease: more freedom in which slots stay active.

- `lambda_node_count_importance`
  Weight of explicit node-count consistency.
  This is now a relative graph-level consistency loss, so `0.1` to `0.5` is a reasonable starting range.
  Increase: stronger pressure to match requested node counts.
  Decrease: more reliance on slot-level existence predictions.

- `lambda_node_label_importance`
  Weight of node-label supervision.
  The loss is normalized per labeled node, so `1.0` is now a reasonable default starting point.
  Increase: better node-label fidelity if labels matter, but can compete with structural learning.
  Decrease: structure dominates over label quality.

- `default_exist_pos_weight`
  Positive-class weight for node existence when auto-balancing is weak or unavailable.
  Increase: penalizes missed existing nodes more strongly.
  Decrease: penalizes false positives relatively more.

- `lambda_direct_edge_importance`
  Weight of direct edge supervision.
  The loss is normalized per supervised pair, so `1.0` is now a reasonable default starting point.
  Increase: stronger learning of adjacency probabilities.
  Decrease: weaker direct structural supervision.

- `lambda_edge_count_importance`
  Weight of explicit edge-count consistency.
  This is now a relative graph-level consistency loss, so `0.1` to `0.5` is a reasonable starting range.
  Increase: stronger pressure to match requested graph density.
  Decrease: decoder relies more on degree and edge-probability tradeoffs.

- `lambda_degree_edge_consistency_importance`
  Weight of consistency between predicted degrees and predicted edges.
  This is now a relative graph-level consistency loss, so `0.1` to `0.5` is a reasonable starting range.
  Increase: tighter structural coherence.
  Decrease: more independence between node-degree and edge heads.

- `lambda_auxiliary_edge_importance`
  Weight of auxiliary locality supervision.
  This usually works best as a regularizer rather than a dominant term, so `0.25` to `0.5` is a reasonable starting range.
  Increase: stronger longer-range structural regularization.
  Decrease: less influence from auxiliary locality targets.

- `lambda_edge_label_importance`
  Weight of edge-label supervision.
  The loss is normalized per labeled pair, so `1.0` is now a reasonable default starting point.
  Increase: better edge-label fidelity, but can distract from structure if labels are noisy.
  Decrease: structure gets priority over label semantics.

#### Conditioning And Sampling

- `pool_condition_tokens`
  Whether to pool condition tokens rather than keeping richer token structure.
  Enable: simpler conditioning pathway, often more stable when the condition representation is noisy.
  Disable: preserves more conditional detail.

- `node_field_sigma`
  Gaussian corruption scale used in score matching and also the default noise scale for the separate guidance predictor.
  Increase: harder denoising task, often smoother but less precise fields.
  Decrease: easier denoising task, sharper fitting, but less robustness.

- `sampling_step_size`
  Step size for iterative node-field sampling.
  Increase: faster movement through latent space, but more instability or overshooting.
  Decrease: more stable and fine-grained updates, but slower convergence.

- `sampling_steps`
  Number of iterative sampling steps used at inference.
  Increase: more refinement, slower generation.
  Decrease: faster generation, rougher outputs.

- `langevin_noise_scale`
  Noise injected during sampling.
  Increase: more diversity and exploration, but noisier trajectories.
  Decrease: more deterministic refinement, but lower diversity.

#### CFG-Specific Parameters

- `cfg_target_mode`
  Explicit target mode for the classifier-free target-conditioning path.
  Use `"classification"` for categorical CFG targets and `"regression"` for scalar numeric CFG targets.
  Leave as `None`: CFG target-conditioning is not configured, and passing CFG training targets will raise an error.

- `cfg_condition_dropout_prob`
  Probability of dropping target-conditioning channels during CFG training.
  Increase: stronger unconditional branch training, but weaker conditional specialization if too high.
  Decrease: stronger conditional specialization, but weaker CFG behavior if too low.

- `cfg_null_target_strategy`
  Null representation used for dropped targets.
  Currently only `"zero"` is supported.

- `target_classification_max_distinct`
  Threshold used only by the separate post-hoc guidance-predictor path when `train_guidance_predictor(..., mode=None)`
  is allowed to infer classification vs regression automatically.
  Increase: more predictor-training target sets are handled as classification.
  Decrease: more predictor-training target sets are handled as regression.

### Main Public Methods

#### `fit(...)`

```python
fit(
    node_batch: NodeGenerationBatch,
    graph_conditioning: GraphConditioningBatch,
    targets: Optional[Sequence[Any]] = None,
    ckpt_path: Optional[str] = None,
)
```

Parameters:

- `node_batch`
  Training node-level supervision.

- `graph_conditioning`
  Training graph-level conditions.

- `targets`
  Optional target values used to activate and train CFG target-conditioning channels.
  Provide targets: enables CFG-capable training only if `cfg_target_mode` was set explicitly on the generator.
  Omit targets: trains without CFG target channels.

- `ckpt_path`
  Optional Lightning checkpoint path used to resume training state.
  Provide a checkpoint path: resumes epoch/optimizer/trainer state from that checkpoint instead of starting fresh.
  Omit it: starts a new training run while still writing fresh checkpoints under `checkpoint_root_dir`.

#### `predict(...)`

```python
predict(
    graph_conditioning: GraphConditioningBatch,
    desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
    guidance_scale: float = 1.0,
) -> GeneratedNodeBatch
```

Parameters:

- `graph_conditioning`
  Graph-level conditions for generation.

- `desired_target`
  Optional CFG target request.
  Provide it: activates CFG if the model was trained with guidance support.
  Omit it: uses the ordinary unguided conditional path.

- `guidance_scale`
  Strength of CFG.
  Increase above `1.0`: stronger pull toward the requested target, often less diversity and more risk of artifacts.
  Decrease toward `1.0`: ordinary conditional generation.
  Decrease toward `0.0`: moves toward the unconditional branch.

#### `set_guidance_predictor(...)`

```python
set_guidance_predictor(
    mode: str,
    output_dimension: Optional[int] = None,
    hidden_dimension: Optional[int] = None,
) -> None
```

Parameters:

- `mode`
  `"classification"` or `"regression"`.

- `output_dimension`
  Number of outputs for the auxiliary predictor.
  Increase for classification: supports more target classes.
  Regression must use `1`.

- `hidden_dimension`
  Hidden size of the separate post-hoc guidance predictor.
  Increase: more expressive predictor, but more overfitting risk and more compute.
  Decrease: simpler predictor, but possibly weaker guidance gradients.

#### `set_guidance_classifier(...)`

```python
set_guidance_classifier(
    num_classes: int,
    hidden_dimension: Optional[int] = None,
) -> None
```

Parameters:

- `num_classes`
  Number of classifier outputs.
  Increase only if the classification target space actually requires it.

- `hidden_dimension`
  Same tradeoff as in `set_guidance_predictor(...)`.

#### `train_guidance_predictor(...)`

```python
train_guidance_predictor(
    node_batch: NodeGenerationBatch,
    graph_conditioning: GraphConditioningBatch,
    targets: Sequence[Any],
    mode: Optional[str] = None,
    learning_rate: float = 1e-3,
    maximum_epochs: int = 30,
    batch_size: Optional[int] = None,
    noise_scale: Optional[float] = None,
) -> None
```

Parameters:

- `targets`
  Classification labels or regression targets for the separate post-hoc predictor.

- `mode`
  Optional explicit predictor mode. If omitted, the implementation infers classification vs regression.

- `learning_rate`
  Increase: faster but less stable predictor fitting.
  Decrease: slower but more stable.

- `maximum_epochs`
  Increase: more predictor fitting capacity.
  Decrease: faster but more underfitting risk.

- `batch_size`
  Increase: smoother gradients and more memory use.
  Decrease: noisier gradients and lower memory use.

- `noise_scale`
  Noise added to node states while training the predictor.
  Increase: more robustness to noisy sampling states, but weaker predictor precision.
  Decrease: sharper predictor fit, but less robustness off-manifold.

#### `train_guidance_classifier(...)`

```python
train_guidance_classifier(
    node_batch: NodeGenerationBatch,
    graph_conditioning: GraphConditioningBatch,
    targets: Sequence[Any],
    learning_rate: float = 1e-3,
    maximum_epochs: int = 30,
    batch_size: Optional[int] = None,
    noise_scale: Optional[float] = None,
) -> None
```

Same interface as `train_guidance_predictor(...)`, fixed to classification mode.

#### `predict_classifier_guided(...)`

```python
predict_classifier_guided(
    graph_conditioning: GraphConditioningBatch,
    desired_class: Union[int, Sequence[Any]],
    classifier_scale: float = 1.0,
) -> GeneratedNodeBatch
```

Parameters:

- `desired_class`
  Requested class target for the separate guidance classifier.

- `classifier_scale`
  Strength of classifier guidance.
  Increase: stronger push toward the requested class, less diversity, more artifact risk if the classifier is imperfect.
  Decrease: weaker target steering.

#### `predict_regression_guided(...)`

```python
predict_regression_guided(
    graph_conditioning: GraphConditioningBatch,
    desired_target: Union[float, Sequence[Any]],
    predictor_scale: float = 1.0,
) -> GeneratedNodeBatch
```

Parameters:

- `desired_target`
  Requested regression target for the separate guidance predictor.

- `predictor_scale`
  Strength of regression guidance.
  Increase: stronger push toward the requested target value, often lower diversity and higher instability.
  Decrease: weaker steering.

### Practical Notes

- `predict(...)` is the CFG path.
- `predict_classifier_guided(...)` and `predict_regression_guided(...)` are the separate post-hoc guidance paths.
- The implementation supports both guidance routes, but not both in the same sampling call.

## `ConditionalNodeFieldGraphDecoder`

This class reconstructs final `networkx.Graph` objects from node-level predictions.

Primary implementation:

- [`../conditional_node_field_graph_generator/conditional_node_field_graph_generator.py`](../conditional_node_field_graph_generator/conditional_node_field_graph_generator.py)

### Constructor

```python
ConditionalNodeFieldGraphDecoder(
    verbose: bool = True,
    existence_threshold: float = 0.5,
    enforce_connectivity: bool = True,
    degree_slack_penalty: float = 1e6,
    warm_start_mst: bool = True,
    n_jobs: int = 1,
)
```

Parameters:

- `verbose`
  Enable or disable decoder logging.

- `existence_threshold`
  Threshold used when deciding whether a node slot exists.
  Increase: fewer nodes survive, which usually yields smaller graphs.
  Decrease: more nodes survive, which usually yields larger graphs.

- `enforce_connectivity`
  Whether the adjacency solver forces connected graphs.
  Enable: decoded graphs are connected, but may distort probabilities to satisfy that constraint.
  Disable: allows disconnected graphs.

- `degree_slack_penalty`
  Penalty for violating target degree constraints in the ILP.
  Increase: stronger degree matching, but greater risk that edge probabilities are overridden.
  Decrease: more flexibility to follow edge probabilities, but weaker degree fidelity.

- `warm_start_mst`
  Whether to initialize the ILP with a maximum spanning tree.
  Enable: often speeds solving and helps connected solutions.
  Disable: more neutral initialization, sometimes slower.

- `n_jobs`
  Parallelism for per-graph decoding.
  Increase: higher throughput on CPU-bound decode workloads.
  Decrease: lower CPU usage and simpler debugging.

### Main Public Method

#### `decode(...)`

```python
decode(
    generated_nodes: GeneratedNodeBatch,
    predicted_node_labels_list: Optional[List[np.ndarray]] = None,
    predicted_edge_probability_matrices: Optional[List[np.ndarray]] = None,
    predicted_edge_labels_list: Optional[List[np.ndarray]] = None,
    predicted_edge_label_matrices: Optional[List[np.ndarray]] = None,
) -> List[nx.Graph]
```

Parameters:

- `generated_nodes`
  Node-level structure predictions to decode.

- `predicted_node_labels_list`
  Optional explicit node labels.
  Provide this when node labels are learned or externally supplied.

- `predicted_edge_probability_matrices`
  Dense edge-probability matrices used by the ILP decoder.
  Higher probabilities favor edge inclusion.
  Lower probabilities favor edge exclusion.

- `predicted_edge_labels_list`
  Optional explicit list of edge labels per graph.

- `predicted_edge_label_matrices`
  Optional dense per-graph edge-label matrices.

## `ConditionalNodeFieldGraphGenerator`

This is the end-to-end orchestration layer. In most user workflows, this is the main high-level class.

### Constructor

```python
ConditionalNodeFieldGraphGenerator(
    graph_vectorizer: Any = None,
    node_graph_vectorizer: Any = None,
    conditional_node_generator_model: Optional[ConditionalNodeGeneratorBase] = None,
    graph_decoder: Optional[ConditionalNodeFieldGraphDecoder] = None,
    verbose: bool = True,
    locality_sample_fraction: float = 1.0,
    locality_horizon: int = 1,
    negative_sample_factor: int = 1,
    locality_sampling_strategy: str = "stratified_preserve",
    locality_target_positive_ratio: Optional[float] = None,
    feasibility_estimator: Any = None,
    use_feasibility_filtering: bool = True,
    max_feasibility_attempts: int = 10,
    feasibility_candidates_per_attempt: int = 4,
    feasibility_failure_mode: str = "return_partial",
)
```

#### Core Components

- `graph_vectorizer`
  Produces graph-level embeddings and graph statistics.
  Better vectorizers improve global conditioning quality.

- `node_graph_vectorizer`
  Produces per-node embeddings.
  Better vectorizers improve node-level supervision quality.

- `conditional_node_generator_model`
  Usually a `ConditionalNodeFieldGenerator`.

- `graph_decoder`
  Usually a `ConditionalNodeFieldGraphDecoder`.

- `verbose`
  Global verbosity for the orchestration layer.

#### Structural Supervision Parameters

- `locality_sample_fraction`
  Fraction of candidate locality supervision pairs used during training.
  Increase toward `1.0`: more supervision, more runtime, less sampling noise.
  Decrease: cheaper training, but weaker structural supervision.

- `locality_horizon`
  Shortest-path horizon used for locality supervision.
  Increase: supervision reaches farther graph neighborhoods, making structure learning more global.
  Decrease toward `1`: supervision is more local and direct.

- `negative_sample_factor`
  Number of negative locality samples relative to positives.
  Increase: stronger pressure against false-positive edges, often sparser predictions.
  Decrease: weaker negative pressure, often denser predictions.

- `locality_sampling_strategy`
  One of `"uniform"`, `"stratified_preserve"`, or `"stratified_target"`.
  `uniform`: simplest sampling, but may drift from the true class balance.
  `stratified_preserve`: tries to preserve the observed positive/negative ratio.
  `stratified_target`: explicitly aims for `locality_target_positive_ratio`.

- `locality_target_positive_ratio`
  Target positive ratio when using `"stratified_target"`.
  Increase: more positive locality pairs, usually encouraging denser local structure.
  Decrease: more negative pairs, usually encouraging more conservative connectivity.

#### Feasibility Filtering Parameters

- `feasibility_estimator`
  Optional predictor that judges whether a decoded graph is acceptable.
  Better estimator quality improves rejection filtering quality.

- `use_feasibility_filtering`
  Whether to apply feasibility filtering during generation.
  Enable: safer outputs, but slower generation and possible partial batches.
  Disable: faster generation, but no rejection of bad outputs.

- `max_feasibility_attempts`
  Maximum rejection-sampling rounds per batch slot.
  Increase: better chance to fill all requested outputs, but slower runtime.
  Decrease: faster failure or fallback.

- `feasibility_candidates_per_attempt`
  Number of candidate graphs generated per still-missing slot at each attempt.
  Increase: higher chance to recover a feasible graph, but more compute.
  Decrease: cheaper each round, but lower acceptance chance.

- `feasibility_failure_mode`
  `"raise"` or `"return_partial"`.
  `"raise"`: strict behavior, useful for pipelines that require a full batch.
  `"return_partial"`: more forgiving behavior, useful for exploratory generation.

### Main Public Methods

#### `fit(...)`

```python
fit(
    graphs: List[nx.Graph],
    train_node_generator: bool = True,
    targets: Optional[Sequence[Any]] = None,
    ckpt_path: Optional[str] = None,
) -> ConditionalNodeFieldGraphGenerator
```

Parameters:

- `graphs`
  Training graphs.

- `train_node_generator`
  Whether to train the node generator during this fit call.
  Enable: full end-to-end training.
  Disable: only fit vectorizers and other orchestration components.

- `targets`
  Optional target values used to train CFG-ready target-conditioning in the node generator.
  Provide targets: enables CFG support in the fitted generator only if `cfg_target_mode` was configured explicitly
  on the node generator.
  Omit targets: no CFG target-conditioning path is trained.

- `ckpt_path`
  Optional checkpoint path forwarded to the underlying node generator fit call.
  Provide a checkpoint path: resumes node-generator training from that checkpoint.
  Omit it: trains from scratch.

#### `encode(...)`

```python
encode(
    graphs: List[nx.Graph],
) -> Tuple[List[np.ndarray], GraphConditioningBatch]
```

Returns:

- list of per-graph node embedding matrices,
- a `GraphConditioningBatch` with graph embeddings, node counts, and edge counts.

#### `set_guidance_predictor(...)`

```python
set_guidance_predictor(
    mode: str,
    output_dimension: Optional[int] = None,
    hidden_dimension: Optional[int] = None,
) -> None
```

Thin orchestration wrapper around the node-generator method of the same name.

#### `set_guidance_classifier(...)`

```python
set_guidance_classifier(
    num_classes: int,
    hidden_dimension: Optional[int] = None,
) -> None
```

Thin orchestration wrapper around the node-generator classifier setup.

#### `train_guidance_predictor(...)`

```python
train_guidance_predictor(
    graphs: List[nx.Graph],
    targets: Sequence[Any],
    mode: Optional[str] = None,
    learning_rate: float = 1e-3,
    maximum_epochs: int = 30,
    batch_size: Optional[int] = None,
    noise_scale: Optional[float] = None,
) -> None
```

Same tuning logic as the node-generator variant, but this method starts from raw graphs and internally builds the required batches.

#### `train_guidance_classifier(...)`

```python
train_guidance_classifier(
    graphs: List[nx.Graph],
    targets: Sequence[Any],
    learning_rate: float = 1e-3,
    maximum_epochs: int = 30,
    batch_size: Optional[int] = None,
    noise_scale: Optional[float] = None,
) -> None
```

Classification-only convenience wrapper.

#### `decode(...)`

```python
decode(
    graph_conditioning: GraphConditioningBatch,
    desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
    guidance_scale: float = 1.0,
    apply_feasibility_filtering: Optional[bool] = None,
) -> List[nx.Graph]
```

Parameters:

- `graph_conditioning`
  Explicit graph-level conditioning to decode from.

- `desired_target`
  Optional CFG target for target-conditioned generation.

- `guidance_scale`
  CFG strength.
  Increase: stronger target steering, less diversity, more artifact risk.
  Decrease: weaker target steering.

- `apply_feasibility_filtering`
  Overrides the object-level default for this call only.
  Enable: safer outputs, slower runtime.
  Disable: faster generation, no rejection filtering.

#### `decode_classifier_guided(...)`

```python
decode_classifier_guided(
    graph_conditioning: GraphConditioningBatch,
    desired_class: Union[int, Sequence[Any]],
    classifier_scale: float = 1.0,
    apply_feasibility_filtering: Optional[bool] = None,
) -> List[nx.Graph]
```

Parameters:

- `desired_class`
  Requested target class for separate post-hoc guidance.

- `classifier_scale`
  Increase: stronger steering toward the target class.
  Decrease: weaker steering, more baseline behavior.

- `apply_feasibility_filtering`
  Same effect as in `decode(...)`.

#### `decode_regression_guided(...)`

```python
decode_regression_guided(
    graph_conditioning: GraphConditioningBatch,
    desired_target: Union[float, Sequence[Any]],
    predictor_scale: float = 1.0,
    apply_feasibility_filtering: Optional[bool] = None,
) -> List[nx.Graph]
```

Parameters:

- `desired_target`
  Requested regression target for separate post-hoc guidance.

- `predictor_scale`
  Increase: stronger push toward the requested value.
  Decrease: weaker steering.

- `apply_feasibility_filtering`
  Same effect as in `decode(...)`.

#### `sample(...)`

```python
sample(
    n_samples: int = 1,
    interpolate_between_n_samples: Optional[int] = None,
    desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
    guidance_scale: float = 1.0,
    apply_feasibility_filtering: Optional[bool] = None,
) -> List[nx.Graph]
```

Parameters:

- `n_samples`
  Number of graphs to sample.
  Increase: more outputs and more runtime.

- `interpolate_between_n_samples`
  If set, each sampled conditioning vector is created by stochastic interpolation among cached training conditions.
  Increase: conditioning moves farther away from exact training examples and may improve variety.
  Decrease or omit: conditioning stays closer to empirical training conditions.

- `desired_target`
  Optional CFG target.

- `guidance_scale`
  Same CFG tradeoff as above.

- `apply_feasibility_filtering`
  Same filtering tradeoff as above.

#### `conditional_sample(...)`

```python
conditional_sample(
    graphs: List[nx.Graph],
    n_samples: int = 1,
    desired_target: Optional[Union[int, float, Sequence[Any]]] = None,
    guidance_scale: float = 1.0,
    apply_feasibility_filtering: Optional[bool] = None,
) -> List[List[nx.Graph]]
```

Parameters:

- `graphs`
  Seed graphs whose conditions are encoded and reused.

- `n_samples`
  Number of decoded samples per input graph.
  Increase: more conditional variants per input.

- `desired_target`
  Optional CFG target.

- `guidance_scale`
  Same CFG tradeoff as above.

- `apply_feasibility_filtering`
  Same filtering tradeoff as above.

#### `sample_classifier_guided(...)`

```python
sample_classifier_guided(
    desired_class: Union[int, Sequence[Any]],
    n_samples: int = 1,
    interpolate_between_n_samples: Optional[int] = None,
    classifier_scale: float = 1.0,
    apply_feasibility_filtering: Optional[bool] = None,
) -> List[nx.Graph]
```

Parameters:

- `desired_class`
  Post-hoc classifier guidance target.

- `n_samples`
  Number of samples to generate.

- `interpolate_between_n_samples`
  Same conditioning-diversity tradeoff as in `sample(...)`.

- `classifier_scale`
  Increase: stronger class steering.
  Decrease: weaker steering.

- `apply_feasibility_filtering`
  Same filtering tradeoff as above.

#### `conditional_sample_classifier_guided(...)`

```python
conditional_sample_classifier_guided(
    graphs: List[nx.Graph],
    desired_class: Union[int, Sequence[Any]],
    n_samples: int = 1,
    classifier_scale: float = 1.0,
    apply_feasibility_filtering: Optional[bool] = None,
) -> List[List[nx.Graph]]
```

Same interface logic as `conditional_sample(...)`, but using separate classifier guidance instead of CFG.

#### `sample_regression_guided(...)`

```python
sample_regression_guided(
    desired_target: Union[float, Sequence[Any]],
    n_samples: int = 1,
    interpolate_between_n_samples: Optional[int] = None,
    predictor_scale: float = 1.0,
    apply_feasibility_filtering: Optional[bool] = None,
) -> List[nx.Graph]
```

Parameters:

- `desired_target`
  Post-hoc regression guidance target.

- `n_samples`
  Number of samples to generate.

- `interpolate_between_n_samples`
  Same conditioning-diversity tradeoff as in `sample(...)`.

- `predictor_scale`
  Increase: stronger regression steering.
  Decrease: weaker steering.

- `apply_feasibility_filtering`
  Same filtering tradeoff as above.

#### `conditional_sample_regression_guided(...)`

```python
conditional_sample_regression_guided(
    graphs: List[nx.Graph],
    desired_target: Union[float, Sequence[Any]],
    n_samples: int = 1,
    predictor_scale: float = 1.0,
    apply_feasibility_filtering: Optional[bool] = None,
) -> List[List[nx.Graph]]
```

Same interface logic as `conditional_sample(...)`, but using separate regression guidance instead of CFG.

## Recommended Mental Model

Use these three levels:

1. `ConditionalNodeFieldGraphGenerator`
   The main end-to-end interface for most users.

2. `ConditionalNodeFieldGenerator`
   The lower-level node generator when you want direct control over batches and node-level outputs.

3. `ConditionalNodeFieldGraphDecoder`
   The reconstruction layer when you want to decode node outputs manually or study decode constraints separately.

## Tuning Shortcuts

If outputs are too small:

- decrease `existence_threshold`,
- decrease `negative_sample_factor`,
- increase `lambda_node_exist_importance`,
- increase `lambda_node_count_importance`.

If outputs are too dense:

- increase `negative_sample_factor`,
- decrease `edge_counts` in conditioning,
- decrease `lambda_direct_edge_importance` only if edge probabilities are overconfident and noisy.

If CFG is too weak:

- increase `guidance_scale`,
- decrease `cfg_condition_dropout_prob` only if conditional specialization is too weak,
- make sure `targets` were actually passed during `fit(...)`.

If post-hoc guidance is too weak:

- increase `classifier_scale` or `predictor_scale`,
- increase separate predictor capacity with `hidden_dimension`,
- train the guidance predictor longer,
- use a moderate `noise_scale` during predictor training so gradients remain useful on sampled states.

If sampling is unstable:

- decrease `sampling_step_size`,
- decrease `guidance_scale` or predictor guidance scale,
- decrease `langevin_noise_scale`,
- increase `sampling_steps`.
