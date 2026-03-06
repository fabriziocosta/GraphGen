"""Reusable workflow helpers shared by demo notebooks."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure local package imports work when notebooks run with cwd=notebooks/.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from AbstractGraph.abstract_graph_operators import compose, cycle, neighborhood, unlabel
from AbstractGraph.feasibility import (
    FeasibilityEstimator,
    FeasibilityEstimatorFeatureCannotExist,
    WithinRangeFeasibilityEstimatorFromNumericalFunction,
)

try:
    from NSPPK.nsppk import NSPPK, NodeNSPPK
except ModuleNotFoundError:
    from nsppk import NSPPK, NodeNSPPK

from equilibrium_matching_decompositional_graph_generator.node_engine import EquilibriumMatchingDecompositionalNodeGenerator
from equilibrium_matching_decompositional_graph_generator.graph_engine import (
    EquilibriumMatchingDecompositionalGraphDecoder,
    EquilibriumMatchingDecompositionalGraphGenerator,
)


def build_dataset(dataset_type, dataset_size=50, size=5, assay_id="651610"):
    if dataset_type == "ARTIFICIAL":
        from equilibrium_matching_decompositional_graph_generator.support import ArtificialGraphDatasetConstructor
        try:
            from notebooks.notebook_utils import offset_neg_graphs, plot_networkx_graphs, select_pos_neg
        except ModuleNotFoundError:
            from notebook_utils import offset_neg_graphs, plot_networkx_graphs, select_pos_neg

        alphabet_size = 3
        graphs, targets = ArtificialGraphDatasetConstructor(
            graph_generator_target_type_pos="cycle",
            graph_generator_context_type_pos="cycle",
            graph_generator_target_type_neg="tree",
            graph_generator_context_type_neg="tree",
            target_size_pos=size,
            context_size_pos=size,
            n_link_edges_pos=1,
            alphabet_size_pos=alphabet_size,
            target_size_neg=size,
            context_size_neg=size,
            n_link_edges_neg=1,
            alphabet_size_neg=alphabet_size,
        ).sample(dataset_size // 2)

        graphs, targets = offset_neg_graphs(graphs, targets, offset=alphabet_size)
        n_graphs_per_line = 8
        pos_graphs, neg_graphs = select_pos_neg(graphs, targets, n_lines=1, n_graphs_per_line=n_graphs_per_line)
        plot_networkx_graphs(pos_graphs, n_cols=n_graphs_per_line)
        plot_networkx_graphs(neg_graphs, n_cols=n_graphs_per_line)
        return graphs, targets

    if dataset_type == "MOLECULAR":
        from coco_grape.data_loader.loader import SupervisedDataSetLoader
        from coco_grape.data_loader.mol.mol_loader import PubChemLoader
        from coco_grape.visualizer.mol_display import draw_molecules

        def pubchem_loader():
            return PubChemLoader().load(assay_id)

        original_graphs, original_targets = SupervisedDataSetLoader(
            pubchem_loader,
            size=dataset_size,
            use_equalized=False,
        ).load()
        original_graphs = np.array(original_graphs, dtype=object)
        original_targets = np.array(original_targets)

        idxs = [idx for idx, graph in enumerate(original_graphs) if nx.number_of_nodes(graph) <= size]
        graphs = original_graphs[idxs].tolist()
        targets = original_targets[idxs]
        draw_molecules(graphs[:14])
        return graphs, targets

    raise ValueError(f"Unsupported dataset_type={dataset_type!r}")


def prepare_experiment(build_dataset_fn: Callable, dataset_size=200, test_size=10, random_state=42, **build_kwargs):
    graphs, targets = build_dataset_fn(dataset_size=dataset_size, **build_kwargs)
    train_graphs, test_graphs, train_targets, test_targets = train_test_split(
        graphs,
        targets,
        test_size=test_size,
        random_state=random_state,
    )
    print(f"train_graphs:{len(train_graphs)}   test_graphs:{len(test_graphs)}")
    return graphs, targets, train_graphs, test_graphs, train_targets, test_targets


def build_graph_generator(
    verbose=2,
    nbits=None,
    node_vectorizer_radius=2,
    node_vectorizer_distance=4,
    node_vectorizer_connector=1,
    node_vectorizer_nbits=None,
    node_vectorizer_dense=True,
    node_vectorizer_parallel=True,
    node_vectorizer_use_edges_as_features=True,
    graph_vectorizer_radius=2,
    graph_vectorizer_distance=4,
    graph_vectorizer_connector=1,
    graph_vectorizer_nbits=None,
    graph_vectorizer_dense=True,
    graph_vectorizer_parallel=True,
    graph_vectorizer_use_edges_as_features=True,
    feasibility_size_quantile=None,
    feasibility_unlabeled_radius=2,
    feasibility_valence_radius=1,
    feasibility_unlabeled_nbits=19,
    feasibility_valence_nbits=19,
    feasibility_cycle_nbits=19,
    feasibility_parallel=True,
    feasibility_backend="dill",
    latent_embedding_dimension=128,
    number_of_transformer_layers=4,
    transformer_attention_head_count=4,
    transformer_dropout=0.2,
    learning_rate=1e-4,
    maximum_epochs=250,
    batch_size=16,
    total_steps=100,
    verbose_epoch_interval=10,
    enable_early_stopping=True,
    early_stopping_monitor="val_total",
    early_stopping_mode="min",
    early_stopping_patience=20,
    early_stopping_min_delta=100.0,
    early_stopping_ema_alpha=0.3,
    restore_best_checkpoint=True,
    important_feature_index=1,
    lambda_degree_importance=5e3,
    default_exist_pos_weight=1.0,
    lambda_node_exist_importance=0,
    lambda_node_count_importance=0.0,
    lambda_node_label_importance=5e4,
    lambda_edge_label_importance=5e3,
    lambda_direct_edge_importance=1e4,
    lambda_edge_count_importance=0.0,
    lambda_degree_edge_consistency_importance=0.0,
    lambda_auxiliary_edge_importance=1.0,
    degree_temperature=1,
    pool_condition_tokens=False,
    equilibrium_matching_sigma=0.2,
    sampling_step_size=0.05,
    sampling_steps=None,
    langevin_noise_scale=0.0,
    cfg_condition_dropout_prob=0.1,
    cfg_null_target_strategy="zero",
    target_classification_max_distinct=20,
    locality_horizon=1,
    locality_sample_fraction=0.5,
    negative_sample_factor=1,
    locality_sampling_strategy="stratified_preserve",
    locality_target_positive_ratio=0.5,
    use_feasibility_filtering=True,
    max_feasibility_attempts=20,
    feasibility_candidates_per_attempt=8,
    feasibility_failure_mode="return_partial",
    decoder_existence_threshold=0.5,
    decoder_enforce_connectivity=True,
    decoder_degree_slack_penalty=1e6,
    decoder_warm_start_mst=True,
    decoder_n_jobs=1,
    artifact_root=None,
    checkpoint_root=None,
):
    if nbits is not None:
        if node_vectorizer_nbits is None:
            node_vectorizer_nbits = nbits
        if graph_vectorizer_nbits is None:
            graph_vectorizer_nbits = nbits
    if node_vectorizer_nbits is None:
        node_vectorizer_nbits = 11
    if graph_vectorizer_nbits is None:
        graph_vectorizer_nbits = 11

    node_graph_vectorizer = NodeNSPPK(
        radius=node_vectorizer_radius,
        distance=node_vectorizer_distance,
        connector=node_vectorizer_connector,
        nbits=node_vectorizer_nbits,
        dense=node_vectorizer_dense,
        parallel=node_vectorizer_parallel,
        use_edges_as_features=node_vectorizer_use_edges_as_features,
    )
    graph_vectorizer = NSPPK(
        radius=graph_vectorizer_radius,
        distance=graph_vectorizer_distance,
        connector=graph_vectorizer_connector,
        nbits=graph_vectorizer_nbits,
        dense=graph_vectorizer_dense,
        parallel=graph_vectorizer_parallel,
        use_edges_as_features=graph_vectorizer_use_edges_as_features,
    )

    feasibility_size = WithinRangeFeasibilityEstimatorFromNumericalFunction(
        numerical_function=lambda graph: len(graph),
        quantile=feasibility_size_quantile,
    )
    feasibility_unlabeled_structure = FeasibilityEstimatorFeatureCannotExist(
        decomposition_function=compose(neighborhood(radius=feasibility_unlabeled_radius), unlabel()),
        nbits=feasibility_unlabeled_nbits,
        parallel=feasibility_parallel,
        backend=feasibility_backend,
    )
    feasibility_valence = FeasibilityEstimatorFeatureCannotExist(
        decomposition_function=neighborhood(radius=feasibility_valence_radius),
        nbits=feasibility_valence_nbits,
        parallel=feasibility_parallel,
        backend=feasibility_backend,
    )
    feasibility_cycle = FeasibilityEstimatorFeatureCannotExist(
        decomposition_function=cycle(),
        nbits=feasibility_cycle_nbits,
        parallel=feasibility_parallel,
        backend=feasibility_backend,
    )
    feasibility_estimator = FeasibilityEstimator(
        [feasibility_size, feasibility_valence, feasibility_cycle, feasibility_unlabeled_structure]
    )

    conditional_node_generator_model = EquilibriumMatchingDecompositionalNodeGenerator(
        latent_embedding_dimension=latent_embedding_dimension,
        number_of_transformer_layers=number_of_transformer_layers,
        transformer_attention_head_count=transformer_attention_head_count,
        transformer_dropout=transformer_dropout,
        learning_rate=learning_rate,
        maximum_epochs=maximum_epochs,
        batch_size=batch_size,
        total_steps=total_steps,
        lambda_degree_importance=lambda_degree_importance,
        lambda_node_exist_importance=lambda_node_exist_importance,
        lambda_node_count_importance=lambda_node_count_importance,
        lambda_node_label_importance=lambda_node_label_importance,
        lambda_edge_label_importance=lambda_edge_label_importance,
        lambda_direct_edge_importance=lambda_direct_edge_importance,
        lambda_edge_count_importance=lambda_edge_count_importance,
        lambda_degree_edge_consistency_importance=lambda_degree_edge_consistency_importance,
        lambda_auxiliary_edge_importance=lambda_auxiliary_edge_importance,
        degree_temperature=degree_temperature,
        equilibrium_matching_sigma=equilibrium_matching_sigma,
        sampling_step_size=sampling_step_size,
        langevin_noise_scale=langevin_noise_scale,
        verbose=verbose,
        verbose_epoch_interval=verbose_epoch_interval,
        enable_early_stopping=enable_early_stopping,
        early_stopping_monitor=early_stopping_monitor,
        early_stopping_mode=early_stopping_mode,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_ema_alpha=early_stopping_ema_alpha,
        restore_best_checkpoint=restore_best_checkpoint,
        important_feature_index=important_feature_index,
        cfg_condition_dropout_prob=cfg_condition_dropout_prob,
        cfg_null_target_strategy=cfg_null_target_strategy,
        target_classification_max_distinct=target_classification_max_distinct,
        default_exist_pos_weight=default_exist_pos_weight,
        artifact_root_dir=str(artifact_root) if artifact_root is not None else None,
        checkpoint_root_dir=str(checkpoint_root) if checkpoint_root is not None else None,
        pool_condition_tokens=pool_condition_tokens,
        sampling_steps=sampling_steps,
    )
    graph_decoder = EquilibriumMatchingDecompositionalGraphDecoder(
        verbose=verbose,
        existence_threshold=decoder_existence_threshold,
        enforce_connectivity=decoder_enforce_connectivity,
        degree_slack_penalty=decoder_degree_slack_penalty,
        warm_start_mst=decoder_warm_start_mst,
        n_jobs=decoder_n_jobs,
    )
    return EquilibriumMatchingDecompositionalGraphGenerator(
        graph_vectorizer=graph_vectorizer,
        node_graph_vectorizer=node_graph_vectorizer,
        conditional_node_generator_model=conditional_node_generator_model,
        graph_decoder=graph_decoder,
        feasibility_estimator=feasibility_estimator,
        locality_sample_fraction=locality_sample_fraction,
        locality_horizon=locality_horizon,
        negative_sample_factor=negative_sample_factor,
        locality_sampling_strategy=locality_sampling_strategy,
        locality_target_positive_ratio=locality_target_positive_ratio,
        use_feasibility_filtering=use_feasibility_filtering,
        max_feasibility_attempts=max_feasibility_attempts,
        feasibility_candidates_per_attempt=feasibility_candidates_per_attempt,
        feasibility_failure_mode=feasibility_failure_mode,
        verbose=verbose,
    )
