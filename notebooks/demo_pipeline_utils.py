"""Reusable workflow helpers shared by demo notebooks."""

from __future__ import annotations

from typing import Callable

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split

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

from eqm_decompositional_graph_generator.eqm_conditional_node_generator import EqMDecompositionalNodeGenerator
from eqm_decompositional_graph_generator.graph_generator import (
    EqMDecompositionalGraphDecoder,
    EqMDecompositionalGraphGenerator,
)


def build_dataset(dataset_type, dataset_size=50, size=5, assay_id="651610"):
    if dataset_type == "ARTIFICIAL":
        from eqm_decompositional_graph_generator.artificial_graph_constructor import ArtificialGraphDatasetConstructor
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
    nbits=11,
    verbose=2,
    maximum_epochs=250,
    batch_size=16,
    total_steps=100,
    early_stopping_min_delta=100.0,
    locality_horizon=1,
    artifact_root=None,
    checkpoint_root=None,
):
    node_graph_vectorizer = NodeNSPPK(
        radius=2,
        distance=4,
        connector=1,
        nbits=nbits,
        dense=True,
        parallel=True,
        use_edges_as_features=True,
    )
    graph_vectorizer = NSPPK(
        radius=2,
        distance=4,
        connector=1,
        nbits=nbits,
        dense=True,
        parallel=True,
        use_edges_as_features=True,
    )

    feasibility_size = WithinRangeFeasibilityEstimatorFromNumericalFunction(
        numerical_function=lambda graph: len(graph),
        quantile=None,
    )
    feasibility_unlabeled_structure = FeasibilityEstimatorFeatureCannotExist(
        decomposition_function=compose(neighborhood(radius=2), unlabel()),
        nbits=19,
        parallel=True,
        backend="dill",
    )
    feasibility_valence = FeasibilityEstimatorFeatureCannotExist(
        decomposition_function=neighborhood(radius=1),
        nbits=19,
        parallel=True,
        backend="dill",
    )
    feasibility_cycle = FeasibilityEstimatorFeatureCannotExist(
        decomposition_function=cycle(),
        nbits=19,
        parallel=True,
        backend="dill",
    )
    feasibility_estimator = FeasibilityEstimator(
        [feasibility_size, feasibility_valence, feasibility_cycle, feasibility_unlabeled_structure]
    )

    conditional_node_generator_model = EqMDecompositionalNodeGenerator(
        latent_embedding_dimension=128,
        number_of_transformer_layers=4,
        transformer_attention_head_count=4,
        transformer_dropout=0.2,
        learning_rate=1e-4,
        maximum_epochs=maximum_epochs,
        batch_size=batch_size,
        total_steps=total_steps,
        lambda_degree_importance=5e3,
        lambda_node_exist_importance=0,
        lambda_node_label_importance=5e4,
        lambda_edge_label_importance=5e3,
        lambda_locality_importance=1e4,
        degree_temperature=1,
        eqm_sigma=0.2,
        sampling_step_size=0.05,
        langevin_noise_scale=0.0,
        verbose=verbose,
        verbose_epoch_interval=10,
        enable_early_stopping=True,
        early_stopping_monitor="val_total",
        early_stopping_mode="min",
        early_stopping_patience=20,
        early_stopping_min_delta=early_stopping_min_delta,
        restore_best_checkpoint=True,
        artifact_root_dir=str(artifact_root) if artifact_root is not None else None,
        checkpoint_root_dir=str(checkpoint_root) if checkpoint_root is not None else None,
    )
    graph_decoder = EqMDecompositionalGraphDecoder(
        verbose=verbose,
        enforce_connectivity=True,
        warm_start_mst=True,
    )
    return EqMDecompositionalGraphGenerator(
        graph_vectorizer=graph_vectorizer,
        node_graph_vectorizer=node_graph_vectorizer,
        conditional_node_generator_model=conditional_node_generator_model,
        graph_decoder=graph_decoder,
        feasibility_estimator=feasibility_estimator,
        locality_sample_fraction=0.5,
        locality_horizon=locality_horizon,
        negative_sample_factor=1,
        locality_sampling_strategy="stratified_preserve",
        locality_target_positive_ratio=0.5,
        max_feasibility_attempts=20,
        feasibility_candidates_per_attempt=8,
        feasibility_failure_mode="return_partial",
        verbose=verbose,
    )
