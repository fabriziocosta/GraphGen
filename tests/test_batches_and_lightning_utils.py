import numpy as np
import pytest
import torch
import warnings

from equilibrium_matching_decompositional_graph_generator.node_engine import (
    GeneratedNodeBatch,
    EquilibriumMatchingDecompositionalNodeGenerator,
    EquilibriumMatchingDecompositionalNodeGeneratorModule,
    GraphConditioningBatch,
    MetricsLogger,
    NodeGenerationBatch,
)
from equilibrium_matching_decompositional_graph_generator.support import run_trainer_fit


def test_graph_conditioning_batch_len():
    batch = GraphConditioningBatch(
        graph_embeddings=np.zeros((4, 8), dtype=float),
        node_counts=np.array([2, 2, 3, 1], dtype=np.int64),
        edge_counts=np.array([1, 1, 2, 0], dtype=np.int64),
    )
    assert len(batch) == 4


def test_node_generation_and_generated_batch_len():
    node_batch = NodeGenerationBatch(
        node_embeddings_list=[np.zeros((2, 4)), np.zeros((3, 4))],
        node_presence_mask=np.ones((2, 3), dtype=bool),
        node_degree_targets=np.zeros((2, 3), dtype=np.int64),
    )
    generated = GeneratedNodeBatch(
        node_embeddings_list=[np.zeros((2, 4)), np.zeros((3, 4)), np.zeros((1, 4))]
    )
    assert len(node_batch) == 2
    assert len(generated) == 3


class _OkTrainer:
    def __init__(self):
        self.called_with = None

    def fit(self, model, train_dataloaders=None, val_dataloaders=None):
        self.called_with = (model, train_dataloaders, val_dataloaders)


class _ExitTrainer:
    def fit(self, model, train_dataloaders=None, val_dataloaders=None):
        raise SystemExit(2)


def test_run_trainer_fit_calls_fit_with_named_loaders():
    trainer = _OkTrainer()
    model = object()
    train_loader = object()
    val_loader = object()

    run_trainer_fit(trainer, model, train_loader, val_loader, context="unit-test")

    assert trainer.called_with == (model, train_loader, val_loader)


def test_run_trainer_fit_wraps_system_exit():
    with pytest.raises(RuntimeError, match="unit-test aborted with SystemExit\\(2\\)"):
        run_trainer_fit(_ExitTrainer(), object(), object(), object(), context="unit-test")


class _WarnTrainer:
    def __init__(self):
        self.called = False

    def fit(self, model, train_dataloaders=None, val_dataloaders=None):
        del model, train_dataloaders, val_dataloaders
        self.called = True
        warnings.warn(
            "The 'train_dataloader' does not have many workers which may be a bottleneck. "
            "Consider increasing the value of the `num_workers` argument` to `num_workers=15` in the `DataLoader` to improve performance.",
            UserWarning,
        )
        warnings.warn(
            "The 'val_dataloader' does not have many workers which may be a bottleneck. "
            "Consider increasing the value of the `num_workers` argument` to `num_workers=15` in the `DataLoader` to improve performance.",
            UserWarning,
        )
        warnings.warn(
            "Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the "
            "`pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem.",
            UserWarning,
        )


def test_run_trainer_fit_suppresses_lightning_worker_warnings():
    trainer = _WarnTrainer()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_trainer_fit(trainer, object(), object(), object(), context="unit-test")

    assert trainer.called is True
    assert caught == []


def test_build_train_val_subsets_reuses_single_example_for_train_and_val():
    dataset = torch.utils.data.TensorDataset(torch.tensor([[1.0]], dtype=torch.float32))

    train_dataset, val_dataset = EquilibriumMatchingDecompositionalNodeGenerator._build_train_val_subsets(dataset)

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    assert train_dataset[0][0].item() == 1.0
    assert val_dataset[0][0].item() == 1.0


def test_build_train_val_subsets_keeps_both_sides_non_empty_for_two_examples():
    dataset = torch.utils.data.TensorDataset(
        torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    )

    train_dataset, val_dataset = EquilibriumMatchingDecompositionalNodeGenerator._build_train_val_subsets(dataset)

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1


def test_build_train_val_subsets_rejects_empty_dataset():
    dataset = torch.utils.data.TensorDataset(torch.empty((0, 1), dtype=torch.float32))

    with pytest.raises(ValueError, match="must contain at least one example"):
        EquilibriumMatchingDecompositionalNodeGenerator._build_train_val_subsets(dataset)


def test_update_ema_metric_tracks_smoothed_validation_signal():
    trainer = type("_Trainer", (), {"callback_metrics": {}, "logged_metrics": {}})()
    pl_module = type("_Module", (), {"_ema_metrics": {}, "early_stopping_ema_alpha": 0.25})()

    first = MetricsLogger._update_ema_metric(trainer, pl_module, "val_equilibrium_matching", 100.0)
    second = MetricsLogger._update_ema_metric(trainer, pl_module, "val_equilibrium_matching", 60.0)

    assert first == pytest.approx(100.0)
    assert second == pytest.approx(90.0)
    assert pl_module._ema_metrics["val_equilibrium_matching"] == pytest.approx(90.0)
    assert trainer.callback_metrics["val_equilibrium_matching_ema"].item() == pytest.approx(90.0)
    assert trainer.logged_metrics["val_equilibrium_matching_ema"].item() == pytest.approx(90.0)


def test_compute_edge_count_loss_matches_target_on_consistent_probabilities():
    edge_probs = torch.tensor(
        [[[0.0, 0.8], [0.8, 0.0]]],
        dtype=torch.float32,
    )
    node_presence_mask = torch.tensor([[True, True]])
    target_edge_counts = torch.tensor([1.0], dtype=torch.float32)

    loss = EquilibriumMatchingDecompositionalNodeGeneratorModule._compute_edge_count_loss(
        edge_probs=edge_probs,
        node_presence_mask=node_presence_mask,
        target_edge_counts=target_edge_counts,
    )

    assert loss.item() == pytest.approx(0.02, abs=1e-6)


def test_compute_degree_edge_consistency_loss_is_zero_when_handshake_identity_matches():
    logits_deg = torch.tensor(
        [[[ -10.0, 10.0], [ -10.0, 10.0]]],
        dtype=torch.float32,
    )
    node_presence_mask = torch.tensor([[True, True]])
    target_edge_counts = torch.tensor([1.0], dtype=torch.float32)

    loss = EquilibriumMatchingDecompositionalNodeGeneratorModule(
        number_of_rows_per_example=2,
        input_feature_dimension=2,
        condition_feature_dimension=3,
        latent_embedding_dimension=4,
        number_of_transformer_layers=1,
        transformer_attention_head_count=1,
        max_degree=1,
    )._compute_degree_edge_consistency_loss(
        logits_deg=logits_deg,
        node_presence_mask=node_presence_mask,
        target_edge_counts=target_edge_counts,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_compute_node_count_loss_is_zero_when_expected_count_matches():
    logits_exist = torch.tensor(
        [[10.0, 10.0, -10.0]],
        dtype=torch.float32,
    )
    target_node_counts = torch.tensor([2.0], dtype=torch.float32)

    loss = EquilibriumMatchingDecompositionalNodeGeneratorModule._compute_node_count_loss(
        logits_exist=logits_exist,
        target_node_counts=target_node_counts,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-4)
