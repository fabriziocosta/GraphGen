import numpy as np
import pytest
import torch
import warnings

import conditional_node_field_graph_generator as graphgen
from conditional_node_field_graph_generator.conditional_node_field_generator import (
    GeneratedNodeBatch,
    ConditionalNodeFieldGenerator,
    ConditionalNodeFieldModule,
    GraphConditioningBatch,
    MetricsLogger,
    NodeGenerationBatch,
)
from conditional_node_field_graph_generator.extensions.demo.storage import (
    find_latest_checkpoint,
    save_graph_generator,
)
from conditional_node_field_graph_generator.extensions.demo.visualization import (
    fit_graph_generator,
)
from conditional_node_field_graph_generator.metrics_visualization import (
    plot_metrics,
)
from conditional_node_field_graph_generator.runtime_utils import run_trainer_fit
from conditional_node_field_graph_generator.training_policy import (
    format_restored_checkpoint_summary,
)


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
        node_presence_mask=np.ones((3, 2), dtype=bool)
    )
    assert len(node_batch) == 2
    assert len(generated) == 3


class _OkTrainer:
    def __init__(self):
        self.called_with = None

    def fit(self, model, train_dataloaders=None, val_dataloaders=None, ckpt_path=None):
        self.called_with = (model, train_dataloaders, val_dataloaders, ckpt_path)


class _ExitTrainer:
    def fit(self, model, train_dataloaders=None, val_dataloaders=None, ckpt_path=None):
        raise SystemExit(2)


def test_run_trainer_fit_calls_fit_with_named_loaders():
    trainer = _OkTrainer()
    model = object()
    train_loader = object()
    val_loader = object()

    run_trainer_fit(trainer, model, train_loader, val_loader, context="unit-test")

    assert trainer.called_with == (model, train_loader, val_loader, None)


def test_run_trainer_fit_forwards_checkpoint_path():
    trainer = _OkTrainer()
    model = object()
    train_loader = object()
    val_loader = object()

    run_trainer_fit(
        trainer,
        model,
        train_loader,
        val_loader,
        context="unit-test",
        ckpt_path="/tmp/resume.ckpt",
    )

    assert trainer.called_with == (model, train_loader, val_loader, "/tmp/resume.ckpt")


def test_run_trainer_fit_wraps_system_exit():
    with pytest.raises(RuntimeError, match="unit-test aborted with SystemExit\\(2\\)"):
        run_trainer_fit(_ExitTrainer(), object(), object(), object(), context="unit-test")


class _WarnTrainer:
    def __init__(self):
        self.called = False

    def fit(self, model, train_dataloaders=None, val_dataloaders=None, ckpt_path=None):
        del model, train_dataloaders, val_dataloaders, ckpt_path
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


def test_find_latest_checkpoint_prefers_last_ckpt(tmp_path):
    root = tmp_path / "checkpoints"
    older = root / "run_old"
    newer = root / "run_new"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "best-001.ckpt").write_text("old")
    (newer / "best-003.ckpt").write_text("best")
    (newer / "last.ckpt").write_text("last")

    latest = find_latest_checkpoint(checkpoint_root=root)

    assert latest is not None
    assert latest.endswith("last.ckpt")


class _FitRecorder:
    def __init__(self):
        self.calls = []

    def fit(self, graphs, targets=None, ckpt_path=None):
        self.calls.append(
            {
                "graphs": graphs,
                "targets": targets,
                "ckpt_path": ckpt_path,
            }
        )


def test_fit_graph_generator_resumes_from_latest_checkpoint(tmp_path):
    recorder = _FitRecorder()
    checkpoint_root = tmp_path / "checkpoints"
    run_dir = checkpoint_root / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "last.ckpt").write_text("checkpoint")

    result = fit_graph_generator(
        recorder,
        train_graphs=["g1", "g2"],
        targets=[1, 0],
        resume_latest_checkpoint=True,
        checkpoint_root=checkpoint_root,
    )

    assert result is recorder
    assert recorder.calls[0]["graphs"] == ["g1", "g2"]
    assert recorder.calls[0]["targets"] == [1, 0]
    assert recorder.calls[0]["ckpt_path"].endswith("last.ckpt")


class _SaveableGenerator:
    def __init__(self, training_size=None):
        self.training_graph_conditioning_ = (
            type("_Conditioning", (), {"__len__": lambda self: training_size})()
            if training_size is not None
            else None
        )


def test_save_graph_generator_includes_training_set_size_in_filename(tmp_path):
    generator = _SaveableGenerator(training_size=42)

    filename = save_graph_generator(
        generator,
        model_name="demo-chem",
        model_dir=tmp_path,
    )

    assert filename.startswith("demo-chem-n42-")
    assert filename.endswith(".pkl")


def test_save_graph_generator_omits_training_set_size_when_unavailable(tmp_path):
    generator = _SaveableGenerator(training_size=None)

    filename = save_graph_generator(
        generator,
        model_name="demo-chem",
        model_dir=tmp_path,
    )

    assert filename.startswith("demo-chem-")
    assert "demo-chem-n" not in filename
    assert filename.endswith(".pkl")


def test_build_train_val_subsets_reuses_single_example_for_train_and_val():
    dataset = torch.utils.data.TensorDataset(torch.tensor([[1.0]], dtype=torch.float32))

    train_dataset, val_dataset = ConditionalNodeFieldGenerator._build_train_val_subsets(dataset)

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    assert train_dataset[0][0].item() == 1.0
    assert val_dataset[0][0].item() == 1.0


def test_package_exports_only_new_primary_names():
    assert sorted(graphgen.__all__) == [
        "ConditionalNodeFieldGenerator",
        "ConditionalNodeFieldGraphDecoder",
        "ConditionalNodeFieldGraphGenerator",
    ]


def test_build_train_val_subsets_keeps_both_sides_non_empty_for_two_examples():
    dataset = torch.utils.data.TensorDataset(
        torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    )

    train_dataset, val_dataset = ConditionalNodeFieldGenerator._build_train_val_subsets(dataset)

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1


def test_build_train_val_subsets_rejects_empty_dataset():
    dataset = torch.utils.data.TensorDataset(torch.empty((0, 1), dtype=torch.float32))

    with pytest.raises(ValueError, match="must contain at least one example"):
        ConditionalNodeFieldGenerator._build_train_val_subsets(dataset)


def test_update_ema_metric_tracks_smoothed_validation_signal():
    trainer = type("_Trainer", (), {"callback_metrics": {}, "logged_metrics": {}})()
    pl_module = type("_Module", (), {"_ema_metrics": {}, "early_stopping_ema_alpha": 0.25})()

    first = MetricsLogger._update_ema_metric(trainer, pl_module, "val_node_field", 100.0)
    second = MetricsLogger._update_ema_metric(trainer, pl_module, "val_node_field", 60.0)

    assert first == pytest.approx(100.0)
    assert second == pytest.approx(90.0)
    assert pl_module._ema_metrics["val_node_field"] == pytest.approx(90.0)
    assert trainer.callback_metrics["val_node_field_ema"].item() == pytest.approx(90.0)
    assert trainer.logged_metrics["val_node_field_ema"].item() == pytest.approx(90.0)


def test_restored_checkpoint_summary_uses_node_field_label():
    summary = format_restored_checkpoint_summary(
        early_stopping_monitor="val_total",
        best_checkpoint_score=12.5,
        best_checkpoint_epoch=3,
        raw_best_val_node_field_loss=8.75,
        stopped_epoch=11,
    )

    assert "raw_val_node_field=8.7500" in summary


def test_compute_edge_count_loss_matches_target_on_consistent_probabilities():
    edge_probs = torch.tensor(
        [[[0.0, 0.8], [0.8, 0.0]]],
        dtype=torch.float32,
    )
    node_presence_mask = torch.tensor([[True, True]])
    target_edge_counts = torch.tensor([1.0], dtype=torch.float32)

    loss = ConditionalNodeFieldModule._compute_edge_count_loss(
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

    loss = ConditionalNodeFieldModule(
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

    loss = ConditionalNodeFieldModule._compute_node_count_loss(
        logits_exist=logits_exist,
        target_node_counts=target_node_counts,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_plot_metrics_accepts_node_field_key():
    plot_metrics(
        train_metrics={"total": [10.0, 9.0], "node_field": [8.0, 7.0]},
        val_metrics={"total": [11.0, 10.0], "node_field": [9.0, 8.0]},
        window=2,
    )
