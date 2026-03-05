import numpy as np
import pytest

from eqm_decompositional_graph_generator.conditional_node_generator_base import (
    GeneratedNodeBatch,
    GraphConditioningBatch,
    NodeGenerationBatch,
)
from eqm_decompositional_graph_generator.lightning_utils import run_trainer_fit


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
