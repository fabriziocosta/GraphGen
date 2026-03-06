"""Training policy helpers for the node engine."""

import contextlib
import logging
import os
import sys
import uuid
from typing import Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


_LITLOGGER_TIP_SNIPPET = "For seamless cloud logging and experiment tracking"


class _SuppressLitLoggerTipFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        return _LITLOGGER_TIP_SNIPPET not in message


def _install_lightning_log_filters() -> None:
    logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
    for existing_filter in logger.filters:
        if isinstance(existing_filter, _SuppressLitLoggerTipFilter):
            return
    logger.addFilter(_SuppressLitLoggerTipFilter())


@contextlib.contextmanager
def suppress_output():
    """Temporarily redirect stdout/stderr to os.devnull."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def build_training_callbacks(
    generator_name: str,
    checkpoint_root_dir: str,
    early_stopping_monitor: str,
    early_stopping_mode: str,
    enable_early_stopping: bool,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    metrics_logger: pl.callbacks.Callback,
) -> Tuple[list, str, ModelCheckpoint]:
    """Build checkpoint and early-stopping callbacks for training."""
    callbacks = [metrics_logger]
    checkpoint_dir = os.path.join(
        checkpoint_root_dir,
        f"{generator_name}_{uuid.uuid4().hex}",
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:03d}-{val_total:.4f}",
        monitor=early_stopping_monitor,
        mode=early_stopping_mode,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        save_weights_only=False,
    )
    callbacks.append(checkpoint_callback)
    if enable_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping_monitor,
                mode=early_stopping_mode,
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
            )
        )
    return callbacks, checkpoint_dir, checkpoint_callback


def create_trainer(
    maximum_epochs: int,
    callbacks: list,
    artifact_root_dir: str,
    train_loader_length: int,
) -> pl.Trainer:
    """Create a Lightning trainer with the node engine defaults."""
    _install_lightning_log_filters()
    return pl.Trainer(
        max_epochs=maximum_epochs,
        callbacks=callbacks,
        logger=True,
        default_root_dir=artifact_root_dir,
        enable_checkpointing=True,
        enable_model_summary=False,
        enable_progress_bar=False,
        log_every_n_steps=max(1, min(10, train_loader_length)),
    )


def format_restored_checkpoint_summary(
    early_stopping_monitor: str,
    best_checkpoint_score,
    best_checkpoint_epoch,
    raw_best_val_eqm,
    stopped_epoch: int,
) -> str:
    """Format the restored-checkpoint summary with key fields first."""
    summary_parts = []
    if best_checkpoint_epoch is not None:
        summary_parts.append(f"best_epoch={best_checkpoint_epoch + 1}")
    summary_parts.append(
        f"{early_stopping_monitor}={best_checkpoint_score:.4f}"
        if best_checkpoint_score is not None
        else f"{early_stopping_monitor}=unknown"
    )
    if raw_best_val_eqm is not None:
        summary_parts.append(f"raw_val_eqm={raw_best_val_eqm:.4f}")
    summary_parts.append(f"stopped_epoch={stopped_epoch}")
    return "Restored best checkpoint: " + ", ".join(summary_parts)
