import sys


def run_trainer_fit(trainer, model, train_loader, val_loader, context: str) -> None:
    """Run Lightning training while surfacing notebook-hostile SystemExit failures."""
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except SystemExit as exc:
        code = exc.code if exc.code is not None else "None"
        argv_preview = " ".join(sys.argv[:5])
        raise RuntimeError(
            f"{context} aborted with SystemExit({code}). "
            "This usually means some code inside the training stack called "
            "CLI-style argument parsing or sys.exit(). "
            f"Current sys.argv starts with: {argv_preview!r}"
        ) from exc
