"""Atomic, validated persistence for long-running held-out backbone training."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from src.training.agent_state import _capture_rng_state, _move_optimizer_state_to_device, _restore_rng_state


STATE_SCHEMA_VERSION = 3


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_backbone_training_state(
    path,
    feature_extractor,
    diffusion_model,
    optimizer,
    completed_epochs,
    best_validation_score,
    history,
    training_signature,
):
    """Atomically save a complete held-out backbone epoch boundary."""
    completed_epochs = int(completed_epochs)
    history = list(history)
    if completed_epochs < 0 or completed_epochs != len(history):
        raise ValueError("completed_epochs must equal the number of history entries.")
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "completed_epochs": completed_epochs,
        "best_validation_score": float(best_validation_score),
        "history": history,
        "training_signature": dict(training_signature),
        "diffusion_model_family": getattr(diffusion_model, "model_family", "unknown"),
        "diffusion_config": getattr(diffusion_model, "diffusion_config", None),
        "feature_extractor_state_dict": feature_extractor.state_dict(),
        "diffusion_model_state_dict": diffusion_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": _capture_rng_state(),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    return path


def restore_backbone_training_state(
    path,
    feature_extractor,
    diffusion_model,
    optimizer,
    device,
    expected_signature,
):
    """Restore a held-out backbone only when its immutable setup matches."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Backbone training state does not exist: {path}")
    payload = _torch_load(path, map_location="cpu")
    required = {
        "schema_version",
        "completed_epochs",
        "best_validation_score",
        "history",
        "training_signature",
        "diffusion_model_family",
        "diffusion_config",
        "feature_extractor_state_dict",
        "diffusion_model_state_dict",
        "optimizer_state_dict",
        "rng_state",
    }
    if not isinstance(payload, dict) or required - set(payload):
        raise ValueError(f"Invalid backbone training state: {path}")
    if int(payload["schema_version"]) != STATE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported backbone training state schema {payload['schema_version']}; "
            f"expected {STATE_SCHEMA_VERSION}."
        )
    if payload["training_signature"] != dict(expected_signature):
        raise ValueError("Backbone training state is incompatible with the requested run configuration.")
    if payload["diffusion_model_family"] != getattr(diffusion_model, "model_family", "unknown"):
        raise ValueError("Backbone training state belongs to a different model family.")
    if payload["diffusion_config"] != getattr(diffusion_model, "diffusion_config", None):
        raise ValueError("Backbone training state diffusion configuration is incompatible with the requested run.")
    history = list(payload["history"])
    completed_epochs = int(payload["completed_epochs"])
    if completed_epochs < 0 or completed_epochs != len(history):
        raise ValueError("Backbone training state has inconsistent completed_epochs/history values.")
    feature_extractor.load_state_dict(payload["feature_extractor_state_dict"])
    diffusion_model.load_state_dict(payload["diffusion_model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device)
    _restore_rng_state(payload["rng_state"])
    return {
        "path": str(path),
        "completed_epochs": completed_epochs,
        "best_validation_score": float(payload["best_validation_score"]),
        "history": history,
    }
