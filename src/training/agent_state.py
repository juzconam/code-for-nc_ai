"""Atomic, validated persistence for long-running residual-agent training."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


STATE_SCHEMA_VERSION = 2


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        return torch.load(path, map_location=map_location)


def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    required = {"python", "numpy", "torch", "cuda"}
    if not isinstance(state, dict) or required - set(state):
        raise ValueError("Training state is missing random-number-generator state.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_agent_training_state(
    path,
    agent,
    optimizer,
    completed_epochs,
    best_val_loss,
    history,
    training_signature,
):
    """Write a complete epoch boundary atomically so runs can resume exactly."""
    completed_epochs = int(completed_epochs)
    history = list(history)
    if completed_epochs < 0 or completed_epochs != len(history):
        raise ValueError("completed_epochs must equal the number of history entries.")

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "completed_epochs": completed_epochs,
        "best_val_loss": float(best_val_loss),
        "history": history,
        "training_signature": dict(training_signature),
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": _capture_rng_state(),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    return path


def restore_agent_training_state(path, agent, optimizer, device, expected_signature):
    """Restore an agent/optimizer state only when its immutable training setup matches."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Agent training state does not exist: {path}")
    payload = _torch_load(path, map_location="cpu")
    required = {
        "schema_version",
        "completed_epochs",
        "best_val_loss",
        "history",
        "training_signature",
        "agent_state_dict",
        "optimizer_state_dict",
        "rng_state",
    }
    if not isinstance(payload, dict) or required - set(payload):
        raise ValueError(f"Invalid agent training state: {path}")
    if int(payload["schema_version"]) != STATE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported agent training state schema {payload['schema_version']}; "
            f"expected {STATE_SCHEMA_VERSION}."
        )
    actual_signature = payload["training_signature"]
    if actual_signature != dict(expected_signature):
        raise ValueError(
            "Agent training state is incompatible with the requested run configuration. "
            f"state={actual_signature}, requested={dict(expected_signature)}"
        )

    history = list(payload["history"])
    completed_epochs = int(payload["completed_epochs"])
    if completed_epochs < 0 or completed_epochs != len(history):
        raise ValueError("Agent training state has inconsistent completed_epochs/history values.")
    agent.load_state_dict(payload["agent_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device)
    _restore_rng_state(payload["rng_state"])
    return {
        "path": str(path),
        "completed_epochs": completed_epochs,
        "best_val_loss": float(payload["best_val_loss"]),
        "history": history,
    }
