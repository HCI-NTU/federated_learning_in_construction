"""Parameter conversion between Ultralytics YOLO and Flower NumPy format.

Flower exchanges model parameters as a list of NumPy arrays. PyTorch models
expose them via `state_dict()`. This module provides the round-trip conversion.

Key utilities:
  - state_dict_to_numpy / numpy_to_state_dict: parameter exchange
  - classify_param_layers: marks each parameter as 'head' or 'shared' (for FedPer)
  - filter_params_for_aggregation: extracts only the parameters that should be
    federated under a given strategy
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn

# Conditional import — ultralytics imports torchvision which is heavy.
# We only need the Detect class for head identification.
try:
    from ultralytics.nn.modules.head import Detect
except ImportError:
    Detect = None  # type: ignore


# ---- State dict ↔ NumPy ----

def state_dict_to_numpy(model: nn.Module) -> list[np.ndarray]:
    """Convert a PyTorch model's state_dict to a list of NumPy arrays.

    Returns independent copies — the arrays do NOT share memory with the
    model's tensors. This is critical for Flower's parameter exchange:
    the client must be able to return parameters that won't be silently
    mutated by subsequent model updates.

    The list order matches the iteration order of state_dict().keys(),
    which is stable for a fixed model definition.
    """
    return [
        v.detach().cpu().numpy().copy()  # .copy() ensures independent memory
        for v in model.state_dict().values()
    ]


def state_dict_keys(model: nn.Module) -> list[str]:
    """Return the parameter key names in state_dict order."""
    return list(model.state_dict().keys())


def numpy_to_state_dict(
    model: nn.Module,
    arrays: list[np.ndarray],
    strict: bool = True,
) -> None:
    """Load NumPy arrays back into a model's state_dict.

    The arrays list must align with the model's state_dict keys, in the same
    order returned by state_dict_to_numpy(model).

    Handles a NumPy quirk: arithmetic on 0-dim arrays (like BatchNorm's
    num_batches_tracked) can produce Python/NumPy scalars instead of 0-dim
    arrays. We coerce all entries to np.ndarray before converting to tensors.

    Args:
        model: PyTorch module to update
        arrays: List of NumPy arrays in state_dict key order
        strict: Whether to require exact key matching (passed to load_state_dict)
    """
    keys = state_dict_keys(model)
    if len(keys) != len(arrays):
        raise ValueError(
            f"Mismatch: model has {len(keys)} state_dict entries, "
            f"got {len(arrays)} arrays."
        )

    # Preserve dtype and device from the original parameters
    current = model.state_dict()
    new_state = {}
    for k, arr in zip(keys, arrays):
        target_tensor = current[k]
        # Coerce scalars to np.ndarray (np.asarray is a no-op for arrays)
        arr_nd = np.asarray(arr)
        new_state[k] = torch.from_numpy(arr_nd).to(
            dtype=target_tensor.dtype, device=target_tensor.device
        )
    model.load_state_dict(new_state, strict=strict)


# ---- Layer classification (for FedPer) ----

def find_detect_module(model: nn.Module) -> nn.Module | None:
    """Locate the Detect head module in an Ultralytics YOLO model.

    Returns the first nn.Module that is an instance of Detect, or None if
    not found (e.g., classification model, or different model variant).
    """
    if Detect is None:
        return None
    for m in model.modules():
        if isinstance(m, Detect):
            return m
    return None


def classify_param_layers(model: nn.Module) -> dict[str, str]:
    """Map each state_dict key to 'head' or 'shared'.

    'head'   = belongs to the detection head (Detect module)
    'shared' = belongs to backbone + neck (everything else)

    For FedPer, 'shared' parameters are federated; 'head' parameters stay local.

    Returns:
        Dict from state_dict key name to 'head' or 'shared'.
    """
    detect = find_detect_module(model)
    if detect is None:
        # Not a detection model, or different structure. Treat everything as shared.
        return {k: "shared" for k in state_dict_keys(model)}

    # Collect the set of parameter and buffer object IDs that belong to the head.
    # We use id() because some param tensors may share storage but be distinct
    # objects in the state_dict.
    head_param_ids = set()
    for p in detect.parameters():
        head_param_ids.add(id(p))
    for b in detect.buffers():
        head_param_ids.add(id(b))

    # Walk named_parameters and named_buffers in the full model to label each.
    # state_dict() iteration order matches named_parameters() then named_buffers()
    # within each module, so we have to match by name, not by iteration position.
    name_to_obj = {}
    for name, p in model.named_parameters():
        name_to_obj[name] = p
    for name, b in model.named_buffers():
        name_to_obj[name] = b

    result: dict[str, str] = {}
    for key in state_dict_keys(model):
        obj = name_to_obj.get(key)
        if obj is not None and id(obj) in head_param_ids:
            result[key] = "head"
        else:
            result[key] = "shared"
    return result


# ---- FedPer filtering ----

def filter_indices_by_role(
    layer_roles: dict[str, str],
    keys_in_order: list[str],
    role: str,
) -> list[int]:
    """Return indices into the parameter list for entries with the given role.

    Used to extract just the 'shared' (backbone+neck) entries from a full
    parameter list, for FedPer aggregation.
    """
    return [i for i, k in enumerate(keys_in_order) if layer_roles[k] == role]


def extract_shared_params(
    all_params: list[np.ndarray],
    layer_roles: dict[str, str],
    keys_in_order: list[str],
) -> list[np.ndarray]:
    """Pull out only the 'shared' (backbone+neck) parameters from a full list."""
    indices = filter_indices_by_role(layer_roles, keys_in_order, "shared")
    return [all_params[i] for i in indices]


def merge_shared_into_full(
    full_params: list[np.ndarray],
    shared_params: list[np.ndarray],
    layer_roles: dict[str, str],
    keys_in_order: list[str],
) -> list[np.ndarray]:
    """Overwrite the 'shared' entries in a full parameter list with new shared values.

    'head' entries are kept from full_params (i.e., from the local client),
    'shared' entries are replaced by the federated values from shared_params.

    Returns a new list — does not mutate inputs.
    """
    shared_indices = filter_indices_by_role(layer_roles, keys_in_order, "shared")
    if len(shared_indices) != len(shared_params):
        raise ValueError(
            f"Shared param count mismatch: model expects {len(shared_indices)}, "
            f"got {len(shared_params)}."
        )

    out = list(full_params)  # shallow copy
    for idx_in_full, new_arr in zip(shared_indices, shared_params):
        out[idx_in_full] = new_arr
    return out


def summarize_roles(layer_roles: dict[str, str]) -> dict:
    """Return a summary of how many params/roles exist (for logging)."""
    counts: dict[str, int] = {}
    for role in layer_roles.values():
        counts[role] = counts.get(role, 0) + 1
    return counts
