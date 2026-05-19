"""Federated learning strategies for the construction FL study.

Three methods are supported:

  - FedAvg  : weighted averaging of full client state_dicts (Flower stock)
  - FedProx : FedAvg + a proximal regularization term applied client-side.
              The server is identical to FedAvg; the proximal_mu coefficient
              is sent to clients via the fit_config.
  - FedPer  : weighted averaging applied to backbone+neck parameters only.
              Each client keeps its own detection head locally across rounds.
              The aggregation logic is structurally the same as FedAvg, but
              clients exchange a shorter parameter list (shared params only)
              and merge them with their local head before training.

All strategies are constructed via build_strategy() so the experiment driver
can select by name from a single entry point.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import flwr as fl
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


logger = logging.getLogger(__name__)


# ---- FedAvg: stock Flower, with a tiny config injector ----

def _make_fit_config_fn(extra_config: dict) -> Callable[[int], dict]:
    """Return a function that produces per-round config sent to each client's fit()."""
    def fit_config(server_round: int) -> dict:
        cfg = {"server_round": server_round, **extra_config}
        return cfg
    return fit_config


# ---- FedProx: stock Flower with proximal_mu in the fit config ----
#
# Flower has a built-in FedProx strategy, but using FedAvg with a configured
# proximal_mu is functionally equivalent for our purposes — the proximal term
# is applied client-side (see YOLOClient._install_fedprox_callback). Keeping
# the server logic identical to FedAvg simplifies result interpretation.


# ---- FedPer ----
#
# FedPer is FedAvg restricted to a subset of the parameters. The clients
# already handle the role-based extract/merge in YOLOClient when
# federate_role="shared" — they only ever send/receive backbone+neck params.
# So the server doesn't need any special handling: from the server's point
# of view, the parameter list is just shorter.
#
# This means FedPer can use plain FedAvg on the server side, provided:
#   - Initial parameters sent to clients are the shared-subset only
#   - Clients are configured with federate_role="shared"
#
# That's it. The strategy IS FedAvg; the difference lives entirely in the
# client wrapper and what gets passed as initial_parameters.


def build_strategy(
    method: str,
    initial_parameters: Optional[Parameters] = None,
    proximal_mu: float = 0.0,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 1,
    min_evaluate_clients: int = 1,
    min_available_clients: int = 1,
    epochs_per_round: int = 1,
    on_fit_config_fn: Optional[Callable[[int], dict]] = None,
    on_evaluate_config_fn: Optional[Callable[[int], dict]] = None,
    fit_metrics_aggregation_fn: Optional[Callable] = None,
    evaluate_metrics_aggregation_fn: Optional[Callable] = None,
) -> fl.server.strategy.Strategy:
    """Build a Flower Strategy by method name.

    Args:
        method: One of "fedavg", "fedprox", "fedper" (case-insensitive).
        initial_parameters: The initial server-held parameters. For "fedper",
            this MUST be the shared-subset only (backbone+neck), not the full
            state_dict. The caller is responsible for extracting it.
        proximal_mu: FedProx proximal coefficient. Ignored for fedavg/fedper.
            Sent to clients via fit_config so they can apply it locally.
        fraction_*: Standard Flower sampling fractions.
        min_*: Standard Flower minimums.
        epochs_per_round: Forwarded to clients via fit_config.
        on_fit_config_fn: Optional override of the per-round fit-config function.
            If None, a default is built that includes epochs_per_round and
            proximal_mu.
        on_evaluate_config_fn: Optional override of the per-round eval config.
        fit_metrics_aggregation_fn / evaluate_metrics_aggregation_fn: Optional
            functions for aggregating client metrics into a server-side summary.

    Returns:
        A configured Flower Strategy.
    """
    method = method.lower()
    valid = {"fedavg", "fedprox", "fedper"}
    if method not in valid:
        raise ValueError(f"Unknown method '{method}'. Must be one of: {sorted(valid)}")

    # Build the per-round fit config based on the method
    if on_fit_config_fn is None:
        extra: dict = {"epochs_per_round": epochs_per_round}
        if method == "fedprox":
            extra["proximal_mu"] = proximal_mu
        on_fit_config_fn = _make_fit_config_fn(extra)

    if on_evaluate_config_fn is None:
        on_evaluate_config_fn = lambda r: {"server_round": r}

    if fit_metrics_aggregation_fn is None:
        fit_metrics_aggregation_fn = _weighted_metrics_avg
    if evaluate_metrics_aggregation_fn is None:
        evaluate_metrics_aggregation_fn = _weighted_metrics_avg

    # All three methods use plain FedAvg as the server strategy. The
    # differences are entirely client-side (epochs/mu via config) or
    # in what subset of parameters is exchanged (FedPer).
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    return strategy


def _weighted_metrics_avg(
    metrics: list[tuple[int, dict[str, Scalar]]],
) -> dict[str, Scalar]:
    """Weighted average of client-reported scalar metrics.

    Each tuple is (num_examples, metrics_dict). Weights are num_examples.
    Non-numeric metric values are ignored.
    """
    if not metrics:
        return {}
    total_weight = sum(n for n, _ in metrics)
    if total_weight == 0:
        return {}

    aggregated: dict[str, float] = {}
    counts: dict[str, int] = {}
    for n, m in metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                aggregated[k] = aggregated.get(k, 0.0) + float(v) * n
                counts[k] = counts.get(k, 0) + n

    return {k: aggregated[k] / counts[k] for k in aggregated}
