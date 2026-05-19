"""Flower server entry point for the construction FL study.

Starts a Flower server with the requested strategy and waits for clients
to connect. Returns the final aggregated parameters and per-round metrics.

Usage (from a script):
    from fl.server import run_fl_server
    history = run_fl_server(
        method="fedper",
        num_rounds=10,
        model_variant="yolo11s.yaml",
        num_clients=3,
        server_address="0.0.0.0:8080",
        results_path="results/exp1_fedper_n4000.json",
    )

The server is intentionally lightweight — it does not own data and does not
run training itself. It only aggregates parameter updates from the clients.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters

from fl.parameter_utils import (
    state_dict_to_numpy,
    state_dict_keys,
    classify_param_layers,
    extract_shared_params,
)
from fl.strategies import build_strategy


logger = logging.getLogger(__name__)


def build_initial_parameters(
    method: str,
    model_variant: str = "yolo11s.yaml",
    nc: int = 17,
) -> tuple[fl.common.Parameters, dict]:
    """Initialize a fresh YOLO model and extract its parameters in Flower format.

    Critical: `nc` must match what the clients use, or the initial parameters
    sent to clients will fail to load (Ultralytics' default nc=80 doesn't match
    the 17-class construction schema). Set explicitly to match your data.yaml.

    For FedPer ('fedper'), only the shared (backbone+neck) parameters are
    returned. For FedAvg/FedProx, the full state_dict is returned.

    Args:
        method: One of 'fedavg', 'fedprox', 'fedper'
        model_variant: YOLO config or weights filename
        nc: Number of detection classes. Must match clients' data.yaml.

    Returns:
        (Flower Parameters, info dict with parameter counts and metadata)
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    logger.info(f"Initializing model {model_variant} (nc={nc}) on server...")

    if model_variant.endswith(".yaml"):
        det_model = DetectionModel(cfg=model_variant, nc=nc)
        torch_model = det_model
    else:
        base_yaml = model_variant.replace(".pt", ".yaml")
        det_model = DetectionModel(cfg=base_yaml, nc=nc)
        pretrained = YOLO(model_variant)
        try:
            det_model.load_state_dict(pretrained.model.state_dict(), strict=False)
        except Exception as e:
            logger.warning(f"Could not transfer pretrained weights to nc={nc} model: {e}")
        torch_model = det_model

    full_params = state_dict_to_numpy(torch_model)
    keys = state_dict_keys(torch_model)
    roles = classify_param_layers(torch_model)

    method = method.lower()
    if method == "fedper":
        shared_params = extract_shared_params(full_params, roles, keys)
        params = ndarrays_to_parameters(shared_params)
        info = {
            "method": method,
            "model_variant": model_variant,
            "nc": nc,
            "n_params_exchanged": len(shared_params),
            "n_params_total": len(full_params),
            "shared_fraction": len(shared_params) / len(full_params),
        }
    else:
        params = ndarrays_to_parameters(full_params)
        info = {
            "method": method,
            "model_variant": model_variant,
            "nc": nc,
            "n_params_exchanged": len(full_params),
            "n_params_total": len(full_params),
            "shared_fraction": 1.0,
        }
    logger.info(f"Initial parameters: {info}")
    return params, info


def run_fl_server(
    method: str,
    num_rounds: int,
    num_clients: int,
    model_variant: str = "yolo11s.yaml",
    nc: int = 17,
    server_address: str = "0.0.0.0:8080",
    proximal_mu: float = 0.01,
    epochs_per_round: int = 1,
    results_path: Optional[str | Path] = None,
) -> dict:
    """Run a federated training session and return per-round history.

    Args:
        method: 'fedavg', 'fedprox', or 'fedper'.
        num_rounds: Number of FL rounds.
        num_clients: Expected number of clients to participate per round.
        model_variant: YOLO variant (e.g., 'yolo11s.yaml' or 'yolo11s.pt').
        nc: Number of detection classes (must match clients' data.yaml).
        server_address: 'host:port' for the Flower server to bind.
        proximal_mu: FedProx coefficient (ignored for fedavg/fedper).
        epochs_per_round: Per-round local epochs each client trains.
        results_path: Optional JSON file to write per-round history to.

    Returns:
        Dict with 'losses_distributed', 'metrics_distributed_fit',
        'metrics_distributed', 'losses_centralized'.
    """
    initial_params, init_info = build_initial_parameters(method, model_variant, nc=nc)

    strategy = build_strategy(
        method=method,
        initial_parameters=initial_params,
        proximal_mu=proximal_mu,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        epochs_per_round=epochs_per_round,
    )

    logger.info(
        f"Starting Flower server: method={method}, num_rounds={num_rounds}, "
        f"clients={num_clients}, address={server_address}"
    )

    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Serialize the history
    serialized = {
        "method": method,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "model_variant": model_variant,
        "init_info": init_info,
        "losses_distributed": [
            {"round": r, "value": v} for r, v in history.losses_distributed
        ],
        "metrics_distributed_fit": {
            k: [{"round": r, "value": v} for r, v in vals]
            for k, vals in history.metrics_distributed_fit.items()
        },
        "metrics_distributed": {
            k: [{"round": r, "value": v} for r, v in vals]
            for k, vals in history.metrics_distributed.items()
        },
    }

    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(serialized, f, indent=2)
        logger.info(f"Wrote results to {results_path}")

    return serialized


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--method", required=True,
        choices=["fedavg", "fedprox", "fedper"],
    )
    parser.add_argument("--num-rounds", type=int, required=True)
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--model-variant", default="yolo11s.yaml")
    parser.add_argument(
        "--nc", type=int, default=17,
        help="Number of detection classes (must match clients' data.yaml)",
    )
    parser.add_argument("--server-address", default="0.0.0.0:8080")
    parser.add_argument("--proximal-mu", type=float, default=0.01)
    parser.add_argument("--epochs-per-round", type=int, default=1)
    parser.add_argument("--results-path", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_fl_server(
        method=args.method,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        model_variant=args.model_variant,
        nc=args.nc,
        server_address=args.server_address,
        proximal_mu=args.proximal_mu,
        epochs_per_round=args.epochs_per_round,
        results_path=args.results_path,
    )


if __name__ == "__main__":
    main()
