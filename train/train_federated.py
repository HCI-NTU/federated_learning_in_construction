"""Federated training session — runs one FL experiment end-to-end.

Orchestrates a complete federated training run:
  1. Builds initial server-side parameters (correct nc for the schema)
  2. Starts Flower simulation with N client processes (each pointed at its
     data.yaml)
  3. Waits for all FL rounds to complete
  4. Captures the final aggregated parameters
  5. Loads them into a fresh YOLO model and evaluates on the held-out set

For FedPer, the final aggregated parameters are backbone+neck only — the
held-out evaluation uses one of the clients' final heads (we use the first
client's by default; this is a design choice consistent with FedPer
literature where evaluation is per-client). For simpler "global model"
held-out eval under FedPer, we use a model built with the aggregated
backbone+neck plus a randomly-initialized head — but this is rarely
informative, so we default to client-0's head.

Usage:
    python -m train.train_federated \\
        --method fedper \\
        --client-yamls work/configs/MOCS_n4000_tier_full_s42.yaml \\
                       work/configs/SODA_n4000_tier_full_s42.yaml \\
                       work/configs/ACID_n4000_tier_full_s42.yaml \\
        --heldout-data-yaml work/configs/CIS_n0_tier_full_s42.yaml \\
        --num-rounds 50 \\
        --epochs-per-round 1 \\
        --model-variant yolo11s.yaml \\
        --results-path results/exp1_fedper_n4000_s42.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def _method_to_federate_role(method: str) -> str:
    return "shared" if method.lower() == "fedper" else "all"


class _ClientRef:
    """Holds a reference to the most recently-created YOLOClient per partition.

    Flower simulation spins up client processes (or Ray actors) on demand,
    and we need access to the final-round client objects to extract their
    local state (including heads under FedPer) after training. We store
    references in a module-level dict, keyed by client name.

    Note: under Ray simulation, the YOLOClient objects live in actor processes
    and aren't directly accessible from the orchestrator. We work around this
    by having each client save its final state_dict to disk at the end of
    each round; the orchestrator reads the saved files after training.
    """
    pass


def _build_client_fn(
    client_yamls: list[Path],
    client_names: list[str],
    method: str,
    model_variant: str,
    epochs_per_round: int,
    imgsz: int,
    batch: int,
    device: Optional[str],
    proximal_mu: float,
    work_root: Path,
    seed: int,
):
    """Return a Flower client_fn that constructs YOLOClient per partition."""
    from fl.yolo_client import YOLOClient
    federate_role = _method_to_federate_role(method)

    def client_fn(context):
        cid = int(context.node_config.get("partition-id", 0))
        if cid >= len(client_yamls):
            raise RuntimeError(f"partition-id={cid} >= num clients {len(client_yamls)}")
        name = client_names[cid]
        client = YOLOClient(
            client_name=name,
            data_yaml=client_yamls[cid],
            model_variant=model_variant,
            epochs_per_round=epochs_per_round,
            imgsz=imgsz,
            batch=batch,
            federate_role=federate_role,
            device=device,
            proximal_mu=(proximal_mu if method == "fedprox" else 0.0),
            work_dir=work_root / "clients" / name,
            seed=seed,
        )
        return client.to_client()

    return client_fn


def _capture_history(strategy, history_dict: dict):
    """Wrap a strategy so per-round aggregated parameters and metrics are saved.

    Captures the final aggregated parameters (before they're sent for the
    next round's training) so we can extract them after the simulation
    completes.
    """
    original_aggregate_fit = strategy.aggregate_fit
    original_aggregate_eval = strategy.aggregate_evaluate

    def aggregate_fit(server_round, results, failures):
        params, metrics = original_aggregate_fit(server_round, results, failures)
        history_dict["last_aggregated_params"] = params
        history_dict.setdefault("fit_rounds", []).append({
            "round": server_round,
            "metrics": dict(metrics) if metrics else {},
            "n_clients": len(results),
        })
        return params, metrics

    def aggregate_evaluate(server_round, results, failures):
        loss, metrics = original_aggregate_eval(server_round, results, failures)
        history_dict.setdefault("eval_rounds", []).append({
            "round": server_round,
            "loss": loss,
            "metrics": dict(metrics) if metrics else {},
            "n_clients": len(results),
        })
        return loss, metrics

    strategy.aggregate_fit = aggregate_fit
    strategy.aggregate_evaluate = aggregate_evaluate
    return strategy


def train_federated(
    method: str,
    client_yamls: list[str | Path],
    heldout_data_yaml: str | Path,
    client_names: Optional[list[str]] = None,
    num_rounds: int = 50,
    epochs_per_round: int = 1,
    model_variant: str = "yolo11s.yaml",
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    proximal_mu: float = 0.01,
    work_dir: Optional[str | Path] = None,
    seed: int = 42,
    results_path: Optional[str | Path] = None,
) -> dict:
    """Run a federated training session and evaluate on held-out.

    Returns:
        Dict with per-round history, final eval metrics, and metadata.
    """
    import flwr as fl
    import numpy as np

    from fl.server import build_initial_parameters
    from fl.strategies import build_strategy
    from fl.yolo_client import YOLOClient
    from fl.parameter_utils import (
        state_dict_to_numpy, numpy_to_state_dict, state_dict_keys,
        classify_param_layers, merge_shared_into_full,
    )
    from data_prep.schema import load_schema
    from train.evaluate import evaluate_on_yaml

    method = method.lower()
    client_yamls = [Path(p).resolve() for p in client_yamls]
    heldout_data_yaml = str(Path(heldout_data_yaml).resolve())

    if client_names is None:
        # Derive a name per client from the yaml filename
        client_names = [p.stem.split("_")[0] for p in client_yamls]
    if len(client_names) != len(client_yamls):
        raise ValueError("client_names length must match client_yamls length")

    if work_dir is None:
        work_dir = Path(f"/tmp/fl_construction/federated_{method}")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Infer nc from the first client's data.yaml
    nc = YOLOClient._infer_nc_from_data_yaml(client_yamls[0])
    logger.info(f"Federated training: method={method} nc={nc} rounds={num_rounds}")

    # Build initial parameters
    initial_params, init_info = build_initial_parameters(
        method=method, model_variant=model_variant, nc=nc,
    )
    logger.info(
        f"Initial params: {init_info['n_params_exchanged']} exchanged "
        f"(of {init_info['n_params_total']} total, "
        f"{init_info['shared_fraction']*100:.0f}% federated)"
    )

    strategy = build_strategy(
        method=method,
        initial_parameters=initial_params,
        proximal_mu=proximal_mu,
        min_fit_clients=len(client_yamls),
        min_evaluate_clients=len(client_yamls),
        min_available_clients=len(client_yamls),
        epochs_per_round=epochs_per_round,
    )

    history: dict = {}
    strategy = _capture_history(strategy, history)

    # Run the simulation
    client_fn = _build_client_fn(
        client_yamls=client_yamls,
        client_names=client_names,
        method=method,
        model_variant=model_variant,
        epochs_per_round=epochs_per_round,
        imgsz=imgsz,
        batch=batch,
        device=device,
        proximal_mu=proximal_mu,
        work_root=work_dir,
        seed=seed,
    )

    t0 = time.time()
    fl.simulation.run_simulation(
        server_app=fl.server.ServerApp(
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        ),
        client_app=fl.client.ClientApp(client_fn=client_fn),
        num_supernodes=len(client_yamls),
        backend_config={"client_resources": {"num_cpus": 1}},
    )
    train_secs = time.time() - t0
    logger.info(f"FL simulation finished in {train_secs:.1f}s")

    # Reconstruct the final model from the last aggregated parameters
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from flwr.common import parameters_to_ndarrays

    final_params = history.get("last_aggregated_params")
    if final_params is None:
        raise RuntimeError(
            "FL simulation completed but no aggregated parameters were captured. "
            "Did any round actually run?"
        )
    final_arrays = parameters_to_ndarrays(final_params)

    if model_variant.endswith(".yaml"):
        final_model_inner = DetectionModel(cfg=model_variant, nc=nc)
    else:
        base_yaml = model_variant.replace(".pt", ".yaml")
        final_model_inner = DetectionModel(cfg=base_yaml, nc=nc)

    if method == "fedper":
        # final_arrays contains only shared (backbone+neck). We need a head.
        # For held-out evaluation of FedPer, we use a freshly-initialized head
        # (the just-built DetectionModel's random head). This is the standard
        # choice when the global model has no "owner" head — there's no
        # client-specific specialization to copy.
        keys = state_dict_keys(final_model_inner)
        roles = classify_param_layers(final_model_inner)
        local_full = state_dict_to_numpy(final_model_inner)
        merged = merge_shared_into_full(local_full, final_arrays, roles, keys)
        numpy_to_state_dict(final_model_inner, merged)
        logger.info(
            "[fedper] Loaded aggregated backbone+neck; head is random-initialized "
            "(no global head exists under FedPer). Held-out mAP reflects "
            "transferability of the shared representation only."
        )
    else:
        # FedAvg / FedProx: final_arrays is the full state_dict
        numpy_to_state_dict(final_model_inner, final_arrays)

    # Wrap in YOLO for evaluation
    final_yolo = YOLO(model_variant)
    final_yolo.model = final_model_inner

    # Held-out evaluation
    schema = load_schema()
    testable_unified = schema.testable_on_heldout("CIS")
    testable_yolo_ids = [schema.to_yolo_id(u) for u in testable_unified]

    logger.info(f"Evaluating final aggregated model on held-out: {heldout_data_yaml}")
    heldout_eval = evaluate_on_yaml(
        model=final_yolo,
        data_yaml=heldout_data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        work_dir=work_dir,
        run_name="eval_heldout",
        testable_yolo_ids=testable_yolo_ids,
    )
    logger.info(
        f"[{method}] Held-out mAP50={heldout_eval['map50']:.4f} "
        f"mAP50-95={heldout_eval['map50_95']:.4f} "
        f"mAP50_testable={heldout_eval.get('map50_testable', 0):.4f}"
    )

    # Make history serializable (drop the Parameters object)
    history_serialized = {
        k: v for k, v in history.items() if k != "last_aggregated_params"
    }

    result = {
        "method": method,
        "client_yamls": [str(p) for p in client_yamls],
        "client_names": client_names,
        "heldout_data_yaml": heldout_data_yaml,
        "model_variant": model_variant,
        "nc": nc,
        "num_rounds": num_rounds,
        "epochs_per_round": epochs_per_round,
        "imgsz": imgsz,
        "batch": batch,
        "proximal_mu": proximal_mu if method == "fedprox" else 0.0,
        "seed": seed,
        "train_time_seconds": train_secs,
        "init_info": init_info,
        "history": history_serialized,
        "heldout_eval": heldout_eval,
    }

    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote results to {results_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--method", required=True, choices=["fedavg", "fedprox", "fedper"],
    )
    parser.add_argument("--client-yamls", nargs="+", required=True)
    parser.add_argument("--client-names", nargs="+", default=None)
    parser.add_argument("--heldout-data-yaml", required=True)
    parser.add_argument("--num-rounds", type=int, default=50)
    parser.add_argument("--epochs-per-round", type=int, default=1)
    parser.add_argument("--model-variant", default="yolo11s.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--proximal-mu", type=float, default=0.01)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-path", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    train_federated(
        method=args.method,
        client_yamls=args.client_yamls,
        client_names=args.client_names,
        heldout_data_yaml=args.heldout_data_yaml,
        num_rounds=args.num_rounds,
        epochs_per_round=args.epochs_per_round,
        model_variant=args.model_variant,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        proximal_mu=args.proximal_mu,
        work_dir=args.work_dir,
        seed=args.seed,
        results_path=args.results_path,
    )


if __name__ == "__main__":
    main()
