"""Client launcher: starts a single Flower client and connects to the server.

In our setup, three of these run concurrently (one per training client),
each pointed at a different data.yaml.

Usage (from a script or the experiment driver):
    python -m fl.run_client \\
        --client-name MOCS \\
        --data-yaml work/configs/MOCS_n4000_tier_full_s42.yaml \\
        --model-variant yolo11s.yaml \\
        --method fedavg \\
        --server-address 127.0.0.1:8080 \\
        --epochs-per-round 1 \\
        --imgsz 640 \\
        --batch 16 \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import flwr as fl

from fl.yolo_client import YOLOClient


def _method_to_federate_role(method: str) -> str:
    return "shared" if method.lower() == "fedper" else "all"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--client-name", required=True)
    parser.add_argument("--data-yaml", required=True, help="Path to Ultralytics data.yaml")
    parser.add_argument("--model-variant", default="yolo11s.yaml")
    parser.add_argument(
        "--method", required=True, choices=["fedavg", "fedprox", "fedper"],
    )
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--epochs-per-round", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--proximal-mu", type=float, default=0.01)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"[{args.client_name}] %(asctime)s %(levelname)s %(message)s",
    )

    federate_role = _method_to_federate_role(args.method)
    proximal_mu = args.proximal_mu if args.method == "fedprox" else 0.0

    work_dir = args.work_dir or f"/tmp/flower_yolo/{args.client_name}"

    client = YOLOClient(
        client_name=args.client_name,
        data_yaml=args.data_yaml,
        model_variant=args.model_variant,
        epochs_per_round=args.epochs_per_round,
        imgsz=args.imgsz,
        batch=args.batch,
        federate_role=federate_role,
        device=args.device,
        proximal_mu=proximal_mu,
        work_dir=work_dir,
        seed=args.seed,
    )

    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
