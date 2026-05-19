"""Isolated per-client training (no federation).

Trains one YOLO11 model on a single client's local data, then evaluates on
both the client's own test split and the held-out CIS test split. This is
the "isolated" baseline — what each company gets if they train alone without
federation.

Three of these (one per training client: MOCS, SODA, ACID) form the isolated
condition for Experiment 1. The held-out CIS mAP is the cross-project
generalization metric; the per-client mAP is the in-domain metric.

Usage:
    python -m train.train_isolated \\
        --client MOCS \\
        --data-yaml work/configs/MOCS_n4000_tier_full_s42.yaml \\
        --heldout-data-yaml work/configs/CIS_n0_tier_full_s42.yaml \\
        --model-variant yolo11s.yaml \\
        --epochs 50 \\
        --imgsz 640 \\
        --batch 16 \\
        --device cuda:0 \\
        --results-path results/exp1_isolated_MOCS_n4000_s42.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import yaml as yamllib


logger = logging.getLogger(__name__)


def train_isolated(
    client: str,
    data_yaml: str | Path,
    heldout_data_yaml: str | Path,
    model_variant: str = "yolo11s.yaml",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    work_dir: Optional[str | Path] = None,
    seed: int = 42,
    results_path: Optional[str | Path] = None,
) -> dict:
    """Train one model on one client's data and evaluate on local + held-out.

    Returns the metrics dict that gets written to results_path.
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    from fl.yolo_client import YOLOClient
    from data_prep.schema import load_schema

    data_yaml = str(Path(data_yaml).resolve())
    heldout_data_yaml = str(Path(heldout_data_yaml).resolve())

    if work_dir is None:
        work_dir = Path(f"/tmp/fl_construction/isolated/{client}")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build the model with the correct nc (same logic as YOLOClient)
    nc = YOLOClient._infer_nc_from_data_yaml(data_yaml)
    logger.info(f"[{client}] Building model {model_variant} with nc={nc}")

    if model_variant.endswith(".yaml"):
        det_model = DetectionModel(cfg=model_variant, nc=nc)
        model = YOLO(model_variant)
        model.model = det_model
    else:
        base_yaml = model_variant.replace(".pt", ".yaml")
        det_model = DetectionModel(cfg=base_yaml, nc=nc)
        pretrained = YOLO(model_variant)
        try:
            det_model.load_state_dict(pretrained.model.state_dict(), strict=False)
        except Exception as e:
            logger.warning(f"Could not transfer pretrained weights: {e}")
        model = YOLO(model_variant)
        model.model = det_model

    # Train
    logger.info(f"[{client}] Starting training: epochs={epochs} imgsz={imgsz} batch={batch}")
    t0 = time.time()
    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(work_dir),
        name="train",
        exist_ok=True,
        verbose=False,
        seed=seed,
        plots=False,
        save=True,
    )
    if device is not None:
        train_kwargs["device"] = device
    model.train(**train_kwargs)
    train_secs = time.time() - t0

    # Determine testable class IDs from the schema (for the held-out subset metric)
    schema = load_schema()
    testable_unified = schema.testable_on_heldout("CIS")
    testable_yolo_ids = [schema.to_yolo_id(u) for u in testable_unified]

    # Evaluate on the client's own val/test split
    from train.evaluate import evaluate_on_yaml
    logger.info(f"[{client}] Evaluating on local test split: {data_yaml}")
    local_eval = evaluate_on_yaml(
        model=model,
        data_yaml=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        work_dir=work_dir,
        run_name="eval_local",
        testable_yolo_ids=testable_yolo_ids,
    )
    logger.info(
        f"[{client}] Local mAP50={local_eval['map50']:.4f} "
        f"mAP50-95={local_eval['map50_95']:.4f}"
    )

    # Evaluate on the held-out CIS test split
    logger.info(f"[{client}] Evaluating on held-out: {heldout_data_yaml}")
    heldout_eval = evaluate_on_yaml(
        model=model,
        data_yaml=heldout_data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        work_dir=work_dir,
        run_name="eval_heldout",
        testable_yolo_ids=testable_yolo_ids,
    )
    logger.info(
        f"[{client}] Held-out mAP50={heldout_eval['map50']:.4f} "
        f"mAP50-95={heldout_eval['map50_95']:.4f} "
        f"mAP50_testable={heldout_eval.get('map50_testable', 0):.4f}"
    )

    result = {
        "method": "isolated",
        "client": client,
        "data_yaml": data_yaml,
        "heldout_data_yaml": heldout_data_yaml,
        "model_variant": model_variant,
        "nc": nc,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "seed": seed,
        "train_time_seconds": train_secs,
        "local_eval": local_eval,
        "heldout_eval": heldout_eval,
    }

    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"[{client}] Wrote results to {results_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--client", required=True, help="Client name (for logging)")
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--heldout-data-yaml", required=True)
    parser.add_argument("--model-variant", default="yolo11s.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-path", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    train_isolated(
        client=args.client,
        data_yaml=args.data_yaml,
        heldout_data_yaml=args.heldout_data_yaml,
        model_variant=args.model_variant,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        work_dir=args.work_dir,
        seed=args.seed,
        results_path=args.results_path,
    )


if __name__ == "__main__":
    main()
