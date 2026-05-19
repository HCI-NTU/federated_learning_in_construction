"""Evaluation helper used by all training scripts.

Wraps Ultralytics `model.val()` and extracts the metrics our experiments
report:
  - mAP@50 (overall and per-class)
  - mAP@50-95 (overall and per-class)
  - per-class breakdowns separating "testable" (in held-out) classes from
    train-only classes

A single trained model is typically evaluated on multiple data.yamls:
  - the client's own test split (per-client metric)
  - the held-out CIS test split (cross-project generalization)

This module abstracts that pattern so each training script just calls
evaluate_on_yaml(model, yaml_path).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml as yamllib


logger = logging.getLogger(__name__)


def _read_class_names(data_yaml: str | Path) -> dict[int, str]:
    """Read the YOLO class names from a data.yaml."""
    with open(data_yaml) as f:
        cfg = yamllib.safe_load(f)
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def evaluate_on_yaml(
    model: Any,
    data_yaml: str | Path,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    work_dir: str | Path | None = None,
    run_name: str = "eval",
    testable_yolo_ids: list[int] | None = None,
) -> dict:
    """Run validation on the given data.yaml and return structured metrics.

    Args:
        model: An Ultralytics YOLO model (already loaded).
        data_yaml: Path to the data.yaml to validate against. The 'val' or
            'test' field of this YAML determines which images are evaluated.
        imgsz: Validation image size.
        batch: Validation batch size.
        device: Torch device. None = auto.
        work_dir: Where Ultralytics writes its run output. None = /tmp default.
        run_name: Subdirectory name under work_dir for this eval run.
        testable_yolo_ids: Optional list of YOLO class IDs to treat as
            "testable" (typically the 7 classes that appear in the held-out).
            If provided, the result contains a `map50_testable` field that
            is the mean of map50 over only these classes.

    Returns:
        dict with keys:
            map50, map50_95          (overall, all classes)
            map50_per_class          {yolo_id: float}
            map50_95_per_class       {yolo_id: float}
            map50_testable           (if testable_yolo_ids provided)
            map50_95_testable        (if testable_yolo_ids provided)
            class_names              (from the data.yaml)
            n_images, n_instances
    """
    data_yaml = str(Path(data_yaml).resolve())
    class_names = _read_class_names(data_yaml)

    val_kwargs = dict(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        verbose=False,
        plots=False,
    )
    if work_dir is not None:
        val_kwargs["project"] = str(work_dir)
        val_kwargs["name"] = run_name
        val_kwargs["exist_ok"] = True
    if device is not None:
        val_kwargs["device"] = device

    results = model.val(**val_kwargs)

    out: dict[str, Any] = {
        "data_yaml": data_yaml,
        "class_names": class_names,
        "map50": float(getattr(results.box, "map50", 0.0) or 0.0),
        "map50_95": float(getattr(results.box, "map", 0.0) or 0.0),
    }

    # Per-class breakdown. Ultralytics exposes per-class metrics in slightly
    # different ways across versions — we try several known attributes.
    per_class_map50: dict[int, float] = {}
    per_class_map50_95: dict[int, float] = {}

    # results.box.maps gives mAP@50-95 per class (an array indexed by class id)
    maps_array = getattr(results.box, "maps", None)
    if maps_array is not None:
        try:
            for cid, v in enumerate(maps_array.tolist()):
                if v > 0 or cid in class_names:
                    per_class_map50_95[cid] = float(v)
        except Exception as e:
            logger.warning(f"Could not read per-class maps: {e}")

    # results.box.ap50 gives mAP@50 per class (array, but only for present classes)
    # Ultralytics also exposes results.box.ap_class_index telling us which class
    # IDs the rows correspond to (since classes with 0 instances are skipped).
    ap50_array = getattr(results.box, "ap50", None)
    class_idx_array = getattr(results.box, "ap_class_index", None)
    if ap50_array is not None and class_idx_array is not None:
        try:
            for cid, v in zip(class_idx_array.tolist(), ap50_array.tolist()):
                per_class_map50[int(cid)] = float(v)
        except Exception as e:
            logger.warning(f"Could not read per-class ap50: {e}")

    out["map50_per_class"] = per_class_map50
    out["map50_95_per_class"] = per_class_map50_95

    # Subset metric over testable classes
    if testable_yolo_ids is not None:
        testable_set = set(testable_yolo_ids)
        testable_map50 = [v for cid, v in per_class_map50.items() if cid in testable_set]
        testable_map50_95 = [v for cid, v in per_class_map50_95.items() if cid in testable_set]
        out["map50_testable"] = (
            sum(testable_map50) / len(testable_map50) if testable_map50 else 0.0
        )
        out["map50_95_testable"] = (
            sum(testable_map50_95) / len(testable_map50_95) if testable_map50_95 else 0.0
        )
        out["n_testable_classes_evaluated"] = len(testable_map50)

    # Dataset size info
    out["n_images"] = int(getattr(results, "nt", 0) or getattr(results, "seen", 0) or 0)
    try:
        # nt_per_class is the per-class instance count
        nt_per_class = getattr(results.box, "nc", None)
        if nt_per_class is None and hasattr(results, "nt_per_class"):
            out["n_instances"] = int(sum(results.nt_per_class))
    except Exception:
        pass

    return out
