"""Smoke test: YOLOClient end-to-end on a tiny synthetic dataset.

Verifies that:
  1. The client instantiates without error
  2. get_parameters returns the expected number of arrays per federate_role
  3. set_parameters round-trips correctly
  4. fit() runs a real Ultralytics training loop without crashing
  5. evaluate() returns valid mAP metrics
  6. Both federate_role='all' and federate_role='shared' (FedPer) paths work

Runs on CPU. Uses a minimal 8-image dataset and 1 epoch — enough to exercise
the training loop without taking forever.
"""
from __future__ import annotations

import logging
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def setup_tiny_dataset(root: Path) -> Path:
    """Build a minimal dataset: 8 train + 4 val images with real bboxes.

    Returns the path to the data.yaml that points at it.
    """
    from tests.make_synthetic_mocs import generate as gen_mocs
    from data_prep.convert_coco import convert_coco_to_yolo
    from data_prep.schema import load_schema
    from tests.make_synthetic_images import render_dataset
    from pipeline.dataset_yaml import build_data_yaml, write_image_list

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    schema = load_schema()

    # Train: 8 images
    train_json = root / "instances_train.json"
    gen_mocs(train_json, n_images=8, seed=0)
    train_labels = root / "labels/train"
    convert_coco_to_yolo(train_json, train_labels, source="MOCS", schema=schema)
    train_images = root / "images/train"
    n_train = render_dataset(train_labels, train_images, width=320, height=240)

    # Val: 4 images
    val_json = root / "instances_val.json"
    gen_mocs(val_json, n_images=4, seed=1)
    val_labels = root / "labels/val"
    convert_coco_to_yolo(val_json, val_labels, source="MOCS", schema=schema)
    val_images = root / "images/val"
    n_val = render_dataset(val_labels, val_images, width=320, height=240)

    print(f"  Train: {n_train} images, Val: {n_val} images")

    # Build image lists
    train_list_path = root / "train_list.txt"
    train_stems = [p.stem for p in train_labels.glob("*.txt")]
    write_image_list(train_list_path, train_images, train_stems)

    val_list_path = root / "val_list.txt"
    val_stems = [p.stem for p in val_labels.glob("*.txt")]
    write_image_list(val_list_path, val_images, val_stems)

    # data.yaml — but Ultralytics expects labels at <dataset>/labels/<split>/
    # mirroring images at <dataset>/images/<split>/. Our layout already matches.
    yaml_path = root / "data.yaml"
    build_data_yaml(
        yaml_path=yaml_path,
        dataset_root=root,
        train_image_list="train_list.txt",
        val_image_list="val_list.txt",
        schema=schema,
        tier="tier_full",
    )
    return yaml_path


def test_client(federate_role: str, data_yaml: Path, work_dir: Path) -> None:
    """Run one full FL round through the client."""
    from fl.yolo_client import YOLOClient

    print(f"\n--- federate_role={federate_role} ---")
    if work_dir.exists():
        shutil.rmtree(work_dir)

    client = YOLOClient(
        client_name=f"test_{federate_role}",
        data_yaml=data_yaml,
        model_variant="yolo11n.yaml",  # smallest variant, build from yaml = no download
        epochs_per_round=1,
        imgsz=320,
        batch=4,
        federate_role=federate_role,
        device="cpu",
        work_dir=work_dir,
        seed=42,
    )

    # 1. get_parameters
    params = client.get_parameters()
    if federate_role == "all":
        expected_n = sum(client._role_summary.values())
    else:
        expected_n = client._role_summary["shared"]
    assert len(params) == expected_n, f"expected {expected_n} arrays, got {len(params)}"
    print(f"  PASS: get_parameters returned {len(params)} arrays")

    # 2. set_parameters round-trip
    client.set_parameters(params)
    params_after = client.get_parameters()
    assert len(params_after) == len(params)
    # Values must match exactly (no training between get and set)
    import numpy as np
    for a, b in zip(params, params_after):
        assert np.array_equal(a, b), "set/get round-trip altered values"
    print(f"  PASS: set/get round-trip preserves values")

    # 3. fit — runs real training
    print(f"  Running fit() with 1 epoch on 8 images...")
    new_params, n_examples, fit_metrics = client.fit(params, config={})
    assert len(new_params) == len(params)
    assert n_examples > 0, f"expected positive example count, got {n_examples}"
    print(f"  PASS: fit() completed, n_examples={n_examples}")

    # 4. evaluate
    print(f"  Running evaluate()...")
    loss, n_val, eval_metrics = client.evaluate(new_params, config={})
    assert isinstance(loss, float)
    assert n_val > 0
    assert "map50" in eval_metrics
    assert "map50_95" in eval_metrics
    print(f"  PASS: evaluate() returned loss={loss:.4f}, map50={eval_metrics['map50']:.4f}")


def main() -> None:
    print("=" * 60)
    print("YOLOClient smoke test")
    print("=" * 60)

    data_root = Path("/tmp/fl_client_test/data")
    work_root = Path("/tmp/fl_client_test/work")

    print("\n[setup] Building tiny synthetic dataset with real images...")
    data_yaml = setup_tiny_dataset(data_root)

    test_client("all", data_yaml, work_root / "fedavg")
    test_client("shared", data_yaml, work_root / "fedper")

    print("\n" + "=" * 60)
    print("YOLOClient smoke test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
