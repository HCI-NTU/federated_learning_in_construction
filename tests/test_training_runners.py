"""Stage 4 training-runner integration test.

Exercises all three Stage 4 training scripts on tiny synthetic data:
  1. train_isolated: trains one model on one client's data, evaluates on
     local + held-out
  2. train_centralized: pools all three clients, trains one model
  3. train_federated: runs FedAvg for 2 rounds across all three clients

This is a plumbing test — verifies that each script produces a results
JSON with the expected fields and that held-out evaluation runs cleanly.
Performance numbers (mAP values) will be ~0 because the data is too tiny
and training too short to actually learn.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def setup_e2e_data(root: Path) -> dict:
    """Create three client datasets + a held-out dataset with real images."""
    from tests.make_synthetic_mocs import generate as gen_mocs
    from tests.make_synthetic_cis import generate as gen_cis
    from data_prep.convert_coco import convert_coco_to_yolo
    from data_prep.schema import load_schema
    from tests.make_synthetic_images import render_dataset
    from pipeline.dataset_yaml import build_data_yaml, write_image_list

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    schema = load_schema()
    setups = {}

    for i, name in enumerate(["client_a", "client_b", "client_c"]):
        d = root / name
        d.mkdir()
        # Train
        train_json = d / "instances_train.json"
        gen_mocs(train_json, n_images=8, seed=i * 10)
        train_labels = d / "labels/train"
        convert_coco_to_yolo(train_json, train_labels, source="MOCS", schema=schema)
        train_images = d / "images/train"
        render_dataset(train_labels, train_images, width=320, height=240)
        # Val
        val_json = d / "instances_val.json"
        gen_mocs(val_json, n_images=4, seed=i * 10 + 1)
        val_labels = d / "labels/val"
        convert_coco_to_yolo(val_json, val_labels, source="MOCS", schema=schema)
        val_images = d / "images/val"
        render_dataset(val_labels, val_images, width=320, height=240)
        # Image lists
        train_list = d / "train_list.txt"
        write_image_list(
            train_list, train_images, [p.stem for p in train_labels.glob("*.txt")]
        )
        val_list = d / "val_list.txt"
        write_image_list(
            val_list, val_images, [p.stem for p in val_labels.glob("*.txt")]
        )
        # data.yaml
        yaml_path = d / "data.yaml"
        build_data_yaml(
            yaml_path=yaml_path,
            dataset_root=d,
            train_image_list="train_list.txt",
            val_image_list="val_list.txt",
            schema=schema,
            tier="tier_full",
        )
        setups[name] = yaml_path

    # Held-out (CIS-like)
    heldout = root / "heldout"
    heldout.mkdir()
    test_json = heldout / "instances_test.json"
    gen_cis(test_json, n_images=8, seed=99)
    test_labels = heldout / "labels/val"  # use 'val' name so data.yaml's val field finds it
    convert_coco_to_yolo(test_json, test_labels, source="CIS", schema=schema)
    test_images = heldout / "images/val"
    render_dataset(test_labels, test_images, width=320, height=240)
    test_list = heldout / "test_list.txt"
    write_image_list(
        test_list, test_images, [p.stem for p in test_labels.glob("*.txt")]
    )
    heldout_yaml = heldout / "data.yaml"
    build_data_yaml(
        yaml_path=heldout_yaml,
        dataset_root=heldout,
        train_image_list="test_list.txt",  # not used for held-out, but required
        val_image_list="test_list.txt",
        schema=schema,
        tier="tier_full",
    )
    setups["heldout"] = heldout_yaml

    return setups


def assert_result_shape(result: dict, expected_keys: list[str], label: str) -> None:
    for k in expected_keys:
        assert k in result, f"{label}: missing key {k}"
    assert "heldout_eval" in result, f"{label}: missing heldout_eval"
    assert "map50" in result["heldout_eval"], f"{label}: heldout_eval missing map50"
    print(f"  PASS: {label} — heldout map50={result['heldout_eval']['map50']:.4f}")


def main() -> None:
    print("=" * 60)
    print("Stage 4 training runners integration test")
    print("=" * 60)

    data_root = Path("/tmp/stage4_test/data")
    work_root = Path("/tmp/stage4_test/work")
    results_root = Path("/tmp/stage4_test/results")

    if results_root.exists():
        shutil.rmtree(results_root)
    if work_root.exists():
        shutil.rmtree(work_root)

    print("\n[setup] Building synthetic clients + heldout...")
    setups = setup_e2e_data(data_root)
    print(f"  Built {len(setups)-1} clients + 1 heldout")

    client_yamls = [setups["client_a"], setups["client_b"], setups["client_c"]]
    heldout_yaml = setups["heldout"]

    # 1. Isolated
    print("\n--- train_isolated ---")
    from train.train_isolated import train_isolated
    res_iso = train_isolated(
        client="client_a",
        data_yaml=client_yamls[0],
        heldout_data_yaml=heldout_yaml,
        model_variant="yolo11n.yaml",
        epochs=1,
        imgsz=320,
        batch=4,
        device="cpu",
        work_dir=work_root / "isolated",
        seed=42,
        results_path=results_root / "isolated.json",
    )
    assert_result_shape(
        res_iso,
        ["method", "client", "local_eval", "heldout_eval", "train_time_seconds"],
        "isolated",
    )
    assert res_iso["method"] == "isolated"

    # 2. Centralized
    print("\n--- train_centralized ---")
    from train.train_centralized import train_centralized
    res_cen = train_centralized(
        client_yamls=client_yamls,
        heldout_data_yaml=heldout_yaml,
        model_variant="yolo11n.yaml",
        epochs=1,
        imgsz=320,
        batch=4,
        device="cpu",
        work_dir=work_root / "centralized",
        seed=42,
        results_path=results_root / "centralized.json",
    )
    assert_result_shape(
        res_cen,
        ["method", "client_yamls", "pooled_val_eval", "heldout_eval", "train_time_seconds"],
        "centralized",
    )
    assert res_cen["method"] == "centralized"

    # 3. Federated — test each of the three methods
    from train.train_federated import train_federated
    for method in ["fedavg", "fedprox", "fedper"]:
        print(f"\n--- train_federated method={method} ---")
        res_fed = train_federated(
            method=method,
            client_yamls=client_yamls,
            client_names=["client_a", "client_b", "client_c"],
            heldout_data_yaml=heldout_yaml,
            num_rounds=2,
            epochs_per_round=1,
            model_variant="yolo11n.yaml",
            imgsz=320,
            batch=4,
            device="cpu",
            proximal_mu=0.01,
            work_dir=work_root / f"federated_{method}",
            seed=42,
            results_path=results_root / f"federated_{method}.json",
        )
        assert_result_shape(
            res_fed,
            ["method", "client_yamls", "heldout_eval", "history", "train_time_seconds"],
            f"federated_{method}",
        )
        assert res_fed["method"] == method
        n_fit_rounds = len(res_fed["history"].get("fit_rounds", []))
        assert n_fit_rounds == 2, f"{method}: expected 2 fit rounds, got {n_fit_rounds}"

    # Cross-verify that FedPer exchanged fewer params than FedAvg/FedProx
    fedavg_json = json.loads((results_root / "federated_fedavg.json").read_text())
    fedper_json = json.loads((results_root / "federated_fedper.json").read_text())
    n_avg = fedavg_json["init_info"]["n_params_exchanged"]
    n_per = fedper_json["init_info"]["n_params_exchanged"]
    assert n_per < n_avg, f"FedPer should exchange fewer: {n_per} >= {n_avg}"
    print(f"\nFedPer/FedAvg exchanged: {n_per}/{n_avg} = {100*n_per/n_avg:.1f}%")

    print("\n" + "=" * 60)
    print("Stage 4 training runners integration test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
