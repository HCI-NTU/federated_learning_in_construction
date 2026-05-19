"""Stage 5 experiment matrix integration test.

Runs a filtered mini-matrix on synthetic data covering all method types:
  - isolated × 1 client
  - centralized
  - fedavg, fedprox, fedper

Then exercises:
  - skip-if-exists logic (re-running should skip everything)
  - CSV aggregation (verifies columns and row count)
  - matrix summary (verifies the 57-spec total)

Total mini-matrix: 5 runs (one per method) at N=8 (synthetic mini), seed=42.
"""
from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def setup_realistic_synthetic_data(data_root: Path) -> None:
    """Create the four datasets in the layout pipeline.build_pipeline expects."""
    from tests.make_synthetic_mocs import generate as gen_mocs
    from tests.make_synthetic_soda import generate as gen_soda
    from tests.make_synthetic_acid import generate as gen_acid
    from tests.make_synthetic_cis import generate as gen_cis
    from tests.make_synthetic_images import render_dataset

    if data_root.exists():
        shutil.rmtree(data_root)
    data_root.mkdir(parents=True)

    # MOCS: train (val→test per project spec)
    (data_root / "MOCS").mkdir()
    (data_root / "MOCS/images/train").mkdir(parents=True)
    (data_root / "MOCS/images/val").mkdir()
    gen_mocs(data_root / "MOCS/instances_train.json", n_images=12, seed=0)
    gen_mocs(data_root / "MOCS/instances_val.json", n_images=6, seed=1)
    # Render labels first to know what stems exist, then create images
    from data_prep.convert_coco import convert_coco_to_yolo
    from data_prep.schema import load_schema
    schema = load_schema()
    convert_coco_to_yolo(
        data_root / "MOCS/instances_train.json", data_root / "MOCS/labels_temp/train",
        source="MOCS", schema=schema,
    )
    render_dataset(data_root / "MOCS/labels_temp/train", data_root / "MOCS/images/train", 320, 240)
    convert_coco_to_yolo(
        data_root / "MOCS/instances_val.json", data_root / "MOCS/labels_temp/val",
        source="MOCS", schema=schema,
    )
    render_dataset(data_root / "MOCS/labels_temp/val", data_root / "MOCS/images/val", 320, 240)
    shutil.rmtree(data_root / "MOCS/labels_temp")

    # SODA: train + test
    (data_root / "SODA/annotations").mkdir(parents=True)
    (data_root / "SODA/images/train").mkdir(parents=True)
    (data_root / "SODA/images/test").mkdir()
    gen_soda(data_root / "SODA/annotations", n_images=20, seed=0)
    # First 12 train, last 8 test (matches the XML names soda_00001..soda_00020)
    for i in range(1, 13):
        # Render via temp YOLO labels
        pass  # for SODA we'll use convert_voc later via the pipeline; just touch files
    # We need images that match XML stems. Use empty images filled by hand.
    from tests.make_synthetic_images import render_image_with_boxes
    for i in range(1, 13):
        render_image_with_boxes(
            data_root / f"SODA/images/train/soda_{i:05d}.jpg", 320, 240, [], seed=i
        )
    for i in range(13, 21):
        render_image_with_boxes(
            data_root / f"SODA/images/test/soda_{i:05d}.jpg", 320, 240, [], seed=i
        )

    # ACID: instances_all.json + flat images dir for pipeline to arrange
    (data_root / "ACID/images").mkdir(parents=True)
    gen_acid(data_root / "ACID/instances_all.json", n_images=40, seed=0)
    for i in range(1, 41):
        render_image_with_boxes(
            data_root / f"ACID/images/acid_{i:05d}.jpg", 320, 240, [], seed=i
        )

    # CIS: test only
    (data_root / "CIS/images/test").mkdir(parents=True)
    gen_cis(data_root / "CIS/instances_test.json", n_images=15, seed=0)
    convert_coco_to_yolo(
        data_root / "CIS/instances_test.json", data_root / "CIS/labels_temp/test",
        source="CIS", schema=schema,
    )
    render_dataset(data_root / "CIS/labels_temp/test", data_root / "CIS/images/test", 320, 240)
    shutil.rmtree(data_root / "CIS/labels_temp")


def main() -> None:
    print("=" * 60)
    print("Stage 5 matrix driver integration test")
    print("=" * 60)

    data_root = Path("/tmp/stage5_test/data")
    work_root = Path("/tmp/stage5_test/work")
    results_root = Path("/tmp/stage5_test/results")
    log_root = Path("/tmp/stage5_test/logs")

    for d in [work_root, results_root, log_root]:
        if d.exists():
            shutil.rmtree(d)

    print("\n[1/6] Setting up synthetic data for all four datasets...")
    setup_realistic_synthetic_data(data_root)

    # 1. Verify matrix generation produces 57 specs at default settings
    print("\n[2/6] Verifying matrix generation...")
    from experiments.matrix import generate_full_matrix, summarize_matrix
    full_specs = generate_full_matrix()
    summary = summarize_matrix(full_specs)
    assert summary["total"] == 57, f"expected 57 specs, got {summary['total']}"
    print(f"  PASS: full matrix has {summary['total']} specs")
    print(f"  by_exp_label: {summary['by_exp_label']}")

    # 2. Build a tiny filtered matrix: one of each method, N=12 (matches MOCS train size)
    print("\n[3/6] Running mini-matrix (5 runs, one per method)...")
    from experiments.run_matrix import run_matrix
    from experiments.matrix import _spec

    mini_specs = [
        _spec(
            exp_label="exp1",
            method="isolated",
            n=8,                                # small N: subsampler will use what's available
            tier="tier_full",
            training_clients=("MOCS", "SODA", "ACID"),
            heldout="CIS",
            seed=42,
            client_for_isolated="MOCS",
        ),
        _spec(
            exp_label="exp1",
            method="centralized",
            n=8,
            tier="tier_full",
            training_clients=("MOCS", "SODA", "ACID"),
            heldout="CIS",
            seed=42,
        ),
        _spec(
            exp_label="exp1",
            method="fedavg",
            n=8,
            tier="tier_full",
            training_clients=("MOCS", "SODA", "ACID"),
            heldout="CIS",
            seed=42,
        ),
        _spec(
            exp_label="exp1",
            method="fedprox",
            n=8,
            tier="tier_full",
            training_clients=("MOCS", "SODA", "ACID"),
            heldout="CIS",
            seed=42,
        ),
        _spec(
            exp_label="exp1",
            method="fedper",
            n=8,
            tier="tier_full",
            training_clients=("MOCS", "SODA", "ACID"),
            heldout="CIS",
            seed=42,
        ),
    ]

    report = run_matrix(
        specs=mini_specs,
        data_root=data_root,
        work_root=work_root,
        results_root=results_root,
        log_root=log_root,
        epochs=1,                  # tiny budget
        model_variant="yolo11n.yaml",
        imgsz=320,
        batch=4,
        device="cpu",
        proximal_mu=0.01,
    )

    print(f"  Summary: {report['summary']}")
    assert report["summary"]["n_ok"] == 5, (
        f"Expected 5 ok runs, got {report['summary']['n_ok']}. "
        f"Failures: {[(k, v) for k, v in report['per_spec'].items() if v['status'] == 'failed']}"
    )
    print(f"  PASS: all 5 specs completed")

    # 3. Re-run to verify skip-existing
    print("\n[4/6] Re-running to verify skip-existing logic...")
    report2 = run_matrix(
        specs=mini_specs,
        data_root=data_root,
        work_root=work_root,
        results_root=results_root,
        log_root=log_root,
        epochs=1,
        model_variant="yolo11n.yaml",
        imgsz=320,
        batch=4,
        device="cpu",
        proximal_mu=0.01,
    )
    assert report2["summary"]["n_skipped"] == 5
    assert report2["summary"]["n_ok"] == 0
    print(f"  PASS: all 5 specs skipped on re-run")

    # 4. Aggregate results into CSV
    print("\n[5/6] Aggregating results to CSV...")
    from experiments.aggregate_results import aggregate
    csv_path = results_root / "aggregated.csv"
    n_rows = aggregate(results_root, csv_path)
    assert n_rows == 5, f"expected 5 rows, got {n_rows}"
    print(f"  PASS: CSV has {n_rows} rows")

    # 5. Verify CSV contents
    print("\n[6/6] Verifying CSV contents...")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames

    assert len(rows) == 5
    # Required columns
    required = [
        "spec_id", "method", "n", "tier", "training_clients",
        "heldout_map50", "heldout_map50_95",
        "heldout_map50_testable", "heldout_map50_95_testable",
    ]
    for col in required:
        assert col in columns, f"missing column: {col}"
    print(f"  PASS: required columns present")

    # All 5 methods accounted for
    methods_in_csv = {r["method"] for r in rows}
    assert methods_in_csv == {"isolated", "centralized", "fedavg", "fedprox", "fedper"}, (
        f"unexpected methods: {methods_in_csv}"
    )
    print(f"  PASS: all 5 methods present in CSV")

    # Per-class columns for the testable classes
    testable_class_cols = [c for c in columns if c.startswith("heldout_map50__")]
    print(f"  Found {len(testable_class_cols)} per-class held-out columns")
    assert len(testable_class_cols) >= 7, (
        f"expected at least 7 per-class cols (one per unified class), got {len(testable_class_cols)}"
    )
    print(f"  PASS: per-class breakdown columns present")

    print("\n" + "=" * 60)
    print("Stage 5 matrix driver integration test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
