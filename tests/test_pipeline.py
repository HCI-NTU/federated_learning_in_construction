"""Stage 2 pipeline integration test.

Sets up synthetic versions of all four datasets in the expected directory
structure, runs build_pipeline.py end-to-end, and verifies that the expected
artifacts (manifests, image lists, data.yaml, filtered labels) are produced
correctly.

This is a smoke test, not a correctness test of individual converters
(see tests/test_converters.py for that).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.make_synthetic_mocs import generate as gen_mocs
from tests.make_synthetic_soda import generate as gen_soda
from tests.make_synthetic_acid import generate as gen_acid
from tests.make_synthetic_cis import generate as gen_cis


def setup_synthetic_data(root: Path) -> None:
    """Create synthetic versions of all four datasets in the expected structure."""
    if root.exists():
        shutil.rmtree(root)

    # Annotation files
    (root / "MOCS").mkdir(parents=True, exist_ok=True)
    (root / "SODA/annotations").mkdir(parents=True, exist_ok=True)
    (root / "ACID").mkdir(parents=True, exist_ok=True)
    (root / "CIS").mkdir(parents=True, exist_ok=True)

    gen_mocs(root / "MOCS/instances_train.json", n_images=300, seed=0)
    gen_mocs(root / "MOCS/instances_val.json", n_images=100, seed=1)
    gen_soda(root / "SODA/annotations", n_images=50, seed=0)
    gen_acid(root / "ACID/instances_all.json", n_images=600, seed=0)
    gen_cis(root / "CIS/instances_test.json", n_images=200, seed=0)

    # Image files (empty placeholders — content not needed for pipeline test)
    image_specs = [
        ("MOCS/images/train", "mocs_", 1, 300),
        ("MOCS/images/val", "mocs_", 1, 100),
        ("SODA/images/train", "soda_", 1, 35),
        ("SODA/images/test", "soda_", 36, 50),
        ("ACID/images/train", "acid_", 1, 600),
        ("CIS/images/test", "cis_", 1, 200),
    ]
    for subdir, prefix, start, end in image_specs:
        d = root / subdir
        d.mkdir(parents=True, exist_ok=True)
        for i in range(start, end + 1):
            (d / f"{prefix}{i:05d}.jpg").touch()


def assert_file_exists(p: Path, label: str) -> None:
    assert p.exists(), f"Missing: {label} ({p})"
    print(f"  PASS: {label}")


def assert_dir_non_empty(p: Path, label: str) -> None:
    assert p.exists() and any(p.iterdir()), f"Empty or missing: {label} ({p})"
    print(f"  PASS: {label}")


def verify_pipeline_outputs(
    work_root: Path, n: int, tier: str, seed: int, clients: list[str]
) -> None:
    """Assert that all expected artifacts were produced."""
    # Summary file
    summary_path = work_root / f"pipeline_summary_n{n}_{tier}_s{seed}.json"
    assert_file_exists(summary_path, "summary JSON")
    summary = json.loads(summary_path.read_text())
    assert summary["n"] == n
    assert summary["tier"] == tier
    assert summary["seed"] == seed
    print(f"  PASS: summary metadata matches request")

    # Per-client artifacts
    for client in clients:
        manifest = work_root / "manifests" / f"{client}_train_n{n}_{tier}_s{seed}.txt"
        assert_file_exists(manifest, f"{client} manifest")
        image_list = (
            work_root / "image_lists" / f"{client}_train_n{n}_{tier}_s{seed}.txt"
        )
        assert_file_exists(image_list, f"{client} train image list")
        yaml = work_root / "configs" / f"{client}_n{n}_{tier}_s{seed}.yaml"
        assert_file_exists(yaml, f"{client} data.yaml")

        # Labels — original always exists; tier-filtered only if non-full tier
        orig_labels = work_root / "labels" / client / "train"
        assert_dir_non_empty(orig_labels, f"{client} original train labels")
        if tier != "tier_full":
            filtered_labels = work_root / f"labels_{tier}" / client / "train"
            assert_dir_non_empty(filtered_labels, f"{client} filtered train labels")

    # Held-out labels (always converted, never subsampled)
    cis_labels = work_root / "labels" / "CIS" / "test"
    assert_dir_non_empty(cis_labels, "CIS held-out labels")


def main() -> None:
    data_root = Path("/tmp/stage2_test/data")
    work_root = Path("/tmp/stage2_test/work")

    print("=" * 60)
    print("Stage 2 pipeline integration test")
    print("=" * 60)
    print("\n[1/3] Setting up synthetic data...")
    setup_synthetic_data(data_root)
    print(f"  PASS: synthetic data at {data_root}")

    # Run the pipeline. Import here so the path is set up first.
    from pipeline.build_pipeline import build_for_client, ensure_acid_split
    from data_prep.schema import load_schema

    schema = load_schema()
    clients = ["MOCS", "SODA", "ACID"]

    for tier in ["tier_full", "tier_shared"]:
        print(f"\n[2/3] Running pipeline with tier={tier}, N=100...")
        # Clean per-tier output
        if work_root.exists():
            shutil.rmtree(work_root)
        work_root.mkdir(parents=True)

        ensure_acid_split(data_root, seed=42)
        for client in clients:
            build_for_client(
                client=client,
                data_root=data_root,
                work_root=work_root,
                n=100,
                tier=tier,
                seed=42,
                schema=schema,
            )
        build_for_client(
            client="CIS",
            data_root=data_root,
            work_root=work_root,
            n=100,
            tier=tier,
            seed=42,
            schema=schema,
        )

        # Write the summary (mirrors what build_pipeline.main() does)
        summary_path = work_root / f"pipeline_summary_n100_{tier}_s42.json"
        summary_path.write_text(json.dumps({
            "n": 100, "tier": tier, "seed": 42,
            "clients": clients, "heldout": "CIS"
        }))

        print(f"\n[3/3] Verifying tier={tier} artifacts...")
        verify_pipeline_outputs(work_root, 100, tier, 42, clients)

    print("\n" + "=" * 60)
    print("Stage 2 integration test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
