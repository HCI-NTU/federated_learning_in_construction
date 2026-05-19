"""Top-level data pipeline orchestrator.

Single CLI entry that runs the full Stage 1+2 pipeline for one experiment
configuration: schema → conversion → ACID split → subsample → tier filter →
data.yaml + image lists.

Idempotent design:
  - Annotation conversion is skipped if labels directory already exists
    with the expected count of files
  - ACID split is skipped if already done
  - Subsample manifests are versioned by (client, split, n, tier, seed)
    and reused if a matching one exists
  - data.yaml is always regenerated (cheap and configuration-sensitive)

Usage:
    # Default: prepare full Experiment 1 setup at N=4000, all 3 clients, tier_full
    python -m pipeline.build_pipeline \\
        --data-root data \\
        --output-root work \\
        --clients MOCS SODA ACID \\
        --n 4000 \\
        --tier tier_full \\
        --seed 42

The output structure:
    work/
        manifests/
            MOCS_train_n4000_tier_full_s42.txt
            ...
        labels_tier_shared/MOCS/train/      (created if tier_shared used)
        configs/
            MOCS_train_n4000_tier_full_s42.yaml
            ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_prep.schema import load_schema
from data_prep.split_acid import split_acid
from data_prep.arrange_acid_images import arrange_acid_images
from data_prep.convert_coco import convert_coco_to_yolo
from data_prep.convert_voc import convert_voc_to_yolo
from pipeline.class_subsets import get_tier_yolo_ids
from pipeline.subsample import stratified_subsample, write_manifest
from pipeline.tier_filter import filter_labels_for_tier
from pipeline.dataset_yaml import build_data_yaml, write_image_list


# Per-client metadata
# Each entry describes how to convert and where the splits live.
# 'split_files' maps split name -> annotation file path (relative to client dir)
CLIENT_CONFIG = {
    "MOCS": {
        "source": "MOCS",
        "format": "coco",
        "splits": {
            # Per project spec: val becomes test (no separate test exists)
            "train": "instances_train.json",
            "test": "instances_val.json",
        },
        "image_dirs": {
            "train": "images/train",
            "test": "images/val",
        },
    },
    "SODA": {
        "source": "SODA",
        "format": "voc",
        "splits": {
            "train": "annotations",  # VOC: directory of XMLs, image_list filters
            "test": "annotations",
        },
        "image_dirs": {
            "train": "images/train",
            "test": "images/test",
        },
    },
    "ACID": {
        "source": "ACID",
        "format": "coco",
        "splits": {
            "train": "instances_train.json",
            "test": "instances_test.json",
        },
        "image_dirs": {
            "train": "images/train",
            "test": "images/test",
        },
        "needs_split": True,
    },
    "CIS": {
        "source": "CIS",
        "format": "coco",
        # Per project spec: only test split is used; train and val are ignored
        "splits": {
            "test": "instances_test.json",
        },
        "image_dirs": {
            "test": "images/test",
        },
    },
}


def ensure_acid_split(
    data_root: Path,
    seed: int = 42,
    acid_raw_images_dir: Path | None = None,
    arrange_mode: str = "move",
) -> None:
    """Split ACID's instances_all.json into train/test if not already done,
    and physically arrange images into images/train and images/test subdirs.

    Args:
        data_root: Root data directory containing ACID/
        seed: Random seed for the split
        acid_raw_images_dir: Where the raw ACID images live before arrangement.
            If None, defaults to data_root/ACID/images. If that flat directory
            does not exist and images are already in images/train and images/test,
            arrangement is skipped (assumed already done).
        arrange_mode: 'move' (default) or 'copy' for the image arrangement step.
    """
    acid_dir = data_root / "ACID"
    all_json = acid_dir / "instances_all.json"
    train_json = acid_dir / "instances_train.json"
    test_json = acid_dir / "instances_test.json"

    # Step 1: split JSONs if needed
    if train_json.exists() and test_json.exists():
        print(f"[ACID] Split JSONs already exist, skipping split.")
    elif not all_json.exists():
        raise FileNotFoundError(
            f"ACID needs splitting but {all_json} does not exist."
        )
    else:
        print(f"[ACID] Splitting {all_json} into train/test...")
        split_acid(all_json, acid_dir, test_ratio=0.2, seed=seed)

    # Step 2: arrange images into train/test subdirs
    # Determine source: if user passed an explicit dir, use it; otherwise check
    # the default flat layout at images/.
    train_img_dir = acid_dir / "images" / "train"
    test_img_dir = acid_dir / "images" / "test"

    # If both subdirs already contain images, assume arrangement is done
    if (
        train_img_dir.is_dir() and any(train_img_dir.iterdir())
        and test_img_dir.is_dir() and any(test_img_dir.iterdir())
    ):
        print(f"[ACID] Image subdirs already populated, skipping arrangement.")
        return

    # Otherwise we need to arrange. Determine source directory.
    if acid_raw_images_dir is None:
        # Default: assume raw images are at data/ACID/images/ (flat)
        default_src = acid_dir / "images"
        if default_src.is_dir() and any(p.is_file() for p in default_src.iterdir()):
            acid_raw_images_dir = default_src
        else:
            print(
                f"[ACID] Warning: could not find raw images to arrange. "
                f"Expected either {default_src} (flat) or images already arranged "
                f"in {train_img_dir} and {test_img_dir}. "
                f"Pass --acid-raw-images-dir to specify a custom location."
            )
            return

    print(
        f"[ACID] Arranging images from {acid_raw_images_dir} into "
        f"images/train and images/test (mode={arrange_mode})..."
    )
    stats = arrange_acid_images(
        acid_dir=acid_dir,
        source_images_dir=acid_raw_images_dir,
        mode=arrange_mode,
    )
    print(
        f"[ACID] Arrangement complete: "
        f"train={stats['train_moved']+stats['train_already_in_place']} images, "
        f"test={stats['test_moved']+stats['test_already_in_place']} images, "
        f"missing={stats['missing']}"
    )


def convert_client_split(
    data_root: Path,
    work_root: Path,
    client: str,
    split: str,
    schema,
) -> Path:
    """Convert one (client, split) to YOLO labels. Returns labels_dir path.

    Idempotent: skips if labels already exist with the expected file count
    (within ±1% tolerance to allow for minor regenerations).
    """
    cfg = CLIENT_CONFIG[client]
    if split not in cfg["splits"]:
        raise ValueError(f"Client {client} has no '{split}' split")

    labels_dir = work_root / "labels" / client / split
    if labels_dir.exists() and any(labels_dir.glob("*.txt")):
        print(f"[{client}/{split}] Labels exist at {labels_dir}, skipping conversion.")
        return labels_dir

    client_dir = data_root / client

    if cfg["format"] == "coco":
        json_path = client_dir / cfg["splits"][split]
        if not json_path.exists():
            raise FileNotFoundError(f"{json_path} does not exist")
        print(f"[{client}/{split}] Converting {json_path} to YOLO...")
        convert_coco_to_yolo(
            json_path=json_path,
            labels_dir=labels_dir,
            source=cfg["source"],
            schema=schema,
        )
    elif cfg["format"] == "voc":
        xml_dir = client_dir / cfg["splits"][split]
        image_dir = client_dir / cfg["image_dirs"][split]
        # Filter XMLs by image stems present in the corresponding image dir
        stem_filter = None
        if image_dir.exists():
            stems = {p.stem for p in image_dir.iterdir() if p.is_file()}
            stem_filter = stems
        print(f"[{client}/{split}] Converting VOC XMLs from {xml_dir} to YOLO...")
        convert_voc_to_yolo(
            xml_dir=xml_dir,
            labels_dir=labels_dir,
            schema=schema,
            image_stem_filter=stem_filter,
        )
    else:
        raise ValueError(f"Unknown format: {cfg['format']}")

    return labels_dir


def build_for_client(
    client: str,
    data_root: Path,
    work_root: Path,
    n: int,
    tier: str,
    seed: int,
    schema,
) -> dict:
    """Run the full pipeline for one client. Returns dict of artifact paths."""
    cfg = CLIENT_CONFIG[client]
    n_classes = schema.num_classes

    # 1. Convert all available splits
    label_dirs = {}
    for split in cfg["splits"]:
        label_dirs[split] = convert_client_split(
            data_root, work_root, client, split, schema
        )

    # CIS has only 'test' — skip subsampling and tier filtering, but DO
    # write a data.yaml pointing at the converted test labels so that
    # downstream evaluators (Stage 4 runners) can find it via a yaml path.
    if "train" not in cfg["splits"]:
        # Build an image list for the held-out's test (or val) split
        test_split = "test" if "test" in cfg["splits"] else next(iter(cfg["splits"]))
        test_image_dir = data_root / client / cfg["image_dirs"][test_split]
        test_stems = [p.stem for p in label_dirs[test_split].glob("*.txt")]
        test_image_list = (
            work_root / "image_lists" / f"{client}_test_{tier}.txt"
        )
        n_test = write_image_list(test_image_list, test_image_dir, test_stems)
        print(f"[{client}] heldout test image list: {n_test} images at {test_image_list}")

        # Apply tier filter to held-out labels if needed, so the val split's
        # label files only contain active classes (matches what the model is
        # trained to predict). This is essential for fair held-out comparison
        # across tiers.
        if tier != "tier_full":
            active_yolo_ids = get_tier_yolo_ids(tier, schema)
            filtered_dir = work_root / f"labels_{tier}" / client / test_split
            if not (filtered_dir.exists() and any(filtered_dir.glob("*.txt"))):
                print(f"[{client}] Filtering heldout {test_split} labels for tier={tier}...")
                filter_labels_for_tier(
                    source_labels_dir=label_dirs[test_split],
                    dest_labels_dir=filtered_dir,
                    active_yolo_ids=active_yolo_ids,
                    stem_filter=None,
                )

        # Build the data.yaml. Use n=0 in the filename to signal "heldout, no
        # subsampling involved" — distinct from training-client yamls.
        yaml_path = work_root / "configs" / f"{client}_n0_{tier}_s{seed}.yaml"
        train_rel = str(test_image_list.relative_to(work_root))
        val_rel = train_rel  # both fields point at the test list (Ultralytics needs both)
        build_data_yaml(
            yaml_path=yaml_path,
            dataset_root=work_root.resolve(),
            train_image_list=train_rel,
            val_image_list=val_rel,
            schema=schema,
            tier=tier,
        )
        print(f"[{client}] Wrote heldout data.yaml: {yaml_path}")

        return {
            "client": client,
            "label_dirs": {k: str(v) for k, v in label_dirs.items()},
            "splits_available": list(cfg["splits"].keys()),
            "test_image_list": str(test_image_list),
            "yaml": str(yaml_path),
        }

    # 2. Subsample train split to N
    active_classes = (
        get_tier_yolo_ids(tier, schema) if tier != "tier_full" else None
    )
    manifest_path = (
        work_root / "manifests" / f"{client}_train_n{n}_{tier}_s{seed}.txt"
    )
    if manifest_path.exists():
        print(f"[{client}] Manifest exists at {manifest_path}, skipping subsample.")
    else:
        print(f"[{client}] Subsampling train to N={n}, tier={tier}, seed={seed}...")
        stems, stats = stratified_subsample(
            labels_dir=label_dirs["train"],
            n=n,
            n_classes=n_classes,
            seed=seed,
            active_classes=active_classes,
        )
        metadata = {
            "client": client,
            "split": "train",
            "tier": tier,
            **stats,
        }
        write_manifest(manifest_path, stems, metadata)
        print(f"[{client}] Wrote {manifest_path} ({stats['n_actual']} stems)")

    # 3. Apply tier filter to labels (only if non-full tier)
    if tier == "tier_full":
        train_labels_dir = label_dirs["train"]
        test_labels_dir = label_dirs.get("test", label_dirs.get("val"))
    else:
        from pipeline.subsample import read_manifest
        stems, _ = read_manifest(manifest_path)
        train_labels_dir = work_root / f"labels_{tier}" / client / "train"
        test_labels_dir = work_root / f"labels_{tier}" / client / "test"

        if not (train_labels_dir.exists() and any(train_labels_dir.glob("*.txt"))):
            print(f"[{client}] Filtering train labels for tier={tier}...")
            filter_labels_for_tier(
                source_labels_dir=label_dirs["train"],
                dest_labels_dir=train_labels_dir,
                active_yolo_ids=active_classes,
                stem_filter=set(stems),
            )

        if "test" in label_dirs and not (
            test_labels_dir.exists() and any(test_labels_dir.glob("*.txt"))
        ):
            print(f"[{client}] Filtering test labels for tier={tier}...")
            filter_labels_for_tier(
                source_labels_dir=label_dirs["test"],
                dest_labels_dir=test_labels_dir,
                active_yolo_ids=active_classes,
                stem_filter=None,  # test set is unrestricted
            )

    # 4. Write image lists
    from pipeline.subsample import read_manifest
    train_stems, _ = read_manifest(manifest_path)

    train_image_dir = data_root / client / cfg["image_dirs"]["train"]
    train_image_list = (
        work_root / "image_lists" / f"{client}_train_n{n}_{tier}_s{seed}.txt"
    )
    n_train_found = write_image_list(train_image_list, train_image_dir, train_stems)
    print(f"[{client}] train image list: {n_train_found} images at {train_image_list}")

    test_split_name = "test" if "test" in cfg["splits"] else None
    test_image_list = None
    if test_split_name:
        test_image_dir = data_root / client / cfg["image_dirs"][test_split_name]
        test_stems = [p.stem for p in test_labels_dir.glob("*.txt")]
        test_image_list = work_root / "image_lists" / f"{client}_test_{tier}.txt"
        n_test_found = write_image_list(test_image_list, test_image_dir, test_stems)
        print(f"[{client}] test image list: {n_test_found} images at {test_image_list}")

    # 5. Build data.yaml
    yaml_path = (
        work_root / "configs" / f"{client}_n{n}_{tier}_s{seed}.yaml"
    )
    # Use absolute paths in the YAML so it works regardless of cwd
    dataset_root = work_root.resolve()
    train_rel = str(train_image_list.relative_to(work_root))
    val_rel = str(test_image_list.relative_to(work_root)) if test_image_list else train_rel
    build_data_yaml(
        yaml_path=yaml_path,
        dataset_root=dataset_root,
        train_image_list=train_rel,
        val_image_list=val_rel,
        schema=schema,
        tier=tier,
    )
    print(f"[{client}] Wrote data.yaml: {yaml_path}")

    return {
        "client": client,
        "manifest": str(manifest_path),
        "train_image_list": str(train_image_list),
        "test_image_list": str(test_image_list) if test_image_list else None,
        "train_labels_dir": str(train_labels_dir),
        "test_labels_dir": str(test_labels_dir) if test_labels_dir else None,
        "yaml": str(yaml_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-root", required=True,
        help="Root directory containing MOCS/, SODA/, ACID/, CIS/ subdirs",
    )
    parser.add_argument(
        "--output-root", required=True,
        help="Where to write labels, manifests, configs, image lists",
    )
    parser.add_argument(
        "--clients", nargs="+", default=["MOCS", "SODA", "ACID"],
        help="Training clients to include",
    )
    parser.add_argument(
        "--heldout", default="CIS",
        help="Held-out dataset (test split only)",
    )
    parser.add_argument(
        "--n", type=int, required=True,
        help="Per-client training images (N for the size sweep)",
    )
    parser.add_argument(
        "--tier",
        default="tier_full",
        choices=["tier_shared", "tier_shared_cross", "tier_full"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schema", default=None)
    parser.add_argument(
        "--acid-raw-images-dir", default=None,
        help="Directory containing the raw ACID image files (before arrangement). "
        "If omitted, defaults to data/ACID/images/ (flat). If images are already "
        "arranged into images/train/ and images/test/, this is ignored.",
    )
    parser.add_argument(
        "--acid-arrange-mode", default="move", choices=["move", "copy"],
        help="Whether to move (default, saves disk) or copy ACID images when "
        "arranging them into train/test subdirs.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    work_root = Path(args.output_root)
    work_root.mkdir(parents=True, exist_ok=True)

    schema = load_schema(args.schema) if args.schema else load_schema()
    print(f"Schema loaded: {schema.num_classes} unified classes\n")

    # Pre-step: ensure ACID is split and images are arranged
    if "ACID" in args.clients or args.heldout == "ACID":
        ensure_acid_split(
            data_root,
            seed=args.seed,
            acid_raw_images_dir=Path(args.acid_raw_images_dir) if args.acid_raw_images_dir else None,
            arrange_mode=args.acid_arrange_mode,
        )

    # Build for each training client
    artifacts = {"clients": {}, "heldout": None}
    for client in args.clients:
        print(f"\n=== Building for client: {client} ===")
        artifacts["clients"][client] = build_for_client(
            client=client,
            data_root=data_root,
            work_root=work_root,
            n=args.n,
            tier=args.tier,
            seed=args.seed,
            schema=schema,
        )

    # Build for held-out (CIS by default) — just convert, no subsampling
    if args.heldout:
        print(f"\n=== Building for held-out: {args.heldout} ===")
        artifacts["heldout"] = build_for_client(
            client=args.heldout,
            data_root=data_root,
            work_root=work_root,
            n=args.n,
            tier=args.tier,
            seed=args.seed,
            schema=schema,
        )

    # Write a summary manifest
    summary_path = work_root / f"pipeline_summary_n{args.n}_{args.tier}_s{args.seed}.json"
    summary = {
        "n": args.n,
        "tier": args.tier,
        "seed": args.seed,
        "clients": list(args.clients),
        "heldout": args.heldout,
        **artifacts,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Pipeline summary: {summary_path} ===")


if __name__ == "__main__":
    main()
