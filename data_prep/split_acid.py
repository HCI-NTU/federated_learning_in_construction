"""ACID train/test splitter using iterative stratification.

ACID is provided as a single instances_all.json file. This script splits it
into instances_train.json and instances_test.json at an 80:20 ratio, using
image-level iterative multi-label stratification.

Why iterative stratification?
  Each ACID image can contain multiple object classes. Simple random splitting
  could push rare classes entirely into one side. Iterative stratification
  (Sechidis et al., 2011) balances each class's image-level presence across
  the splits as closely as possible.

Output preserves COCO structure: the same 'info', 'licenses', and 'categories'
sections are copied; 'images' and 'annotations' are partitioned per split.

Usage:
    python -m data_prep.split_acid \\
        --input data/ACID/instances_all.json \\
        --output-dir data/ACID \\
        --test-ratio 0.2 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def build_multilabel_matrix(
    images: list[dict],
    annotations: list[dict],
    category_ids: list[int],
) -> tuple[np.ndarray, list[int]]:
    """Build a binary (n_images, n_categories) presence matrix.

    Returns:
        Y: presence matrix, Y[i, j] = 1 if image i contains category_ids[j]
        image_ids_ordered: image_id corresponding to each row of Y
    """
    # Map image_id -> set of category_ids present in that image
    image_to_cats: dict[int, set[int]] = defaultdict(set)
    for ann in annotations:
        image_to_cats[ann["image_id"]].add(ann["category_id"])

    image_ids_ordered = [img["id"] for img in images]
    cat_to_col = {c: j for j, c in enumerate(category_ids)}

    Y = np.zeros((len(image_ids_ordered), len(category_ids)), dtype=np.int8)
    for i, image_id in enumerate(image_ids_ordered):
        for cat in image_to_cats[image_id]:
            if cat in cat_to_col:
                Y[i, cat_to_col[cat]] = 1
    return Y, image_ids_ordered


def report_split_balance(
    images_train: list[dict],
    images_test: list[dict],
    annotations: list[dict],
    categories: list[dict],
) -> None:
    """Print per-class image-count and instance-count breakdown for train/test."""
    train_ids = {img["id"] for img in images_train}
    test_ids = {img["id"] for img in images_test}

    # Per-class image counts (image is "present" if class appears in it)
    img_train: dict[int, int] = defaultdict(int)
    img_test: dict[int, int] = defaultdict(int)
    inst_train: dict[int, int] = defaultdict(int)
    inst_test: dict[int, int] = defaultdict(int)

    # Track which images contain which categories
    image_to_cats: dict[int, set[int]] = defaultdict(set)
    for ann in annotations:
        image_to_cats[ann["image_id"]].add(ann["category_id"])
        if ann["image_id"] in train_ids:
            inst_train[ann["category_id"]] += 1
        elif ann["image_id"] in test_ids:
            inst_test[ann["category_id"]] += 1

    for image_id, cats in image_to_cats.items():
        if image_id in train_ids:
            for c in cats:
                img_train[c] += 1
        elif image_id in test_ids:
            for c in cats:
                img_test[c] += 1

    cat_name = {c["id"]: c["name"] for c in categories}

    print()
    print(f"Split balance (n_train_images={len(images_train)}, n_test_images={len(images_test)})")
    print(f"{'cat_id':<8}{'name':<20}{'imgs_tr':>9}{'imgs_te':>9}{'%test':>8}{'inst_tr':>10}{'inst_te':>10}{'%test':>8}")
    print("-" * 82)
    for cat_id in sorted(cat_name.keys()):
        tr_i = img_train[cat_id]
        te_i = img_test[cat_id]
        tr_a = inst_train[cat_id]
        te_a = inst_test[cat_id]
        pct_img = 100 * te_i / (tr_i + te_i) if (tr_i + te_i) > 0 else 0
        pct_inst = 100 * te_a / (tr_a + te_a) if (tr_a + te_a) > 0 else 0
        print(
            f"{cat_id:<8}{cat_name[cat_id]:<20}{tr_i:>9}{te_i:>9}{pct_img:>7.1f}%{tr_a:>10}{te_a:>10}{pct_inst:>7.1f}%"
        )


def split_acid(
    input_path: str | Path,
    output_dir: str | Path,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Split ACID instances_all.json into train and test JSONs.

    Returns:
        (train_path, test_path)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]
    category_ids = [c["id"] for c in categories]

    # Filter out images with no annotations — they have no labels to stratify on
    # and contribute nothing to detection training. We'll log them and exclude.
    image_ids_with_anns = {ann["image_id"] for ann in annotations}
    images_with_anns = [img for img in images if img["id"] in image_ids_with_anns]
    images_without = [img for img in images if img["id"] not in image_ids_with_anns]
    if images_without:
        print(f"Note: {len(images_without)} images have no annotations and are excluded from the split.")

    Y, image_ids_ordered = build_multilabel_matrix(
        images_with_anns, annotations, category_ids
    )
    X = np.arange(len(image_ids_ordered)).reshape(-1, 1)

    # Iterative multi-label stratification
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError:
        raise ImportError(
            "Missing dependency 'iterative-stratification'. "
            "Install with: pip install iterative-stratification"
        )

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    train_idx, test_idx = next(msss.split(X, Y))

    train_image_ids = {image_ids_ordered[i] for i in train_idx}
    test_image_ids = {image_ids_ordered[i] for i in test_idx}

    # Partition images and annotations
    images_train = [img for img in images_with_anns if img["id"] in train_image_ids]
    images_test = [img for img in images_with_anns if img["id"] in test_image_ids]
    annotations_train = [
        ann for ann in annotations if ann["image_id"] in train_image_ids
    ]
    annotations_test = [
        ann for ann in annotations if ann["image_id"] in test_image_ids
    ]

    # Build output COCO JSONs (preserve top-level fields)
    coco_train = {
        **{k: v for k, v in coco.items() if k not in ("images", "annotations")},
        "images": images_train,
        "annotations": annotations_train,
    }
    coco_test = {
        **{k: v for k, v in coco.items() if k not in ("images", "annotations")},
        "images": images_test,
        "annotations": annotations_test,
    }

    train_path = output_dir / "instances_train.json"
    test_path = output_dir / "instances_test.json"
    with open(train_path, "w") as f:
        json.dump(coco_train, f)
    with open(test_path, "w") as f:
        json.dump(coco_test, f)

    print(f"Wrote {train_path} ({len(images_train)} images, {len(annotations_train)} annotations)")
    print(f"Wrote {test_path}  ({len(images_test)} images, {len(annotations_test)} annotations)")

    report_split_balance(images_train, images_test, annotations, categories)

    return train_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input", required=True, help="Path to ACID instances_all.json"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write instances_train.json and instances_test.json",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_acid(args.input, args.output_dir, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
