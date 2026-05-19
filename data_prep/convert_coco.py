"""COCO JSON to YOLO format converter.

Reads a COCO-format JSON file and writes per-image YOLO label files.
Each YOLO label line: `class_id cx cy w h` with all coords normalized to [0, 1].
The class_id is the 0-indexed YOLO id derived from the unified schema.

Supports three source datasets via the --source argument:
  - MOCS: single COCO category_id -> unified mapping
  - CIS:  multiple source category_ids may map to one unified class
          (e.g., people-helmet + people-no-helmet -> worker)
  - ACID: single COCO category_id -> unified mapping

Annotations whose source category_id is not in the unified schema are skipped
(with a count reported at the end). Bboxes with zero or negative width/height
are skipped.

Usage:
    python -m data_prep.convert_coco \\
        --input data/MOCS/instances_train.json \\
        --images-dir data/MOCS/images/train \\
        --labels-dir data/MOCS/labels/train \\
        --source MOCS
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .schema import load_schema, ClassSchema


def _lookup_for_source(schema: ClassSchema, source: str):
    """Return the appropriate lookup function for a given source dataset."""
    lookup = {
        "MOCS": schema.lookup_mocs,
        "CIS": schema.lookup_cis,
        "ACID": schema.lookup_acid,
    }
    if source not in lookup:
        raise ValueError(
            f"Unknown source '{source}'. Must be one of: {list(lookup.keys())}"
        )
    return lookup[source]


def convert_coco_to_yolo(
    json_path: str | Path,
    labels_dir: str | Path,
    source: str,
    schema: Optional[ClassSchema] = None,
) -> dict:
    """Convert a COCO JSON to per-image YOLO label files.

    Args:
        json_path: Path to COCO-format JSON file
        labels_dir: Output directory for .txt label files (one per image)
        source: 'MOCS', 'CIS', or 'ACID' — selects the schema lookup
        schema: Optional pre-loaded ClassSchema (default: load from configs/)

    Returns:
        Stats dict with: n_images, n_anns_written, n_anns_skipped_unmapped,
        n_anns_skipped_invalid_bbox, per_class_counts (yolo_id -> count)
    """
    json_path = Path(json_path)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if schema is None:
        schema = load_schema()
    lookup = _lookup_for_source(schema, source)

    with open(json_path) as f:
        coco = json.load(f)

    # Build image_id -> (filename, w, h) map
    images_by_id: dict[int, dict] = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    n_anns_written = 0
    n_anns_skipped_unmapped = 0
    n_anns_skipped_invalid_bbox = 0
    per_class_counts: dict[int, int] = defaultdict(int)
    unmapped_source_ids: set[int] = set()

    for image_id, image in images_by_id.items():
        img_w = image["width"]
        img_h = image["height"]
        # YOLO label filename = image filename stem + .txt
        # Image file_name may include subdir prefix; strip to basename
        stem = Path(image["file_name"]).stem
        label_path = labels_dir / f"{stem}.txt"

        lines: list[str] = []
        for ann in anns_by_image.get(image_id, []):
            unified_id = lookup(ann["category_id"])
            if unified_id is None:
                n_anns_skipped_unmapped += 1
                unmapped_source_ids.add(ann["category_id"])
                continue

            # COCO bbox: [x, y, w, h] in absolute pixels (top-left origin)
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                n_anns_skipped_invalid_bbox += 1
                continue

            # Convert to YOLO format: [cx, cy, w, h] normalized
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Clamp to [0, 1] in case of bboxes that extend slightly past image edge
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            yolo_id = schema.to_yolo_id(unified_id)
            lines.append(f"{yolo_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            per_class_counts[yolo_id] += 1
            n_anns_written += 1

        # Write the label file. For images with no valid annotations, write an
        # empty file — Ultralytics treats this as a negative sample (image with
        # no labels), which is fine.
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

    stats = {
        "source": source,
        "input": str(json_path),
        "n_images": len(images_by_id),
        "n_anns_written": n_anns_written,
        "n_anns_skipped_unmapped": n_anns_skipped_unmapped,
        "n_anns_skipped_invalid_bbox": n_anns_skipped_invalid_bbox,
        "unmapped_source_ids": sorted(unmapped_source_ids),
        "per_class_counts": dict(per_class_counts),
    }
    return stats


def _print_stats(stats: dict, schema: ClassSchema) -> None:
    print(f"\nConverted {stats['source']}: {stats['input']}")
    print(f"  Images:              {stats['n_images']}")
    print(f"  Annotations written: {stats['n_anns_written']}")
    print(f"  Skipped (unmapped):  {stats['n_anns_skipped_unmapped']}")
    if stats["unmapped_source_ids"]:
        print(f"    Unmapped source category_ids: {stats['unmapped_source_ids']}")
        print(f"    (These categories are not in the unified schema — verify if intentional)")
    print(f"  Skipped (bad bbox):  {stats['n_anns_skipped_invalid_bbox']}")
    print(f"\n  Per-class instance counts (YOLO id : count):")
    yolo_id_to_name = {i: name for i, name in enumerate(schema.class_names)}
    for yolo_id in sorted(stats["per_class_counts"].keys()):
        name = yolo_id_to_name[yolo_id]
        count = stats["per_class_counts"][yolo_id]
        print(f"    {yolo_id:>3}  {name:<25} {count:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, help="Path to COCO JSON file")
    parser.add_argument(
        "--labels-dir",
        required=True,
        help="Output directory for YOLO .txt label files",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["MOCS", "CIS", "ACID"],
        help="Which source dataset's schema mapping to use",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Optional path to class_schema.json (defaults to configs/class_schema.json)",
    )
    # --images-dir is accepted but unused — listed for orchestrator-friendly CLI.
    # The converter only reads the JSON; images are referenced by the dataset.yaml,
    # not by the label conversion step.
    parser.add_argument(
        "--images-dir",
        default=None,
        help="(Optional, unused) Path to image directory — kept for orchestration uniformity",
    )
    args = parser.parse_args()

    schema = load_schema(args.schema) if args.schema else load_schema()
    stats = convert_coco_to_yolo(
        json_path=args.input,
        labels_dir=args.labels_dir,
        source=args.source,
        schema=schema,
    )
    _print_stats(stats, schema)


if __name__ == "__main__":
    main()
