"""Tier-based label filter.

Reads original YOLO label files and writes filtered copies that retain only
annotations for classes in the active tier. The original labels are
untouched, so multiple tiers can coexist without re-running conversion.

Filtered labels live in a tier-specific subdirectory:

    labels/train/                       (original, all classes)
    labels_tier_shared/train/           (only tier_shared classes)
    labels_tier_shared_cross/train/     (only tier_shared_cross classes)

For tier_full, no filtering is needed — point data.yaml at the original
labels directory directly.

If an image's filtered label file would be empty (no active-class
annotations), the image is excluded from the filtered split (rather than
written as an empty file — empty labels make Ultralytics treat the image as
a hard negative, which is not what we want for tier-restricted training).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from data_prep.schema import load_schema
from pipeline.class_subsets import get_tier_yolo_ids


def filter_labels_for_tier(
    source_labels_dir: str | Path,
    dest_labels_dir: str | Path,
    active_yolo_ids: Iterable[int],
    stem_filter: Optional[set[str]] = None,
) -> dict:
    """Filter YOLO label files to retain only active class annotations.

    Args:
        source_labels_dir: Original labels directory (from data_prep)
        dest_labels_dir: Where to write filtered labels
        active_yolo_ids: Set of YOLO class IDs to keep
        stem_filter: If provided, only process labels for these stems
            (typically the subsampled manifest)

    Returns:
        Stats dict: n_input, n_kept, n_dropped_empty, n_lines_total,
        n_lines_kept, n_lines_dropped
    """
    source_dir = Path(source_labels_dir)
    dest_dir = Path(dest_labels_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    active = frozenset(active_yolo_ids)

    n_input = 0
    n_kept = 0
    n_dropped_empty = 0
    n_lines_total = 0
    n_lines_kept = 0

    for txt_path in sorted(source_dir.glob("*.txt")):
        if stem_filter is not None and txt_path.stem not in stem_filter:
            continue
        n_input += 1

        kept_lines: list[str] = []
        for line in txt_path.read_text().strip().split("\n"):
            if not line:
                continue
            n_lines_total += 1
            try:
                class_id = int(line.split()[0])
            except (ValueError, IndexError):
                continue
            if class_id in active:
                kept_lines.append(line)
                n_lines_kept += 1

        if not kept_lines:
            # Image has no active-class annotations — exclude from tier split
            n_dropped_empty += 1
            continue

        dest_path = dest_dir / txt_path.name
        dest_path.write_text("\n".join(kept_lines) + "\n")
        n_kept += 1

    stats = {
        "source": str(source_dir),
        "dest": str(dest_dir),
        "n_active_classes": len(active),
        "n_input_files": n_input,
        "n_output_files": n_kept,
        "n_dropped_empty": n_dropped_empty,
        "n_lines_total": n_lines_total,
        "n_lines_kept": n_lines_kept,
        "n_lines_dropped": n_lines_total - n_lines_kept,
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source-labels", required=True)
    parser.add_argument("--dest-labels", required=True)
    parser.add_argument(
        "--tier",
        required=True,
        choices=["tier_shared", "tier_shared_cross", "tier_full"],
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest file (from subsample.py). If given, only "
        "labels for stems in the manifest are filtered.",
    )
    parser.add_argument("--schema", default=None)
    args = parser.parse_args()

    schema = load_schema(args.schema) if args.schema else load_schema()
    active = get_tier_yolo_ids(args.tier, schema)

    stem_filter = None
    if args.manifest:
        from pipeline.subsample import read_manifest
        stems, _ = read_manifest(args.manifest)
        stem_filter = set(stems)

    if args.tier == "tier_full":
        print(
            "Tier 'tier_full' does not require filtering. "
            "Use the original labels directory directly."
        )
        return

    stats = filter_labels_for_tier(
        source_labels_dir=args.source_labels,
        dest_labels_dir=args.dest_labels,
        active_yolo_ids=active,
        stem_filter=stem_filter,
    )

    print(f"Tier filter ({args.tier}):")
    print(f"  Active YOLO ids:     {sorted(active)}")
    print(f"  Input label files:   {stats['n_input_files']}")
    print(f"  Output label files:  {stats['n_output_files']}")
    print(f"  Dropped (empty):     {stats['n_dropped_empty']}")
    print(f"  Total annotations:   {stats['n_lines_total']}")
    print(f"  Kept annotations:    {stats['n_lines_kept']}")
    print(f"  Dropped annotations: {stats['n_lines_dropped']}")


if __name__ == "__main__":
    main()
