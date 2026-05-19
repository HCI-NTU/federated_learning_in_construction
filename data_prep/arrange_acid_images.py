"""ACID image arrangement utility.

After `split_acid.py` produces `instances_train.json` and `instances_test.json`,
this script physically moves (or optionally copies) ACID image files into the
corresponding `images/train/` and `images/test/` subdirectories, based on the
filenames recorded in each split JSON.

Why this exists
---------------
ACID is distributed as a single `instances_all.json` with all images in one
directory (e.g., `data/ACID/images/`). The pipeline expects images at
`data/ACID/images/train/` and `data/ACID/images/test/` after splitting. This
utility bridges that gap.

Default behavior is to MOVE files (one-time operation, no disk duplication).
Use --copy to copy instead if you want to preserve the source directory.

Idempotent: if a file is already in its target subdir, it's left alone. If
a referenced file is missing entirely, a warning is logged.

Usage:
    python -m data_prep.arrange_acid_images \\
        --acid-dir data/ACID \\
        --source-images-dir data/ACID/images          # if all images are flat here
        # or
        --source-images-dir data/ACID/images/train    # if user already put them here

    # To copy instead of move:
        --mode copy
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


def _read_filenames(json_path: Path) -> list[str]:
    """Extract image file_name entries from a COCO JSON."""
    with open(json_path) as f:
        coco = json.load(f)
    return [img["file_name"] for img in coco["images"]]


def _stem_to_extension_index(source_dir: Path) -> dict[str, str]:
    """Index image files in source_dir by stem -> filename (with extension).

    Allows lookup when the COCO JSON's file_name has a different extension
    than the actual files on disk, or when the file_name lacks a directory
    prefix that the actual file has.
    """
    index: dict[str, str] = {}
    if not source_dir.is_dir():
        return index
    for p in source_dir.iterdir():
        if p.is_file():
            index[p.stem] = p.name
    return index


def arrange_acid_images(
    acid_dir: str | Path,
    source_images_dir: str | Path,
    mode: str = "move",
) -> dict:
    """Move/copy ACID images into images/train and images/test based on split JSONs.

    Args:
        acid_dir: Path to data/ACID (contains instances_train.json, instances_test.json)
        source_images_dir: Directory holding all the raw image files
        mode: 'move' (default) or 'copy'

    Returns:
        Stats dict.
    """
    acid_dir = Path(acid_dir)
    source_images_dir = Path(source_images_dir)

    train_json = acid_dir / "instances_train.json"
    test_json = acid_dir / "instances_test.json"
    if not train_json.exists() or not test_json.exists():
        raise FileNotFoundError(
            f"Expected {train_json} and {test_json} to exist. "
            f"Run split_acid.py first."
        )

    train_filenames = _read_filenames(train_json)
    test_filenames = _read_filenames(test_json)

    target_train = acid_dir / "images" / "train"
    target_test = acid_dir / "images" / "test"
    target_train.mkdir(parents=True, exist_ok=True)
    target_test.mkdir(parents=True, exist_ok=True)

    # Index source images by stem (in case JSON file_name differs slightly from disk)
    source_index = _stem_to_extension_index(source_images_dir)

    if mode not in ("move", "copy"):
        raise ValueError("mode must be 'move' or 'copy'")
    op = shutil.move if mode == "move" else shutil.copy2

    stats = {
        "mode": mode,
        "source_dir": str(source_images_dir),
        "train_moved": 0,
        "test_moved": 0,
        "train_already_in_place": 0,
        "test_already_in_place": 0,
        "missing": 0,
        "missing_files": [],
    }

    def _arrange_one(filename: str, target_dir: Path, split_label: str) -> None:
        target_path = target_dir / Path(filename).name

        # Skip if already in target (idempotent)
        if target_path.exists():
            stats[f"{split_label}_already_in_place"] += 1
            return

        # Look up actual file in source
        source_path = None
        # First try: exact filename match
        candidate = source_images_dir / Path(filename).name
        if candidate.exists():
            source_path = candidate
        else:
            # Second: try by stem (handles extension mismatches)
            stem = Path(filename).stem
            if stem in source_index:
                source_path = source_images_dir / source_index[stem]

        if source_path is None or not source_path.exists():
            stats["missing"] += 1
            if len(stats["missing_files"]) < 10:
                stats["missing_files"].append(filename)
            return

        op(str(source_path), str(target_path))
        stats[f"{split_label}_moved"] += 1

    for fn in train_filenames:
        _arrange_one(fn, target_train, "train")
    for fn in test_filenames:
        _arrange_one(fn, target_test, "test")

    return stats


def _print_stats(stats: dict) -> None:
    print(f"\nACID image arrangement ({stats['mode']}):")
    print(f"  Source dir:             {stats['source_dir']}")
    print(f"  Train: moved={stats['train_moved']}, already in place={stats['train_already_in_place']}")
    print(f"  Test:  moved={stats['test_moved']}, already in place={stats['test_already_in_place']}")
    if stats["missing"] > 0:
        print(f"  Missing: {stats['missing']} files")
        print(f"    Examples: {stats['missing_files']}")
        print(
            "    (These images are referenced in the split JSONs but not "
            "found on disk. Verify the source directory is correct.)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--acid-dir", required=True,
        help="Path to data/ACID (must contain instances_train.json and instances_test.json)",
    )
    parser.add_argument(
        "--source-images-dir", required=True,
        help="Directory containing the raw ACID image files",
    )
    parser.add_argument(
        "--mode", default="move", choices=["move", "copy"],
        help="'move' (default) saves disk; 'copy' preserves the source directory",
    )
    args = parser.parse_args()

    stats = arrange_acid_images(
        acid_dir=args.acid_dir,
        source_images_dir=args.source_images_dir,
        mode=args.mode,
    )
    _print_stats(stats)


if __name__ == "__main__":
    main()
