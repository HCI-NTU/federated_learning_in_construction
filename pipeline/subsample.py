"""Stratified per-client image subsampler.

Selects N image stems from a YOLO-format dataset, balanced by class presence
using iterative multi-label stratification. Reproducible via seed.

The output is a "manifest" — a list of image stems (filenames without
extension). Downstream components (data.yaml generator, tier filter) read
this manifest to know which images to include for a given (client, N, seed)
configuration.

If N exceeds the number of available images with annotations, all available
images are used and a warning is logged. The actual size used is recorded
in the manifest header.

Manifest format (one stem per line, plus a header):
    # client=MOCS split=train n_requested=4000 n_actual=4000 seed=42 tier=tier_full
    mocs_00001
    mocs_00002
    ...
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


def _build_presence_matrix(
    stems: list[str], labels_dir: Path, n_classes: int
) -> np.ndarray:
    """For each stem, build a binary presence vector across n_classes.

    Y[i, c] = 1 iff stem i has at least one annotation of YOLO class c.
    """
    Y = np.zeros((len(stems), n_classes), dtype=np.int8)
    for i, stem in enumerate(stems):
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                class_id = int(line.split()[0])
            except (ValueError, IndexError):
                continue
            if 0 <= class_id < n_classes:
                Y[i, class_id] = 1
    return Y


def stratified_subsample(
    labels_dir: str | Path,
    n: int,
    n_classes: int,
    seed: int = 42,
    active_classes: Optional[frozenset[int]] = None,
) -> tuple[list[str], dict]:
    """Subsample N image stems via iterative multi-label stratification.

    Args:
        labels_dir: Directory of YOLO .txt label files
        n: Target number of images
        n_classes: Total YOLO class count (== schema.num_classes)
        seed: Random seed
        active_classes: If provided, only stratify on these YOLO ids.
            Useful when subsampling for a restricted tier — ensures the
            sample is balanced over the classes that will actually be active.
            Note: this affects sampling strategy only; the labels themselves
            are not modified here (use tier_filter for that).

    Returns:
        (list of selected stems, stats dict)
    """
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")

    # Discover all label files
    label_files = sorted(labels_dir.glob("*.txt"))
    all_stems = [p.stem for p in label_files]

    # Exclude stems whose label file is empty (no annotations) — these contain
    # no learnable signal for stratification purposes.
    stems_with_labels: list[str] = []
    for stem, path in zip(all_stems, label_files):
        if path.stat().st_size > 0:
            stems_with_labels.append(stem)

    n_available = len(stems_with_labels)
    if n_available == 0:
        raise ValueError(f"No annotated images found in {labels_dir}")

    # If requested N >= available, return all
    if n >= n_available:
        if n > n_available:
            print(
                f"Warning: requested n={n} but only {n_available} annotated images "
                f"available in {labels_dir}. Using all available.",
                file=sys.stderr,
            )
        return stems_with_labels, {
            "n_requested": n,
            "n_actual": n_available,
            "n_available": n_available,
            "seed": seed,
            "method": "use_all",
        }

    # Build full presence matrix
    Y_full = _build_presence_matrix(stems_with_labels, labels_dir, n_classes)

    # Optionally restrict stratification to active classes only
    if active_classes is not None:
        active_cols = sorted(active_classes)
        Y_strat = Y_full[:, active_cols]
        # Drop stems with zero active-class annotations — they contribute
        # nothing useful under the active tier.
        nonzero_mask = Y_strat.sum(axis=1) > 0
        if not nonzero_mask.all():
            n_dropped = int((~nonzero_mask).sum())
            print(
                f"Note: dropping {n_dropped} stems with no active-class "
                f"annotations under the requested tier.",
                file=sys.stderr,
            )
            stems_with_labels = [s for s, keep in zip(stems_with_labels, nonzero_mask) if keep]
            Y_strat = Y_strat[nonzero_mask]
        n_available = len(stems_with_labels)
        if n >= n_available:
            if n > n_available:
                print(
                    f"Warning: after active-class filtering, only {n_available} "
                    f"images remain (requested n={n}). Using all available.",
                    file=sys.stderr,
                )
            return stems_with_labels, {
                "n_requested": n,
                "n_actual": n_available,
                "n_available": n_available,
                "seed": seed,
                "method": "use_all_after_filter",
            }
    else:
        Y_strat = Y_full

    # Iterative multi-label stratification to pick exactly N
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError:
        raise ImportError(
            "Missing dependency 'iterative-stratification'. "
            "Install with: pip install iterative-stratification"
        )

    # iterstrat doesn't support choosing N directly — it splits into two
    # groups by ratio. We use test_size=n/n_available so the "test" side
    # has exactly N items (it picks the right side via the ratio).
    target_ratio = n / n_available
    X = np.arange(n_available).reshape(-1, 1)

    # Edge case: at extreme ratios near 0 or 1, iterstrat can be unstable.
    # Clamp to a safe range and adjust if needed.
    if target_ratio >= 0.99:
        return stems_with_labels[:n], {
            "n_requested": n,
            "n_actual": min(n, n_available),
            "n_available": n_available,
            "seed": seed,
            "method": "trivial_almost_all",
        }

    # Use iterstrat to pick a balanced subset of size N
    # We treat the "test" partition as our selected subset.
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=target_ratio, random_state=seed
    )
    _, selected_idx = next(msss.split(X, Y_strat))

    # iterstrat may give us slightly more or fewer than exactly N due to
    # discrete sample counts; adjust to exactly N if possible.
    if len(selected_idx) != n:
        # Shuffle the deficit/surplus via the same seed for reproducibility
        rng = np.random.default_rng(seed)
        if len(selected_idx) > n:
            # Too many — randomly drop down to N
            selected_idx = rng.choice(selected_idx, size=n, replace=False)
        else:
            # Too few — randomly add from the unselected pool
            unselected = np.setdiff1d(np.arange(n_available), selected_idx)
            n_to_add = n - len(selected_idx)
            extras = rng.choice(unselected, size=n_to_add, replace=False)
            selected_idx = np.concatenate([selected_idx, extras])

    selected_stems = [stems_with_labels[i] for i in sorted(selected_idx)]

    return selected_stems, {
        "n_requested": n,
        "n_actual": len(selected_stems),
        "n_available": n_available,
        "seed": seed,
        "method": "iterative_stratified",
    }


def write_manifest(
    manifest_path: str | Path,
    stems: list[str],
    metadata: dict,
) -> None:
    """Write a manifest file with a header and one stem per line."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    header_parts = " ".join(f"{k}={v}" for k, v in metadata.items())
    lines = [f"# {header_parts}"] + stems
    manifest_path.write_text("\n".join(lines) + "\n")


def read_manifest(manifest_path: str | Path) -> tuple[list[str], dict]:
    """Read a manifest file. Returns (stems, metadata)."""
    manifest_path = Path(manifest_path)
    lines = manifest_path.read_text().strip().split("\n")
    metadata: dict = {}
    stems = []
    for line in lines:
        if line.startswith("#"):
            for part in line.lstrip("# ").split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    metadata[k] = v
        elif line:
            stems.append(line)
    return stems, metadata


def report_subsample_balance(
    selected_stems: list[str],
    labels_dir: Path,
    n_classes: int,
    schema_names: Optional[list[str]] = None,
) -> None:
    """Print per-class image-presence counts in the selected subsample."""
    counts: dict[int, int] = defaultdict(int)
    instance_counts: dict[int, int] = defaultdict(int)
    for stem in selected_stems:
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            continue
        seen_in_image: set[int] = set()
        for line in label_path.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                class_id = int(line.split()[0])
            except (ValueError, IndexError):
                continue
            instance_counts[class_id] += 1
            seen_in_image.add(class_id)
        for cid in seen_in_image:
            counts[cid] += 1

    print(f"\nSubsample composition ({len(selected_stems)} images):")
    print(f"{'yolo_id':<10}{'name':<25}{'images':>10}{'instances':>14}")
    print("-" * 60)
    for yid in sorted(counts.keys()):
        name = schema_names[yid] if schema_names else f"class_{yid}"
        print(f"{yid:<10}{name:<25}{counts[yid]:>10}{instance_counts[yid]:>14}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--labels-dir", required=True, help="Directory of YOLO .txt labels"
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Target number of images"
    )
    parser.add_argument(
        "--output", required=True, help="Output manifest path"
    )
    parser.add_argument(
        "--client", required=True, help="Client name (e.g., MOCS)"
    )
    parser.add_argument(
        "--split", default="train", help="Split name (train/test)"
    )
    parser.add_argument(
        "--tier",
        default="tier_full",
        help="Tier name for active-class filtering during stratification",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--schema", default=None, help="Path to class_schema.json"
    )
    args = parser.parse_args()

    from data_prep.schema import load_schema
    schema = load_schema(args.schema) if args.schema else load_schema()

    active_classes = None
    if args.tier != "tier_full":
        from pipeline.class_subsets import get_tier_yolo_ids
        active_classes = get_tier_yolo_ids(args.tier, schema)

    stems, stats = stratified_subsample(
        labels_dir=args.labels_dir,
        n=args.n,
        n_classes=schema.num_classes,
        seed=args.seed,
        active_classes=active_classes,
    )

    metadata = {
        "client": args.client,
        "split": args.split,
        "tier": args.tier,
        **stats,
    }
    write_manifest(args.output, stems, metadata)
    print(f"Wrote manifest: {args.output}")
    print(f"  n_requested={stats['n_requested']}, n_actual={stats['n_actual']}, "
          f"n_available={stats['n_available']}, method={stats['method']}")

    report_subsample_balance(
        stems, Path(args.labels_dir), schema.num_classes, schema.class_names
    )


if __name__ == "__main__":
    main()
