# Data Pipeline (Stage 2)

Orchestrates the data preparation pipeline on top of Stage 1's converters:
subsampling, tier-based class restriction, and Ultralytics-compatible YAML
generation.

## Components

- `class_subsets.py` — Tier definitions (Tier Shared, Tier Shared+Cross, Tier
  Full) derived from the unified schema.
- `subsample.py` — Stratified subsampler. Selects N images per client using
  iterative multi-label stratification, reproducible via seed. Optionally
  restricts stratification to active classes for a given tier.
- `tier_filter.py` — Non-destructive label filter. Rewrites YOLO label files
  to retain only annotations for the active tier's classes.
- `dataset_yaml.py` — Generates the Ultralytics `data.yaml` config that
  defines training/validation paths and the class list.
- `build_pipeline.py` — Top-level orchestrator. One CLI entry point runs
  Stage 1 conversions + Stage 2 subsampling/filtering/YAML generation for a
  given (clients, N, tier, seed) configuration.

## Usage

The orchestrator handles everything in one command:

    python -m pipeline.build_pipeline \
        --data-root data \
        --output-root work \
        --clients MOCS SODA ACID \
        --heldout CIS \
        --n 4000 \
        --tier tier_full \
        --seed 42

This produces, under `work/`:

```
work/
├── labels/                              # Original Stage 1 YOLO labels
│   ├── MOCS/{train,test}/
│   ├── SODA/{train,test}/
│   ├── ACID/{train,test}/
│   └── CIS/test/
├── labels_tier_shared/                  # Only if --tier tier_shared (or tier_shared_cross)
│   └── ...
├── manifests/                            # Subsample manifests, one per client/N/tier
│   ├── MOCS_train_n4000_tier_full_s42.txt
│   └── ...
├── image_lists/                          # Image path lists for Ultralytics
│   ├── MOCS_train_n4000_tier_full_s42.txt
│   ├── MOCS_test_tier_full.txt
│   └── ...
├── configs/                              # Ultralytics data.yaml files
│   ├── MOCS_n4000_tier_full_s42.yaml
│   └── ...
└── pipeline_summary_n4000_tier_full_s42.json   # Run metadata
```

## Idempotency

The orchestrator is idempotent at each stage:

- **Conversion** is skipped if the target labels directory already exists
  with content.
- **ACID split** is skipped if `instances_train.json` and `instances_test.json`
  are already present.
- **Subsample manifests** are reused if a matching one (same N, tier, seed)
  already exists.
- **Tier-filtered labels** are reused if the target directory already
  contains the expected files.
- **`data.yaml` files** are always regenerated (cheap and config-sensitive).

To force re-run, delete the relevant directory or file.

## Tier definitions

| Tier | YOLO IDs | Description |
|---|---|---|
| `tier_shared` | 0–6 | The 7 classes testable on CIS held-out |
| `tier_shared_cross` | 0–8 | tier_shared + tower_crane + mobile_crane |
| `tier_full` | 0–16 | All 17 unified classes (no filtering) |

See `class_subsets.py` for the precise definitions, which are derived from
the schema rather than hardcoded.

## Subsampling

`subsample.py` uses iterative multi-label stratification on per-image class
presence vectors. This ensures balanced class coverage in the selected
subset rather than uniform random sampling, which can drop rare classes
entirely at small N.

When `--tier` is restricted (non-full), stratification is restricted to
active classes — the subsampler preferentially selects images that
contribute to the active tier. Images with no active-class annotations are
excluded from the candidate pool.

If requested N exceeds available annotated images, the subsampler falls
back to using all available and logs a warning. The actual count is
recorded in the manifest header.

## Image directory expectations

The pipeline assumes images live at paths matching the convention from the
top-level README:

```
data/<CLIENT>/images/<split>/<stem>.<ext>
```

The image list files (in `work/image_lists/`) reference absolute paths to
the originals. No images are copied or moved. Make sure your real datasets
follow this layout — image filename stems must match the annotation file's
internal references.

For ACID specifically, after the 80:20 split, the image files must be
present in either `images/train/` or `images/test/`. If your raw ACID has
all images in a single directory, you'll need to either:
1. Physically split the images by reading the manifest and copying, or
2. Modify the pipeline's `CLIENT_CONFIG` to point both `train` and `test`
   image_dirs at the same source directory (the image lists will still be
   correct based on the stems-per-split JSON).
