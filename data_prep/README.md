# Data Preparation (Stage 1)

This module converts the four source datasets (MOCS, SODA, ACID, CIS) into a
unified YOLO format keyed on the schema in `configs/class_schema.json`.

## Components

- `schema.py` — Loads the unified class schema and provides per-source lookups.
- `split_acid.py` — Splits ACID's `instances_all.json` into train/test at 80:20
  using image-level iterative multi-label stratification.
- `arrange_acid_images.py` — Physically moves (or copies) ACID images into
  `images/train/` and `images/test/` subdirectories based on the split JSONs.
  Called automatically by `pipeline/build_pipeline.py` after splitting.
- `convert_coco.py` — Converts MOCS, CIS, and ACID COCO JSON annotations to YOLO
  format. Single source of truth via `--source MOCS|CIS|ACID`.
- `convert_voc.py` — Converts SODA Pascal VOC XML annotations to YOLO format.

All converters produce per-image `.txt` files with lines of the form:

    class_id cx cy w h

where `class_id` is the 0-indexed YOLO class id (from `unified_id - 1`), and
all coordinates are normalized to [0, 1].

## Setup

    pip install -r requirements.txt

## Usage

### Step 1: Split ACID

ACID arrives as a single `instances_all.json`. Split it 80:20 first:

    python -m data_prep.split_acid \
        --input data/ACID/instances_all.json \
        --output-dir data/ACID \
        --test-ratio 0.2 \
        --seed 42

This writes `data/ACID/instances_train.json` and `data/ACID/instances_test.json`.

### Step 2: Convert COCO datasets (MOCS, CIS, ACID)

For each dataset and split:

    python -m data_prep.convert_coco \
        --input data/MOCS/instances_train.json \
        --labels-dir data/MOCS/labels/train \
        --source MOCS

Following the per-dataset split rules from the project spec:

- **MOCS:** train → train, val → test
- **CIS:** test only (train/val unused; CIS is held-out)
- **ACID:** post-split train → train, test → test

### Step 3: Convert SODA VOC XML

SODA's XMLs live in a flat directory; split is defined by which images are in
`images/train` vs `images/test`. Use the `--image-list` filter to convert only
XMLs corresponding to one split:

    # Generate image list for the train split
    ls data/SODA/images/train > /tmp/soda_train_list.txt

    python -m data_prep.convert_voc \
        --input-dir data/SODA/annotations \
        --labels-dir data/SODA/labels/train \
        --image-list /tmp/soda_train_list.txt

Same for the test split.

## Class schema

The schema is `configs/class_schema.json`. It defines 17 unified classes;
of these, 7 are testable on the CIS held-out:

  worker, excavator, bulldozer, wheel_loader, dump_truck,
  concrete_mixer_truck, roller_compactor

The remaining 10 classes either appear only in training clients (8 classes,
used for per-client and cross-client evaluation) or only in CIS held-out
(2 classes — precast_panel and precast_panel_truck — excluded from evaluation
since no training source provides them).

## Notes

- Annotations with `category_id`s (or VOC `<name>` strings) not in the schema
  are skipped with a count and the unmapped IDs/names reported.
- Bboxes with zero or negative width/height are skipped.
- Images with no annotations after filtering get an empty `.txt` file
  (Ultralytics treats this as a negative sample, which is correct behavior).
- The schema for SODA keys on the XML `<name>` value `"person"`, which is the
  actual value in the VOC files. (The schema's `"id"` field for SODA is
  unused by the VOC converter.)
