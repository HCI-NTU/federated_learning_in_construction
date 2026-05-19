# Federated Object Detection Across Construction Projects with Partial Semantic Overlap

A study of federated learning (FL) for construction object detection, framed
around partial semantic overlap between client projects.

This README is the canonical project reference: research framing, experimental
design, datasets, methods, and constraints. Implementation details for each
stage live in their respective module READMEs.

---

## Research framing

### Problem reformulation

Most federated learning research on object detection assumes clients share a
common label space, with statistical differences (size, class balance) as the
main heterogeneity. Construction violates this assumption in a specific way:
**construction projects exhibit partial and asymmetric semantic overlap.**
Different projects label different subsets of objects, driven by what each
operation cares about — a worker-safety monitoring program produces
person-only annotations; an earthmoving productivity program produces
equipment-only annotations; a broader program produces both.

This reframes the federated detection problem from *"can FL match centralized
training"* to *"under what conditions can clients with partially overlapping
label spaces collaboratively learn a shared detector, and what aggregation
strategies remain effective as semantic divergence increases?"*

### Research question

> How do federated learning strategies affect collaborative object detection
> across construction projects with partial semantic overlap between clients?

Answered through three sub-questions:

1. How does federated learning compare to isolated and centralized training
   under realistic construction data conditions?
2. To what extent do unique (non-overlapping) classes in each client's label
   space help or hinder learning of shared classes?
3. How sensitive are FL methods to the inclusion or exclusion of individual
   clients with distinct label scopes?

### Contributions

- **Conceptual.** A reformulation of federated object detection in
  construction as collaborative learning across partially overlapping label
  spaces.
- **Empirical.** A systematic comparison of FL paradigms (FedAvg, FedProx,
  FedPer) against isolated and centralized baselines, across realistic
  per-client data scales and class-space configurations, using four real
  construction datasets.
- **Practical.** A characterization of when federated learning is beneficial
  for construction object detection — and when it is not — with specific
  attention to partial semantic overlap and label-space asymmetry.

Safety-critical applications (PPE compliance, hazard detection, worker
proximity monitoring) motivate the need for collaborative learning in the
introduction and implications, but do not define the experimental scope.

---

## Datasets and federation setup

### Training clients (3)

| Client | Unified classes | Role |
|---|---|---|
| MOCS | 13 | Broad — worker + diverse equipment |
| SODA | 1 | Specialist — worker only |
| ACID | 10 | Equipment-focused — earthmoving emphasis |

### Held-out (1)

| Source | Unified classes | Role |
|---|---|---|
| CIS | 9 (incl. 2 unmatched in training) | Cross-project generalization evaluation |

Only CIS's `test` split is used. Train and val are ignored.

### Unified class schema

17 unified classes total. Of these:

- **7 testable on held-out CIS**: worker, excavator, bulldozer, wheel_loader,
  dump_truck, concrete_mixer_truck, roller_compactor
- **8 trained but not in held-out**: tower_crane, mobile_crane,
  concrete_pump_truck, backhoe_loader, grader, pile_driver, crane_hook,
  other_vehicle (used for per-client and cross-client evaluation)
- **2 in held-out only, not in any training source**: precast_panel,
  precast_panel_truck (excluded from evaluation — no method can predict them)

See `configs/class_schema.json` for the full schema mapping each source
dataset's category IDs to unified class IDs.

### Data split conventions

The project follows these rules across all datasets:

- If a dataset has separate `test`, the `val` split is merged into `train`
  as `trainval`.
- If a dataset has no separate `test`, the `val` split is used as `test`.
- **MOCS:** has train + val, no test → `val` becomes test
- **CIS:** has train + val + test → only `test` is used (CIS is held-out)
- **SODA:** has train + test → used as-is
- **ACID:** has only `instances_all.json` → split 80:20 to train/test via
  iterative multi-label stratification

### Data availability (verified)

All three training clients have **at least 8000 training images** available
after splitting. This supports the full per-client size sweep without
fallback.

| Client | Train images available |
|---|---|
| MOCS | ≥ 8000 |
| SODA | ≥ 8000 |
| ACID | ~8000 (post 80:20 split from ~10000 total) |

---

## Methods compared

| Method | Role | What it tests |
|---|---|---|
| Isolated | Lower bound | Privacy-perfect but data-starved per-client training |
| Centralized | Upper bound | What full data pooling would deliver |
| FedAvg | Standard FL baseline | Naive federation under partial overlap |
| FedProx | Drift-controlled FL | Whether proximal regularization closes the gap |
| FedPer | Structural personalization (federate backbone + neck, keep heads local) | Whether per-client heads resolve label-space asymmetry |

**Detector.** Ultralytics YOLO11. Default variant: `yolo11s`. Configurable
via experiment config.

**FL framework.** Flower. Single-machine simulation, three client processes.

---

## Experimental design

The study comprises three coordinated experiments, all sharing the same data
pipeline and held-out evaluation.

### Experiment 1: Learning paradigm comparison

All 5 methods, per-client size sweep at **N ∈ {1000, 2000, 4000, 8000}**,
full class space (17 classes), all 3 training clients.

≈ 28 runs. Headline result: held-out CIS mAP as a function of N, one line
per method.

### Experiment 2: Semantic overlap ablation

Class space restricted to three tiers, at a single representative N (to be
determined after Experiment 1):

| Tier | Active training classes | Description |
|---|---|---|
| Shared (7) | The 7 testable classes (worker + 6 shared equipment) | Pure shared-semantics control |
| Shared + Cross (9) | The 7 + tower_crane + mobile_crane (shared MOCS↔ACID, not in held-out) | Tests cross-client auxiliary supervision |
| Full (17) | All unified classes | Realistic partial overlap (= Experiment 1 baseline) |

Held-out evaluation is fixed at the 7 testable classes across all tiers, so
the comparison directly measures how adding non-evaluable training classes
affects evaluable-class performance.

Tier Full reuses Experiment 1 results at the chosen N. Tiers Shared and
Shared+Cross add 2 tiers × 5 methods = 10 runs.

### Experiment 3: Client composition ablation

The role of each client is isolated by training with subsets of the three
clients, at the same single N as Experiment 2. Full class space. FL methods
only (FedAvg, FedProx, FedPer):

- Drop-SODA (MOCS + ACID only)
- Drop-MOCS (SODA + ACID only)
- Drop-ACID (MOCS + SODA only)

The all-three baseline is taken from Experiment 1 at the chosen N.

3 FL methods × 3 drop configurations = 9 runs.

### Total experimental matrix

**Approximately 47 runs.** May expand if N for Experiments 2/3 is run at
multiple size points instead of one.

The choice of N for Experiments 2 and 3 is deferred — it will be set after
Experiment 1 reveals where method differences are most visible. In the code,
N is configurable everywhere; no value is hardcoded.

---

## Evaluation

### Primary metric

Held-out CIS mAP — both mAP@50 and mAP@50-95 — computed over the 7 testable
unified classes.

### Secondary metrics

- Per-client mAP on each training client's own test split (local performance
  retention)
- Per-class mAP breakdown on the held-out, separated into shared vs. unique
  class groupings
- Cross-client mAP for tower_crane and mobile_crane (MOCS ↔ ACID transfer)
- FL-vs-centralized gap as a function of per-client data size

### Reporting

Each method × configuration combination produces a single metrics record.
The full results table is exported as CSV, supporting both per-experiment
analysis and cross-experiment synthesis.

---

## Scope and limitations

The study does **not** address:

- Visual / contextual domain shift between projects (acknowledged as a
  confound, not the focus). The semantic overlap dimension is more
  construction-distinctive and admits cleaner quantification.
- Temporal evolution within projects (separate research direction).
- Cross-organizational privacy mechanisms beyond model-weight exchange
  (e.g., differential privacy, secure aggregation). These are orthogonal
  additions, not the contribution.
- Algorithm-level innovation in FL aggregation. The contribution is
  empirical and conceptual, using established FL methods.

---

## Predicted result patterns

Stated explicitly so the design is falsifiable:

| Prediction | Confidence |
|---|---|
| Centralized > all FL methods > best isolated client on held-out generalization | High (near-deterministic given data-pooling advantage) |
| FedPer ≥ FedProx ≥ FedAvg on per-client mAP | High (FedPer's design intent) |
| FedPer ≥ FedProx ≥ FedAvg on held-out mAP | Moderate (depends on backbone benefit) |
| FL-vs-isolated gap shrinks as per-client N grows | High (documented in FL literature) |
| Restricting to Tier Shared improves vs. Tier Full | Uncertain (the finding is the contribution) |
| Removing SODA hurts worker mAP, helps equipment mAP under FedAvg; both effects diminished under FedPer | Moderate (the FedPer-defense story) |

Contradicting predictions are themselves publishable findings. The design
ensures every outcome is informative.

---

## Project structure

```
fl-construction/
├── README.md                       ← this file
├── requirements.txt
├── configs/
│   └── class_schema.json           ← unified class schema
├── data_prep/                      ← Stage 1: dataset conversion to YOLO
│   ├── README.md
│   ├── schema.py
│   ├── split_acid.py
│   ├── arrange_acid_images.py
│   ├── convert_coco.py
│   └── convert_voc.py
├── pipeline/                       ← Stage 2: dataset orchestration
│   ├── README.md
│   ├── class_subsets.py
│   ├── subsample.py
│   ├── tier_filter.py
│   ├── dataset_yaml.py
│   └── build_pipeline.py
├── fl/                             ← Stage 3: Flower + YOLO integration
│   ├── README.md
│   ├── parameter_utils.py
│   ├── yolo_client.py
│   ├── strategies.py
│   ├── server.py
│   └── run_client.py
├── train/                          ← Stage 4: per-method training runners
│   ├── README.md
│   ├── evaluate.py
│   ├── train_isolated.py
│   ├── train_centralized.py
│   └── train_federated.py
├── experiments/                    ← Stage 5: matrix driver + aggregator
│   ├── README.md
│   ├── matrix.py
│   ├── run_matrix.py
│   └── aggregate_results.py
├── tests/                          ← regression tests for each stage
│   ├── make_synthetic_mocs.py
│   ├── make_synthetic_cis.py
│   ├── make_synthetic_acid.py
│   ├── make_synthetic_soda.py
│   ├── make_synthetic_images.py
│   ├── test_converters.py
│   ├── test_pipeline.py
│   ├── test_yolo_client.py
│   ├── test_fl_integration.py
│   ├── test_training_runners.py
│   └── test_matrix_driver.py
└── data/                           ← raw datasets (user-provided)
    ├── MOCS/
    ├── SODA/
    ├── ACID/
    └── CIS/
```

## Dataset directory layout (expected)

```
data/MOCS/
    instances_train.json
    instances_val.json          (becomes test)
    images/train/
    images/val/

data/SODA/
    annotations/*.xml           (one per image, named to match)
    images/train/
    images/test/

data/ACID/
    instances_all.json          (split to train/test by pipeline)
    images/                     (place all raw images here; pipeline moves them)
    images/train/               (populated by pipeline)
    images/test/                (populated by pipeline)

data/CIS/
    instances_test.json         (only test split is used)
    images/test/
```

For ACID, the pipeline runs two steps automatically:
1. **Split** `instances_all.json` 80:20 → `instances_train.json` + `instances_test.json`
2. **Arrange** raw images from a flat directory (default `data/ACID/images/`) into `images/train/` and `images/test/` subdirs based on the split.

If your raw ACID images live elsewhere, pass `--acid-raw-images-dir <path>` to the orchestrator. Use `--acid-arrange-mode copy` if you want to preserve the source directory (default is `move`, which saves disk).

## Running the study

Once data is in place at `data/<DATASET>/`, the full study runs in three
commands:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the experimental matrix (57 specs by default)
python -m experiments.run_matrix \
    --data-root data \
    --output-root work \
    --results-root results \
    --epochs 50 \
    --model-variant yolo11s.yaml \
    --imgsz 640 \
    --batch 16 \
    --device cuda:0 \
    --exp23-n 4000

# 3. Aggregate results into a single CSV
python -m experiments.aggregate_results \
    --results-root results \
    --output-csv results/aggregated.csv
```

The matrix driver is resumable — re-running the same command picks up
where it left off. See `experiments/README.md` for filters, the dry-run
mode, and the full list of options.

To run the study in stages (recommended for first execution):

```bash
# Just Experiment 1
python -m experiments.run_matrix --filter-exp exp1 ...

# Pick the right N for Experiments 2 and 3 based on Experiment 1 results
# then run the rest
python -m experiments.run_matrix --filter-exp exp2 exp3_drop_MOCS exp3_drop_SODA exp3_drop_ACID \
    --exp23-n <chosen N> ...
```

## Build status

| Stage | Module | Status |
|---|---|---|
| 1 | `data_prep/` — schema, converters, ACID splitter | ✓ built, tested |
| 2 | `pipeline/` — subsampler, tier filter, data.yaml, orchestrator | ✓ built, tested |
| 3 | `fl/` — Flower client, strategies, server | ✓ built, tested |
| 4 | `train/` — isolated, centralized, federated runners | ✓ built, tested |
| 5 | `experiments/` — matrix definitions, driver, results aggregator | ✓ built, tested |

The project is complete. To run the full study end-to-end, see
`experiments/README.md`.

See `data_prep/README.md` for Stage 1 usage. Subsequent stages will document
their own components.
