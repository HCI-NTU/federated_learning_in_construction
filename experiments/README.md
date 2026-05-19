# Experiment Matrix Driver (Stage 5)

The orchestration layer that turns the 57-run experimental matrix into a
single command. Manages the full workflow: data preparation → per-spec
training → results aggregation.

## Components

- `matrix.py` — Defines the experimental matrix as a list of run specs.
  Each spec uniquely identifies one training configuration
  (method, N, tier, training_clients, seed).
- `run_matrix.py` — The driver. Iterates over specs, calls
  `pipeline.build_pipeline` to prepare data (idempotent), then dispatches to
  the appropriate Stage 4 runner. Skip-if-exists for resume; per-spec log
  files for debugging.
- `aggregate_results.py` — Reads all per-spec result JSONs and writes a flat
  CSV with one row per run, headline metrics in columns. Ready for pandas
  analysis.

## The matrix at default settings

```
Total: 57 specs
  exp1 (Experiment 1: paradigm comparison): 28 specs
  exp2 (Experiment 2: semantic overlap):    14 specs
  exp3 (Experiment 3: client composition):  15 specs

By method:
  isolated:    24  (3 clients × 4 N values for exp1, plus exp2/exp3 isolated baselines)
  centralized:  6
  fedavg:       9
  fedprox:      9
  fedper:       9

By N value:
  1000:  7  (exp1 only)
  2000:  7  (exp1 only)
  4000: 36  (exp1 + all of exp2 + exp3)
  8000:  7  (exp1 only)

By tier:
  tier_full:        43
  tier_shared:       7  (exp2 only)
  tier_shared_cross: 7  (exp2 only)
```

The N used for Experiments 2 and 3 (`--exp23-n`, default 4000) is the
"representative" size — choose it based on Experiment 1 results showing
where method differences are most visible.

Inspect the matrix without running anything:

```bash
python -m experiments.matrix
python -m experiments.matrix --show-specs   # print every spec id
```

## Workflow

### One command for the full study

```bash
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
```

This runs all 57 specs sequentially. Each spec:
1. Prepares its data via the pipeline (idempotent — already-converted
   datasets are skipped)
2. Calls the appropriate Stage 4 runner
3. Writes a JSON results file to `results/<spec_id>.json`
4. Writes a per-spec log to `results/logs/<spec_id>.log`

### Resume after interruption

Just re-run the same command. Specs whose result JSON already exists are
skipped:

```
2026-05-19 12:34 INFO [3/57] SKIP (exists): exp1__fedavg__n2000__tier_full__MOCS-SODA-ACID__s42
2026-05-19 12:34 INFO [4/57] START: exp1__fedprox__n2000__tier_full__MOCS-SODA-ACID__s42
```

### Run only a subset

Filters compose:

```bash
# Just Experiment 1 fedper runs
python -m experiments.run_matrix \
    --filter-exp exp1 \
    --filter-method fedper \
    ...

# Just the N=1000 column across all methods
python -m experiments.run_matrix --filter-n 1000 ...

# Skip Experiments 2 and 3 entirely
python -m experiments.run_matrix --skip-exp2 --skip-exp3 ...
```

### Dry run

Print what would run, without launching:

```bash
python -m experiments.run_matrix --dry-run ...
```

### Aggregate into CSV

After all (or some) runs complete:

```bash
python -m experiments.aggregate_results \
    --results-root results \
    --output-csv results/aggregated.csv
```

One row per spec, with these columns:

| Column | Purpose |
|---|---|
| `spec_id` | Unique identifier — joins back to the JSON file |
| `exp_label` | exp1 / exp2 / exp3_drop_X |
| `method` | isolated / centralized / fedavg / fedprox / fedper |
| `n`, `tier`, `training_clients`, `seed` | Spec parameters |
| `client_for_isolated` | Which client (isolated runs only) |
| `train_time_seconds` | Wall-clock training time |
| `n_params_exchanged`, `shared_fraction` | FL communication cost (federated only) |
| `heldout_map50`, `heldout_map50_95` | Held-out mAP, all classes |
| `heldout_map50_testable`, `heldout_map50_95_testable` | **Headline metric** |
| `heldout_n_testable_classes_evaluated` | Sanity check — should be 7 on real data |
| `local_map50`, `local_map50_95` | Local val mAP (isolated only) |
| `pooled_val_map50`, `pooled_val_map50_95` | Pooled val mAP (centralized only) |
| `heldout_map50__<class_name>` | Per-class held-out mAP (one column per unified class) |

The headline figure for Experiment 1 is `heldout_map50_testable` plotted
against `n`, with one line per `method`.

## Failure handling

When a single spec fails (OOM, transient error, corrupted data), the driver
logs the traceback to that spec's log file and continues with the rest of
the matrix. The `matrix_run_report.json` summarizes per-spec status:

```json
{
  "summary": {"n_total": 57, "n_ok": 55, "n_skipped": 0, "n_failed": 2},
  "per_spec": {
    "exp1__fedavg__n8000__...": {"status": "failed", "error": "OOM", ...},
    "exp1__fedper__n8000__...": {"status": "ok", "elapsed_seconds": 4302.1, ...},
    ...
  }
}
```

To retry only failed specs: delete their result JSON files, then re-run
the matrix command (skip-existing will avoid re-running the successful
ones).

## Estimated wall-clock cost

For a single A100-class GPU at the default settings (50 epochs,
yolo11s, imgsz=640, batch=16):

- **Isolated** at N=4000: ~30–45 min per client × 3 clients = ~2 hours
- **Centralized** at N=4000: ~45–90 min (3× data per client)
- **Federated** at N=4000 with 50 rounds × 1 epoch: ~2–4 hours
- Total over the full sweep with 4 N values: roughly **5–8 GPU-days** for all 57 runs

The driver is sequential by design. For parallel execution across multiple
GPUs, partition the matrix by `--filter-method` or `--filter-n` and launch
parallel processes — each writes to the same `results/` directory and
won't collide because spec IDs are unique.

## Important design choices

### N for Experiments 2 and 3 is deferred

`--exp23-n` defaults to 4000 but should be chosen *after* Experiment 1
results are inspected. Re-running with a different `--exp23-n` produces
new spec IDs (because the spec ID includes N), so previously-completed
exp2/exp3 runs at a different N are preserved.

### Held-out yamls are produced by the pipeline

The pipeline writes a held-out data.yaml even though there's no subsample
manifest (held-out is never subsampled). This is what `pipeline.build_pipeline`
does at the end of `build_for_client(client="CIS", ...)`. The filename
uses `n0` as a marker that no subsampling was applied:
`work/configs/CIS_n0_tier_full_s42.yaml`.

### Tier filtering for held-out evaluation

When a non-full tier is active, the pipeline also filters the held-out
labels to retain only active classes. This is essential for fair
comparison: a model trained on tier_shared should only be evaluated on
tier_shared classes. The filtered labels live at
`work/labels_<tier>/CIS/test/`.
