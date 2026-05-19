# Training Runners (Stage 4)

Top-level scripts that execute one training configuration each. The Stage 5
experiment driver calls these repeatedly with different parameters to fill
out the experimental matrix.

## Components

- `evaluate.py` — `evaluate_on_yaml(model, data_yaml, ...)`: shared helper
  that runs Ultralytics `model.val()` against a data.yaml and extracts the
  structured metrics (overall mAP, per-class mAP, "testable" subset mAP).
  Used by all three training runners.
- `train_isolated.py` — Trains one YOLO11 model on one client's local data.
  Evaluates on both the client's own test split and the held-out CIS split.
  Produces the per-client isolated baselines.
- `train_centralized.py` — Pools image lists from multiple clients into a
  combined data.yaml, trains a single model on the union. The "upper bound"
  baseline.
- `train_federated.py` — Runs one FL session (FedAvg / FedProx / FedPer)
  via Flower simulation, captures the final aggregated parameters, evaluates
  the resulting global model on the held-out CIS split.

## Output

Every runner writes a single JSON results file with a common structure:

```json
{
  "method": "fedper",                  // or "isolated", "centralized", "fedavg", "fedprox"
  "model_variant": "yolo11s.yaml",
  "nc": 17,
  "epochs": 50,                        // or num_rounds/epochs_per_round for federated
  "imgsz": 640,
  "batch": 16,
  "seed": 42,
  "train_time_seconds": 1832.4,

  "heldout_eval": {
    "data_yaml": "...",
    "class_names": {0: "worker", ...},
    "map50": 0.412,                    // overall held-out mAP@50
    "map50_95": 0.241,
    "map50_per_class": {0: 0.55, 1: 0.38, ...},
    "map50_95_per_class": {...},
    "map50_testable": 0.467,           // mean over the 7 testable classes
    "map50_95_testable": 0.272,
    "n_testable_classes_evaluated": 7,
    "n_images": 2400
  },

  // Method-specific extras:
  "client": "MOCS",                    // isolated only
  "local_eval": {...},                 // isolated only
  "client_yamls": [...],               // centralized + federated
  "pooled_val_eval": {...},            // centralized only
  "history": {                         // federated only
    "fit_rounds": [{round: 1, metrics: {...}, n_clients: 3}, ...],
    "eval_rounds": [...]
  },
  "init_info": {                       // federated only
    "n_params_exchanged": 378,
    "n_params_total": 499,
    "shared_fraction": 0.758
  }
}
```

The headline metric for Experiment 1 is `heldout_eval.map50_testable`
(or `map50_95_testable`). The experiment driver collects this across all
runs into a CSV.

## Usage

### Isolated

```bash
python -m train.train_isolated \
    --client MOCS \
    --data-yaml work/configs/MOCS_n4000_tier_full_s42.yaml \
    --heldout-data-yaml work/configs/CIS_n0_tier_full_s42.yaml \
    --model-variant yolo11s.yaml \
    --epochs 50 \
    --imgsz 640 \
    --batch 16 \
    --device cuda:0 \
    --results-path results/exp1_isolated_MOCS_n4000_s42.json
```

### Centralized

```bash
python -m train.train_centralized \
    --client-yamls work/configs/MOCS_n4000_tier_full_s42.yaml \
                   work/configs/SODA_n4000_tier_full_s42.yaml \
                   work/configs/ACID_n4000_tier_full_s42.yaml \
    --heldout-data-yaml work/configs/CIS_n0_tier_full_s42.yaml \
    --model-variant yolo11s.yaml \
    --epochs 50 \
    --results-path results/exp1_centralized_n4000_s42.json
```

### Federated

```bash
python -m train.train_federated \
    --method fedper \
    --client-yamls work/configs/MOCS_n4000_tier_full_s42.yaml \
                   work/configs/SODA_n4000_tier_full_s42.yaml \
                   work/configs/ACID_n4000_tier_full_s42.yaml \
    --client-names MOCS SODA ACID \
    --heldout-data-yaml work/configs/CIS_n0_tier_full_s42.yaml \
    --num-rounds 50 \
    --epochs-per-round 1 \
    --model-variant yolo11s.yaml \
    --results-path results/exp1_fedper_n4000_s42.json
```

For FedProx, also pass `--proximal-mu 0.01` (default).

## Design notes

### Epoch budgets

For comparable comparisons:
- **Isolated**: `epochs = E_total` (full budget)
- **Centralized**: `epochs = E_total` (same budget, but on N×3 data)
- **Federated**: `num_rounds × epochs_per_round = E_total`

For a 50-epoch budget, FL with `epochs_per_round=1` runs 50 rounds; with
`epochs_per_round=2`, 25 rounds. Both should be tried for a thorough
comparison, but `epochs_per_round=1` is the most common FL literature default.

### FedPer held-out evaluation

FedPer aggregates only backbone+neck; each client keeps its own head. There
is no "global head," which raises the question: how do you evaluate the
"global FedPer model" on a held-out set?

This study uses the convention of evaluating with a **freshly-initialized
random head** plus the aggregated backbone+neck. The held-out mAP under
this configuration reflects the *transferability of the shared
representation*, not the performance of any specific client's specialized
head.

This is one of several defensible choices in the literature (alternatives:
average all clients' heads, use the largest client's head, evaluate
per-client). The choice should be stated explicitly in the methods section.

### nc consistency

All three runners use the same `nc` resolution as `fl.yolo_client`:
infer from the data.yaml's `names` field. This guarantees that the same
model architecture is built whether you're running isolated, centralized,
or federated training on the same data — making the comparison fair.

### Reproducibility

Every runner takes `--seed` and forwards it to Ultralytics' training loop.
For federated, the seed governs initial parameters and per-client training;
each client uses the same seed (so all clients start from the same global
model each round — this is Flower's standard behavior).
