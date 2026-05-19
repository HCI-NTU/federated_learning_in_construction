"""Results aggregator.

Reads all per-spec JSON result files in a results directory and produces a
flat CSV ready for analysis. Each row is one run; columns are spec
parameters + headline metrics.

CSV columns:
  spec_id, exp_label, method, n, tier, training_clients, heldout,
  client_for_isolated (or empty), seed,
  train_time_seconds, n_params_exchanged (or empty for non-federated),
  heldout_map50, heldout_map50_95,
  heldout_map50_testable, heldout_map50_95_testable,
  heldout_n_testable_classes_evaluated,
  local_map50, local_map50_95            (isolated only)
  pooled_val_map50, pooled_val_map50_95  (centralized only)
  per_class_map50_<name>                  (one column per testable class)

Usage:
    python -m experiments.aggregate_results \\
        --results-root results \\
        --output-csv results/aggregated.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def _flat_row_for(result: dict, schema_class_names: list[str]) -> dict:
    """Flatten one result JSON into a CSV row dict."""
    he = result.get("heldout_eval", {}) or {}
    le = result.get("local_eval", {}) or {}
    pe = result.get("pooled_val_eval", {}) or {}
    init = result.get("init_info", {}) or {}

    row: dict[str, Any] = {
        "spec_id": result.get("spec_id", ""),
        "exp_label": result.get("exp_label", ""),
        "method": result.get("method", ""),
        "n": result.get("n", ""),
        "tier": result.get("tier", ""),
        "training_clients": "|".join(result.get("training_clients", [])),
        "heldout": result.get("heldout", ""),
        "client_for_isolated": result.get("client", "") if result.get("method") == "isolated" else "",
        "seed": result.get("seed", ""),
        "model_variant": result.get("model_variant", ""),
        "train_time_seconds": result.get("train_time_seconds", ""),
        "n_params_exchanged": init.get("n_params_exchanged", ""),
        "n_params_total": init.get("n_params_total", ""),
        "shared_fraction": init.get("shared_fraction", ""),
        "heldout_map50": he.get("map50", ""),
        "heldout_map50_95": he.get("map50_95", ""),
        "heldout_map50_testable": he.get("map50_testable", ""),
        "heldout_map50_95_testable": he.get("map50_95_testable", ""),
        "heldout_n_testable_classes_evaluated": he.get("n_testable_classes_evaluated", ""),
        "local_map50": le.get("map50", ""),
        "local_map50_95": le.get("map50_95", ""),
        "pooled_val_map50": pe.get("map50", ""),
        "pooled_val_map50_95": pe.get("map50_95", ""),
    }

    # Per-class held-out mAP50 broken out by class name
    he_per_class = he.get("map50_per_class", {}) or {}
    for cid, cname in enumerate(schema_class_names):
        # JSON keys come back as strings even though they're ints
        val = he_per_class.get(cid, he_per_class.get(str(cid), ""))
        row[f"heldout_map50__{cname}"] = val

    return row


def aggregate(
    results_root: str | Path,
    output_csv: str | Path,
    schema_path: str | Path | None = None,
) -> int:
    """Walk results_root, build a CSV at output_csv. Returns row count."""
    from data_prep.schema import load_schema

    schema = load_schema(schema_path) if schema_path else load_schema()
    class_names = schema.class_names

    results_root = Path(results_root)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for json_path in sorted(results_root.glob("*.json")):
        if json_path.name == "matrix_run_report.json":
            continue  # skip the matrix-level summary
        try:
            with open(json_path) as f:
                result = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read {json_path}: {e}")
            continue
        if not isinstance(result, dict):
            continue
        if "heldout_eval" not in result:
            continue  # not a training-runner result
        rows.append(_flat_row_for(result, class_names))

    if not rows:
        logger.warning(f"No result JSONs found in {results_root}")
        with open(output_csv, "w", newline="") as f:
            f.write("")
        return 0

    # Stable column order: take the union of keys across rows, sorted with
    # known columns first
    known_first = [
        "spec_id", "exp_label", "method", "n", "tier", "training_clients",
        "heldout", "client_for_isolated", "seed", "model_variant",
        "train_time_seconds",
        "n_params_exchanged", "n_params_total", "shared_fraction",
        "heldout_map50", "heldout_map50_95",
        "heldout_map50_testable", "heldout_map50_95_testable",
        "heldout_n_testable_classes_evaluated",
        "local_map50", "local_map50_95",
        "pooled_val_map50", "pooled_val_map50_95",
    ]
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    rest = sorted(all_keys - set(known_first))
    columns = known_first + rest

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info(f"Wrote {len(rows)} rows × {len(columns)} cols to {output_csv}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--schema", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    aggregate(args.results_root, args.output_csv, args.schema)


if __name__ == "__main__":
    main()
