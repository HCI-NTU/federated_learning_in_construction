"""Experiment matrix driver.

Walks the full experimental matrix one spec at a time:
  1. Prepares data (via pipeline.build_pipeline) — idempotent
  2. Runs the appropriate Stage 4 training script
  3. Skips runs whose results file already exists (for resume)
  4. Captures per-run logs for debugging

Designed for long-running studies: a single Python process iterates through
all specs sequentially. Resumability is essential — if the process dies
after 30 runs, restarting picks up at run 31 by detecting existing result
files.

Usage:
    python -m experiments.run_matrix \\
        --data-root data \\
        --output-root work \\
        --results-root results \\
        --exp23-n 4000 \\
        --epochs 50 \\
        --model-variant yolo11s.yaml \\
        --imgsz 640 \\
        --batch 16 \\
        --device cuda:0

    # Filter to a subset for testing:
    python -m experiments.run_matrix \\
        --filter-method fedper \\
        --filter-n 1000 \\
        ...

    # Dry run (no actual training, just print what would run):
    python -m experiments.run_matrix --dry-run ...
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

from experiments.matrix import (
    generate_full_matrix,
    summarize_matrix,
    build_spec_id,
    DEFAULT_N_SWEEP,
    DEFAULT_EXP2_3_N,
    DEFAULT_TRAINING_CLIENTS,
    DEFAULT_HELDOUT,
    DEFAULT_SEEDS,
)


logger = logging.getLogger(__name__)


def _client_yaml_path(work_root: Path, client: str, n: int, tier: str, seed: int) -> Path:
    """Match the naming convention used by pipeline.build_pipeline."""
    return work_root / "configs" / f"{client}_n{n}_{tier}_s{seed}.yaml"


def _heldout_yaml_path(work_root: Path, heldout: str, tier: str, seed: int) -> Path:
    """Held-out client uses the same naming; the orchestrator names heldout
    yamls with n=<requested-n> by convention but only the test split matters."""
    # The pipeline writes heldout configs with the same n as the run,
    # but the held-out test split content doesn't depend on n. We pick
    # an arbitrary n value that's been prepared for any run, or fall back
    # to scanning the configs dir.
    configs_dir = work_root / "configs"
    pattern = f"{heldout}_n*_{tier}_s{seed}.yaml"
    matches = sorted(configs_dir.glob(pattern))
    if matches:
        return matches[0]
    # No held-out yaml exists yet — return a default that build_pipeline will create
    return configs_dir / f"{heldout}_n0_{tier}_s{seed}.yaml"


def results_path_for_spec(results_root: Path, spec: dict) -> Path:
    return results_root / f"{spec['spec_id']}.json"


def prepare_data_for_spec(
    spec: dict,
    data_root: Path,
    work_root: Path,
    schema_path: Path | None = None,
    acid_raw_images_dir: Path | None = None,
) -> None:
    """Run the pipeline orchestrator for one spec's data needs.

    Idempotent: the pipeline itself skips already-converted/subsampled data.
    """
    from pipeline.build_pipeline import (
        ensure_acid_split, build_for_client,
    )
    from data_prep.schema import load_schema

    schema = load_schema(schema_path) if schema_path else load_schema()

    clients_needed = list(spec["training_clients"])
    if "ACID" in clients_needed or spec["heldout"] == "ACID":
        ensure_acid_split(
            data_root, seed=spec["seed"],
            acid_raw_images_dir=acid_raw_images_dir,
            arrange_mode="move",
        )

    for client in clients_needed:
        build_for_client(
            client=client,
            data_root=data_root,
            work_root=work_root,
            n=spec["n"],
            tier=spec["tier"],
            seed=spec["seed"],
            schema=schema,
        )

    # Held-out — always prepare (CIS test split)
    build_for_client(
        client=spec["heldout"],
        data_root=data_root,
        work_root=work_root,
        n=spec["n"],
        tier=spec["tier"],
        seed=spec["seed"],
        schema=schema,
    )


def run_one_spec(
    spec: dict,
    data_root: Path,
    work_root: Path,
    results_root: Path,
    epochs: int,
    model_variant: str,
    imgsz: int,
    batch: int,
    device: str | None,
    proximal_mu: float,
    schema_path: Path | None = None,
    acid_raw_images_dir: Path | None = None,
) -> dict:
    """Run one experiment spec end-to-end. Returns the result dict."""
    # 1. Prepare data
    prepare_data_for_spec(
        spec, data_root, work_root, schema_path, acid_raw_images_dir,
    )

    # 2. Locate the relevant data.yaml files
    heldout_yaml = _heldout_yaml_path(work_root, spec["heldout"], spec["tier"], spec["seed"])

    client_yamls = [
        _client_yaml_path(work_root, c, spec["n"], spec["tier"], spec["seed"])
        for c in spec["training_clients"]
    ]
    for cy in client_yamls:
        if not cy.exists():
            raise FileNotFoundError(
                f"Expected client data.yaml not found: {cy}. "
                f"The pipeline should have produced it."
            )

    results_path = results_path_for_spec(results_root, spec)

    # 3. Dispatch to the right Stage 4 runner
    work_dir_for_run = work_root / "training_runs" / spec["spec_id"]

    if spec["method"] == "isolated":
        from train.train_isolated import train_isolated
        client = spec["client_for_isolated"]
        client_yaml = _client_yaml_path(
            work_root, client, spec["n"], spec["tier"], spec["seed"]
        )
        result = train_isolated(
            client=client,
            data_yaml=client_yaml,
            heldout_data_yaml=heldout_yaml,
            model_variant=model_variant,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            work_dir=work_dir_for_run,
            seed=spec["seed"],
            results_path=results_path,
        )

    elif spec["method"] == "centralized":
        from train.train_centralized import train_centralized
        result = train_centralized(
            client_yamls=client_yamls,
            heldout_data_yaml=heldout_yaml,
            model_variant=model_variant,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            work_dir=work_dir_for_run,
            seed=spec["seed"],
            results_path=results_path,
        )

    elif spec["method"] in ("fedavg", "fedprox", "fedper"):
        from train.train_federated import train_federated
        result = train_federated(
            method=spec["method"],
            client_yamls=client_yamls,
            client_names=list(spec["training_clients"]),
            heldout_data_yaml=heldout_yaml,
            num_rounds=epochs,        # FL convention: 1 epoch per round
            epochs_per_round=1,
            model_variant=model_variant,
            imgsz=imgsz,
            batch=batch,
            device=device,
            proximal_mu=proximal_mu,
            work_dir=work_dir_for_run,
            seed=spec["seed"],
            results_path=results_path,
        )

    else:
        raise ValueError(f"Unknown method: {spec['method']}")

    # Tag the result with the spec id so post-hoc analysis can recover it
    result["spec_id"] = spec["spec_id"]
    result["exp_label"] = spec["exp_label"]
    result["n"] = spec["n"]
    result["tier"] = spec["tier"]
    result["training_clients"] = spec["training_clients"]
    result["heldout"] = spec["heldout"]
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_matrix(
    specs: list[dict],
    data_root: Path,
    work_root: Path,
    results_root: Path,
    log_root: Path,
    epochs: int = 50,
    model_variant: str = "yolo11s.yaml",
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    proximal_mu: float = 0.01,
    schema_path: Path | None = None,
    acid_raw_images_dir: Path | None = None,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict:
    """Run all specs sequentially. Returns a per-spec status report."""
    results_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    status_report: dict[str, dict] = {}
    t_all = time.time()

    for i, spec in enumerate(specs):
        sid = spec["spec_id"]
        results_path = results_path_for_spec(results_root, spec)

        if skip_existing and results_path.exists():
            logger.info(f"[{i+1}/{len(specs)}] SKIP (exists): {sid}")
            status_report[sid] = {"status": "skipped_existing", "result_path": str(results_path)}
            continue

        if dry_run:
            logger.info(f"[{i+1}/{len(specs)}] DRY RUN: {sid}")
            status_report[sid] = {"status": "dry_run"}
            continue

        logger.info(f"[{i+1}/{len(specs)}] START: {sid}")
        t0 = time.time()
        log_path = log_root / f"{sid}.log"

        # Per-run log handler so each spec's output goes to its own file
        per_run_handler = logging.FileHandler(log_path)
        per_run_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
        root_logger = logging.getLogger()
        root_logger.addHandler(per_run_handler)

        try:
            run_one_spec(
                spec=spec,
                data_root=data_root,
                work_root=work_root,
                results_root=results_root,
                epochs=epochs,
                model_variant=model_variant,
                imgsz=imgsz,
                batch=batch,
                device=device,
                proximal_mu=proximal_mu,
                schema_path=schema_path,
                acid_raw_images_dir=acid_raw_images_dir,
            )
            elapsed = time.time() - t0
            logger.info(f"[{i+1}/{len(specs)}] DONE ({elapsed:.0f}s): {sid}")
            status_report[sid] = {
                "status": "ok",
                "elapsed_seconds": elapsed,
                "result_path": str(results_path),
                "log_path": str(log_path),
            }
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"[{i+1}/{len(specs)}] FAILED ({elapsed:.0f}s): {sid} — {e}")
            logger.error(traceback.format_exc())
            status_report[sid] = {
                "status": "failed",
                "elapsed_seconds": elapsed,
                "error": str(e),
                "log_path": str(log_path),
            }
            # Continue with the rest of the matrix — one failure shouldn't
            # block the whole study. Errors are captured for review.
        finally:
            root_logger.removeHandler(per_run_handler)
            per_run_handler.close()

    total_elapsed = time.time() - t_all
    summary = {
        "n_total": len(specs),
        "n_ok": sum(1 for s in status_report.values() if s["status"] == "ok"),
        "n_skipped": sum(1 for s in status_report.values() if s["status"] == "skipped_existing"),
        "n_failed": sum(1 for s in status_report.values() if s["status"] == "failed"),
        "n_dry_run": sum(1 for s in status_report.values() if s["status"] == "dry_run"),
        "total_elapsed_seconds": total_elapsed,
    }
    return {"summary": summary, "per_spec": status_report}


def _filter_specs(
    specs: list[dict],
    methods: list[str] | None,
    ns: list[int] | None,
    tiers: list[str] | None,
    exp_labels: list[str] | None,
) -> list[dict]:
    out = specs
    if methods:
        out = [s for s in out if s["method"] in methods]
    if ns:
        out = [s for s in out if s["n"] in ns]
    if tiers:
        out = [s for s in out if s["tier"] in tiers]
    if exp_labels:
        out = [s for s in out if s["exp_label"] in exp_labels]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True,
                        help="Working directory for pipeline outputs (labels, manifests, configs)")
    parser.add_argument("--results-root", required=True,
                        help="Where per-spec JSON results are written")
    parser.add_argument("--log-root", default=None,
                        help="Where per-spec stdout/stderr logs are written. "
                             "Default: <results-root>/logs")
    parser.add_argument("--exp23-n", type=int, default=DEFAULT_EXP2_3_N,
                        help="Single N value for Experiments 2 and 3")
    parser.add_argument("--n-sweep", type=int, nargs="+", default=list(DEFAULT_N_SWEEP),
                        help="N values for Experiment 1 sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (= num_rounds for federated, epochs_per_round=1)")
    parser.add_argument("--model-variant", default="yolo11s.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--proximal-mu", type=float, default=0.01)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--acid-raw-images-dir", default=None)

    parser.add_argument("--filter-method", nargs="+", default=None,
                        help="Run only specs whose method is in this list")
    parser.add_argument("--filter-n", type=int, nargs="+", default=None)
    parser.add_argument("--filter-tier", nargs="+", default=None)
    parser.add_argument("--filter-exp", nargs="+", default=None,
                        help="e.g., exp1 exp2 exp3_drop_MOCS")

    parser.add_argument("--skip-exp2", action="store_true")
    parser.add_argument("--skip-exp3", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Re-run specs even if their results JSON exists")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    data_root = Path(args.data_root).resolve()
    work_root = Path(args.output_root).resolve()
    results_root = Path(args.results_root).resolve()
    log_root = Path(args.log_root) if args.log_root else results_root / "logs"

    # Generate matrix
    specs = generate_full_matrix(
        n_sweep=tuple(args.n_sweep),
        exp23_n=args.exp23_n,
        seeds=tuple(args.seeds),
        include_exp2=not args.skip_exp2,
        include_exp3=not args.skip_exp3,
    )

    # Apply filters
    specs = _filter_specs(
        specs,
        methods=args.filter_method,
        ns=args.filter_n,
        tiers=args.filter_tier,
        exp_labels=args.filter_exp,
    )

    logger.info(f"Matrix: {len(specs)} specs after filtering")
    logger.info(f"Summary: {summarize_matrix(specs)}")

    report = run_matrix(
        specs=specs,
        data_root=data_root,
        work_root=work_root,
        results_root=results_root,
        log_root=log_root,
        epochs=args.epochs,
        model_variant=args.model_variant,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        proximal_mu=args.proximal_mu,
        schema_path=Path(args.schema) if args.schema else None,
        acid_raw_images_dir=Path(args.acid_raw_images_dir) if args.acid_raw_images_dir else None,
        dry_run=args.dry_run,
        skip_existing=not args.no_skip_existing,
    )

    # Write a master report
    report_path = results_root / "matrix_run_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Matrix run report: {report_path}")
    logger.info(f"Final summary: {report['summary']}")


if __name__ == "__main__":
    main()
