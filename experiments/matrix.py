"""Experimental matrix definitions.

Generates the full list of run specs covering the three experiments:

  Experiment 1: Learning paradigm comparison
    - Methods: isolated, centralized, fedavg, fedprox, fedper
    - N: {1000, 2000, 4000, 8000}
    - Tier: tier_full
    - Clients: all 3 (MOCS, SODA, ACID)
    - ≈ 28 runs (isolated counts 3× per N, others 1× per N)

  Experiment 2: Semantic overlap ablation
    - Methods: same 5
    - N: single representative value (configurable, default 4000)
    - Tier: tier_shared, tier_shared_cross (Full reuses Exp 1)
    - Clients: all 3
    - 10 runs (2 tiers × 5 methods)

  Experiment 3: Client composition ablation
    - Methods: fedavg, fedprox, fedper (FL only)
    - N: same as Experiment 2
    - Tier: tier_full
    - Clients: drop-MOCS / drop-SODA / drop-ACID (and 1 isolated baseline
              per dropped configuration to compare against)
    - 9 runs (3 FL × 3 drops) + 6 isolated runs for the 2-client baselines

  Total: ≈ 53 runs (a few more than the original 47 estimate because
  Experiment 3 now includes isolated baselines for each drop config so
  the comparison is apples-to-apples).

Each spec is a dict that uniquely identifies one training run and can be
passed to the appropriate Stage 4 runner.
"""
from __future__ import annotations

from typing import Any


# Defaults for the experiment matrix. Override via run_matrix.py CLI.
DEFAULT_TRAINING_CLIENTS = ("MOCS", "SODA", "ACID")
DEFAULT_HELDOUT = "CIS"
DEFAULT_SEEDS = (42,)
DEFAULT_N_SWEEP = (1000, 2000, 4000, 8000)
DEFAULT_EXP2_3_N = 4000  # placeholder; pick based on Experiment 1 results
DEFAULT_METHODS_FULL = ("isolated", "centralized", "fedavg", "fedprox", "fedper")
DEFAULT_METHODS_FL = ("fedavg", "fedprox", "fedper")


def _spec(
    *,
    exp_label: str,
    method: str,
    n: int,
    tier: str,
    training_clients: tuple[str, ...],
    heldout: str,
    seed: int,
    client_for_isolated: str | None = None,
) -> dict[str, Any]:
    """Build one run spec dict. Used by all the experiment generators below."""
    s: dict[str, Any] = {
        "exp_label": exp_label,
        "method": method,
        "n": n,
        "tier": tier,
        "training_clients": list(training_clients),
        "heldout": heldout,
        "seed": seed,
    }
    if method == "isolated":
        if client_for_isolated is None:
            raise ValueError("client_for_isolated required for method='isolated'")
        s["client_for_isolated"] = client_for_isolated
    s["spec_id"] = build_spec_id(s)
    return s


def build_spec_id(spec: dict) -> str:
    """A short stable id for the spec, used for filenames."""
    parts = [
        spec["exp_label"],
        spec["method"],
        f"n{spec['n']}",
        spec["tier"],
        "-".join(spec["training_clients"]),
        f"s{spec['seed']}",
    ]
    if spec.get("client_for_isolated"):
        parts.append(f"client_{spec['client_for_isolated']}")
    return "__".join(parts)


def generate_experiment_1(
    n_sweep: tuple[int, ...] = DEFAULT_N_SWEEP,
    training_clients: tuple[str, ...] = DEFAULT_TRAINING_CLIENTS,
    heldout: str = DEFAULT_HELDOUT,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    methods: tuple[str, ...] = DEFAULT_METHODS_FULL,
) -> list[dict]:
    """Experiment 1: methods × N sweep, full class space, all clients."""
    specs = []
    for seed in seeds:
        for n in n_sweep:
            for method in methods:
                if method == "isolated":
                    # One isolated run per client
                    for client in training_clients:
                        specs.append(_spec(
                            exp_label="exp1",
                            method=method,
                            n=n,
                            tier="tier_full",
                            training_clients=training_clients,
                            heldout=heldout,
                            seed=seed,
                            client_for_isolated=client,
                        ))
                else:
                    specs.append(_spec(
                        exp_label="exp1",
                        method=method,
                        n=n,
                        tier="tier_full",
                        training_clients=training_clients,
                        heldout=heldout,
                        seed=seed,
                    ))
    return specs


def generate_experiment_2(
    n: int = DEFAULT_EXP2_3_N,
    training_clients: tuple[str, ...] = DEFAULT_TRAINING_CLIENTS,
    heldout: str = DEFAULT_HELDOUT,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    methods: tuple[str, ...] = DEFAULT_METHODS_FULL,
) -> list[dict]:
    """Experiment 2: tier_shared and tier_shared_cross at a single N.

    Tier_full results are reused from Experiment 1 at the same N — not
    re-run here.
    """
    specs = []
    for seed in seeds:
        for tier in ("tier_shared", "tier_shared_cross"):
            for method in methods:
                if method == "isolated":
                    for client in training_clients:
                        specs.append(_spec(
                            exp_label="exp2",
                            method=method,
                            n=n,
                            tier=tier,
                            training_clients=training_clients,
                            heldout=heldout,
                            seed=seed,
                            client_for_isolated=client,
                        ))
                else:
                    specs.append(_spec(
                        exp_label="exp2",
                        method=method,
                        n=n,
                        tier=tier,
                        training_clients=training_clients,
                        heldout=heldout,
                        seed=seed,
                    ))
    return specs


def generate_experiment_3(
    n: int = DEFAULT_EXP2_3_N,
    full_training_clients: tuple[str, ...] = DEFAULT_TRAINING_CLIENTS,
    heldout: str = DEFAULT_HELDOUT,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    fl_methods: tuple[str, ...] = DEFAULT_METHODS_FL,
    include_isolated_baselines: bool = True,
) -> list[dict]:
    """Experiment 3: drop one client at a time. FL methods only by default.

    For each "drop" configuration, the all-three baseline is reused from
    Experiment 1 at the same N.

    If include_isolated_baselines=True, also generates per-client isolated
    runs for the 2-client configurations so that "FL with these 2 clients"
    can be compared against "isolated from one of these 2 clients."
    """
    specs = []
    full_set = set(full_training_clients)
    for seed in seeds:
        for drop in full_training_clients:
            remaining = tuple(c for c in full_training_clients if c != drop)
            # FL methods on the 2-client subset
            for method in fl_methods:
                specs.append(_spec(
                    exp_label=f"exp3_drop_{drop}",
                    method=method,
                    n=n,
                    tier="tier_full",
                    training_clients=remaining,
                    heldout=heldout,
                    seed=seed,
                ))
            # Optional isolated baselines for the 2-client subset
            if include_isolated_baselines:
                for client in remaining:
                    specs.append(_spec(
                        exp_label=f"exp3_drop_{drop}",
                        method="isolated",
                        n=n,
                        tier="tier_full",
                        training_clients=remaining,
                        heldout=heldout,
                        seed=seed,
                        client_for_isolated=client,
                    ))
    return specs


def generate_full_matrix(
    n_sweep: tuple[int, ...] = DEFAULT_N_SWEEP,
    exp23_n: int = DEFAULT_EXP2_3_N,
    training_clients: tuple[str, ...] = DEFAULT_TRAINING_CLIENTS,
    heldout: str = DEFAULT_HELDOUT,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    include_exp2: bool = True,
    include_exp3: bool = True,
    include_exp3_isolated_baselines: bool = True,
) -> list[dict]:
    """Return the full deduplicated list of specs across all three experiments."""
    specs: list[dict] = []
    specs.extend(generate_experiment_1(
        n_sweep=n_sweep, training_clients=training_clients,
        heldout=heldout, seeds=seeds,
    ))
    if include_exp2:
        specs.extend(generate_experiment_2(
            n=exp23_n, training_clients=training_clients,
            heldout=heldout, seeds=seeds,
        ))
    if include_exp3:
        specs.extend(generate_experiment_3(
            n=exp23_n, full_training_clients=training_clients,
            heldout=heldout, seeds=seeds,
            include_isolated_baselines=include_exp3_isolated_baselines,
        ))
    # Dedupe by spec_id (shouldn't be any duplicates if logic is correct,
    # but a safety net)
    seen = set()
    deduped = []
    for s in specs:
        if s["spec_id"] in seen:
            continue
        seen.add(s["spec_id"])
        deduped.append(s)
    return deduped


def summarize_matrix(specs: list[dict]) -> dict:
    """Counts breakdown for inspection."""
    by_exp: dict[str, int] = {}
    by_method: dict[str, int] = {}
    by_n: dict[int, int] = {}
    by_tier: dict[str, int] = {}
    for s in specs:
        by_exp[s["exp_label"]] = by_exp.get(s["exp_label"], 0) + 1
        by_method[s["method"]] = by_method.get(s["method"], 0) + 1
        by_n[s["n"]] = by_n.get(s["n"], 0) + 1
        by_tier[s["tier"]] = by_tier.get(s["tier"], 0) + 1
    return {
        "total": len(specs),
        "by_exp_label": by_exp,
        "by_method": by_method,
        "by_n": by_n,
        "by_tier": by_tier,
    }


def main() -> None:
    """Print the matrix summary, for inspection."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspect the experimental matrix")
    parser.add_argument("--exp23-n", type=int, default=DEFAULT_EXP2_3_N)
    parser.add_argument("--show-specs", action="store_true",
                        help="Print every spec id, not just the summary")
    args = parser.parse_args()

    specs = generate_full_matrix(exp23_n=args.exp23_n)
    summary = summarize_matrix(specs)
    print(json.dumps(summary, indent=2))
    if args.show_specs:
        print()
        for s in specs:
            print(s["spec_id"])


if __name__ == "__main__":
    main()
