"""Class subset definitions for Experiment 2 tiers.

Each tier defines which unified classes are *active* during training.
Held-out evaluation always uses the 7 testable classes regardless of tier,
so the comparison directly measures how varying training class coverage
affects evaluable-class performance.

Tier definitions are derived from the schema, not hardcoded — if the schema
changes, the tiers update automatically.

Tiers:
  - tier_shared       : 7 classes (worker + 6 testable equipment)
  - tier_shared_cross : 9 classes (above + tower_crane + mobile_crane)
  - tier_full         : 17 classes (all unified)

Use:
    from pipeline.class_subsets import get_tier_yolo_ids

    yolo_ids = get_tier_yolo_ids("tier_shared", schema)
    # -> frozenset({0, 1, 2, 3, 4, 5, 6})

The returned set contains 0-indexed YOLO class IDs. Pass it to the tier
filter to drop labels for inactive classes.
"""
from __future__ import annotations

from typing import Iterable

from data_prep.schema import ClassSchema


TIER_NAMES = ("tier_shared", "tier_shared_cross", "tier_full")


def _unified_ids_in_multiple_training_sources(
    schema: ClassSchema, training_sources: Iterable[str], min_sources: int
) -> list[int]:
    """Unified IDs that appear in at least `min_sources` of the training sources."""
    result = []
    for entry in schema.entries:
        count = sum(1 for s in training_sources if entry[s])
        if count >= min_sources:
            result.append(entry["unified_id"])
    return result


def _to_yolo_id_set(schema: ClassSchema, unified_ids: Iterable[int]) -> frozenset[int]:
    return frozenset(schema.to_yolo_id(uid) for uid in unified_ids)


def get_tier_yolo_ids(tier: str, schema: ClassSchema) -> frozenset[int]:
    """Return the set of 0-indexed YOLO class IDs active in the given tier.

    Args:
        tier: One of TIER_NAMES
        schema: Loaded ClassSchema

    Returns:
        frozenset of YOLO class IDs

    Tier definitions:
        tier_shared:
            The classes testable on the CIS held-out set. These appear in at
            least one training client AND in CIS. This is the cleanest
            "shared semantics" definition because evaluation is fully aligned.

        tier_shared_cross:
            tier_shared, plus classes shared across at least 2 training
            clients but NOT in CIS held-out. With current data, this adds
            tower_crane and mobile_crane (both in MOCS and ACID).
            These act as auxiliary cross-client supervision: they help train
            the backbone but cannot be evaluated on held-out.

        tier_full:
            All unified classes, including singleton classes that appear in
            only one training client. Matches the default Experiment 1
            configuration. SODA-unique, MOCS-unique, and ACID-unique classes
            all contribute to local heads but cannot be evaluated across the
            full federation.
    """
    if tier == "tier_full":
        return _to_yolo_id_set(
            schema, [e["unified_id"] for e in schema.entries]
        )

    if tier == "tier_shared":
        # Testable on held-out CIS — present in some training source AND in CIS
        return _to_yolo_id_set(schema, schema.testable_on_heldout("CIS"))

    if tier == "tier_shared_cross":
        # Start from tier_shared
        active = set(schema.testable_on_heldout("CIS"))
        # Add classes in >=2 training sources that aren't already in CIS
        training_sources = ("MOCS", "SODA", "ACID")
        cross_client = _unified_ids_in_multiple_training_sources(
            schema, training_sources, min_sources=2
        )
        # Filter to those NOT in CIS (i.e., not already in tier_shared)
        for uid in cross_client:
            entry = next(e for e in schema.entries if e["unified_id"] == uid)
            if not entry["CIS"]:
                active.add(uid)
        return _to_yolo_id_set(schema, active)

    raise ValueError(f"Unknown tier '{tier}'. Must be one of {TIER_NAMES}")


def get_tier_class_names(tier: str, schema: ClassSchema) -> list[str]:
    """Return active class names in YOLO id order, with inactive slots as '_inactive'.

    YOLO data.yaml requires a `names:` list with one entry per class id.
    For inactive classes, we use a placeholder name so the indexing stays
    aligned with the model's output channels. The model still has all 17
    output channels — we just don't train them via filtered labels.
    """
    active_yolo_ids = get_tier_yolo_ids(tier, schema)
    full_names = schema.class_names  # in YOLO id order
    result = []
    for i, name in enumerate(full_names):
        if i in active_yolo_ids:
            result.append(name)
        else:
            result.append(f"_inactive_{name}")
    return result


def report_tier(tier: str, schema: ClassSchema) -> None:
    """Print tier composition for inspection."""
    yolo_ids = get_tier_yolo_ids(tier, schema)
    names = schema.class_names
    print(f"\nTier: {tier}")
    print(f"  Active YOLO ids: {sorted(yolo_ids)}")
    print(f"  Active count:    {len(yolo_ids)} / {len(names)}")
    print(f"  Active classes:")
    for yid in sorted(yolo_ids):
        print(f"    {yid:>3}  {names[yid]}")


if __name__ == "__main__":
    # Self-check
    from data_prep.schema import load_schema
    schema = load_schema()
    for tier in TIER_NAMES:
        report_tier(tier, schema)
