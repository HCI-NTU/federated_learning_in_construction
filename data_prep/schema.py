"""Schema loader for unified class mapping.

Loads the unified class schema and provides lookup utilities for converting
source dataset class IDs/names to unified YOLO class IDs.

The unified schema is the single source of truth for which source classes
map to which unified class. All converters key off this file.

Unified YOLO class IDs are 0-indexed (unified_id - 1) for YOLO format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class ClassSchema:
    """Wraps the unified class schema with per-source lookup tables."""

    def __init__(self, schema_path: str | Path):
        with open(schema_path) as f:
            self.entries = json.load(f)

        # Build per-source lookup: source_class_id -> unified_id (1-indexed)
        # COCO datasets (MOCS, ACID): single source class per unified entry
        # CIS: list of source classes per unified entry (e.g., worker = helmet + no-helmet)
        self._mocs_id_to_unified: dict[int, int] = {}
        self._acid_id_to_unified: dict[int, int] = {}
        self._cis_id_to_unified: dict[int, int] = {}

        # SODA uses VOC XML <name> string, not numeric ID
        self._soda_name_to_unified: dict[str, int] = {}

        for entry in self.entries:
            uid = entry["unified_id"]

            if entry["MOCS"]:
                self._mocs_id_to_unified[entry["MOCS"]["id"]] = uid
            if entry["ACID"]:
                self._acid_id_to_unified[entry["ACID"]["id"]] = uid
            if entry["CIS"]:
                for src in entry["CIS"]:
                    self._cis_id_to_unified[src["id"]] = uid
            if entry["SODA"]:
                self._soda_name_to_unified[entry["SODA"]["name"]] = uid

        # Build unified_id -> unified_name lookup
        self._unified_id_to_name: dict[int, str] = {
            e["unified_id"]: e["unified_name"] for e in self.entries
        }

    @property
    def num_classes(self) -> int:
        """Total number of unified classes (= YOLO model output channels)."""
        return len(self.entries)

    @property
    def class_names(self) -> list[str]:
        """Unified class names in YOLO order (yolo_id = unified_id - 1)."""
        # Sort by unified_id to guarantee order
        return [
            self._unified_id_to_name[uid]
            for uid in sorted(self._unified_id_to_name.keys())
        ]

    def to_yolo_id(self, unified_id: int) -> int:
        """Convert 1-indexed unified_id to 0-indexed YOLO class id."""
        return unified_id - 1

    def lookup_mocs(self, source_id: int) -> Optional[int]:
        """Map MOCS source category_id -> unified_id (1-indexed). None if not in schema."""
        return self._mocs_id_to_unified.get(source_id)

    def lookup_acid(self, source_id: int) -> Optional[int]:
        return self._acid_id_to_unified.get(source_id)

    def lookup_cis(self, source_id: int) -> Optional[int]:
        return self._cis_id_to_unified.get(source_id)

    def lookup_soda(self, source_name: str) -> Optional[int]:
        """Map SODA VOC <name> string -> unified_id."""
        return self._soda_name_to_unified.get(source_name)

    def classes_in_source(self, source: str) -> list[int]:
        """Return unified_ids that have entries for the given source.

        source: one of 'MOCS', 'SODA', 'ACID', 'CIS'
        """
        return [e["unified_id"] for e in self.entries if e[source]]

    def testable_on_heldout(self, heldout_source: str = "CIS") -> list[int]:
        """Unified IDs that are both in some training source AND in the held-out source.

        These are the classes for which cross-project generalization can be measured.
        """
        training_sources = ["MOCS", "SODA", "ACID"]
        result = []
        for e in self.entries:
            in_training = any(e[s] for s in training_sources)
            in_heldout = e[heldout_source] is not None
            if in_training and in_heldout:
                result.append(e["unified_id"])
        return result


def load_schema(path: str | Path = None) -> ClassSchema:
    """Convenience loader. Defaults to project's configs/class_schema.json."""
    if path is None:
        # Default: configs/class_schema.json at repo root
        path = Path(__file__).parent.parent / "configs" / "class_schema.json"
    return ClassSchema(path)


if __name__ == "__main__":
    # Quick self-check when run as a script
    schema = load_schema()
    print(f"Loaded schema with {schema.num_classes} unified classes")
    print(f"Class names (YOLO order): {schema.class_names}")
    print()
    print(f"MOCS contributes: {len(schema.classes_in_source('MOCS'))} classes")
    print(f"SODA contributes: {len(schema.classes_in_source('SODA'))} classes")
    print(f"ACID contributes: {len(schema.classes_in_source('ACID'))} classes")
    print(f"CIS contains:     {len(schema.classes_in_source('CIS'))} classes")
    print()
    testable = schema.testable_on_heldout("CIS")
    print(f"Testable on CIS held-out: {len(testable)} classes")
    print(f"  IDs: {testable}")
    print(f"  Names: {[schema._unified_id_to_name[u] for u in testable]}")
