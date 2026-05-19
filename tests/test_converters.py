"""End-to-end verification test for all four data converters.

This is the mapping verification — it doesn't just check 'does it run,' it
checks 'did each source id/name land in the EXPECTED YOLO id.' This catches
schema-lookup bugs that an unverified smoke test would miss.

How it works:
  1. Generate synthetic data for each dataset
  2. Run the appropriate converter
  3. For each source category, look up its expected YOLO id from the schema
  4. Read the converted YOLO label files and count instances per YOLO id
  5. Compare actual vs. expected counts — they must match exactly
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# Make data_prep importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_prep.schema import load_schema
from data_prep.convert_coco import convert_coco_to_yolo
from data_prep.convert_voc import convert_voc_to_yolo
from tests.make_synthetic_mocs import generate as gen_mocs
from tests.make_synthetic_cis import generate as gen_cis
from tests.make_synthetic_acid import generate as gen_acid
from tests.make_synthetic_soda import generate as gen_soda


def count_yolo_labels(labels_dir: Path) -> dict[int, int]:
    """Count instances per YOLO class id across all .txt files in a directory."""
    counts: dict[int, int] = defaultdict(int)
    for txt_file in labels_dir.glob("*.txt"):
        for line in txt_file.read_text().strip().split("\n"):
            if not line:
                continue
            class_id = int(line.split()[0])
            counts[class_id] += 1
    return dict(counts)


def expected_yolo_counts_from_coco(
    coco_path: Path, source: str, schema
) -> dict[int, int]:
    """Compute the expected per-YOLO-id count by replaying the schema mapping
    against the source COCO JSON, independently of the converter."""
    with open(coco_path) as f:
        coco = json.load(f)

    lookup = {
        "MOCS": schema.lookup_mocs,
        "CIS": schema.lookup_cis,
        "ACID": schema.lookup_acid,
    }[source]

    expected: dict[int, int] = defaultdict(int)
    for ann in coco["annotations"]:
        unified_id = lookup(ann["category_id"])
        if unified_id is None:
            continue
        # Skip invalid bboxes (the converter does too)
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        yolo_id = schema.to_yolo_id(unified_id)
        expected[yolo_id] += 1
    return dict(expected)


def expected_yolo_counts_from_voc(xml_dir: Path, schema) -> dict[int, int]:
    """Independent re-derivation of expected counts for SODA VOC XMLs."""
    import xml.etree.ElementTree as ET
    expected: dict[int, int] = defaultdict(int)
    for xml_path in xml_dir.glob("*.xml"):
        tree = ET.parse(xml_path)
        for obj in tree.getroot().findall("object"):
            name_elem = obj.find("name")
            bbox = obj.find("bndbox")
            if name_elem is None or bbox is None:
                continue
            name = name_elem.text.strip() if name_elem.text else ""
            unified_id = schema.lookup_soda(name)
            if unified_id is None:
                continue
            try:
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
            except (AttributeError, ValueError):
                continue
            if xmax - xmin <= 0 or ymax - ymin <= 0:
                continue
            yolo_id = schema.to_yolo_id(unified_id)
            expected[yolo_id] += 1
    return dict(expected)


def assert_match(actual: dict[int, int], expected: dict[int, int], label: str):
    if actual != expected:
        print(f"\nFAIL: {label}")
        print(f"  Expected: {dict(sorted(expected.items()))}")
        print(f"  Actual:   {dict(sorted(actual.items()))}")
        missing = {k: v for k, v in expected.items() if actual.get(k, 0) != v}
        print(f"  Mismatched yolo_ids: {missing}")
        raise AssertionError(f"{label}: counts do not match")
    print(f"  PASS: {label} — {sum(actual.values())} instances across {len(actual)} classes")


def run_test_for_coco_source(source: str, tmp_root: Path, schema) -> None:
    print(f"\n--- Testing {source} (COCO JSON) ---")
    tmp = tmp_root / source.lower()
    tmp.mkdir(parents=True, exist_ok=True)

    coco_path = tmp / f"instances_{source.lower()}.json"
    gen_func = {"MOCS": gen_mocs, "CIS": gen_cis, "ACID": gen_acid}[source]
    gen_func(coco_path)

    labels_dir = tmp / "labels"
    convert_coco_to_yolo(
        json_path=coco_path,
        labels_dir=labels_dir,
        source=source,
        schema=schema,
    )

    actual = count_yolo_labels(labels_dir)
    expected = expected_yolo_counts_from_coco(coco_path, source, schema)
    assert_match(actual, expected, f"{source} per-class instance counts")

    # Extra: verify that the YOLO ids in output are exactly the set of ids
    # that the schema can produce for this source.
    expected_yolo_id_set = {
        schema.to_yolo_id(uid)
        for uid in schema.classes_in_source(source)
    }
    # We only expect to *see* a yolo_id in output if it was sampled in the
    # synthetic data, which it always is given our weights. So:
    assert set(actual.keys()).issubset(expected_yolo_id_set), (
        f"{source} produced YOLO ids outside the schema's source coverage: "
        f"{set(actual.keys()) - expected_yolo_id_set}"
    )
    print(f"  PASS: {source} produces only YOLO ids within its source coverage")


def run_test_for_soda(tmp_root: Path, schema) -> None:
    print(f"\n--- Testing SODA (VOC XML) ---")
    tmp = tmp_root / "soda"
    tmp.mkdir(parents=True, exist_ok=True)

    xml_dir = tmp / "annotations"
    gen_soda(xml_dir)

    labels_dir = tmp / "labels"
    convert_voc_to_yolo(xml_dir=xml_dir, labels_dir=labels_dir, schema=schema)

    actual = count_yolo_labels(labels_dir)
    expected = expected_yolo_counts_from_voc(xml_dir, schema)
    assert_match(actual, expected, "SODA per-class instance counts")

    # SODA only has 'person' -> worker (yolo_id 0). Verify nothing else appears.
    assert set(actual.keys()) == {0}, (
        f"SODA produced unexpected YOLO ids: {set(actual.keys())} (expected {{0}})"
    )
    print("  PASS: SODA produces only YOLO id 0 (worker)")


def main() -> None:
    schema = load_schema()
    print(f"Schema loaded: {schema.num_classes} unified classes")

    tmp_root = Path("/tmp/converter_verification")
    if tmp_root.exists():
        import shutil
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)

    run_test_for_coco_source("MOCS", tmp_root, schema)
    run_test_for_coco_source("CIS", tmp_root, schema)
    run_test_for_coco_source("ACID", tmp_root, schema)
    run_test_for_soda(tmp_root, schema)

    print("\n" + "=" * 60)
    print("ALL CONVERTER MAPPING TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
