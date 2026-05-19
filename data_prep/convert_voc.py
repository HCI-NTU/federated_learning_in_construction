"""Pascal VOC XML to YOLO format converter (SODA).

Reads per-image VOC XML annotation files and writes YOLO-format label files.
Used exclusively for SODA, which provides one XML per image.

VOC XML format expected (standard):
    <annotation>
        <filename>image001.jpg</filename>
        <size>
            <width>1920</width>
            <height>1080</height>
        </size>
        <object>
            <name>person</name>
            <bndbox>
                <xmin>100</xmin>
                <ymin>200</ymin>
                <xmax>300</xmax>
                <ymax>500</ymax>
            </bndbox>
        </object>
        ...
    </annotation>

The <name> string is the lookup key (e.g., 'person' for SODA). Mapping is
defined in the unified schema's SODA entries.

Usage:
    python -m data_prep.convert_voc \\
        --input-dir data/SODA/annotations \\
        --labels-dir data/SODA/labels/train \\
        --image-list-filter <optional list of image stems to include>
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .schema import load_schema, ClassSchema


def _text(elem: Optional[ET.Element], default: str = "") -> str:
    return elem.text.strip() if elem is not None and elem.text else default


def _parse_voc_xml(xml_path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    """Parse a single VOC XML file.

    Returns:
        (image_width, image_height, list of (name, xmin, ymin, xmax, ymax))
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path}: missing <size> element")
    width = int(_text(size.find("width"), "0"))
    height = int(_text(size.find("height"), "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"{xml_path}: invalid image dimensions {width}x{height}")

    objects = []
    for obj in root.findall("object"):
        name = _text(obj.find("name"))
        bndbox = obj.find("bndbox")
        if not name or bndbox is None:
            continue
        try:
            # VOC bboxes may be floats in some variants
            xmin = float(_text(bndbox.find("xmin"), "0"))
            ymin = float(_text(bndbox.find("ymin"), "0"))
            xmax = float(_text(bndbox.find("xmax"), "0"))
            ymax = float(_text(bndbox.find("ymax"), "0"))
        except ValueError:
            continue
        objects.append((name, xmin, ymin, xmax, ymax))

    return width, height, objects


def convert_voc_to_yolo(
    xml_dir: str | Path,
    labels_dir: str | Path,
    schema: Optional[ClassSchema] = None,
    image_stem_filter: Optional[set[str]] = None,
) -> dict:
    """Convert a directory of VOC XML files to per-image YOLO label files.

    Args:
        xml_dir: Directory containing .xml files (one per image)
        labels_dir: Output directory for .txt label files
        schema: Optional pre-loaded ClassSchema
        image_stem_filter: If provided, only process XMLs whose stem is in this set.
            Useful for filtering SODA to only the images that exist in train/ or test/.

    Returns:
        Stats dict.
    """
    xml_dir = Path(xml_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if schema is None:
        schema = load_schema()

    xml_files = sorted(xml_dir.glob("*.xml"))
    if image_stem_filter is not None:
        xml_files = [p for p in xml_files if p.stem in image_stem_filter]

    n_anns_written = 0
    n_anns_skipped_unmapped = 0
    n_anns_skipped_invalid_bbox = 0
    n_xml_errors = 0
    per_class_counts: dict[int, int] = defaultdict(int)
    unmapped_names: set[str] = set()

    for xml_path in xml_files:
        try:
            width, height, objects = _parse_voc_xml(xml_path)
        except Exception as e:
            n_xml_errors += 1
            print(f"Warning: failed to parse {xml_path.name}: {e}")
            continue

        label_path = labels_dir / f"{xml_path.stem}.txt"
        lines: list[str] = []
        for name, xmin, ymin, xmax, ymax in objects:
            unified_id = schema.lookup_soda(name)
            if unified_id is None:
                n_anns_skipped_unmapped += 1
                unmapped_names.add(name)
                continue

            bw = xmax - xmin
            bh = ymax - ymin
            if bw <= 0 or bh <= 0:
                n_anns_skipped_invalid_bbox += 1
                continue

            cx = (xmin + bw / 2) / width
            cy = (ymin + bh / 2) / height
            nw = bw / width
            nh = bh / height

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            yolo_id = schema.to_yolo_id(unified_id)
            lines.append(f"{yolo_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            per_class_counts[yolo_id] += 1
            n_anns_written += 1

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

    stats = {
        "source": "SODA",
        "input_dir": str(xml_dir),
        "n_xml_files": len(xml_files),
        "n_xml_errors": n_xml_errors,
        "n_anns_written": n_anns_written,
        "n_anns_skipped_unmapped": n_anns_skipped_unmapped,
        "n_anns_skipped_invalid_bbox": n_anns_skipped_invalid_bbox,
        "unmapped_names": sorted(unmapped_names),
        "per_class_counts": dict(per_class_counts),
    }
    return stats


def _print_stats(stats: dict, schema: ClassSchema) -> None:
    print(f"\nConverted SODA: {stats['input_dir']}")
    print(f"  XML files processed: {stats['n_xml_files']}")
    print(f"  XML parse errors:    {stats['n_xml_errors']}")
    print(f"  Annotations written: {stats['n_anns_written']}")
    print(f"  Skipped (unmapped):  {stats['n_anns_skipped_unmapped']}")
    if stats["unmapped_names"]:
        print(f"    Unmapped <name> values: {stats['unmapped_names']}")
        print(f"    (These names are not in the unified schema — verify if intentional)")
    print(f"  Skipped (bad bbox):  {stats['n_anns_skipped_invalid_bbox']}")
    print(f"\n  Per-class instance counts:")
    yolo_id_to_name = {i: name for i, name in enumerate(schema.class_names)}
    for yolo_id in sorted(stats["per_class_counts"].keys()):
        name = yolo_id_to_name[yolo_id]
        count = stats["per_class_counts"][yolo_id]
        print(f"    {yolo_id:>3}  {name:<25} {count:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input-dir", required=True, help="Directory of VOC XML files")
    parser.add_argument(
        "--labels-dir", required=True, help="Output directory for YOLO .txt files"
    )
    parser.add_argument(
        "--image-list",
        default=None,
        help="Optional path to a text file listing image filenames (one per line) "
        "to filter which XMLs to process. Used to split SODA into train/test.",
    )
    parser.add_argument("--schema", default=None)
    args = parser.parse_args()

    schema = load_schema(args.schema) if args.schema else load_schema()

    stem_filter = None
    if args.image_list:
        with open(args.image_list) as f:
            stem_filter = {Path(line.strip()).stem for line in f if line.strip()}

    stats = convert_voc_to_yolo(
        xml_dir=args.input_dir,
        labels_dir=args.labels_dir,
        schema=schema,
        image_stem_filter=stem_filter,
    )
    _print_stats(stats, schema)


if __name__ == "__main__":
    main()
