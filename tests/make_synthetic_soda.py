"""Generate synthetic SODA-like VOC XMLs for testing the VOC converter.

SODA has only one class: 'person'. The generator creates per-image XMLs with
1-5 person bboxes each. Also generates one XML with an unknown <name> to
exercise the unmapped-name code path.
"""
from __future__ import annotations

import random
from pathlib import Path
import sys


XML_TEMPLATE = """<annotation>
    <filename>{filename}</filename>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>3</depth>
    </size>
{objects}
</annotation>
"""

OBJECT_TEMPLATE = """    <object>
        <name>{name}</name>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>"""


def generate(output_dir: Path, n_images: int = 50, seed: int = 0) -> None:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, n_images + 1):
        filename = f"soda_{i:05d}.jpg"
        width, height = 1920, 1080

        # Most images get 'person'; a few get an unknown class for testing.
        n_objs = rng.randint(1, 5)
        objects = []
        for _ in range(n_objs):
            if rng.random() < 0.02 and i > 10:
                # 2% chance of an unmapped name, after first 10 images
                name = "vehicle"  # not in SODA schema
            else:
                name = "person"
            bw = rng.randint(40, 200)
            bh = rng.randint(80, 400)
            xmin = rng.randint(0, width - bw)
            ymin = rng.randint(0, height - bh)
            xmax = xmin + bw
            ymax = ymin + bh
            objects.append(
                OBJECT_TEMPLATE.format(
                    name=name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
                )
            )

        xml_text = XML_TEMPLATE.format(
            filename=filename,
            width=width,
            height=height,
            objects="\n".join(objects),
        )
        (output_dir / f"soda_{i:05d}.xml").write_text(xml_text)

    print(f"Wrote {n_images} XML files to {output_dir}")


if __name__ == "__main__":
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_soda_xml")
    generate(output)
