"""Generate a synthetic MOCS-like COCO JSON for testing the COCO converter.

The synthetic data exercises all 13 MOCS source category IDs from the unified
schema, ensuring the converter correctly maps each one to its expected unified
YOLO id.

MOCS categories (id : name -> unified_id, yolo_id):
   1  Worker          -> 1, yolo=0
   2  Static crane    -> 8, yolo=7
   3  Hanging head    -> 14, yolo=13
   4  Crane           -> 9, yolo=8
   5  Roller          -> 7, yolo=6
   6  Bulldozer       -> 3, yolo=2
   7  Excavator       -> 2, yolo=1
   8  Truck           -> 5, yolo=4
   9  Loader          -> 4, yolo=3
  10  Pump truck      -> 10, yolo=9
  11  Concrete mixer  -> 6, yolo=5
  12  Pile driving    -> 13, yolo=12
  13  Other vehicle   -> 17, yolo=16
"""
from __future__ import annotations

import json
import random
from pathlib import Path
import sys


MOCS_CATEGORIES = [
    {"id": 1, "name": "Worker"},
    {"id": 2, "name": "Static crane"},
    {"id": 3, "name": "Hanging head"},
    {"id": 4, "name": "Crane"},
    {"id": 5, "name": "Roller"},
    {"id": 6, "name": "Bulldozer"},
    {"id": 7, "name": "Excavator"},
    {"id": 8, "name": "Truck"},
    {"id": 9, "name": "Loader"},
    {"id": 10, "name": "Pump truck"},
    {"id": 11, "name": "Concrete mixer"},
    {"id": 12, "name": "Pile driving"},
    {"id": 13, "name": "Other vehicle"},
]

# Give every class at least a few instances so the per-class test is meaningful
CLASS_WEIGHTS = {
    1: 200,   # Worker - common
    7: 100,   # Excavator
    8: 80,    # Truck
    11: 60,   # Concrete mixer
    6: 50,    # Bulldozer
    5: 40,    # Roller
    9: 40,    # Loader
    4: 30,    # Crane
    2: 20,    # Static crane
    10: 20,   # Pump truck
    3: 15,    # Hanging head
    12: 10,   # Pile driving
    13: 10,   # Other vehicle
}


def generate(output_path: Path, n_images: int = 300, seed: int = 0) -> None:
    rng = random.Random(seed)

    images = []
    annotations = []
    next_image_id = 1
    next_ann_id = 1

    cat_ids = list(CLASS_WEIGHTS.keys())
    cat_weights = [CLASS_WEIGHTS[c] for c in cat_ids]

    for _ in range(n_images):
        img_w, img_h = 1920, 1080
        images.append({
            "id": next_image_id,
            "file_name": f"mocs_{next_image_id:05d}.jpg",
            "width": img_w,
            "height": img_h,
        })

        # 1-5 objects per image
        n_objs = rng.choices([1, 2, 3, 4, 5], weights=[20, 30, 25, 15, 10])[0]
        for _ in range(n_objs):
            cat_id = rng.choices(cat_ids, weights=cat_weights)[0]
            bw = rng.randint(50, 400)
            bh = rng.randint(50, 400)
            bx = rng.randint(0, img_w - bw)
            by = rng.randint(0, img_h - bh)
            annotations.append({
                "id": next_ann_id,
                "image_id": next_image_id,
                "category_id": cat_id,
                "bbox": [bx, by, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            next_ann_id += 1
        next_image_id += 1

    coco = {
        "info": {"description": "Synthetic MOCS-like test data"},
        "licenses": [],
        "categories": MOCS_CATEGORIES,
        "images": images,
        "annotations": annotations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f)
    print(f"Wrote {output_path}: {len(images)} images, {len(annotations)} annotations")


if __name__ == "__main__":
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_mocs.json")
    generate(output)
