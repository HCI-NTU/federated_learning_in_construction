"""Generate a synthetic CIS-like COCO JSON for testing the COCO converter.

The critical test case here is the multi-source mapping for 'worker':
  - CIS source id=6 ('people-helmet')    -> unified worker
  - CIS source id=7 ('people-no-helmet') -> unified worker
Both should land in YOLO id 0 after conversion.

CIS categories (id : name -> unified_id, yolo_id):
   0  PC                -> 16, yolo=15  (precast_panel)
   1  PC-truck          -> 15, yolo=14  (precast_panel_truck)
   2  dozer             -> 3,  yolo=2   (bulldozer)
   3  dump-truck        -> 5,  yolo=4   (dump_truck)
   4  excavator         -> 2,  yolo=1   (excavator)
   5  mixer             -> 6,  yolo=5   (concrete_mixer_truck)
   6  people-helmet     -> 1,  yolo=0   (worker) -- multi-source
   7  people-no-helmet  -> 1,  yolo=0   (worker) -- multi-source
   8  roller            -> 7,  yolo=6   (roller_compactor)
   9  wheel-loader      -> 4,  yolo=3   (wheel_loader)

CIS uses 0-indexed category IDs, unlike MOCS and ACID (1-indexed).
This is also a test of the schema lookup correctly reading the per-source IDs.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
import sys


CIS_CATEGORIES = [
    {"id": 0, "name": "PC"},
    {"id": 1, "name": "PC-truck"},
    {"id": 2, "name": "dozer"},
    {"id": 3, "name": "dump-truck"},
    {"id": 4, "name": "excavator"},
    {"id": 5, "name": "mixer"},
    {"id": 6, "name": "people-helmet"},
    {"id": 7, "name": "people-no-helmet"},
    {"id": 8, "name": "roller"},
    {"id": 9, "name": "wheel-loader"},
]

CLASS_WEIGHTS = {
    6: 120,  # people-helmet - common (worker)
    7: 80,   # people-no-helmet - common (worker)
    4: 90,   # excavator
    3: 70,   # dump-truck
    9: 50,   # wheel-loader
    2: 50,   # dozer
    5: 40,   # mixer
    8: 30,   # roller
    0: 25,   # PC
    1: 15,   # PC-truck
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
        img_w, img_h = 1280, 720
        images.append({
            "id": next_image_id,
            "file_name": f"cis_{next_image_id:05d}.jpg",
            "width": img_w,
            "height": img_h,
        })

        n_objs = rng.choices([1, 2, 3, 4, 5], weights=[15, 30, 30, 15, 10])[0]
        for _ in range(n_objs):
            cat_id = rng.choices(cat_ids, weights=cat_weights)[0]
            bw = rng.randint(40, 350)
            bh = rng.randint(40, 350)
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
        "info": {"description": "Synthetic CIS-like test data"},
        "licenses": [],
        "categories": CIS_CATEGORIES,
        "images": images,
        "annotations": annotations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f)
    print(f"Wrote {output_path}: {len(images)} images, {len(annotations)} annotations")


if __name__ == "__main__":
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_cis.json")
    generate(output)
