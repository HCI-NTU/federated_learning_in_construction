"""Generate a synthetic ACID-like COCO JSON for testing the splitter.

The synthetic data mimics ACID's structure:
- 10 categories with IDs 1-12 (matching the actual schema IDs for ACID)
- Realistic class imbalance: common classes (excavator, dump_truck) abundant,
  rare classes (grader, backhoe_loader) scarce
- Multiple objects per image, with co-occurrence patterns
- Some images with no annotations (to test edge-case handling)
"""
from __future__ import annotations

import json
import random
from pathlib import Path


# Match the ACID category IDs from the unified schema
ACID_CATEGORIES = [
    {"id": 1, "name": "backhoe_loader"},
    {"id": 2, "name": "cement_truck"},
    {"id": 3, "name": "compactor"},
    {"id": 4, "name": "dozer"},
    {"id": 5, "name": "dump_truck"},
    {"id": 6, "name": "excavator"},
    {"id": 7, "name": "grader"},
    {"id": 8, "name": "mobile_crane"},
    {"id": 9, "name": "tower_crane"},
    {"id": 10, "name": "wheel_loader"},
]

# Approximate per-class frequencies — common classes get more, rare classes less
CLASS_WEIGHTS = {
    6: 200,   # excavator: very common
    5: 150,   # dump_truck: very common
    4: 100,   # dozer: common
    3: 80,    # compactor: common
    10: 70,   # wheel_loader: common
    8: 40,    # mobile_crane: moderate
    2: 30,    # cement_truck: moderate
    9: 20,    # tower_crane: less common
    1: 12,    # backhoe_loader: rare
    7: 8,     # grader: very rare
}

def generate(output_path: Path, n_images: int = 600, seed: int = 0) -> None:
    rng = random.Random(seed)

    images = []
    annotations = []
    next_image_id = 1
    next_ann_id = 1

    cat_ids = list(CLASS_WEIGHTS.keys())
    cat_weights = [CLASS_WEIGHTS[c] for c in cat_ids]

    for _ in range(n_images):
        img_w, img_h = 1280, 720
        img = {
            "id": next_image_id,
            "file_name": f"acid_{next_image_id:05d}.jpg",
            "width": img_w,
            "height": img_h,
        }
        images.append(img)

        # 5% of images get zero annotations (test edge case)
        if rng.random() < 0.05:
            next_image_id += 1
            continue

        # Otherwise 1-4 objects per image
        n_objs = rng.choices([1, 2, 3, 4], weights=[40, 35, 20, 5])[0]
        for _ in range(n_objs):
            cat_id = rng.choices(cat_ids, weights=cat_weights)[0]
            # Random bbox in COCO format [x, y, w, h]
            bw = rng.randint(50, 400)
            bh = rng.randint(50, 300)
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
        "info": {"description": "Synthetic ACID-like test data"},
        "licenses": [],
        "categories": ACID_CATEGORIES,
        "images": images,
        "annotations": annotations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f)
    print(f"Wrote {output_path}: {len(images)} images, {len(annotations)} annotations")


if __name__ == "__main__":
    import sys
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_acid.json")
    generate(output)
