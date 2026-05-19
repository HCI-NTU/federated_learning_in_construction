"""Generate synthetic JPEG images with visible content for real YOLO training.

Earlier tests created empty-file placeholders, sufficient for pipeline-level
testing (which only checks filenames). Actual YOLO training requires real
image bytes — this script generates simple shapes that vary by class.

The shapes are deliberately easy to learn: solid-colored rectangles on a
gradient background, with each class getting a distinct color. This isn't
a meaningful detection task — it exists purely to verify that the FL client
can run a real Ultralytics training loop end-to-end without crashing.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Deterministic palette — color per YOLO class id 0..16
CLASS_COLORS = [
    (255, 0, 0),     # 0 worker            red
    (0, 200, 0),     # 1 excavator         green
    (0, 100, 255),   # 2 bulldozer         blue
    (255, 200, 0),   # 3 wheel_loader      yellow
    (200, 0, 200),   # 4 dump_truck        magenta
    (0, 200, 200),   # 5 concrete_mixer    cyan
    (255, 128, 0),   # 6 roller_compactor  orange
    (128, 0, 200),   # 7 tower_crane       purple
    (180, 80, 80),   # 8 mobile_crane      brown-red
    (80, 180, 80),   # 9 concrete_pump     olive
    (60, 120, 180),  # 10 backhoe          steel-blue
    (220, 220, 60),  # 11 grader           lime
    (180, 80, 180),  # 12 pile_driver      pink-purple
    (140, 140, 140), # 13 crane_hook       gray
    (80, 80, 80),    # 14 precast_truck    dark gray
    (200, 100, 50),  # 15 precast_panel    rust
    (60, 60, 60),    # 16 other_vehicle    near-black
]


def render_image_with_boxes(
    out_path: Path,
    width: int,
    height: int,
    boxes: list[tuple[int, float, float, float, float]],
    seed: int = 0,
) -> None:
    """Render a JPEG with rectangles for each box (boxes are YOLO format).

    boxes is a list of (class_id, cx, cy, w, h) with cx/cy/w/h in [0, 1].
    """
    rng = random.Random(seed)
    # Gradient background
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    bg_array = np.zeros((height, width, 3), dtype=np.uint8)
    bg_color = (rng.randint(60, 180), rng.randint(60, 180), rng.randint(60, 180))
    for y in range(height):
        for x in range(width):
            bg_array[y, x] = bg_color
    img = Image.fromarray(bg_array)

    draw = ImageDraw.Draw(img)
    for class_id, cx, cy, bw, bh in boxes:
        if not (0 <= class_id < len(CLASS_COLORS)):
            continue
        color = CLASS_COLORS[class_id]
        x1 = int((cx - bw / 2) * width)
        y1 = int((cy - bh / 2) * height)
        x2 = int((cx + bw / 2) * width)
        y2 = int((cy + bh / 2) * height)
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(255, 255, 255), width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=85)


def render_dataset(
    label_dir: Path, image_dir: Path, width: int = 320, height: int = 240
) -> int:
    """For each .txt label in label_dir, render the corresponding image.

    Returns the count of images rendered.
    """
    label_dir = Path(label_dir)
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for label_path in label_dir.glob("*.txt"):
        boxes = []
        for line in label_path.read_text().strip().split("\n"):
            if not line:
                continue
            parts = line.split()
            try:
                class_id = int(parts[0])
                cx, cy, bw, bh = (float(x) for x in parts[1:5])
                boxes.append((class_id, cx, cy, bw, bh))
            except (ValueError, IndexError):
                continue
        img_path = image_dir / f"{label_path.stem}.jpg"
        render_image_with_boxes(img_path, width, height, boxes, seed=hash(label_path.stem) & 0xFFFF)
        count += 1
    return count


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: make_synthetic_images.py <labels_dir> <output_image_dir>")
        sys.exit(1)
    n = render_dataset(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Rendered {n} images")
