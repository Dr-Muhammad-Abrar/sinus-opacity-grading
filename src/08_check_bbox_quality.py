"""
Stage 8 (diagnostic) — Bounding box quality check.

The contact-sheet scan revealed two issues:
  1. A black ruler/scale-bar artifact baked into many images
  2. Several crops that look like thin horizontal slivers rather than
     plausible sinus cavity regions — suggesting some "maxillary sinus"
     bounding box annotations may be degenerate (extremely flat, wrong
     height) rather than actual anatomy

This script measures every sinus bbox's aspect ratio and flags outliers
automatically, instead of relying on visual scanning. It also does a
simple, cheap heuristic check for the black ruler bar (a solid band of
very dark pixels spanning most of the image width).

Saves:
    outputs/labels/bbox_quality_report.csv
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import pandas as pd
from PIL import Image
from coco_lite import COCO

ASPECT_RATIO_FLAG = 3.0    # width/height (or height/width) above this is flagged as likely degenerate
DARK_BAND_FRACTION = 0.85  # if this fraction of a horizontal strip's pixels are near-black, flag as likely ruler/scale-bar


def check_dark_band(image, bbox, pad_ratio):
    """Cheap heuristic: look at the top strip of the padded crop region
    for a suspiciously uniform, near-black horizontal band (the ruler
    artifact typically appears as a solid black bar with white text)."""
    x, y, w, h = bbox
    pad_x, pad_y = w * pad_ratio, h * pad_ratio
    left = max(0, int(x - pad_x))
    top = max(0, int(y - pad_y))
    right = min(image.width, int(x + w + pad_x))
    bottom = min(image.height, int(y + h + pad_y))
    crop = np.asarray(image.crop((left, top, right, bottom)), dtype=np.float32)

    if crop.shape[0] < 10:
        return False

    strip_height = max(5, crop.shape[0] // 8)
    top_strip = crop[:strip_height, :]
    dark_fraction = (top_strip < 30).mean()  # near-black pixels
    return dark_fraction > DARK_BAND_FRACTION


def main():
    rows = []
    for split in ["train", "valid", "test"]:
        coco = COCO(config.ANNOTATIONS[split])
        cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
        sinus_id = next((k for k, v in cats.items() if v.strip().lower() == config.SINUS_CLASS_NAME), None)
        if sinus_id is None:
            continue

        for img_id in sorted(coco.getImgIds(catIds=[sinus_id])):
            img_info = coco.loadImgs([img_id])[0]
            ann = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], catIds=[sinus_id]))[0]
            x, y, w, h = ann["bbox"]
            aspect_ratio = max(w / h, h / w) if h > 0 and w > 0 else float("inf")

            img_path = os.path.join(config.IMAGES[split], img_info["file_name"])
            image = Image.open(img_path).convert("L")
            has_dark_band = check_dark_band(image, ann["bbox"], config.PAD_RATIO)

            rows.append({
                "split": split, "image_id": img_id, "file_name": img_info["file_name"],
                "bbox_w": w, "bbox_h": h, "aspect_ratio": aspect_ratio,
                "likely_degenerate_box": aspect_ratio > ASPECT_RATIO_FLAG,
                "likely_ruler_artifact": has_dark_band,
            })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(os.path.dirname(config.LABELS_CSV), "bbox_quality_report.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Checked {len(df)} sinus boxes.\n")
    print("Aspect ratio stats:")
    print(df["aspect_ratio"].describe())

    n_degenerate = df["likely_degenerate_box"].sum()
    n_ruler = df["likely_ruler_artifact"].sum()
    print(f"\nFlagged as likely degenerate box (aspect ratio > {ASPECT_RATIO_FLAG}): "
          f"{n_degenerate} / {len(df)} ({100*n_degenerate/len(df):.1f}%)")
    print(f"Flagged as likely ruler/scale-bar artifact: "
          f"{n_ruler} / {len(df)} ({100*n_ruler/len(df):.1f}%)")

    both = df[df["likely_degenerate_box"] & df["likely_ruler_artifact"]]
    print(f"Flagged for BOTH issues: {len(both)}")

    print(f"\nFull report saved to: {out_csv}")
    print("\nOpen this CSV, sort by 'aspect_ratio' descending, and look at the image_ids "
          "with the highest values first — those are your most likely degenerate boxes.")


if __name__ == "__main__":
    main()
