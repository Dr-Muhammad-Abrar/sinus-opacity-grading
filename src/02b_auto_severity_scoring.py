"""
Stage 2 (automated) — Pixel-intensity-based severity scoring.

Rationale:
    A healthy, air-filled maxillary sinus appears as a dark, radiolucent
    cavity on a panoramic X-ray. Mucosal thickening, opacification, or a
    retention cyst/polyp increases radiographic density inside that
    cavity, making it appear lighter/cloudier. This script measures that
    directly and objectively, with no human judgment involved:

    1. Per-image contrast normalization (autocontrast), to reduce the
       effect of exposure/device differences between images.
    2. Mean pixel intensity measured inside a SHRUNK, CENTERED inner
       region of the sinus bounding box (trimming INNER_MARGIN_RATIO
       from each side) — not the full box. The full box includes the
       bright cortical boundary line at the sinus floor/walls, which is
       naturally high-contrast regardless of cavity opacification; an
       earlier version of this script that used the full box produced
       Grad-CAM++ activations concentrated on that boundary line rather
       than the cavity interior, suggesting the score was dominated by
       edge brightness rather than true opacification. Trimming the
       margin isolates the cavity interior instead.
    3. Population-level percentile thresholds (33rd / 66th) convert the
       continuous intensity score into 3 ordinal tiers:
           0 = Normal        (darkest third — clearest cavity)
           1 = Mild           (middle third)
           2 = Moderate/severe (brightest third — most opacified)

Output format matches what 03_roi_extraction.py expects, so this is a
drop-in replacement for the manual grading tool's CSV.

Limitation to state explicitly in your paper: this is an objective
image-intensity proxy for opacification, not a clinically validated
diagnosis. It should ideally be validated against a radiologist's
grading on a subset (see 02c_radiologist_subset_export.py).
"""

import sys
import os
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
from PIL import Image, ImageOps
from coco_lite import COCO

INNER_MARGIN_RATIO = 0.2  # trim 20% off each side of the box, keeping the centered inner 60% x 60%
BBOX_QUALITY_REPORT = os.path.join(config.OUTPUT_DIR, "labels", "bbox_quality_report.csv")
ASPECT_RATIO_FLAG = 3.0


def load_excluded_ids():
    """Loads the set of (split, image_id) pairs flagged as degenerate
    bounding boxes by 08_check_bbox_quality.py, so they're skipped
    entirely -- this propagates the fix through every downstream stage,
    since everything reads from this script's output."""
    if not os.path.exists(BBOX_QUALITY_REPORT):
        print("No bbox_quality_report.csv found -- run 08_check_bbox_quality.py first "
              "if you want degenerate boxes excluded. Proceeding without exclusion.")
        return set()
    import pandas as pd
    report = pd.read_csv(BBOX_QUALITY_REPORT)
    bad = report[report["aspect_ratio"] > ASPECT_RATIO_FLAG]
    excluded = set(zip(bad["split"], bad["image_id"]))
    print(f"Excluding {len(excluded)} images flagged as degenerate boxes "
          f"(aspect ratio > {ASPECT_RATIO_FLAG}).")
    return excluded


def compute_intensity_score(image_path, bbox):
    """Returns mean pixel intensity inside a shrunk, centered inner
    region of the sinus bbox (trimming the cortical boundary margin),
    after per-image autocontrast normalization."""
    image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image, cutoff=1)  # per-image contrast normalization

    x, y, w, h = bbox
    trim_x, trim_y = w * INNER_MARGIN_RATIO, h * INNER_MARGIN_RATIO
    left = max(0, int(x + trim_x))
    top = max(0, int(y + trim_y))
    right = min(image.width, int(x + w - trim_x))
    bottom = min(image.height, int(y + h - trim_y))
    crop = image.crop((left, top, right, bottom))

    arr = np.asarray(crop, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def main():
    excluded_ids = load_excluded_ids()
    records = []  # (split, image_id, file_name, bbox, mean_intensity, std_intensity)

    for split in ["train", "valid", "test"]:
        coco = COCO(config.ANNOTATIONS[split])
        cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
        sinus_id = next((k for k, v in cats.items() if v.strip().lower() == config.SINUS_CLASS_NAME), None)
        if sinus_id is None:
            continue

        for img_id in sorted(coco.getImgIds(catIds=[sinus_id])):
            if (split, img_id) in excluded_ids:
                continue
            img_info = coco.loadImgs([img_id])[0]
            ann = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], catIds=[sinus_id]))[0]
            bbox = ann["bbox"]

            img_path = os.path.join(config.IMAGES[split], img_info["file_name"])
            mean_i, std_i = compute_intensity_score(img_path, bbox)
            records.append([split, img_id, img_info["file_name"], bbox, mean_i, std_i])

    # Percentile thresholds computed across the WHOLE population (all splits
    # together), so grade boundaries are consistent regardless of split.
    all_means = np.array([r[4] for r in records])
    p33, p66 = np.percentile(all_means, [33, 66])
    print(f"Population thresholds — P33: {p33:.2f}, P66: {p66:.2f}")
    print(f"(mean intensity range: {all_means.min():.2f} to {all_means.max():.2f})")

    def grade_from_mean(m):
        if m <= p33:
            return "0", config.GRADE_LABELS["0"]
        elif m <= p66:
            return "1", config.GRADE_LABELS["1"]
        else:
            return "2", config.GRADE_LABELS["2"]

    os.makedirs(os.path.dirname(config.LABELS_CSV), exist_ok=True)
    with open(config.LABELS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image_id", "file_name", "bbox", "grade", "grade_label",
                          "mean_intensity", "std_intensity"])
        counts = {"0": 0, "1": 0, "2": 0}
        for split, img_id, file_name, bbox, mean_i, std_i in records:
            grade, grade_label = grade_from_mean(mean_i)
            counts[grade] += 1
            writer.writerow([split, img_id, file_name, bbox, grade, grade_label, mean_i, std_i])

    print(f"\nSaved {len(records)} automated severity labels to: {config.LABELS_CSV}")
    print("Grade distribution:", {config.GRADE_LABELS[k]: v for k, v in counts.items()})


if __name__ == "__main__":
    main()