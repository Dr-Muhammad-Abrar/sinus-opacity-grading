"""
Stage 3 — ROI extraction.
Reads your graded labels CSV (from 02_grading_tool.py) and crops each
sinus ROI, padded and resized, saving it into:
    outputs/rois/<split>/<grade>/<image_id>.png

This keeps the original train/valid/test split from the dataset intact
(no re-shuffling), and also runs a leakage check across splits by
comparing file names, since the original split's leakage-safety hasn't
been independently verified yet.
"""

import sys
import os
import ast
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import pandas as pd
from PIL import Image
from coco_lite import COCO

DARK_ROW_THRESHOLD = 30      # pixel intensity below this counts as "near-black"
DARK_ROW_FRACTION = 0.85      # fraction of a row's pixels that must be near-black to flag that row
MAX_BAND_SEARCH_FRACTION = 0.35  # only search the top 35% of the crop for a band (avoid eating real anatomy)

# Active cropping was tried and found to introduce inconsistent aspect-ratio
# distortion (variable per-image height after stripping, then force-resized
# to a fixed square), which measurably reduced Grad-CAM++ IoU (0.143 -> 0.100)
# once the alignment bug was fixed and it was measured correctly. Reverted to
# detection-only: this documents which images WOULD be affected (useful for
# the paper's limitations section) without altering the actual pixels used
# for training/evaluation. Set to True to re-enable active stripping.
APPLY_BAND_STRIPPING = False


def strip_ruler_band(crop):
    """Detects a ruler/scale-bar artifact as a contiguous run of
    near-black rows at the TOP of the crop (where it consistently
    appeared across the dataset) and removes those rows, shifting the
    crop's top boundary down past the artifact before resizing. Returns
    the (possibly shortened) crop and a bool indicating whether
    anything was removed."""
    arr = np.asarray(crop, dtype=np.float32)
    h, w = arr.shape
    max_search_rows = int(h * MAX_BAND_SEARCH_FRACTION)

    band_end = 0
    for row in range(max_search_rows):
        dark_fraction = (arr[row] < DARK_ROW_THRESHOLD).mean()
        if dark_fraction > DARK_ROW_FRACTION:
            band_end = row + 1
        else:
            if band_end > 0:
                break  # band ended, stop (don't eat further unrelated dark rows)

    if band_end > 0:
        return crop.crop((0, band_end, w, h)), True, band_end
    return crop, False, 0


def clear_roi_dir():
    """Removes any previously extracted crops before writing fresh ones.
    Without this, re-running this script after labels change (e.g. after
    excluding degenerate boxes, or after refining the severity scoring)
    leaves stale files sitting in old grade folders, silently inflating
    the dataset with duplicates/outdated images."""
    if os.path.exists(config.ROI_DIR):
        print(f"Clearing existing ROI folder: {config.ROI_DIR}")
        shutil.rmtree(config.ROI_DIR)


def check_filename_leakage(df):
    """Warn if the exact same file_name appears in more than one split."""
    dupes = df.groupby("file_name")["split"].nunique()
    leaking = dupes[dupes > 1]
    if len(leaking):
        print(f"WARNING: {len(leaking)} file names appear in more than one split — possible leakage:")
        print(leaking.head(20))
    else:
        print("No duplicate file names across splits — leakage check passed.")


def extract_all():
    clear_roi_dir()
    df = pd.read_csv(config.LABELS_CSV)
    check_filename_leakage(df)

    coco_objs = {s: COCO(config.ANNOTATIONS[s]) for s in ["train", "valid", "test"]}

    for grade in config.GRADE_LABELS:
        for split in ["train", "valid", "test"]:
            os.makedirs(os.path.join(config.ROI_DIR, split, grade), exist_ok=True)

    n_saved = 0
    n_band_stripped = 0
    metadata_rows = []
    for _, row in df.iterrows():
        split, img_id, file_name, bbox, grade = (
            row["split"], int(row["image_id"]), row["file_name"], row["bbox"], str(row["grade"])
        )
        x, y, w, h = ast.literal_eval(bbox) if isinstance(bbox, str) else bbox
        pad_x, pad_y = w * config.PAD_RATIO, h * config.PAD_RATIO

        img_path = os.path.join(config.IMAGES[split], file_name)
        image = Image.open(img_path).convert("L")
        left = max(0, int(x - pad_x))
        top = max(0, int(y - pad_y))
        right = min(image.width, int(x + w + pad_x))
        bottom = min(image.height, int(y + h + pad_y))
        crop = image.crop((left, top, right, bottom))
        pre_strip_w, pre_strip_h = crop.size

        crop, would_strip, detected_band_end = strip_ruler_band(crop)
        if would_strip:
            n_band_stripped += 1
        if not APPLY_BAND_STRIPPING:
            # Revert to the un-stripped crop for actual training/eval use,
            # but keep the detection result for the metadata/reporting.
            crop = image.crop((left, top, right, bottom))
            band_end = 0
        else:
            band_end = detected_band_end

        crop = crop.resize((config.IMG_SIZE, config.IMG_SIZE))

        out_path = os.path.join(config.ROI_DIR, split, grade, f"{img_id}.png")
        crop.save(out_path)
        n_saved += 1

        metadata_rows.append({
            "split": split, "image_id": img_id, "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
            "pad_x": pad_x, "pad_y": pad_y, "pre_strip_w": pre_strip_w, "pre_strip_h": pre_strip_h,
            "band_end": band_end,
        })

    print(f"Saved {n_saved} ROI crops to {config.ROI_DIR}")
    action_word = "stripped" if APPLY_BAND_STRIPPING else "detected (not applied, see APPLY_BAND_STRIPPING)"
    print(f"Ruler/scale-bar band {action_word} in {n_band_stripped}/{n_saved} images")

    meta_df = pd.DataFrame(metadata_rows)
    meta_csv = os.path.join(os.path.dirname(config.LABELS_CSV), "roi_crop_metadata.csv")
    meta_df.to_csv(meta_csv, index=False)
    print(f"Saved crop metadata (needed for correct Grad-CAM box alignment) to: {meta_csv}")

    # Class balance summary — check this before training
    print("\nClass balance per split:")
    print(df.groupby(["split", "grade_label"]).size())


if __name__ == "__main__":
    extract_all()