"""
Stage 2c (optional but recommended) — Radiologist validation subset.

Pulls a stratified sample across the automated score range (not just
random images — deliberately spans low/mid/high intensity so the
validation actually tests the full range) and exports:
    outputs/radiologist_subset/<n>_<split>_<image_id>.png   (the crops)
    outputs/radiologist_subset/validation_sheet.csv          (to fill in)

Give your radiologist the PNG folder + the CSV. They fill in the
"radiologist_grade" column (0/1/2) without seeing the automated grade
column (it's included for your records but hide/hand them a copy
without it if you want a blind check, which is the more rigorous test).

Run 02d_validate_against_radiologist.py afterward to compute agreement.
"""

import sys
import os
import csv
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
from PIL import Image

N_PER_TIER = 15  # ~45 total images, spanning all 3 automated tiers evenly
SUBSET_DIR = os.path.join(config.OUTPUT_DIR, "radiologist_subset")


def main():
    df = pd.read_csv(config.LABELS_CSV)
    os.makedirs(SUBSET_DIR, exist_ok=True)

    random.seed(42)
    sampled_rows = []
    for grade in ["0", "1", "2"]:
        tier_df = df[df["grade"].astype(str) == grade]
        n = min(N_PER_TIER, len(tier_df))
        sampled_rows.append(tier_df.sample(n=n, random_state=42))

    subset = pd.concat(sampled_rows).reset_index(drop=True)

    sheet_rows = []
    for i, row in subset.iterrows():
        img_path = os.path.join(config.IMAGES[row["split"]], row["file_name"])
        bbox = eval(row["bbox"]) if isinstance(row["bbox"], str) else row["bbox"]
        x, y, w, h = bbox
        pad_x, pad_y = w * config.PAD_RATIO, h * config.PAD_RATIO

        image = Image.open(img_path).convert("L")
        left = max(0, int(x - pad_x))
        top = max(0, int(y - pad_y))
        right = min(image.width, int(x + w + pad_x))
        bottom = min(image.height, int(y + h + pad_y))
        crop = image.crop((left, top, right, bottom))

        out_name = f"{i:03d}_{row['split']}_{row['image_id']}.png"
        crop.save(os.path.join(SUBSET_DIR, out_name))

        sheet_rows.append({
            "sample_id": out_name,
            "split": row["split"],
            "image_id": row["image_id"],
            "automated_grade": row["grade"],
            "radiologist_grade": "",  # to be filled in by the radiologist
        })

    sheet_path = os.path.join(SUBSET_DIR, "validation_sheet.csv")
    with open(sheet_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "image_id",
                                                 "automated_grade", "radiologist_grade"])
        writer.writeheader()
        writer.writerows(sheet_rows)

    print(f"Exported {len(subset)} images to: {SUBSET_DIR}")
    print(f"Validation sheet: {sheet_path}")
    print("\nFor a blind validation, make a copy of validation_sheet.csv with the")
    print("'automated_grade' column removed before giving it to your radiologist.")


if __name__ == "__main__":
    main()
