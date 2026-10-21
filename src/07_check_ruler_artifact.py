"""
Stage 7 (diagnostic) — Ruler/scale-bar artifact check.

One of the Grad-CAM++ overlays (image 1475) showed what looked like a
calibration ruler baked into the X-ray itself, with the model's
attention concentrated near it. This script builds "contact sheets" --
grids of many small thumbnails with their image IDs labeled -- so you
can quickly scan all 233 sinus crops at once and judge how common this
artifact actually is, rather than opening 233 files individually.

Saves:
    outputs/figures/contact_sheet_1.png, _2.png, ... (batches of 40)
"""

import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from PIL import Image, ImageDraw, ImageFont
from coco_lite import COCO

PER_SHEET = 40
THUMB_SIZE = 140
COLS = 8
LABEL_HEIGHT = 18


def build_contact_sheets():
    crops = []  # list of (image_id, split, PIL image)

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
            pad_x, pad_y = w * config.PAD_RATIO, h * config.PAD_RATIO

            img_path = os.path.join(config.IMAGES[split], img_info["file_name"])
            image = Image.open(img_path).convert("L")
            left = max(0, int(x - pad_x))
            top = max(0, int(y - pad_y))
            right = min(image.width, int(x + w + pad_x))
            bottom = min(image.height, int(y + h + pad_y))
            crop = image.crop((left, top, right, bottom))
            crops.append((img_id, split, crop))

    print(f"Collected {len(crops)} crops. Building contact sheets ({PER_SHEET} per sheet)...")

    n_sheets = math.ceil(len(crops) / PER_SHEET)
    for sheet_idx in range(n_sheets):
        batch = crops[sheet_idx * PER_SHEET: (sheet_idx + 1) * PER_SHEET]
        rows = math.ceil(len(batch) / COLS)

        sheet_w = COLS * THUMB_SIZE
        sheet_h = rows * (THUMB_SIZE + LABEL_HEIGHT)
        sheet = Image.new("L", (sheet_w, sheet_h), color=255)
        draw = ImageDraw.Draw(sheet)

        for i, (img_id, split, crop) in enumerate(batch):
            r, c = divmod(i, COLS)
            thumb = crop.copy()
            thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
            x_off = c * THUMB_SIZE + (THUMB_SIZE - thumb.width) // 2
            y_off = r * (THUMB_SIZE + LABEL_HEIGHT) + (THUMB_SIZE - thumb.height) // 2
            sheet.paste(thumb, (x_off, y_off))
            label_y = r * (THUMB_SIZE + LABEL_HEIGHT) + THUMB_SIZE
            draw.text((c * THUMB_SIZE + 4, label_y), f"{split[:1]}{img_id}", fill=0)

        out_path = os.path.join(config.FIGURES_DIR, f"contact_sheet_{sheet_idx + 1}.png")
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        sheet.save(out_path)
        print(f"Saved: {out_path}  ({len(batch)} images)")

    print(f"\nDone. Open the {n_sheets} contact sheet(s) in outputs/figures/ and scan for any "
          "ruler markings, scale bars, watermarks, or numbers baked into the image (like the "
          "'0 1 2 3...' seen in image 1475). Note the image IDs (e.g. 't1475') of any you spot.")


if __name__ == "__main__":
    build_contact_sheets()
