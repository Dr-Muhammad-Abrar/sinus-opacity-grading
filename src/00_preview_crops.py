"""
Utility — Preview crops (no grading).

Saves every sinus ROI as a plain PNG file into outputs/preview_crops/,
named by split and image ID, so you can open the folder in Windows
Explorer (switch to "Large icons" view) and actually browse all 233
images side by side at once — much easier than judging them one at a
time through the grading tool window.

Run this, then open:
    outputs/preview_crops/
in File Explorer and look through them before grading for real.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from PIL import Image
from coco_lite import COCO

PREVIEW_DIR = os.path.join(config.OUTPUT_DIR, "preview_crops")


def main():
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    n_saved = 0

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

            out_path = os.path.join(PREVIEW_DIR, f"{split}_{img_id}.png")
            crop.save(out_path)
            n_saved += 1

    print(f"Saved {n_saved} preview crops to: {PREVIEW_DIR}")
    print("Open that folder in File Explorer (switch to Large Icons or Extra Large Icons view)")
    print("and scroll through them to see whether they actually look visually different from each other.")


if __name__ == "__main__":
    main()
