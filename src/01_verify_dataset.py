"""
Stage 1 — Dataset verification.
Confirms class names/IDs and reports sinus + pathology counts per split.
Run this first, any time you re-download or change the dataset.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from coco_lite import COCO

PATHOLOGY_NAMES = ["periapical lesion", "bone loss", "cyst", "root resorption"]


def verify_split(split_name):
    print(f"\n===== {split_name} =====")
    coco = COCO(config.ANNOTATIONS[split_name])
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    sinus_id = [k for k, v in cats.items() if v.strip().lower() == config.SINUS_CLASS_NAME]
    pathology_ids = [k for k, v in cats.items() if v.strip().lower() in PATHOLOGY_NAMES]

    if not sinus_id:
        print("WARNING: no sinus class found in this split.")
        return

    sinus_imgs = set(coco.getImgIds(catIds=sinus_id))
    pathology_imgs = set(coco.getImgIds(catIds=pathology_ids))

    print("Total images:", len(coco.getImgIds()))
    print("Sinus ROIs:", len(sinus_imgs))
    print("Sinus + pathology overlap:", len(sinus_imgs & pathology_imgs))
    print("Sinus, no pathology overlap:", len(sinus_imgs - pathology_imgs))


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        verify_split(split)
