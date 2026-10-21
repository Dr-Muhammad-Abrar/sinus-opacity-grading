"""
Twin check (read-only): finds near-identical crops inside each split and across
splits, using the ROI crops already saved in outputs/rois. Run after 03_roi_extraction.py.
In the train split of the Kaggle export used here, copies of one radiograph have
consecutive ids (odd id, next even id); this script verifies that pattern.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
from PIL import Image

THRESH = 0.97  # correlation above this = near-identical


def load_split(split):
    root = os.path.join(config.ROI_DIR, split)
    imgs = {}
    for g in os.listdir(root):
        gdir = os.path.join(root, g)
        for f in os.listdir(gdir):
            if f.lower().endswith(".png"):
                im = Image.open(os.path.join(gdir, f)).convert("L").resize((64, 64))
                imgs[int(os.path.splitext(f)[0])] = np.asarray(im, dtype=float)
    return imgs


def corr(a, b):
    x, y = a.ravel(), b.ravel()
    if x.std() < 1e-6 or y.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


data = {s: load_split(s) for s in ["train", "valid", "test"]}

for split, imgs in data.items():
    ids = sorted(imgs)
    odd_even = [(i, i + 1) for i in ids if i % 2 == 1 and (i + 1) in imgs]
    even_odd = [(i, i + 1) for i in ids if i % 2 == 0 and (i + 1) in imgs]
    c_pair = [corr(imgs[a], imgs[b]) for a, b in odd_even]
    c_ctrl = [corr(imgs[a], imgs[b]) for a, b in even_odd]
    has_twin = 0
    for i in ids:
        best = max((corr(imgs[i], imgs[j]) for j in ids if j != i), default=0)
        has_twin += best > THRESH
    print(f"\n=== {split}: {len(ids)} images ===")
    print(f"images with a near-duplicate in the same split: {has_twin}/{len(ids)}")
    if odd_even:
        print(f"(odd id, next id) pairs: {len(odd_even)}, near-identical: "
              f"{sum(c > THRESH for c in c_pair)}, lowest corr: {min(c_pair):.3f}")
    if even_odd:
        print(f"control (even id, next id) pairs: {len(even_odd)}, near-identical: "
              f"{sum(c > THRESH for c in c_ctrl)}")

for other in ["valid", "test"]:
    leaks = 0
    for j, b in data[other].items():
        best = max((corr(a, b) for a in data["train"].values()), default=0)
        leaks += best > THRESH
    print(f"\n{other} images that have a near-duplicate in train: {leaks}/{len(data[other])}")
