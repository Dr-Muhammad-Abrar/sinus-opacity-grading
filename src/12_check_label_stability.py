"""
Step 6 - Label stability check (read-only).

The twin copies of one X-ray look identical, yet 28 of 71 twin pairs got
different severity grades. This script measures WHY: is the brightness score
itself noisy (copies differ a lot), or is it stable but sliced by cutoffs that
cut through the middle of the distribution (so tiny differences flip grades)?
It only reads outputs/labels/sinus_severity_labels.csv and changes nothing.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import pandas as pd

df = pd.read_csv(config.LABELS_CSV)
df["grade"] = df["grade"].astype(int)
mi = df["mean_intensity"].to_numpy()
p33, p66 = np.percentile(mi, [33, 66])
print(f"images: {len(df)} | cutoffs P33={p33:.2f}, P66={p66:.2f} | "
      f"spread (std over all images)={mi.std():.2f} | middle-band width={p66 - p33:.2f}")

# ---- build twin pairs in train: (odd id, next even id) share (id+1)//2 ----
tr = df[df["split"] == "train"].copy()
tr["pair"] = (tr["image_id"] + 1) // 2
rows = []
for _, g in tr.groupby("pair"):
    if len(g) == 2:
        g = g.sort_values("image_id")
        a, b = g.iloc[0], g.iloc[1]
        rows.append((int(a.image_id), int(b.image_id), a.grade, b.grade,
                     a.mean_intensity, b.mean_intensity))
pairs = pd.DataFrame(rows, columns=["id_a", "id_b", "grade_a", "grade_b", "mi_a", "mi_b"])
pairs["d"] = pairs["mi_b"] - pairs["mi_a"]          # second copy minus first copy
pairs["flip"] = pairs["grade_a"] != pairs["grade_b"]
n_pairs, n_flip = len(pairs), int(pairs["flip"].sum())
print(f"\ntwin pairs: {n_pairs} | pairs with DIFFERENT grades: {n_flip} ({100 * n_flip / n_pairs:.0f}%)")

absd = pairs["d"].abs()
print("\n--- how different are the two copies of the same X-ray? ---")
print(f"|difference in brightness score|: median={absd.median():.2f}, "
      f"90th pct={absd.quantile(0.9):.2f}, max={absd.max():.2f}")
print(f"signed difference (2nd copy minus 1st): mean={pairs['d'].mean():+.2f}, "
      f"SD={pairs['d'].std():.2f}, share where 2nd copy is brighter={100 * (pairs['d'] > 0).mean():.0f}%")

# reliability: how much of the score variation is real (between X-rays) vs copy-to-copy noise
sw2 = (pairs["d"] ** 2).mean() / 2                      # within-pair variance
pair_means = (pairs["mi_a"] + pairs["mi_b"]) / 2
sb2 = max(pair_means.var(ddof=1) - sw2 / 2, 0)          # between-X-ray variance
icc = sb2 / (sb2 + sw2) if (sb2 + sw2) > 0 else float("nan")
print(f"reliability of the brightness score across twin copies (ICC): {icc:.3f}")

# how many images sit so close to a cutoff that a copy-sized change could flip them
med = absd.median()
dist = np.minimum(np.abs(mi - p33), np.abs(mi - p66))
for mult in (1, 2, 3):
    n_close = int((dist <= mult * med).sum())
    print(f"images within {mult} x median-copy-difference ({mult * med:.2f}) of a cutoff: "
          f"{n_close}/{len(df)} ({100 * n_close / len(df):.0f}%)")

print("\n--- examples of twin pairs whose grades differ ---")
print(pairs[pairs["flip"]].head(8).round(2).to_string(index=False))

print("\nHOW TO READ THIS")
print("* ICC high (>0.95) and signed mean near 0: the score is stable; the flips come from cutoffs")
print("  slicing through the dense middle of the distribution. Fix = drop images near cutoffs, or")
print("  predict the continuous score instead of three classes.")
print("* Signed mean clearly away from 0 (one copy consistently brighter): the copies differ by a")
print("  brightness change, so the label tracks exposure, not anatomy.")
