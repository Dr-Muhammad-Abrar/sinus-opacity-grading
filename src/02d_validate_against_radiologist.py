"""
Stage 2d — Validate automated scores against the radiologist's subset grades.

Run this after your radiologist has filled in the "radiologist_grade"
column in outputs/radiologist_subset/validation_sheet.csv.

Reports:
    - Cohen's kappa (chance-corrected agreement between automated and
      radiologist grades)
    - Spearman correlation (treats grades as ordinal)
    - A simple confusion matrix

This is the number that goes in your paper to justify using the
automated score as a valid severity proxy.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
from simple_metrics import confusion_matrix


def spearman_corr(x, y):
    """Pure-NumPy Spearman correlation with a normal-approximation p-value."""
    import numpy as np
    def rank(a):
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        return ranks
    rx, ry = rank(np.asarray(x)), rank(np.asarray(y))
    rho = np.corrcoef(rx, ry)[0, 1]
    n = len(x)
    if n > 3 and abs(rho) < 1:
        t_stat = rho * np.sqrt((n - 2) / (1 - rho**2))
        from math import erf, sqrt
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
    else:
        p_value = float("nan")
    return rho, p_value


def cohen_kappa_score(y1, y2, weights=None):
    """Pure-NumPy Cohen's kappa, with optional 'linear' weighting."""
    import numpy as np
    y1, y2 = np.asarray(y1), np.asarray(y2)
    labels = sorted(set(y1) | set(y2))
    n_labels = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}
    cm = confusion_matrix(y1, y2, labels=labels).astype(float)
    n = cm.sum()

    if weights == "linear":
        w = np.zeros((n_labels, n_labels))
        for i in range(n_labels):
            for j in range(n_labels):
                w[i, j] = abs(i - j)
    else:
        w = 1 - np.eye(n_labels)

    row_marg = cm.sum(axis=1)
    col_marg = cm.sum(axis=0)
    expected = np.outer(row_marg, col_marg) / n

    observed_disagreement = (w * cm).sum() / n
    expected_disagreement = (w * expected).sum() / n
    return 1 - observed_disagreement / expected_disagreement if expected_disagreement > 0 else float("nan")

SHEET_PATH = os.path.join(config.OUTPUT_DIR, "radiologist_subset", "validation_sheet.csv")


def main():
    df = pd.read_csv(SHEET_PATH)
    df = df.dropna(subset=["radiologist_grade"])
    df["radiologist_grade"] = df["radiologist_grade"].astype(int)
    df["automated_grade"] = df["automated_grade"].astype(int)

    if len(df) == 0:
        print("No filled-in radiologist grades found yet — ask your radiologist to complete the sheet first.")
        return

    kappa = cohen_kappa_score(df["automated_grade"], df["radiologist_grade"], weights="linear")
    rho, p_value = spearman_corr(df["automated_grade"], df["radiologist_grade"])
    cm = confusion_matrix(df["radiologist_grade"], df["automated_grade"], labels=[0, 1, 2])

    print(f"n = {len(df)} validated samples\n")
    print(f"Cohen's kappa (linear-weighted): {kappa:.3f}")
    print(f"Spearman correlation: rho = {rho:.3f}, p = {p_value:.4f}\n")
    print("Confusion matrix (rows = radiologist, cols = automated):")
    print("              Auto:0  Auto:1  Auto:2")
    for i, row_label in enumerate(["Rad:0", "Rad:1", "Rad:2"]):
        print(f"{row_label:12s} {cm[i][0]:6d}  {cm[i][1]:6d}  {cm[i][2]:6d}")

    print("\nInterpretation guide (Landis & Koch):")
    print("  <0.20 slight | 0.21-0.40 fair | 0.41-0.60 moderate | 0.61-0.80 substantial | >0.80 almost perfect")


if __name__ == "__main__":
    main()
