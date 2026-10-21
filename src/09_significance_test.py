"""
Stage 9 — Paired significance test: ConvNeXt-Tiny vs. ResNet-50.

Compares the two backbones using a PAIRED test across the 5 fold
scores (both models were evaluated on the identical fold splits, since
both scripts use the same SEED=42 stratified k-fold), which is the
correct test here -- an unpaired test would ignore that both models
saw exactly the same data in each fold.

IMPORTANT CAVEAT, stated plainly: n=5 (one score per fold) gives very
limited statistical power. This test can tell you whether the observed
difference is larger than you'd expect from fold-to-fold noise alone,
but it cannot rule out a real difference existing that this small a
sample simply can't detect. Report the result with that caveat, not as
a definitive answer.

No scipy/sklearn dependency (paired t-test implemented directly, with
a hardcoded critical-value table for df=4 rather than needing an
incomplete-beta p-value calculator).
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import pandas as pd

# Two-tailed critical t-values for df=4 (5 paired folds - 1)
T_CRITICAL = {"0.10": 2.132, "0.05": 2.776, "0.01": 4.604}


def paired_t_test(a, b, corrected=False):
    """Paired t-test on per-fold scores.
    corrected=False : ordinary paired t-test (assumes the 5 fold scores are independent).
    corrected=True  : Nadeau & Bengio (2003) corrected resampled t-test. CV folds share
                      training data, so their scores are NOT independent; the correction
                      widens the standard error by sqrt(1/k + n_test/n_train) (k=5, ratio 1/4)."""
    diffs = np.array(a) - np.array(b)
    n = len(diffs)
    mean_diff = diffs.mean()
    std_diff = diffs.std(ddof=1)
    if corrected:
        se = std_diff * np.sqrt(1.0 / n + 1.0 / 4.0)
    else:
        se = std_diff / np.sqrt(n)
    t_stat = mean_diff / se if se > 0 else float("inf")
    return t_stat, mean_diff, std_diff, n - 1


def verdict_for(t_stat):
    abs_t = abs(t_stat)
    if abs_t > T_CRITICAL["0.05"]:
        return "statistically significant at alpha=0.05"
    elif abs_t > T_CRITICAL["0.10"]:
        return "significant only at the weaker alpha=0.10 threshold"
    return "NOT statistically significant (alpha=0.10 or 0.05)"


def report(metric_name, convnext_scores, resnet_scores):
    t_naive, mean_diff, std_diff, dof = paired_t_test(convnext_scores, resnet_scores, corrected=False)
    t_corr, _, _, _ = paired_t_test(convnext_scores, resnet_scores, corrected=True)
    print(f"\n--- {metric_name} ---")
    print(f"ConvNeXt-Tiny per fold: {np.round(convnext_scores, 3).tolist()}")
    print(f"ResNet-50 per fold:     {np.round(resnet_scores, 3).tolist()}")
    print(f"Mean difference (ConvNeXt - ResNet50): {mean_diff:+.4f} (SD of differences: {std_diff:.4f})")
    print(f"Ordinary paired t (assumes independent folds): t={t_naive:.3f}, df={dof} -> {verdict_for(t_naive)}")
    print(f"Corrected t (Nadeau-Bengio, USE THIS ONE):      t={t_corr:.3f}, df={dof} -> {verdict_for(t_corr)}")
    print("Caveat: only 5 folds, and twin pairs mean fewer independent images than the "
          "fold sizes suggest, so power is very low. 'Not significant' does NOT prove the "
          "models are equal; it only means no difference is detectable here.")
    return t_corr, verdict_for(t_corr)


def main():
    labels_dir = os.path.dirname(config.LABELS_CSV)
    convnext_df = pd.read_csv(os.path.join(labels_dir, "kfold_results.csv"))
    resnet_df = pd.read_csv(os.path.join(labels_dir, "resnet50_kfold_results.csv"))

    print("Paired significance test: ConvNeXt-Tiny vs. ResNet-50")
    print("(paired on identical fold splits, SEED=42 stratified k-fold)")

    report("Accuracy", convnext_df["accuracy"].values, resnet_df["accuracy"].values)
    report("Macro F1", convnext_df["macro_f1"].values, resnet_df["macro_f1"].values)


if __name__ == "__main__":
    main()
