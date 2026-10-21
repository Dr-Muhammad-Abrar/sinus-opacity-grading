"""
Step 14 - Rebuild the paper's data figures directly from your result files.

Reads (all in outputs/labels):
    kfold_results.csv, resnet50_kfold_results.csv      per-fold scores (04b, 04c)
    cnn_predictions.csv                                ConvNeXt test predictions (04b)
    gradcam_iou_scores.csv                             per-image IoU (05)
    sinus_severity_labels.csv                          opacity scores + grades (02b)
    sad_ode_results.csv                                fitted beta, gamma (06)
    resnet50_test_predictions.csv                      OPTIONAL (04c does not write it by default)
Writes PNG figures to outputs/paper_figures/ and prints a CHECK of every number
against the values currently written in the paper, so any mismatch is visible.
Only numpy, pandas and matplotlib are used (no scipy / sklearn).
Figure 1 (pipeline drawing) is a schematic and is not rebuilt here.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = os.path.dirname(config.LABELS_CSV)
OUT = os.path.join(config.OUTPUT_DIR, "paper_figures")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, RED, GRAY = "#2E86AB", "#E67E22", "#27AE60", "#C0392B", "#7F8C8D"
CHECKS = []


def check(name, value, expected, tol):
    ok = abs(value - expected) <= tol
    CHECKS.append((ok, name, value, expected))


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def macro_f1(y, p, labels=(0, 1, 2)):
    y, p = np.asarray(y), np.asarray(p)
    f = []
    for c in labels:
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f))


def confusion(y, p, labels=(0, 1, 2)):
    cm = np.zeros((3, 3), int)
    for t, q in zip(y, p): cm[labels.index(int(t)), labels.index(int(q))] += 1
    return cm


# ------------------------------------------------------------ load
cn = pd.read_csv(os.path.join(LAB, "kfold_results.csv"))
rn = pd.read_csv(os.path.join(LAB, "resnet50_kfold_results.csv"))
pred = pd.read_csv(os.path.join(LAB, "cnn_predictions.csv"))
iou = pd.read_csv(os.path.join(LAB, "gradcam_iou_scores.csv"))
lab = pd.read_csv(config.LABELS_CSV)
ode = pd.read_csv(os.path.join(LAB, "sad_ode_results.csv")).iloc[0]

# ------------------------------------------------------------ numbers
cn_acc, cn_f1 = cn["accuracy"].to_numpy(), cn["macro_f1"].to_numpy()
rn_acc, rn_f1 = rn["accuracy"].to_numpy(), rn["macro_f1"].to_numpy()
check("ConvNeXt CV accuracy mean", cn_acc.mean(), 0.623, 0.0006); check("ConvNeXt CV accuracy SD", cn_acc.std(ddof=1), 0.120, 0.0006)
check("ConvNeXt CV macro-F1 mean", cn_f1.mean(), 0.600, 0.0006); check("ConvNeXt CV macro-F1 SD", cn_f1.std(ddof=1), 0.149, 0.0006)
check("ResNet-50 CV accuracy mean", rn_acc.mean(), 0.595, 0.0006); check("ResNet-50 CV accuracy SD", rn_acc.std(ddof=1), 0.149, 0.0006)
check("ResNet-50 CV macro-F1 mean", rn_f1.mean(), 0.584, 0.0006); check("ResNet-50 CV macro-F1 SD", rn_f1.std(ddof=1), 0.164, 0.0006)

d = cn_acc - rn_acc
t_corr = d.mean() / (d.std(ddof=1) * np.sqrt(1 / len(d) + 1 / 4))
check("Corrected paired t (accuracy)", t_corr, 0.34, 0.006)

y_cn, p_cn = pred["true_grade"].to_numpy(), pred["pred_grade"].to_numpy()
cm_cn = confusion(y_cn, p_cn)
k_cn, n_cn = int((y_cn == p_cn).sum()), len(y_cn)
check("ConvNeXt test accuracy", k_cn / n_cn, 0.727, 0.001); check("ConvNeXt test macro-F1", macro_f1(y_cn, p_cn), 0.726, 0.001)
check("Test images", n_cn, 11, 0)

rn_pred_path = os.path.join(LAB, "resnet50_test_predictions.csv")
if os.path.exists(rn_pred_path):
    rp = pd.read_csv(rn_pred_path); cm_rn = confusion(rp["true_grade"], rp["pred_grade"]); rn_src = "from resnet50_test_predictions.csv"
else:
    cm_rn = np.array([[4, 0, 0], [0, 3, 1], [0, 0, 3]]); rn_src = "TYPED from the console output (04c does not save predictions)"
k_rn, n_rn = int(np.trace(cm_rn)), int(cm_rn.sum())
check("ResNet-50 test accuracy", k_rn / n_rn, 0.909, 0.001)

v = iou["iou_with_sinus_box"].to_numpy(); ok_mask = iou["correct"].astype(bool).to_numpy()
rng = np.random.RandomState(0); boots = [rng.choice(v, len(v)).mean() for _ in range(10000)]
ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
check("IoU mean", v.mean(), 0.167, 0.0006); check("IoU CI low", ci_lo, 0.113, 0.003); check("IoU CI high", ci_hi, 0.224, 0.003)
check("IoU correct mean", v[ok_mask].mean(), 0.179, 0.0006); check("IoU incorrect mean", v[~ok_mask].mean(), 0.135, 0.0006)

mi = lab["mean_intensity"].to_numpy()
p33, p66 = np.percentile(mi, [33, 66])
tr = lab[lab["split"] == "train"].copy(); tr["pair"] = (tr["image_id"] + 1) // 2
rows = []
for _, g in tr.groupby("pair"):
    if len(g) == 2:
        g = g.sort_values("image_id"); a, b = g.iloc[0], g.iloc[1]
        rows.append((a.image_id, b.image_id, int(a.grade), int(b.grade), a.mean_intensity, b.mean_intensity))
pairs = pd.DataFrame(rows, columns=["id_a", "id_b", "grade_a", "grade_b", "mi_a", "mi_b"])
pairs["d"] = pairs["mi_b"] - pairs["mi_a"]; pairs["flip"] = pairs["grade_a"] != pairs["grade_b"]
absd = pairs["d"].abs(); med = absd.median()
sw2 = (pairs["d"] ** 2).mean() / 2; pm = (pairs["mi_a"] + pairs["mi_b"]) / 2
sb2 = max(pm.var(ddof=1) - sw2 / 2, 0); icc = sb2 / (sb2 + sw2)
dist = np.minimum(np.abs(mi - p33), np.abs(mi - p66))
near = [100 * float((dist <= m * med).mean()) for m in (1, 2, 3)]
check("Twin pairs", len(pairs), 71, 0); check("Pairs with different grades", int(pairs["flip"].sum()), 28, 0)
check("Median |copy difference|", med, 10.49, 0.006); check("90th pct |copy difference|", absd.quantile(0.9), 21.79, 0.006)
check("Cutoff P33", p33, 109.50, 0.006); check("Cutoff P66", p66, 125.52, 0.006); check("Middle band width", p66 - p33, 16.02, 0.012)
check("SD of score over images", mi.std(), 20.97, 0.006); check("ICC", icc, 0.820, 0.0006)
check("% images within 1 x median of cutoff", near[0], 66, 0.6); check("% within 2 x", near[1], 82, 0.6); check("% within 3 x", near[2], 93, 0.6)

# ------------------------------------------------------------ Figure 2
fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.8))
names = ["Median copy-to-copy\ndifference", "90th percentile\ncopy-to-copy diff.", "Width of the\nmiddle grade band", "SD of score\nacross images"]
vals = [med, absd.quantile(0.9), p66 - p33, mi.std()]
a.barh(names[::-1], vals[::-1], color=[RED, RED, BLUE, GRAY][::-1])
for i, x in enumerate(vals[::-1]): a.text(x + 0.4, i, f"{x:.1f}", va="center", fontsize=9)
a.set_xlabel("Brightness-score units"); a.set_xlim(0, max(vals) * 1.25)
a.set_title("(a) Noise between two copies of one X-ray\nvs. the width of the grades", fontsize=10)
b.bar(["1 x", "2 x", "3 x"], near, color=ORANGE)
for i, x in enumerate(near): b.text(i, x + 1.5, f"{x:.0f}%", ha="center", fontsize=9)
b.set_ylim(0, 105); b.set_ylabel(f"% of the {len(lab)} images")
b.set_xlabel(f"Distance to the nearest grade cutoff,\nin multiples of the median copy difference ({med:.1f})")
b.set_title("(b) Images close enough to a cutoff\nto change grade between copies", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_label_stability.png"), dpi=200); plt.close()

# ------------------------------------------------------------ Figure 2b (real twin pairs)
fig, ax = plt.subplots(figsize=(5.2, 4.8))
same = ~pairs["flip"]
ax.scatter(pairs.loc[same, "mi_a"], pairs.loc[same, "mi_b"], s=26, color=BLUE, label=f"Same grade ({int(same.sum())} pairs)")
ax.scatter(pairs.loc[~same, "mi_a"], pairs.loc[~same, "mi_b"], s=26, color=RED, label=f"Different grades ({int((~same).sum())} pairs)")
lo, hi = min(mi.min(), pairs[["mi_a", "mi_b"]].min().min()) - 3, max(mi.max(), pairs[["mi_a", "mi_b"]].max().max()) + 3
ax.plot([lo, hi], [lo, hi], color="#999", lw=1, ls=":")
for c in (p33, p66):
    ax.axvline(c, color="k", lw=0.8, ls="--"); ax.axhline(c, color="k", lw=0.8, ls="--")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel("Opacity score, first copy"); ax.set_ylabel("Opacity score, second copy")
ax.set_title("Twin pairs: score of the two copies of one X-ray\n(dashed lines = grade cutoffs)", fontsize=10); ax.legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2b_twin_scatter.png"), dpi=200); plt.close()

# ------------------------------------------------------------ Figure 3
fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
f = cn["fold"].to_numpy()
for x, p, q in zip(f, cn_acc, rn_acc): a.plot([x, x], [p, q], color="#bbb", lw=1.2, zorder=0)
a.plot(f, cn_acc, "o-", color=BLUE, lw=2, ms=7, label="ConvNeXt-Tiny"); a.plot(rn["fold"], rn_acc, "s-", color=ORANGE, lw=2, ms=7, label="ResNet-50")
a.set_xticks(f); a.set_xlabel("Fold (n = " + ", ".join(str(int(x)) for x in cn["n_val"]) + ")"); a.set_ylabel("Accuracy")
a.set_title(f"(a) Per-fold accuracy (corrected paired t = {t_corr:.2f})", fontsize=10); a.legend(); a.grid(axis="y", alpha=.25)
a.set_ylim(min(cn_acc.min(), rn_acc.min()) - 0.1, max(cn_acc.max(), rn_acc.max()) + 0.1)
x = np.arange(2); w = 0.34
te = [k_cn / n_cn, k_rn / n_rn]; ci = [wilson(k_cn, n_cn), wilson(k_rn, n_rn)]
b.bar(x - w / 2, [cn_acc.mean(), rn_acc.mean()], w, yerr=[cn_acc.std(ddof=1), rn_acc.std(ddof=1)], capsize=4, color=BLUE, label="CV mean $\\pm$ SD")
b.bar(x + w / 2, te, w, yerr=[[te[i] - ci[i][0] for i in range(2)], [ci[i][1] - te[i] for i in range(2)]], capsize=4, color=GREEN, label=f"Test (n = {n_cn}), 95% Wilson CI")
b.set_xticks(x); b.set_xticklabels(["ConvNeXt-Tiny", "ResNet-50"]); b.set_ylim(0, 1.35); b.set_yticks([0, .2, .4, .6, .8, 1.0])
b.set_ylabel("Accuracy"); b.set_title("(b) Cross-validation and test accuracy", fontsize=10); b.legend(fontsize=8, loc="upper center"); b.grid(axis="y", alpha=.25)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_models.png"), dpi=200); plt.close()

# ------------------------------------------------------------ Figure 4
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.9)); labs = ["Low", "Interm.", "High"]
for ax_, (t, cm) in zip(axs, {f"ConvNeXt-Tiny ({k_cn}/{n_cn} correct)": cm_cn, f"ResNet-50 ({k_rn}/{n_rn} correct)": cm_rn}.items()):
    ax_.imshow(cm, cmap="Blues", vmin=0, vmax=max(4, cm.max()))
    ax_.set_xticks(range(3)); ax_.set_yticks(range(3)); ax_.set_xticklabels(labs); ax_.set_yticklabels(labs)
    ax_.set_xlabel("Predicted"); ax_.set_ylabel("True (opacity tertile)"); ax_.set_title(t, fontsize=10)
    for i in range(3):
        for j in range(3):
            ax_.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] >= 3 else "black", fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig4_confusion.png"), dpi=200); plt.close()

# ------------------------------------------------------------ Figure 5
fig, ax = plt.subplots(figsize=(8.2, 3.7))
ax.axhspan(ci_lo, ci_hi, color="#ddd", alpha=.6, label=f"95% bootstrap CI of the mean ({ci_lo:.2f}-{ci_hi:.2f})")
ax.bar(range(len(v)), v, color=[BLUE if o else ORANGE for o in ok_mask])
ax.axhline(v.mean(), color="k", lw=1.2, ls="--", label=f"Mean IoU = {v.mean():.3f}")
ax.set_xticks(range(len(v))); ax.set_xticklabels([str(i) for i in iou["image_id"]], rotation=45, fontsize=8)
ax.set_xlabel("Test image id (blue = correctly classified, orange = misclassified)"); ax.set_ylabel("IoU with annotated box")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig5_iou.png"), dpi=200); plt.close()

# ------------------------------------------------------------ Figure A1 (pure NumPy RK4)
beta, gamma = float(ode["beta_fit"]), float(ode["gamma_fit"])
def rhs(y): S, A, D = y; return np.array([-beta * S * A, beta * S * A - gamma * A, gamma * A])
n = 2000; t = np.linspace(0, 1, n); y = np.array([0.99, 0.01, 0.0]); traj = np.zeros((3, n)); traj[:, 0] = y
for i in range(1, n):
    h = t[i] - t[i - 1]; k1 = rhs(y); k2 = rhs(y + h / 2 * k1); k3 = rhs(y + h / 2 * k2); k4 = rhs(y + h * k3)
    y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); traj[:, i] = y
fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.plot(t, traj[0], lw=2.2, color=BLUE, label="S (low opacity)"); ax.plot(t, traj[1], lw=2.2, color=ORANGE, label="A (intermediate)")
ax.plot(t, traj[2], lw=2.2, color=RED, label="D (high opacity)")
ax.set_xlabel("Pseudo-time (arbitrary unit)"); ax.set_ylabel("Proportion of the sample")
ax.set_title(f"Fitted SAD curve ($\\beta$ = {beta:.2f}, $\\gamma$ = {gamma:.2f})", fontsize=10); ax.legend(); ax.grid(alpha=.25)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figA1_sad.png"), dpi=200); plt.close()
check("SAD end state S", traj[0, -1], 0.331, 0.002); check("SAD end state D", traj[2, -1], 0.340, 0.002)

# ------------------------------------------------------------ report
print(f"Figures written to: {OUT}")
print(f"ResNet-50 confusion matrix: {rn_src}\n")
print("CHECK against the numbers currently in the paper:")
bad = 0
for ok, name, val, exp in CHECKS:
    print(f"  {'OK  ' if ok else 'DIFF'}  {name}: file gives {val:.4g}, paper says {exp:g}")
    bad += (not ok)
print(f"\n{len(CHECKS) - bad} of {len(CHECKS)} numbers match." + ("" if not bad else f"  {bad} DIFFER - tell me which, and I will correct the paper."))
