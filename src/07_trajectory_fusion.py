"""
Stage 7 — Trajectory fusion: per-patient continuous progression score.

For every image in the cleaned dataset (not just the test split), this:
  1. Runs the trained final ConvNeXt-Tiny model to get a predicted
     probability vector (P_Normal, P_Mild, P_Severe) per image.
  2. Reconstructs the fitted SAD trajectory S(t), A(t), D(t) using the
     beta/gamma already fitted in 06_ode_fitting.py.
  3. Finds, for each patient, the pseudo-time t* whose simulated state
     is closest to that patient's predicted probability vector --
     turning a 3-class label into a continuous position on the curve.
  4. Propagates parameter uncertainty into a rough band on t*, using
     the already-computed beta/gamma 95% CI bounds (an approximation,
     not a full re-propagated bootstrap -- stated explicitly).
  5. CRITICAL VALIDATION GATE: checks whether t* carries real
     information beyond the discrete grade, by computing the WITHIN-
     GRADE spread of t*. If patients within the same grade all get
     nearly identical t* values, the fusion isn't adding anything and
     should not be presented as a richer score in the paper.

Saves:
    outputs/labels/trajectory_fusion_results.csv
    outputs/figures/tstar_by_grade.png
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def sad_rhs(y, beta, gamma):
    S, A, D = y
    return np.array([-beta*S*A, beta*S*A - gamma*A, gamma*A])


def rk4_integrate(beta, gamma, t_grid, epsilon=0.01):
    """Fixed-step RK4 integrator, pure NumPy -- avoids scipy.integrate,
    whose import chain (via scipy.sparse._csparsetools) can be blocked
    by Windows Application Control policies on some locked-down
    machines. This 3-equation system is smooth and well-behaved, so a
    fixed-step RK4 at this grid density is accurate to well beyond the
    precision needed here."""
    y = np.array([1 - epsilon, epsilon, 0.0])
    n = len(t_grid)
    out = np.zeros((3, n))
    out[:, 0] = y
    for i in range(1, n):
        dt = t_grid[i] - t_grid[i-1]
        k1 = sad_rhs(y, beta, gamma)
        k2 = sad_rhs(y + dt/2*k1, beta, gamma)
        k3 = sad_rhs(y + dt/2*k2, beta, gamma)
        k4 = sad_rhs(y + dt*k3, beta, gamma)
        y = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        out[:, i] = y
    return out


def spearman_corr(x, y):
    """Pure-NumPy Spearman correlation (rank-based Pearson), with a
    normal-approximation p-value -- avoids scipy.stats, which can pull
    in the same blocked import chain."""
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
        # Normal approximation to the t-distribution p-value (fine for n this large)
        from math import erf, sqrt
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
    else:
        p_value = float("nan")
    return rho, p_value

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
T_GRID = np.linspace(0, 1.0, 500)  # fine grid to search for t*


def get_trajectory(beta, gamma, epsilon=0.01):
    return rk4_integrate(beta, gamma, T_GRID, epsilon)


class ImageListDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from PIL import Image
        path, split, image_id, grade = self.records[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), split, image_id, grade


def collect_all_roi_paths():
    records = []
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(config.ROI_DIR, split)
        for grade in os.listdir(split_dir):
            grade_dir = os.path.join(split_dir, grade)
            for fname in os.listdir(grade_dir):
                image_id = os.path.splitext(fname)[0]
                records.append((os.path.join(grade_dir, fname), split, image_id, int(grade)))
    return records


def find_tstar(prob_vec, trajectory):
    """Nearest point on the trajectory (Euclidean distance) to a
    predicted probability vector."""
    diffs = trajectory.T - np.array(prob_vec)  # (500, 3)
    dists = np.linalg.norm(diffs, axis=1)
    idx = np.argmin(dists)
    return T_GRID[idx], dists[idx]


def main():
    # ---- Load fitted ODE parameters ----
    ode_results = pd.read_csv(os.path.join(os.path.dirname(config.LABELS_CSV), "sad_ode_results.csv")).iloc[0]
    beta_fit, gamma_fit = ode_results["beta_fit"], ode_results["gamma_fit"]
    beta_lo, beta_hi = ode_results["beta_ci_low"], ode_results["beta_ci_high"]
    gamma_lo, gamma_hi = ode_results["gamma_ci_low"], ode_results["gamma_ci_high"]
    print(f"Using fitted beta={beta_fit:.4f}, gamma={gamma_fit:.4f}")

    traj_central = get_trajectory(beta_fit, gamma_fit)
    # Rough uncertainty band: extreme corners of the CI box (approximation, not full bootstrap propagation)
    traj_low = get_trajectory(beta_lo, gamma_hi)   # slower growth corner
    traj_high = get_trajectory(beta_hi, gamma_lo)  # faster growth corner

    # ---- Load trained model ----
    ckpt_path = os.path.join(config.MODELS_DIR, "convnext_tiny_final.pt")
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    records = collect_all_roi_paths()
    print(f"Running inference on {len(records)} images (full cleaned dataset)...")
    dataset = ImageListDataset(records, eval_tf)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    results = []
    with torch.no_grad():
        for imgs, splits, image_ids, grades in loader:
            imgs = imgs.to(DEVICE)
            probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
            for i in range(len(probs)):
                p = probs[i]
                t_star, dist = find_tstar(p, traj_central)
                t_low, _ = find_tstar(p, traj_low)
                t_high, _ = find_tstar(p, traj_high)
                severity_index = 1 * p[1] + 2 * p[2]  # continuous, independent of curve saturation
                results.append({
                    "split": splits[i], "image_id": image_ids[i].item() if hasattr(image_ids[i], "item") else image_ids[i],
                    "true_grade": grades[i].item(),
                    "prob_0": p[0], "prob_1": p[1], "prob_2": p[2],
                    "severity_index": severity_index,
                    "t_star": t_star, "t_star_band_low": min(t_low, t_high),
                    "t_star_band_high": max(t_low, t_high), "match_distance": dist,
                })

    df = pd.DataFrame(results)
    out_csv = os.path.join(os.path.dirname(config.LABELS_CSV), "trajectory_fusion_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} trajectory-matched scores to: {out_csv}")

    # ---- VALIDATION GATE (primary score: CNN-derived severity index) ----
    print("\n===== Validation: does the severity index add information beyond the discrete grade? =====")
    rho, p_value = spearman_corr(df["true_grade"], df["severity_index"])
    print(f"Spearman correlation, severity_index vs. true grade: rho={rho:.3f}, p={p_value:.4f}")

    grade_names = {0: "Normal", 1: "Mild", 2: "Moderate/severe"}
    print("\nWithin-grade spread of severity_index (primary check):")
    overall_meaningful = True
    for g in [0, 1, 2]:
        sub = df[df["true_grade"] == g]["severity_index"]
        spread = sub.std()
        print(f"  Grade {g} ({grade_names[g]}): n={len(sub)}, mean={sub.mean():.3f}, "
              f"std={spread:.3f}, range=[{sub.min():.3f}, {sub.max():.3f}]")
        if spread < 0.02:
            overall_meaningful = False

    print()
    if overall_meaningful:
        print("RESULT: Within-grade spread is present. These values are IN-SAMPLE (the final model "
              "was trained on most of these images) and the spread partly reflects unstable labels "
              "between near-identical copies. Treat the index as descriptive, not as a validated score.")
    else:
        print("RESULT: Even the raw CNN probabilities show collapsed spread within at least one "
              "grade -- in that case the model itself is very confident/saturated for that class, "
              "which is a finding about the classifier, not an artifact of curve matching.")

    print("\n----- Secondary diagnostic: curve-based t* (for reference only) -----")
    print("Note: t* is expected to collapse in saturated regions of the fitted trajectory "
          "(beta >> gamma causes fast saturation) -- this is a property of the fitted dynamics, "
          "not a bug, and is reported as a limitation rather than used as the primary score.")
    for g in [0, 1, 2]:
        sub = df[df["true_grade"] == g]["t_star"]
        print(f"  Grade {g} ({grade_names[g]}): t* std={sub.std():.3f}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for g in [0, 1, 2]:
        sub = df[df["true_grade"] == g]["severity_index"]
        jitter = np.random.normal(g, 0.05, size=len(sub))
        ax.scatter(jitter, sub, alpha=0.6, s=25, label=grade_names[g])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([grade_names[g] for g in [0, 1, 2]])
    ax.set_ylabel("Continuous severity index (0-2)")
    ax.set_title(f"CNN-derived continuous severity index by grade (Spearman rho={rho:.3f})")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig_path = os.path.join(config.FIGURES_DIR, "severity_index_by_grade.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nSaved diagnostic plot to: {fig_path}")


if __name__ == "__main__":
    main()