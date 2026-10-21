"""
Stage 6 — SAD ODE fitting and mathematical validation (multi-split version).

Model:
    dS/dt = -beta * S * A
    dA/dt =  beta * S * A - gamma * A
    dD/dt =  gamma * A

WHY THIS VERSION IS DIFFERENT FROM A SINGLE POOLED FIT:
    Fitting (beta, gamma) to match ONE pooled population snapshot is an
    exactly-determined problem (2 parameters, 2 independent constraints,
    since S+A+D=1) — least squares can drive the residual to exactly
    zero almost regardless of the data, which is a mathematical
    inevitability, not evidence of a good model.

    This version instead fits (beta, gamma) so that the SAME simulated end-state is as
    close as possible to the proportions of the THREE splits (train, valid, test) at
    once. NOTE: minimizing squared error against one common end-state returns the
    unweighted MEAN of the three proportion vectors, so the fit still reproduces those
    mean proportions exactly. The reported RMSE and R^2 therefore measure only how
    much the splits differ from each other (sampling variability), NOT how well the
    ODE describes the data. In addition, the grades are tertiles, so each class holds
    about one third of the images by construction.

    Caveat to state plainly in your paper: train/valid/test are not
    independent clinical populations — they're three random subsamples
    of the same underlying dataset (and the automated labels were
    thresholded on POOLED percentiles, so the three splits are expected
    to look fairly similar by construction). This check therefore
    mainly validates whether the calibrated point is consistent with
    ordinary sampling variability across subsamples of one dataset — a
    real, independent population (a different clinic, a different
    dataset) would be a stronger test than what's shown here. Say this
    explicitly rather than overselling the R^2.

Saves:
    outputs/figures/sad_trajectory.png
    outputs/figures/sad_loss_surface.png
    outputs/labels/sad_ode_results.csv
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

T_FINAL = 1.0
EPSILON = 0.01
N_BOOTSTRAP = 1000
SEED = 42


def sad_rhs(t, y, beta, gamma):
    S, A, D = y
    dS = -beta * S * A
    dA = beta * S * A - gamma * A
    dD = gamma * A
    return [dS, dA, dD]


def simulate(beta, gamma, t_final=T_FINAL, epsilon=EPSILON):
    y0 = [1 - epsilon, epsilon, 0.0]
    sol = solve_ivp(sad_rhs, [0, t_final], y0, args=(beta, gamma),
                     dense_output=True, rtol=1e-8, atol=1e-10)
    S, A, D = sol.y[:, -1]
    return S, A, D, sol


def residuals_multi(params, empirical_list):
    """Residuals against MULTIPLE empirical proportion sets (one per
    split), all compared to the SAME simulated end-state -- this is
    what makes the system over-determined."""
    beta, gamma = params
    if beta <= 0 or gamma <= 0:
        return np.array([1e3] * (3 * len(empirical_list)))
    S, A, D = simulate(beta, gamma)[:3]
    simulated = np.array([S, A, D])
    all_residuals = []
    for emp in empirical_list:
        all_residuals.extend(simulated - np.array(emp))
    return np.array(all_residuals)


def fit_params_multi(empirical_list, x0=(1.0, 1.0)):
    result = least_squares(residuals_multi, x0=x0, args=(empirical_list,), bounds=(1e-4, 50))
    return result


def load_split_proportions():
    df = pd.read_csv(config.LABELS_CSV)
    proportions_by_split = {}
    for split in ["train", "valid", "test"]:
        sub = df[df["split"] == split]
        counts = sub["grade"].astype(str).value_counts().reindex(["0", "1", "2"], fill_value=0)
        total = counts.sum()
        proportions_by_split[split] = (counts / total).values
        print(f"{split} (n={total}): Normal={proportions_by_split[split][0]:.3f}, "
              f"Mild={proportions_by_split[split][1]:.3f}, "
              f"Moderate/severe={proportions_by_split[split][2]:.3f}")
    return proportions_by_split, df


def bootstrap_ci_multi(df, n=N_BOOTSTRAP):
    """Stratified bootstrap: resample WITHIN each split separately, so
    the resampled data preserves the same 3-split structure each time."""
    rng = np.random.RandomState(SEED)
    grades_by_split = {s: df[df["split"] == s]["grade"].astype(str).values
                        for s in ["train", "valid", "test"]}

    betas, gammas = [], []
    for _ in range(n):
        emp_list = []
        for split, grades in grades_by_split.items():
            sample = rng.choice(grades, size=len(grades), replace=True)
            counts = pd.Series(sample).value_counts().reindex(["0", "1", "2"], fill_value=0)
            emp_list.append((counts / len(grades)).values)
        try:
            res = fit_params_multi(emp_list)
            if res.success:
                betas.append(res.x[0])
                gammas.append(res.x[1])
        except Exception:
            continue

    return np.array(betas), np.array(gammas)


def check_conservation(beta, gamma):
    _, _, _, sol = simulate(beta, gamma)
    totals = sol.y.sum(axis=0)
    max_deviation = np.max(np.abs(totals - 1.0))
    print(f"Conservation check (S+A+D should equal 1 at every t): "
          f"max deviation = {max_deviation:.2e}")
    return max_deviation


def analyze_equilibria(beta, gamma, S0):
    R0 = beta * S0 / gamma
    eigenvalue_at_start = beta * S0 - gamma
    print("\n===== Equilibrium and stability analysis =====")
    print("Equilibria: the continuum of points (S, 0, D) with S+D=1 "
          "(any point where A=0)")
    print(f"Linearized eigenvalue in the A-direction at t=0: {eigenvalue_at_start:.4f}")
    print(f"R0-style threshold (beta*S0/gamma): {R0:.4f}")
    if R0 > 1:
        print("R0 > 1: A initially grows before settling to a new A=0 equilibrium "
              "(epidemic-like spread through the population).")
    else:
        print("R0 <= 1: A decays monotonically from the start, no initial growth phase.")
    return R0


def goodness_of_fit_multi(beta, gamma, empirical_list, split_names):
    S, A, D = simulate(beta, gamma)[:3]
    simulated = np.array([S, A, D])
    print(f"\nSimulated end-state (same for all splits): S={S:.3f}, A={A:.3f}, D={D:.3f}")

    all_residuals = []
    for name, emp in zip(split_names, empirical_list):
        residual = simulated - np.array(emp)
        all_residuals.extend(residual)
        print(f"  {name}: empirical=({emp[0]:.3f}, {emp[1]:.3f}, {emp[2]:.3f}) "
              f"| residual=({residual[0]:+.3f}, {residual[1]:+.3f}, {residual[2]:+.3f})")

    all_residuals = np.array(all_residuals)
    rmse = np.sqrt(np.mean(all_residuals ** 2))
    all_empirical = np.concatenate(empirical_list)
    ss_res = np.sum(all_residuals ** 2)
    ss_tot = np.sum((all_empirical - np.mean(all_empirical)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"\nOverall (across all 3 splits, 9 values / 6 independent): RMSE={rmse:.4f}, R^2={r2:.4f}")
    print("(Note: the fit reproduces the mean of the split proportions, so RMSE and R^2 only "
          "measure split-to-split differences, not model adequacy.)")
    return rmse, r2


def plot_trajectory(beta, gamma, save_path):
    _, _, _, sol = simulate(beta, gamma)
    t_eval = np.linspace(0, T_FINAL, 200)
    y = sol.sol(t_eval)

    plt.figure(figsize=(7, 5))
    plt.plot(t_eval, y[0], label="S (Normal)", linewidth=2)
    plt.plot(t_eval, y[1], label="A (Mild)", linewidth=2)
    plt.plot(t_eval, y[2], label="D (Moderate/severe)", linewidth=2)
    plt.xlabel("Pseudo-time (arbitrary unit, population-level calibration)")
    plt.ylabel("Population proportion")
    plt.title(f"Fitted SAD trajectory (beta={beta:.3f}, gamma={gamma:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved trajectory plot to: {save_path}")


def plot_loss_surface(empirical_list, beta_fit, gamma_fit, save_path):
    beta_range = np.linspace(max(0.01, beta_fit * 0.2), beta_fit * 3, 40)
    gamma_range = np.linspace(max(0.01, gamma_fit * 0.2), gamma_fit * 3, 40)
    loss_grid = np.zeros((len(gamma_range), len(beta_range)))

    for i, g in enumerate(gamma_range):
        for j, b in enumerate(beta_range):
            r = residuals_multi([b, g], empirical_list)
            loss_grid[i, j] = np.sum(r ** 2)

    plt.figure(figsize=(7, 5))
    cp = plt.contourf(beta_range, gamma_range, np.log10(loss_grid + 1e-10), levels=30, cmap="viridis")
    plt.colorbar(cp, label="log10(sum squared residuals)")
    plt.scatter([beta_fit], [gamma_fit], color="red", marker="x", s=100, label="Fitted (beta, gamma)")
    plt.xlabel("beta")
    plt.ylabel("gamma")
    plt.title("Loss surface (multi-split) -- a sharp, isolated minimum supports identifiability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved loss surface plot to: {save_path}")


def main():
    proportions_by_split, df = load_split_proportions()
    split_names = ["train", "valid", "test"]
    empirical_list = [proportions_by_split[s] for s in split_names]

    result = fit_params_multi(empirical_list)
    beta_fit, gamma_fit = result.x
    print(f"\nFitted parameters (joint across 3 splits): beta={beta_fit:.4f}, gamma={gamma_fit:.4f}")

    check_conservation(beta_fit, gamma_fit)
    rmse, r2 = goodness_of_fit_multi(beta_fit, gamma_fit, empirical_list, split_names)
    R0 = analyze_equilibria(beta_fit, gamma_fit, S0=1 - EPSILON)

    print("\nRunning stratified bootstrap (this may take a little while)...")
    betas, gammas = bootstrap_ci_multi(df)
    beta_ci = np.percentile(betas, [2.5, 97.5])
    gamma_ci = np.percentile(gammas, [2.5, 97.5])
    correlation = np.corrcoef(betas, gammas)[0, 1]

    print(f"\n===== Bootstrap results (n={len(betas)} successful fits) =====")
    print(f"beta:  {np.mean(betas):.4f}  (95% CI: {beta_ci[0]:.4f} - {beta_ci[1]:.4f})")
    print(f"gamma: {np.mean(gammas):.4f}  (95% CI: {gamma_ci[0]:.4f} - {gamma_ci[1]:.4f})")
    print(f"Correlation between beta and gamma estimates: {correlation:.3f}")
    if abs(correlation) > 0.9:
        print("WARNING: beta and gamma are highly correlated -- weak identifiability. "
              "Report this honestly.")
    else:
        print("Correlation is moderate -- parameters appear reasonably identifiable.")

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plot_trajectory(beta_fit, gamma_fit, os.path.join(config.FIGURES_DIR, "sad_trajectory.png"))
    plot_loss_surface(empirical_list, beta_fit, gamma_fit,
                       os.path.join(config.FIGURES_DIR, "sad_loss_surface.png"))

    out_csv = os.path.join(os.path.dirname(config.LABELS_CSV), "sad_ode_results.csv")
    pd.DataFrame([{
        "beta_fit": beta_fit, "gamma_fit": gamma_fit,
        "beta_ci_low": beta_ci[0], "beta_ci_high": beta_ci[1],
        "gamma_ci_low": gamma_ci[0], "gamma_ci_high": gamma_ci[1],
        "beta_gamma_correlation": correlation,
        "R0": R0, "rmse": rmse, "r2": r2,
        "train_S": proportions_by_split["train"][0], "train_A": proportions_by_split["train"][1],
        "train_D": proportions_by_split["train"][2],
        "valid_S": proportions_by_split["valid"][0], "valid_A": proportions_by_split["valid"][1],
        "valid_D": proportions_by_split["valid"][2],
        "test_S": proportions_by_split["test"][0], "test_A": proportions_by_split["test"][1],
        "test_D": proportions_by_split["test"][2],
    }]).to_csv(out_csv, index=False)
    print(f"\nSaved all results to: {out_csv}")


if __name__ == "__main__":
    main()