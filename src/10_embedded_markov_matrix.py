"""
Stage 10 — Embedded discrete-time Markov transition matrix.

Derives a discrete-time transition probability matrix from the already
-fitted beta, gamma -- no new data, no new fitting, purely a
reformulation of the same calibrated dynamics into a more clinically
readable form ("X% chance of moving from Mild to Severe per pseudo-
time unit").

IMPORTANT MATHEMATICAL CAVEAT (read before writing this into the
paper): the SAD model's S -> A transition rate is beta*A(t), which
depends on the co-evolving population state A(t) -- this is a
frequency-dependent (mean-field) rate, NOT a constant per-individual
hazard. A textbook embedded Markov chain (P = exp(Q * dt) for a fixed
generator Q) is only exactly valid for CONSTANT transition rates. So:

  - The A -> D transition IS exactly constant-rate (dD/dt = gamma*A
    means every Affected individual has a constant hazard gamma of
    becoming Diseased, independent of population state) -- its
    embedded transition probability below is EXACT.

  - The S -> A transition is only approximately constant-rate. Here it
    is approximated using the time-averaged force of transition,
    beta * mean(A(t)) over the calibrated pseudo-time window -- this
    is a mean-field approximation, not an exact embedding, and must be
    described as such in the paper. Do not present this cell as exact.

Saves:
    outputs/labels/embedded_markov_matrix.csv
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
import pandas as pd

T_FINAL = 1.0
EPSILON = 0.01
DT = 1.0  # one full pseudo-time unit per step, matching the calibrated window


def sad_rhs(y, beta, gamma):
    S, A, D = y
    return np.array([-beta*S*A, beta*S*A - gamma*A, gamma*A])


def rk4_trajectory(beta, gamma, n_steps=500, t_final=T_FINAL, epsilon=EPSILON):
    t_grid = np.linspace(0, t_final, n_steps)
    y = np.array([1 - epsilon, epsilon, 0.0])
    out = np.zeros((3, n_steps))
    out[:, 0] = y
    for i in range(1, n_steps):
        dt = t_grid[i] - t_grid[i-1]
        k1 = sad_rhs(y, beta, gamma)
        k2 = sad_rhs(y + dt/2*k1, beta, gamma)
        k3 = sad_rhs(y + dt/2*k2, beta, gamma)
        k4 = sad_rhs(y + dt*k3, beta, gamma)
        y = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        out[:, i] = y
    return t_grid, out


def main():
    ode_results = pd.read_csv(os.path.join(os.path.dirname(config.LABELS_CSV), "sad_ode_results.csv")).iloc[0]
    beta, gamma = ode_results["beta_fit"], ode_results["gamma_fit"]
    print(f"Using fitted beta={beta:.4f}, gamma={gamma:.4f}")

    t_grid, trajectory = rk4_trajectory(beta, gamma)
    A_t = trajectory[1]
    A_mean = float(np.sum((A_t[:-1] + A_t[1:]) / 2 * np.diff(t_grid))) / T_FINAL  # manual trapezoidal rule
    print(f"Time-averaged A(t) over the calibrated window: {A_mean:.4f}")

    lambda_SA = beta * A_mean  # mean-field approximation, see docstring
    lambda_AD = gamma          # exact constant-rate

    P_SA = 1 - np.exp(-lambda_SA * DT)
    P_SS = 1 - P_SA
    P_AD = 1 - np.exp(-lambda_AD * DT)
    P_AA = 1 - P_AD

    matrix = pd.DataFrame(
        [[P_SS, P_SA, 0.0], [0.0, P_AA, P_AD], [0.0, 0.0, 1.0]],
        index=["From Normal (S)", "From Mild (A)", "From Moderate/severe (D)"],
        columns=["To Normal (S)", "To Mild (A)", "To Moderate/severe (D)"],
    )

    print("\nEmbedded discrete-time transition matrix (one pseudo-time unit):")
    print(matrix.round(4))
    print(f"\nP(Normal -> Mild) = {P_SA:.4f}  [mean-field approximation, see docstring]")
    print(f"P(Mild -> Moderate/severe) = {P_AD:.4f}  [exact, constant-rate]")
    print("P(Moderate/severe -> Moderate/severe) = 1.0000  [absorbing state, by model construction]")

    out_csv = os.path.join(os.path.dirname(config.LABELS_CSV), "embedded_markov_matrix.csv")
    matrix.to_csv(out_csv)
    print(f"\nSaved matrix to: {out_csv}")


if __name__ == "__main__":
    main()