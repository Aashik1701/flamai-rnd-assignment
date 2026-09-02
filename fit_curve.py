"""
Recover (theta, M, X) for the parametric curve

    x(t) = t*cos(theta) - exp(M*|t|)*sin(0.3t)*sin(theta) + X
    y(t) = 42 + t*sin(theta) + exp(M*|t|)*sin(0.3t)*cos(theta)      , 6 < t < 60

from an unordered point cloud (xy_data.csv) sampled from the curve.

Method
------
The CSV rows are NOT in t-order (checked: adjacent rows jump by O(10) units,
but the curve's own speed over a uniform 1500-point t-grid in [6,60] is
O(0.03) -- so any pointwise t-indexed fit is meaningless). This is really a
"fit a curve to an unordered point cloud" problem, so instead:

1. For candidate parameters, densely sample the model curve over t in [6,60].
2. Score the candidate with a Chamfer-L1 distance: for every data point, the
   L1 distance to its nearest point on the dense candidate curve, averaged.
   (scipy's cKDTree with p=1 gives exact-nearest-neighbour L1 lookup, so this
   is cheap even for dense curves.)
3. Minimize that score with a bounded global optimizer (differential
   evolution) over the given parameter ranges, then polish locally
   (Nelder-Mead) with a much denser curve sample for precision.
4. Report theta/M/X, the achieved L1 fit error, and a Desmos-style
   parametric expression (per the assignment's required submission format).
5. Plot the data cloud against the recovered curve for visual verification.
"""

import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- data ----
DATA_PATH = "xy_data.csv"
T_LO, T_HI = 6.0, 60.0
THETA_LO_DEG, THETA_HI_DEG = 0.0, 50.0
M_LO, M_HI = -0.05, 0.05
X_LO, X_HI = 0.0, 100.0


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df[["x", "y"]].to_numpy(dtype=float)


# --------------------------------------------------------------- model ----
def curve_xy(t, theta, M, X):
    """Vectorized model curve. theta in radians."""
    env = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(theta) - env * np.sin(theta) + X
    y = 42.0 + t * np.sin(theta) + env * np.cos(theta)
    return x, y


def sample_curve(theta, M, X, n):
    t = np.linspace(T_LO, T_HI, n)
    x, y = curve_xy(t, theta, M, X)
    return np.column_stack([x, y])


# ------------------------------------------------------------- scoring ----
def chamfer_l1(params, data, n_curve):
    theta, M, X = params
    curve_pts = sample_curve(theta, M, X, n_curve)
    tree = cKDTree(curve_pts)
    dist, _ = tree.query(data, k=1, p=1)  # exact L1 nearest-neighbour distance
    return dist.mean()


def make_objective(data, n_curve):
    def obj(params):
        return chamfer_l1(params, data, n_curve)
    return obj


# -------------------------------------------------------------- fitting ----
def fit(data):
    bounds = [
        (np.radians(THETA_LO_DEG), np.radians(THETA_HI_DEG)),
        (M_LO, M_HI),
        (X_LO, X_HI),
    ]

    t0 = time.time()
    # Stage 1: bounded global search (coarse curve sampling -> fast).
    de_result = differential_evolution(
        make_objective(data, n_curve=1500),
        bounds=bounds,
        popsize=25,
        maxiter=150,
        tol=1e-10,
        mutation=(0.4, 1.5),
        recombination=0.8,
        seed=0,
        polish=False,
    )
    t1 = time.time()

    # Stage 2: local polish with a much denser curve sample for precision.
    nm_result = minimize(
        make_objective(data, n_curve=8000),
        x0=de_result.x,
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 4000, "maxfev": 4000},
    )
    t2 = time.time()

    theta, M, X = nm_result.x
    final_score = chamfer_l1(nm_result.x, data, n_curve=20000)

    return {
        "theta_rad": theta,
        "theta_deg": np.degrees(theta),
        "M": M,
        "X": X,
        "de_score": de_result.fun,
        "nm_score": nm_result.fun,
        "final_l1_chamfer": final_score,
        "de_time_s": t1 - t0,
        "nm_time_s": t2 - t1,
    }


# ----------------------------------------------------------- reporting ----
def desmos_expression(theta, M, X):
    return (
        f"\\left(t*\\cos({theta:.6f})-e^{{{M:.6f}\\left|t\\right|}}\\cdot"
        f"\\sin(0.3t)\\sin({theta:.6f})+{X:.6f},42+"
        f"t*\\sin({theta:.6f})+e^{{{M:.6f}\\left|t\\right|}}\\cdot"
        f"\\sin(0.3t)\\cos({theta:.6f})\\right)"
    )


def plot_fit(data, theta, M, X, l1_score, out_path="fit_result.png"):
    fitted = sample_curve(theta, M, X, 6000)

    fig, axes = plt.subplots(1, 2, figsize=(22, 11))

    ax = axes[0]
    ax.scatter(data[:, 0], data[:, 1], s=14, c="steelblue", alpha=0.55,
               edgecolors="none", label=f"CSV data (n={len(data)})")
    ax.plot(fitted[:, 0], fitted[:, 1], c="crimson", lw=2.6, label="Recovered curve", zorder=3)
    ax.set_title("Fit overlay: recovered curve vs. supplied data", fontsize=16, pad=14)
    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("y", fontsize=13)
    ax.legend(fontsize=12, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=11)

    param_text = (
        "Recovered parameters\n"
        r"$\theta$" + f" = {np.degrees(theta):.4f}°  ({theta:.6f} rad)\n"
        f"M = {M:.6f}\n"
        f"X = {X:.6f}\n"
        f"\nmean L1 fit error = {l1_score:.5f}"
    )
    ax.text(
        0.02, 0.02, param_text, transform=ax.transAxes,
        fontsize=13, va="bottom", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="gray", alpha=0.92),
    )

    tree = cKDTree(fitted)
    dist, _ = tree.query(data, k=1, p=1)
    ax2 = axes[1]
    ax2.hist(dist, bins=60, color="darkorange", edgecolor="black", linewidth=0.3)
    ax2.set_title(f"Per-point L1 residual to fitted curve\nmean={dist.mean():.5f}, max={dist.max():.5f}",
                  fontsize=16, pad=14)
    ax2.set_xlabel("L1 distance", fontsize=13)
    ax2.set_ylabel("count", fontsize=13)
    ax2.tick_params(labelsize=11)

    eq_text = (
        r"$x(t)=t\cos\theta-e^{M|t|}\sin(0.3t)\sin\theta+X$" + "\n"
        r"$y(t)=42+t\sin\theta+e^{M|t|}\sin(0.3t)\cos\theta$" + "\n"
        f"6 < t < 60"
    )
    ax2.text(
        0.98, 0.98, eq_text, transform=ax2.transAxes,
        fontsize=12, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="gray", alpha=0.92),
    )

    fig.suptitle(
        f"θ = {np.degrees(theta):.4f}°   |   M = {M:.6f}   |   X = {X:.6f}",
        fontsize=19, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    print(f"[saved] {out_path}")


# ----------------------------------------------------------------- main ----
def main():
    data = load_data()
    print(f"Loaded {len(data)} points from {DATA_PATH}")

    result = fit(data)

    theta, M, X = result["theta_rad"], result["M"], result["X"]

    print("\n=== Recovered parameters ===")
    print(f"theta = {theta:.6f} rad = {result['theta_deg']:.4f} deg")
    print(f"M     = {M:.6f}")
    print(f"X     = {X:.6f}")
    print(f"\nDE stage score (coarse):  {result['de_score']:.6f}  ({result['de_time_s']:.1f}s)")
    print(f"NM polish score (dense):  {result['nm_score']:.6f}  ({result['nm_time_s']:.1f}s)")
    print(f"Final mean L1 Chamfer distance (data -> curve): {result['final_l1_chamfer']:.6f}")

    expr = desmos_expression(theta, M, X)
    print("\n=== Desmos / submission expression ===")
    print(expr)

    with open("fit_result.json", "w") as f:
        json.dump({**result, "desmos_expression": expr}, f, indent=2)
    print("\n[saved] fit_result.json")

    plot_fit(data, theta, M, X, result["final_l1_chamfer"])


if __name__ == "__main__":
    main()
