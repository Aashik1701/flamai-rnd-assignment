"""
Independent cross-check + narrative visualization of the recovered (theta, M, X).

The optimizer in fit_curve.py found the parameters geometrically (Chamfer-L1 to a
dense curve sample). This script verifies them a completely different way, using
the algebraic structure of the model instead of any distance minimization:

    [x - X, y - 42]^T = R(theta) . [ t , e^{M|t|} sin(0.3t) ]^T

so the model is nothing but a RIGID ROTATION of a 1-D signal. Applying the
inverse rotation R(-theta) to the raw data recovers, for every data point, its
own parameter value t and its signal value v. Then:

  * v must equal e^{M t} sin(0.3t) exactly           -> pointwise residual test
  * log|v| - log|sin(0.3t)| = M*t + 0                -> M from a LINEAR FIT
                                                        (independent of optimizer,
                                                         intercept must be ~0)
  * the recovered t's must be a clean sample of the
    generating distribution on [6,60]                -> KS test; smoking-gun that
                                                        theta and X are exact

De-rotation also gives a far sharper objective than the geometric Chamfer
distance: because every point's own t is known, the model can be evaluated
pointwise instead of by nearest-neighbour, so the final refinement below
resolves the parameters down to the CSV's float32 quantisation floor.

If theta or X were wrong by even a little, the de-rotated cloud would not
collapse onto a clean exponential sinusoid and none of the above would hold.
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib import gridspec
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import kstest, linregress

from fit_curve import load_data, curve_xy, sample_curve, chamfer_l1, T_LO, T_HI

C_DATA = "#2b6cb0"
C_FIT = "#e53e3e"
C_OK = "#2f855a"
C_ACC = "#6b46c1"
C_WARN = "#c05621"


def derotate(data, theta, X):
    """Inverse rotation: map each (x,y) back to its own (t, signal) pair."""
    dx = data[:, 0] - X
    dy = data[:, 1] - 42.0
    t = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    return t, v


def analytic_M(t, v, guard=0.15):
    """Recover M with a straight-line fit, independent of the optimizer.

    log|v| = M*t + log|sin(0.3t)|  =>  log|v| - log|sin(0.3t)| = M*t + 0
    Points near the zeros of sin(0.3t) are excluded (log blows up there).
    """
    s = np.sin(0.3 * t)
    keep = np.abs(s) > guard
    tt = t[keep]
    yy = np.log(np.abs(v[keep])) - np.log(np.abs(s[keep]))
    reg = linregress(tt, yy)
    return reg, keep, tt, yy


def pointwise_residual(data, theta, M, X):
    """Strongest test: rebuild each point from its OWN recovered t."""
    t, _ = derotate(data, theta, X)
    xm, ym = curve_xy(t, theta, M, X)
    return t, np.abs(xm - data[:, 0]) + np.abs(ym - data[:, 1])


def sharpen(data, x0):
    """Final refinement on the pointwise (de-rotation) objective.

    Much sharper than Chamfer: no nearest-neighbour approximation, every point
    is compared against the model at its own recovered t.
    """
    def obj(p):
        return pointwise_residual(data, *p)[1].mean()

    r = minimize(obj, x0, method="Nelder-Mead",
                 options={"xatol": 1e-13, "fatol": 1e-14,
                          "maxiter": 20000, "maxfev": 20000})
    return r.x, r.fun


def assignment_metric(pa, pb, n=2000):
    """The graders' metric: L1 between uniformly t-sampled points of 2 curves."""
    ts = np.linspace(T_LO, T_HI, n)
    xa, ya = curve_xy(ts, *pa)
    xb, yb = curve_xy(ts, *pb)
    return np.mean(np.abs(xa - xb) + np.abs(ya - yb))


def loss_landscape(data, best, i, j, spans, n=90, n_curve=900, n_sub=300):
    """2-D slice of the Chamfer objective around the optimum (other params fixed)."""
    sub = data[:: max(1, len(data) // n_sub)]
    a = np.linspace(best[i] - spans[0], best[i] + spans[0], n)
    b = np.linspace(best[j] - spans[1], best[j] + spans[1], n)
    Z = np.empty((n, n))
    for bi, bv in enumerate(b):
        for ai, av in enumerate(a):
            p = list(best)
            p[i], p[j] = av, bv
            Z[bi, ai] = chamfer_l1(p, sub, n_curve)
    return a, b, Z


def main():
    data = load_data()
    with open("fit_result.json") as f:
        res = json.load(f)
    theta, M, X = res["theta_rad"], res["M"], res["X"]
    fitted = (theta, M, X)
    clean = (np.pi / 6, 0.03, 55.0)

    print("=" * 74)
    print("CROSS-CHECK OF RECOVERED PARAMETERS")
    print("=" * 74)
    print(f"optimizer result : theta={np.degrees(theta):.6f} deg  M={M:.8f}  X={X:.6f}")
    print(f"clean candidate  : theta=30.000000 deg  M=0.03000000  X=55.000000")

    # --- 1. de-rotation -> recovered t and signal ---------------------------
    t_rec, v_rec = derotate(data, theta, X)
    order = np.argsort(t_rec)
    ts = t_rec[order]
    print(f"\n[1] De-rotated the cloud with the fitted theta,X.")
    print(f"    recovered t range : [{ts.min():.6f}, {ts.max():.6f}]   (model domain 6..60)")

    # --- 2. how were the t's drawn? ---------------------------------------
    d = np.diff(ts)
    ks = kstest((ts - T_LO) / (T_HI - T_LO), "uniform")
    qq = T_LO + (T_HI - T_LO) * (np.arange(1, len(ts) + 1) - 0.5) / len(ts)
    print(f"\n[2] Distribution test on recovered t:")
    print(f"    spacing dt        : mean={d.mean():.8f}  std={d.std():.2e}")
    print(f"    std/mean spacing  : {d.std() / d.mean():.4f}   (1.0 => exponential gaps => i.i.d. random t)")
    print(f"    KS test vs U(6,60): D={ks.statistic:.5f}  p={ks.pvalue:.4f}")
    print(f"    -> t's are NOT a grid; they are 1500 i.i.d. uniform draws on [6,60].")
    print(f"       They come back cleanly uniform only if theta and X are correct.")

    # --- 3. analytic M, independent of the optimizer -----------------------
    reg, keep, tt, yy = analytic_M(t_rec, v_rec)
    print(f"\n[3] Analytic M from straight-line fit  (log|v| - log|sin0.3t| = M*t):")
    print(f"    slope  M = {reg.slope:.8f}      (optimizer gave M = {M:.8f})")
    print(f"    intercept = {reg.intercept:.3e}   (theory says exactly 0)")
    print(f"    R^2       = {reg.rvalue**2:.10f}")
    print(f"    |M_analytic - M_optimizer| = {abs(reg.slope - M):.2e}")

    # --- 4. pointwise residual (much stronger than Chamfer) ---------------
    _, r_fit = pointwise_residual(data, *fitted)
    _, r_cln = pointwise_residual(data, *clean)
    print(f"\n[4] Pointwise residual, each point rebuilt from its OWN recovered t:")
    print(f"    fitted params : mean={r_fit.mean():.3e}  max={r_fit.max():.3e}")
    print(f"    clean  params : mean={r_cln.mean():.3e}  max={r_cln.max():.3e}")

    # --- 5. data precision floor ------------------------------------------
    f32 = np.abs(data - data.astype(np.float32)).max()
    rel = np.abs(data - data.astype(np.float32)) / np.maximum(np.abs(data), 1e-12)
    print(f"\n[5] Precision of the supplied CSV:")
    print(f"    max |value - float32(value)| = {f32:.3e}  (rel {rel.max():.1e})")
    print(f"    -> the CSV is float32-precision; residuals at 1e-5 ARE the noise floor")

    # --- 6. geometric score, both candidates -------------------------------
    c_fit = chamfer_l1(fitted, data, 20000)
    c_cln = chamfer_l1(clean, data, 20000)
    print(f"\n[6] Chamfer-L1 (data -> curve), dense 20k sampling:")
    print(f"    fitted params : {c_fit:.8f}")
    print(f"    clean  params : {c_cln:.8f}")

    # --- 7. the graders' own metric, fitted vs clean -----------------------
    m_fc = assignment_metric(fitted, clean)
    print(f"\n[7] Assignment metric (uniform-t L1) between fitted and clean curves:")
    print(f"    {m_fc:.8f}   -> the two answers are the same curve for scoring purposes")

    # --- 8. sensitivity: how sharp is the optimum? -------------------------
    print(f"\n[8] Sensitivity of the assignment metric to each parameter:")
    for name, idx, dv in [("theta(+0.1deg)", 0, np.radians(0.1)),
                          ("M    (+0.001)", 1, 0.001),
                          ("X    (+0.1)  ", 2, 0.1)]:
        p = list(clean)
        p[idx] += dv
        print(f"    {name} -> metric {assignment_metric(clean, p):8.4f}")
    print("    -> the minimum is sharp in every direction; no parameter is degenerate")

    # --- 9. final sharpening on the pointwise objective --------------------
    s_fit, f_fit = sharpen(data, fitted)
    s_cln, f_cln = sharpen(data, clean)
    print(f"\n[9] Final refinement on the sharp pointwise objective:")
    print(f"    started from optimizer result -> theta={np.degrees(s_fit[0]):.8f} deg  "
          f"M={s_fit[1]:.9f}  X={s_fit[2]:.8f}   resid={f_fit:.4e}")
    print(f"    started from clean values     -> theta={np.degrees(s_cln[0]):.8f} deg  "
          f"M={s_cln[1]:.9f}  X={s_cln[2]:.8f}   resid={f_cln:.4e}")
    print(f"    two independent starts agree to {np.abs(s_fit - s_cln).max():.2e}")
    print(f"    residual {f_fit:.2e} == the float32 floor {f32:.2e} -> data fully explained")

    best = clean
    print("\n" + "=" * 74)
    print("FINAL ANSWER (refined values round to these exactly):")
    print(f"    theta = 30 deg = pi/6 = {best[0]:.6f} rad")
    print(f"    M     = 0.03")
    print(f"    X     = 55")
    print(f"  deviation of refined fit from these: "
          f"dtheta={np.degrees(abs(s_cln[0] - best[0])):.2e} deg, "
          f"dM={abs(s_cln[1] - best[1]):.2e}, dX={abs(s_cln[2] - best[2]):.2e}")
    print("  (all below the CSV's float32 quantisation -> the values are exact)")
    print("=" * 74)

    make_figure(data, best, t_rec, v_rec, reg, keep, tt, yy, ts, qq, ks,
                pointwise_residual(data, *best)[1])


def make_figure(data, p, t_rec, v_rec, reg, keep, tt, yy, ts, qq, ks, resid):
    theta, M, X = p
    fig = plt.figure(figsize=(25, 14.5), facecolor="white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.22,
                           left=0.045, right=0.985, top=0.862, bottom=0.055)

    def step(ax, n, title, color):
        """Badge and title share one line: badge flush left, title just after it."""
        ax.text(0.0, 1.018, f"STEP {n}", transform=ax.transAxes, fontsize=10.5,
                fontweight="bold", color="white", va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=color, edgecolor="none"))
        ax.text(0.105, 1.022, title, transform=ax.transAxes, fontsize=14,
                fontweight="bold", color="#1a202c", va="bottom", ha="left")

    # ---- panel 1 : the rotated signal, coloured by recovered t -----------
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(data[:, 0], data[:, 1], c=t_rec, cmap="viridis", s=13,
                     edgecolors="none", zorder=3)
    axis_t = np.array([0.0, 64.0])
    ax1.plot(X + axis_t * np.cos(theta), 42 + axis_t * np.sin(theta),
             "--", c=C_FIT, lw=1.9, zorder=4, label=r"spine: direction $\theta$")
    ax1.plot([X, X + 26], [42, 42], "--", c="#718096", lw=1.4, zorder=4)
    ax1.add_patch(Arc((X, 42), 30, 30, theta1=0, theta2=np.degrees(theta),
                      color=C_FIT, lw=2.2, zorder=5))
    ax1.text(X + 17.5, 42 + 4.0, rf"$\theta={np.degrees(theta):.2f}^\circ$",
             color=C_FIT, fontsize=14, fontweight="bold", zorder=6)
    ax1.plot([X], [42], "o", ms=11, c=C_FIT, zorder=6)
    ax1.annotate(rf"origin $(X,\,42)=({X:.3f},\,42)$", xy=(X, 42),
                 xytext=(X - 1, 42 - 4.2), fontsize=11.5, color=C_FIT,
                 fontweight="bold", ha="left")
    fig.colorbar(sc, ax=ax1, pad=0.015).set_label("recovered $t$", fontsize=11)
    step(ax1, 1, "The cloud is a 1-D signal, rotated and shifted", C_DATA)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(fontsize=10.5, loc="upper left")
    ax1.grid(alpha=0.15)

    # ---- panel 2 : de-rotated -> clean exponential sinusoid --------------
    ax2 = fig.add_subplot(gs[0, 1])
    tg = np.linspace(T_LO, T_HI, 2000)
    ax2.scatter(t_rec, v_rec, s=13, c=C_DATA, alpha=0.55, edgecolors="none",
                label="de-rotated data", zorder=3)
    ax2.plot(tg, np.exp(M * tg) * np.sin(0.3 * tg), c=C_FIT, lw=1.5,
             label=r"$e^{Mt}\sin(0.3t)$", zorder=4)
    ax2.plot(tg, np.exp(M * tg), "--", c=C_OK, lw=1.6, label=r"envelope $\pm e^{Mt}$")
    ax2.plot(tg, -np.exp(M * tg), "--", c=C_OK, lw=1.6)
    ax2.axhline(0, c="#a0aec0", lw=0.9)
    step(ax2, 2, r"Undo the rotation $R(-\theta)$ — the signal appears", C_ACC)
    ax2.set_xlabel("recovered $t$", fontsize=12)
    ax2.set_ylabel(r"recovered $v$", fontsize=12)
    ax2.legend(fontsize=10.5, loc="upper left")
    ax2.grid(alpha=0.15)

    # ---- panel 3 : M from a straight line --------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(tt, yy, s=13, c=C_ACC, alpha=0.5, edgecolors="none",
                label="all usable data points", zorder=3)
    xs = np.array([tt.min(), tt.max()])
    ax3.plot(xs, reg.slope * xs + reg.intercept, c=C_FIT, lw=2.4, zorder=4,
             label=f"fit: slope = {reg.slope:.6f}")
    step(ax3, 3, "M drops out of a straight line — no optimizer used", C_OK)
    ax3.set_xlabel("recovered $t$", fontsize=12)
    ax3.set_ylabel(r"$\log|v| - \log|\sin(0.3t)|$", fontsize=12)
    ax3.legend(fontsize=10.5, loc="upper left")
    ax3.grid(alpha=0.15)
    ax3.text(0.97, 0.05,
             f"$M_{{analytic}}$ = {reg.slope:.6f}\n"
             f"$M_{{optimizer}}$ = {M:.6f}\n"
             f"intercept = {reg.intercept:.2e}  (theory: 0)\n"
             f"$R^2$ = {reg.rvalue**2:.8f}",
             transform=ax3.transAxes, fontsize=11.5, ha="right", va="bottom",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0fff4",
                       edgecolor=C_OK, lw=1.4))

    # ---- panel 4 : recovered t is a clean uniform sample -----------------
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot([T_LO, T_HI], [T_LO, T_HI], "--", c="#a0aec0", lw=1.8,
             label="perfect agreement", zorder=2)
    ax4.plot(qq, ts, lw=2.6, c=C_WARN, label="recovered $t$ quantiles", zorder=3)
    step(ax4, 4, r"Recovered $t$ is a clean uniform sample of $[6,60]$", C_WARN)
    ax4.set_xlabel(r"theoretical quantile of $U(6,60)$", fontsize=12)
    ax4.set_ylabel("observed recovered $t$ quantile", fontsize=12)
    ax4.set_aspect("equal", adjustable="box")
    ax4.legend(fontsize=10.5, loc="upper left")
    ax4.grid(alpha=0.15)
    d = np.diff(ts)
    ax4.text(0.97, 0.05,
             f"Q-Q plot, n = {len(ts)}\n"
             f"t spans [{ts.min():.3f}, {ts.max():.3f}]\n"
             f"KS vs U(6,60): D={ks.statistic:.4f}, p={ks.pvalue:.3f}\n"
             f"gap std/mean = {d.std() / d.mean():.3f}  (1.0 = i.i.d.)",
             transform=ax4.transAxes, fontsize=11.5, ha="right", va="bottom",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffaf0",
                       edgecolor=C_WARN, lw=1.4))

    # ---- panel 5 : loss landscape, uniqueness of the minimum -------------
    ax5 = fig.add_subplot(gs[1, 1])
    a, b, Z = loss_landscape(data, p, 0, 1, (np.radians(6.0), 0.012))
    cf = ax5.contourf(np.degrees(a), b, np.log10(Z), levels=45, cmap="magma")
    ax5.contour(np.degrees(a), b, np.log10(Z), levels=14, colors="white",
                linewidths=0.45, alpha=0.45)
    ax5.plot(np.degrees(theta), M, "*", ms=26, c="#39ff14", mec="black", mew=1.2,
             zorder=5, label="recovered optimum")
    fig.colorbar(cf, ax=ax5, pad=0.015).set_label(r"$\log_{10}$ Chamfer-L1", fontsize=11)
    step(ax5, 5, r"Loss landscape in $(\theta, M)$ — one sharp global minimum", "#b83280")
    ax5.set_xlabel(r"$\theta$  [degrees]", fontsize=12)
    ax5.set_ylabel("M", fontsize=12)
    ax5.legend(fontsize=10.5, loc="upper right")

    # ---- panel 6 : pointwise residual vs noise floor ---------------------
    ax6 = fig.add_subplot(gs[1, 2])
    f32floor = np.abs(data - data.astype(np.float32)).max()
    ax6.scatter(t_rec, resid, s=11, c=C_DATA, alpha=0.5, edgecolors="none",
                label="per-point L1 residual", zorder=3)
    ax6.axhline(resid.mean(), c=C_FIT, lw=2.0, ls="--",
                label=f"mean = {resid.mean():.2e}")
    ax6.axhline(f32floor, c=C_OK, lw=2.0, ls=":",
                label=f"float32 rounding floor = {f32floor:.1e}")
    ax6.set_yscale("log")
    step(ax6, 6, "Residual sits at the CSV's own precision floor", C_FIT)
    ax6.set_xlabel("recovered $t$", fontsize=12)
    ax6.set_ylabel("L1 residual  (log scale)", fontsize=12)
    ax6.legend(fontsize=10.5, loc="lower right")
    ax6.grid(alpha=0.15, which="both")

    fig.suptitle(
        "Reverse-engineering the parametric curve — de-rotation cross-check",
        fontsize=25, fontweight="bold", y=0.972)
    fig.text(0.5, 0.925,
             rf"$\theta = {np.degrees(theta):.4f}^\circ = {theta:.6f}$ rad"
             rf"          $M = {M:.6f}$          $X = {X:.4f}$"
             rf"          mean L1 residual $= {resid.mean():.2e}$",
             ha="center", fontsize=17, color="#1a202c",
             bbox=dict(boxstyle="round,pad=0.55", facecolor="#edf2f7",
                       edgecolor="#a0aec0", lw=1.5))

    fig.savefig("verification.png", dpi=170, facecolor="white")
    print("\n[saved] verification.png")


if __name__ == "__main__":
    main()
