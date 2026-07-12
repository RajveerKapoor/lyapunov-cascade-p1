#!/usr/bin/env python3
"""Synthetic-tail calibration of the cascade operator T — paper section 6.

Analysis for the manuscript:

Writes outputs under results/ and paper/figures/.

Question decided here (the platykurtic vs leptokurtic dichotomy):
  (a) does T's fixed point track a genuine (regularly-varying) tail index — conservation?
  (b) does T default light-tailed (Gumbel-domain) inputs to the exponential fixed point,
      whose estimator reading is alpha*(f) = 1 + [(1/f) E1(ln(1/f))]^{-1} (~3.35 at f=0.2)?
  (c) where do the observed non-mixed systems (3.03-3.31) sit relative to (b)?

Everything runs through the cascade operator (cascade.py).
(cascade_once + bootstrap_cascade; Clauset MLE 1 + n/sum(log), xmin = 80th pctile).
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.path.join(_ROOT, "data")
RESULTS = os.path.join(_ROOT, "results")
FIGS = os.path.join(_ROOT, "paper", "figures")


def _data(rel):
    # resolve a legacy relative data path to the flat data/ directory
    return os.path.join(DATA, os.path.basename(rel))
sys.path.insert(0, _HERE)
import cascade  # noqa: E402

from scipy.special import exp1  # noqa: E402

OUT_DIR = RESULTS
FIG_DIR = FIGS
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N = 500_000
B = 250                      # bootstrap reps (draft precision; state in paper)
SEED = 0xF1BE
rng = np.random.default_rng(SEED)


def pareto(alpha, n, xm=1.0, r=rng):
    """P(X > x) = (x/xm)^-alpha, x >= xm  (true Pareto, not Lomax)."""
    u = r.random(n)
    return xm * u ** (-1.0 / alpha)


def mixture(bulk, tail, r=rng):
    x = np.concatenate([bulk, tail])
    return r.permutation(x)


def analytic_exp_reading(f):
    """Asymptotic Clauset+1 reading of Exp(1) at tail fraction f: 1 + 1/((1/f) E1(ln(1/f)))."""
    u = np.log(1.0 / f)
    return 1.0 + 1.0 / ((1.0 / f) * exp1(u))


# ---------------------------------------------------------------- families
FAMILIES = [
    # name, generator, EVT domain, true tail index (None = light), bootstrap?
    ("exp",        lambda: rng.exponential(1.0, N),                    "Gumbel",  None, True),
    ("exp_seed2",  lambda: np.random.default_rng(2077).exponential(1.0, N), "Gumbel", None, False),
    ("halfnormal", lambda: np.abs(rng.normal(0, 1, N)),                "Gumbel",  None, True),
    ("gauss_n31",  lambda: np.abs(rng.normal(3, 1, N)),                "Gumbel",  None, False),
    ("gauss_narrow", lambda: np.abs(rng.normal(1.0, 0.1, N)),          "Gumbel",  None, True),
    ("lognorm_s05", lambda: rng.lognormal(0, 0.5, N),                  "Gumbel",  None, False),
    ("lognorm_s10", lambda: rng.lognormal(0, 1.0, N),                  "Gumbel",  None, True),
    ("weibull_k15", lambda: rng.weibull(1.5, N),                       "Gumbel",  None, False),
    ("weibull_k07", lambda: rng.weibull(0.7, N),                       "Gumbel",  None, True),
    ("gamma_3",    lambda: rng.gamma(3.0, 1.0, N),                     "Gumbel",  None, False),
    ("uniform",    lambda: rng.random(N),                              "Weibull", None, False),
    ("pareto_a25", lambda: pareto(2.5, N),                             "Frechet", 2.5,  True),
    ("pareto_a40", lambda: pareto(4.0, N),                             "Frechet", 4.0,  True),
    ("pareto_a60", lambda: pareto(6.0, N),                             "Frechet", 6.0,  False),
    ("mix_bulk_p30", lambda: mixture(np.abs(rng.normal(1, 0.25, int(0.88 * N))),
                                     pareto(3.0, N - int(0.88 * N), xm=2.0)), "Frechet", 3.0, True),
    ("mix_bulk_p40", lambda: mixture(np.abs(rng.normal(1, 0.25, int(0.88 * N))),
                                     pareto(4.0, N - int(0.88 * N), xm=2.0)), "Frechet", 4.0, True),
    ("mix_bulk_p50", lambda: mixture(np.abs(rng.normal(1, 0.25, int(0.88 * N))),
                                     pareto(5.0, N - int(0.88 * N), xm=2.0)), "Frechet", 5.0, False),
]


def run_family(name, gen, domain, true_alpha, do_boot):
    x = np.asarray(gen())
    x = x[x > 0]
    t0 = time.time()
    pt = cascade.cascade_once(x)
    row = {
        "name": name, "domain": domain, "true_alpha": true_alpha, "n": int(len(x)),
        "alpha0": float(pt["alphas"][0]) if pt["alphas"] else None,
        "alphas": [round(float(a), 4) for a in pt["alphas"]],
        "alpha_star": float(pt["alpha_star"]),
        "r": float(pt["r"]), "r2": float(pt["r2"]),
        "n_levels": len(pt["alphas"]),
    }
    if do_boot:
        bt = cascade.bootstrap_cascade(x, B=B, seed=SEED)
        row["ci95"] = [float(v) for v in bt["alpha_star_ci95"]]
        row["se"] = float(bt["alpha_star_se"])
    row["secs"] = round(time.time() - t0, 1)
    print(f"[battery] {name:14s} dom={domain:7s} true={str(true_alpha):4s} "
          f"a0={row['alpha0']:.3f} a*={row['alpha_star']:.4f} r={row['r']:.3f} "
          f"ci={row.get('ci95', '-')} ({row['secs']}s)", flush=True)
    return row


def main():
    t_start = time.time()
    results = {"seed": SEED, "N": N, "B": B, "runs": [], "meta": {}}

    # 0) analytic reading curve
    fgrid = np.linspace(0.05, 0.45, 81)
    results["analytic"] = {
        "f_grid": [float(f) for f in fgrid],
        "alpha_star_exp_f": [float(analytic_exp_reading(f)) for f in fgrid],
        "alpha_star_exp_f02": float(analytic_exp_reading(0.2)),
        "alpha_star_exp_f01": float(analytic_exp_reading(0.1)),
        "alpha_star_exp_f03": float(analytic_exp_reading(0.3)),
    }
    print(f"[analytic] alpha*_Exp(f=0.2) = {results['analytic']['alpha_star_exp_f02']:.6f}", flush=True)

    # 1) the battery
    for spec in FAMILIES:
        results["runs"].append(run_family(*spec))

    # 2) Exp at n = 5M (fixed-point precision) + tail-fraction sweep at n = 500k
    x5 = np.random.default_rng(SEED).exponential(1.0, 5_000_000)
    pt5 = cascade.cascade_once(x5)
    results["exp_5M"] = {"alpha0": float(pt5["alphas"][0]), "alpha_star": float(pt5["alpha_star"]),
                         "alphas": [round(float(a), 4) for a in pt5["alphas"]],
                         "r": float(pt5["r"]), "r2": float(pt5["r2"])}
    print(f"[exp 5M] a0={pt5['alphas'][0]:.4f} a*={pt5['alpha_star']:.4f}", flush=True)

    xe = np.random.default_rng(SEED).exponential(1.0, N)
    fsweep = {}
    for f in (0.1, 0.2, 0.3):
        p = cascade.cascade_once(xe, tail_fraction=f)
        fsweep[str(f)] = {"alpha0": float(p["alphas"][0]), "alpha_star": float(p["alpha_star"]),
                          "analytic": float(analytic_exp_reading(f))}
        print(f"[f-sweep] f={f}: a0={p['alphas'][0]:.4f} a*={p['alpha_star']:.4f} "
              f"analytic={analytic_exp_reading(f):.4f}", flush=True)
    results["exp_fsweep"] = fsweep

    # 2b) REAL on-disk FTLE arrays — the decisive f-sweep
    #     (flat alpha*(f) = conserved genuine tail; sliding along the analytic curve = light-tail reading)
    import glob as _glob
    real_specs = [("dp_canonical_50k_T15", "lyapunov_all_50k.npy")]
    for pth in sorted([os.path.join(DATA, _n) for _n in ("coupled_quartic_ftle_n500k.npy", "coupled_quartic_lam0p2_ftle_n500k.npy", "lorenz_ftle_n500k_T100.npy", "standard_map_K5_ftle_n500k.npy")]):
        real_specs.append((os.path.basename(pth).replace("_ftle_n500k.npy", "").replace(".npy", ""),
                           os.path.basename(pth)))
    real = {}
    for label, rel in real_specs:
        pth = _data(rel)
        if not os.path.exists(pth):
            continue
        arr = np.load(pth)
        arr = arr[arr > 0.01]          # chaotic threshold
        row = {"n_chaotic": int(len(arr)), "path": rel, "fsweep": {}}
        for f in (0.1, 0.2, 0.3):
            p = cascade.cascade_once(arr, tail_fraction=f)
            row["fsweep"][str(f)] = {"alpha0": float(p["alphas"][0]), "alpha_star": float(p["alpha_star"]),
                                     "r": float(p["r"]), "r2": float(p["r2"]),
                                     "alphas": [round(float(a), 4) for a in p["alphas"]]}
            print(f"[real-fsweep] {label:28s} f={f}: a*={p['alpha_star']:.4f} "
                  f"(a0={p['alphas'][0]:.3f}, n={len(arr)})", flush=True)
        real[label] = row
    results["real_fsweep"] = real

    # 2c) synthetic conserved-tail reference f-sweep (bulk + Pareto alpha=4, independent rng)
    rmix = np.random.default_rng(SEED + 1)
    xm4 = mixture(np.abs(rmix.normal(1, 0.25, int(0.88 * N))),
                  pareto(4.0, N - int(0.88 * N), xm=2.0, r=rmix), r=rmix)
    msweep = {}
    for f in (0.1, 0.2, 0.3):
        p = cascade.cascade_once(xm4, tail_fraction=f)
        msweep[str(f)] = {"alpha_star": float(p["alpha_star"]), "alpha0": float(p["alphas"][0])}
        print(f"[mix4-fsweep] f={f}: a*={p['alpha_star']:.4f}", flush=True)
    results["mix4_fsweep"] = msweep

    # 3) DP canonical per-level curve (on-disk reproduction)
    dp_path = os.path.join(RESULTS, "dp_canonical_repro.json")
    dp_alphas, dp_astar = None, None
    if os.path.exists(dp_path):
        d = json.load(open(dp_path))
        dp_alphas = d["R1"]["mean"]["alphas"]
        dp_astar = d["R1"]["mean"]["alpha_star"]
        results["dp_canonical"] = {"alphas": dp_alphas, "alpha_star": dp_astar,
                                   "r": d["R1"]["mean"]["r"], "r2": d["R1"]["mean"]["r2"]}

    results["meta"]["total_secs"] = round(time.time() - t_start, 1)
    with open(os.path.join(OUT_DIR, "synthetic_battery_results.json"), "w") as fh:
        json.dump(results, fh, indent=1)

    # ---------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    AEXP = results["analytic"]["alpha_star_exp_f02"]

    def get(name):
        return next(r for r in results["runs"] if r["name"] == name)

    # Fig 1 — cascade flows alpha_k vs k
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if dp_alphas:
        ax.plot(range(len(dp_alphas)), dp_alphas, "o-", color="black", lw=1.8,
                label=f"double pendulum (measured), $\\alpha^*={dp_astar:.2f}$")
    for nm, lab, c, m in [("gauss_n31", "synthetic $|N(3,1)|$ (light tail)", "#666666", "s"),
                          ("exp", "synthetic Exp(1) (fixed point)", "#1f77b4", "^"),
                          ("mix_bulk_p40", "synthetic bulk + Pareto $\\alpha=4$ tail", "#d62728", "v")]:
        rr = get(nm)
        ax.plot(range(len(rr["alphas"])), rr["alphas"], m + "-", color=c, lw=1.4, ms=4, label=lab)
    ax.axhline(AEXP, color="#1f77b4", ls=":", lw=1.2)
    ax.text(0.2, AEXP - 0.28, f"$\\alpha^*_{{\\mathrm{{Exp}}}}={AEXP:.3f}$", color="#1f77b4", fontsize=9)
    ax.axhline(4.0, color="#d62728", ls=":", lw=1.0)
    ax.set_xlabel("cascade level $k$")
    ax.set_ylabel(r"fitted tail exponent $\alpha_k$")
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(2.4, 11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_flows.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig1_flows.png"), dpi=200)

    # Fig 2 — the two classes across real systems (measured values)
    IN = [("DP canonical ($n=5{\\times}10^6$)", 3.710, 0.056),
          ("DP variant 1", 3.789, 0.213), ("DP variant 2", 3.854, 0.210),
          ("DP variant 3", 3.758, 0.217), ("DP variant 4", 3.748, 0.170),
          ("DP variant 5", 4.091, 0.176), ("DP variant 6", 3.595, 0.318),
          ("DP variant 7", 3.959, 0.171),
          ("DP $8{\\times}8$ $(m,l)$ map", 3.896, 0.193),
          ("DP 169 geometries", 3.868, 0.197),
          ("standard map $K{=}1$", 4.382, 0.327)]
    EX = [("Lorenz-63", 3.031, 0.066), ("standard map $K{=}5$", 3.038, 0.114),
          ("coupled quartic $\\lambda{=}0.1$", 3.239, 0.127),
          ("coupled quartic $\\lambda{=}0.2$", 3.312, 0.109)]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    y = 0
    ylabels = []
    for nm, v, se in EX:
        ax.errorbar(v, y, xerr=1.96 * se, fmt="o", color="#1f77b4", capsize=2)
        ylabels.append(nm); y += 1
    y += 1
    ylabels.append("")
    for nm, v, se in IN:
        ax.errorbar(v, y, xerr=1.96 * se, fmt="s", color="#d62728", capsize=2)
        ylabels.append(nm); y += 1
    ax.axvline(AEXP, color="#1f77b4", ls=":", lw=1.2)
    ax.text(AEXP + 0.01, -0.8, "$\\alpha^*_{\\mathrm{Exp}}$", color="#1f77b4", fontsize=9)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.set_xlabel(r"cascade fixed point $\alpha^*$  (95% CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_classes.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig2_classes.png"), dpi=200)

    # Fig 3 — conservation: true tail index vs measured alpha*
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    xs, ys, es = [], [], []
    for r_ in results["runs"]:
        if r_["true_alpha"]:
            xs.append(r_["true_alpha"]); ys.append(r_["alpha_star"])
            es.append(1.96 * r_.get("se", 0.0))
    ax.errorbar(xs, ys, yerr=es, fmt="o", color="#d62728", capsize=2, label="Fréchet-domain inputs")
    grid = np.linspace(2, 6.5, 10)
    ax.plot(grid, grid, "k--", lw=1, label="$\\alpha^*=\\alpha_{\\mathrm{true}}$ (conservation)")
    lt = [r_["alpha_star"] for r_ in results["runs"] if r_["true_alpha"] is None and r_["domain"] == "Gumbel"]
    ax.axhspan(min(lt), max(lt), color="#1f77b4", alpha=0.15,
               label="Gumbel-domain inputs (no true $\\alpha$)")
    ax.axhline(AEXP, color="#1f77b4", ls=":", lw=1.2)
    ax.set_xlabel(r"input tail index $\alpha_{\mathrm{true}}$")
    ax.set_ylabel(r"cascade fixed point $\alpha^*$")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_conservation.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig3_conservation.png"), dpi=200)

    # Fig 4 — the exponential reading alpha*_Exp(f): analytic curve + measured
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(results["analytic"]["f_grid"], results["analytic"]["alpha_star_exp_f"],
            "-", color="black", lw=1.5, label=r"analytic $1+[(1/f)E_1(\ln(1/f))]^{-1}$")
    for f in (0.1, 0.2, 0.3):
        ax.plot(f, fsweep[str(f)]["alpha_star"], "o", color="#1f77b4", ms=7)
    ax.plot([], [], "o", color="#1f77b4", label="measured (Exp(1), $n=5{\\times}10^5$)")
    ax.set_xlabel("tail fraction $f$")
    ax.set_ylabel(r"$\alpha^*_{\mathrm{Exp}}(f)$")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_reading.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig4_reading.png"), dpi=200)

    # Fig 5 — THE discriminator: alpha*(f) for real systems vs the exponential reading curve
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(results["analytic"]["f_grid"], results["analytic"]["alpha_star_exp_f"], "-",
            color="black", lw=1.6, label="exponential fixed-point reading (analytic)")
    fvals = [0.1, 0.2, 0.3]
    ax.plot(fvals, [msweep[str(f)]["alpha_star"] for f in fvals], "d--", color="#2ca02c",
            lw=1.3, ms=6, label=r"synthetic bulk + Pareto $\alpha{=}4$ (conserved ref.)")
    for label, row in real.items():
        ys = [row["fsweep"][str(f)]["alpha_star"] for f in fvals]
        ax.plot(fvals, ys, "o-", lw=1.4, ms=5, label=label.replace("_", " "))
    ax.set_xlabel("tail fraction $f$")
    ax.set_ylabel(r"cascade fixed point $\alpha^*(f)$")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_fsweep.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig5_fsweep.png"), dpi=200)

    print(f"[battery] DONE in {results['meta']['total_secs']}s -> "
          f"{OUT_DIR}/synthetic_battery_results.json + 5 figures", flush=True)


if __name__ == "__main__":
    main()
