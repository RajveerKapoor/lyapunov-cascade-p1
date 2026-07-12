#!/usr/bin/env python3
"""Addendum — mechanism experiments for the paper.

Decides, on the cascade operator (cascade.py):
  A) Full per-level flows + f-sweeps + moments for ALL real FTLE arrays on disk
     (DP 50k/500k/5M + 3 DP geometry variants + quartic x2 + Lorenz + K=5),
     with drop-level-0 fits where alpha0 overflows the curve-fit p0 bounds,
     and a fit-free 'plateau' statistic as robustness.
  B) Fresh generation of the standard map K=1 (mixed-phase) FTLE at n=500k,
     N_iter=1000 — completes the K=1 vs K=5 matched pair with shape moments.
  C) Matched-Gaussian null per system: |N(mu,sigma)| with the system's own
     chaotic (mu,sigma) — does bulk Gaussianity alone reproduce the flow?
  D) Gaussian-family universality check: level-1 of ANY Gaussian bulk is exactly
     half-normal (mu cancels in A-B), so flows should coincide from level 1 on.
  E) Deep cascade at n=16M (Exp seed vs Gaussian seed, 25 levels) + per-level
     Kolmogorov distance to Exp(1): is the ~3.87 Gaussian plateau metastable
     (drifting toward the unique exponential fixed point, Puri-Rubin 1970)?
"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats as st
from scipy.special import exp1

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.path.join(_ROOT, "data")
RESULTS = os.path.join(_ROOT, "results")
FIGS = os.path.join(_ROOT, "paper", "figures")


def _data(rel):
    # resolve a legacy relative data path to the flat data/ directory
    return os.path.join(DATA, os.path.basename(rel))
sys.path.insert(0, _HERE)
sys.path.insert(0, _HERE)
import cascade  # noqa: E402

OUT_DIR = RESULTS
FIG_DIR = FIGS
SEED = 0xADD
Z08 = st.norm.ppf(0.8)                      # 0.8416
M08 = st.norm.pdf(Z08) / 0.2 - Z08          # 0.5582 (Gaussian excess log-slope, 1st order)


def a0_pred_gauss(cv):
    """First-order level-0 Clauset reading of a Gaussian bulk |N(mu,sigma)| at f=0.2."""
    return 1.0 + (1.0 / cv + Z08) / M08


def moments(x):
    return {"n": int(len(x)), "mean": float(np.mean(x)), "std": float(np.std(x)),
            "cv": float(np.std(x) / np.mean(x)), "skew": float(st.skew(x)),
            "kurt": float(st.kurtosis(x))}


def ks_to_exp(x):
    """Kolmogorov distance of mean-normalized sample to Exp(1)."""
    z = np.sort(x) / np.mean(x)
    n = len(z)
    emp_hi = np.arange(1, n + 1) / n
    emp_lo = np.arange(0, n) / n
    theo = 1.0 - np.exp(-z)
    return float(max(np.max(np.abs(emp_hi - theo)), np.max(np.abs(emp_lo - theo))))


def plateau(alphas, ns):
    """Fit-free plateau: median of fitted alpha at levels k>=2 with n_k >= 500 (last <=5)."""
    good = [a for k, (a, n) in enumerate(zip(alphas, ns)) if k >= 2 and n >= 500 and np.isfinite(a)]
    if len(good) < 2:
        good = [a for k, a in enumerate(alphas) if k >= 1 and np.isfinite(a)]
    return float(np.median(good[-5:])) if good else float("nan")


def analyze(x, f=0.2, max_levels=25, want_ks=False):
    alphas, xmins, ns, ks, levels = cascade.build_power_law_tree(x, tail_fraction=f,
                                                                 max_levels=max_levels)
    a_fit, C, r, r2, se = cascade.fit_convergence(alphas)
    a_d0, _, r_d0, r2_d0, _ = cascade.fit_convergence(alphas[1:]) if len(alphas) > 5 \
        else (float("nan"),) * 5
    out = {"alphas": [round(float(a), 4) for a in alphas], "ns": [int(n) for n in ns],
           "astar_fit": float(a_fit), "r": float(r), "r2": float(r2),
           "astar_drop0": float(a_d0), "r_drop0": float(r_d0),
           "plateau": plateau(alphas, ns)}
    if want_ks:
        out["ks_to_exp"] = [round(ks_to_exp(lv), 4) for lv in levels if len(lv) >= 200]
    return out


def full_row(label, arr, fs=(0.1, 0.2, 0.3), want_ks=True):
    x = arr[arr > 0.01]
    row = {"label": label, "moments": moments(x),
           "a0_pred_gauss_from_cv": float(a0_pred_gauss(np.std(x) / np.mean(x)))}
    for f in fs:
        row[f"f{f}"] = analyze(x, f=f, want_ks=(want_ks and f == 0.2))
    m = row["moments"]
    a2 = row["f0.2"]
    print(f"[row] {label:22s} cv={m['cv']:.4f} sk={m['skew']:+.2f} ku={m['kurt']:+.1f} | "
          f"a0={a2['alphas'][0]:.2f}(pred {row['a0_pred_gauss_from_cv']:.2f}) "
          f"a*fit={a2['astar_fit']:.3f} a*d0={a2['astar_drop0']:.3f} plat={a2['plateau']:.3f}",
          flush=True)
    return row


def main():
    t0 = time.time()
    res = {"seed": SEED, "rows": {}, "matched_gauss": {}, "gauss_family": {}, "deep": {}}

    # ---------------- A) real arrays
    real = [
        ("dp_50k_T15",       "lyapunov_all_50k.npy"),
        ("dp_500k_T15",      "ftle_n500k_lagrangian_T15_wave0c.npy"),
        ("dp_5M_T15",        "ftle_n5M_lagrangian_T15.npy"),
        ("dp_m0p5_l1_500k",  "disc067_ftle_dp_m0p5_l1_n500k_T15.npy"),
        ("dp_m2_l0p5_500k",  "disc067_ftle_dp_m2_l0p5_n500k_T15.npy"),
        ("dp_m0p5_l2_500k",  "disc067_ftle_dp_m0p5_l2_n500k_T15.npy"),
        ("quartic_l01",      "coupled_quartic_ftle_n500k.npy"),
        ("quartic_l02",      "coupled_quartic_lam0p2_ftle_n500k.npy"),
        ("lorenz_T100",      "lorenz_ftle_n500k_T100.npy"),
        ("stdmap_K5",        "standard_map_K5_ftle_n500k.npy"),
    ]
    arrays = {}
    for label, rel in real:
        p = _data(rel)
        if not os.path.exists(p):
            print(f"[skip] {label}: missing {rel}", flush=True)
            continue
        arr = np.load(p)
        arrays[label] = arr
        try:
            res["rows"][label] = full_row(label, arr)
        except Exception as e:
            print(f"[ERR] {label}: {e}", flush=True)

    # ---------------- B) fresh standard map K=1 (mixed-phase pair for K=5)
    try:
        import standard_map as sm
        tK = time.time()
        out = sm.compute_lyapunov_standard_map(500_000, K=1.0, N_iter=1000, seed_entropy=SEED)
        arr = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=float)
        arrays["stdmap_K1_fresh"] = arr
        res["rows"]["stdmap_K1_fresh"] = full_row("stdmap_K1_fresh", arr)
        res["rows"]["stdmap_K1_fresh"]["gen_secs"] = round(time.time() - tK, 1)
        print(f"[K1] generated n=500k in {time.time()-tK:.0f}s", flush=True)
    except Exception as e:
        print(f"[ERR] K=1 generation: {e}", flush=True)

    # ---------------- C) matched-Gaussian null per system
    rng = np.random.default_rng(SEED)
    for label in list(res["rows"].keys()):
        try:
            m = res["rows"][label]["moments"]
            g = np.abs(rng.normal(m["mean"], m["std"], m["n"]))
            res["matched_gauss"][label] = full_row(f"gaussmatch[{label}]", g,
                                                   fs=(0.2,), want_ks=(label in
                                                   ("dp_500k_T15", "lorenz_T100")))
        except Exception as e:
            print(f"[ERR] matched {label}: {e}", flush=True)

    # ---------------- D) Gaussian-family universality (level-1 == half-normal for ALL cv)
    for cv in (0.05, 0.1, 1 / 3, 0.6):
        try:
            g = np.abs(rng.normal(1.0, cv, 500_000))
            res["gauss_family"][f"cv{cv:.3f}"] = analyze(g, want_ks=False)
            a = res["gauss_family"][f"cv{cv:.3f}"]
            print(f"[gaussfam] cv={cv:.3f}: a0={a['alphas'][0]:.2f} "
                  f"a1={a['alphas'][1] if len(a['alphas'])>1 else float('nan'):.3f} "
                  f"a*d0={a['astar_drop0']:.3f} plat={a['plateau']:.3f}", flush=True)
        except Exception as e:
            print(f"[ERR] gaussfam cv={cv}: {e}", flush=True)

    # ---------------- E) deep cascade n=16M — metastability of the Gaussian plateau
    for label, gen in [("exp_16M", lambda: np.random.default_rng(7).exponential(1.0, 16_000_000)),
                       ("gauss_16M_cv033", lambda: np.abs(np.random.default_rng(7)
                                                          .normal(1.0, 1 / 3, 16_000_000)))]:
        try:
            tD = time.time()
            res["deep"][label] = analyze(gen(), max_levels=25, want_ks=True)
            d = res["deep"][label]
            print(f"[deep] {label}: levels={len(d['alphas'])} "
                  f"alphas[0,1,5,10,15,20]={[d['alphas'][i] if i < len(d['alphas']) else None for i in (0,1,5,10,15,20)]} "
                  f"plat={d['plateau']:.3f} ({time.time()-tD:.0f}s)", flush=True)
            print(f"       ks_to_exp={d.get('ks_to_exp', [])}", flush=True)
        except Exception as e:
            print(f"[ERR] deep {label}: {e}", flush=True)

    res["analytic_f02"] = {"alpha_star_exp": 1.0 + 1.0 / (5 * float(exp1(np.log(5)))),
                           "a0_gauss_formula": "1 + (1/cv + 0.8416)/0.5582"}
    res["total_secs"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT_DIR, "addendum_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    # ---------------- figures 6-8
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    AEXP = res["analytic_f02"]["alpha_star_exp"]

    # Fig 6 — mechanism: real flow vs matched-Gaussian flow (DP and Lorenz)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, sys_lab, ttl in [(axes[0], "dp_500k_T15", "double pendulum ($n=5{\\times}10^5$)"),
                             (axes[1], "lorenz_T100", "Lorenz-63 ($n=5{\\times}10^5$)")]:
        if sys_lab in res["rows"]:
            a = res["rows"][sys_lab]["f0.2"]["alphas"]
            ax.plot(range(len(a)), a, "o-", color="black", lw=1.6, label="measured FTLE")
        if sys_lab in res["matched_gauss"]:
            a = res["matched_gauss"][sys_lab]["f0.2"]["alphas"]
            ax.plot(range(len(a)), a, "s--", color="#d62728", lw=1.3,
                    label="matched Gaussian $|N(\\mu,\\sigma)|$")
        ax.axhline(AEXP, color="#1f77b4", ls=":", lw=1.2)
        ax.set_xlabel("cascade level $k$")
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.set_ylim(2.2, 10)
    axes[0].set_ylabel(r"$\alpha_k$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_matched.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig6_matched.png"), dpi=200)

    # Fig 7 — slow manifold: KS distance to Exp(1) per level
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for lab, src, c, m in [("Exp seed ($16$M)", res["deep"].get("exp_16M"), "#1f77b4", "^"),
                           ("Gaussian seed ($16$M)", res["deep"].get("gauss_16M_cv033"), "#d62728", "s"),
                           ("double pendulum", res["rows"].get("dp_500k_T15", {}).get("f0.2"), "black", "o"),
                           ("coupled quartic $\\lambda{=}0.1$",
                            res["rows"].get("quartic_l01", {}).get("f0.2"), "#2ca02c", "v"),
                           ("Lorenz-63", res["rows"].get("lorenz_T100", {}).get("f0.2"), "#9467bd", "d")]:
        if src and src.get("ks_to_exp"):
            ax.semilogy(range(len(src["ks_to_exp"])), src["ks_to_exp"], m + "-",
                        color=c, lw=1.3, ms=4, label=lab)
    ax.set_xlabel("cascade level $k$")
    ax.set_ylabel("Kolmogorov distance to Exp(1)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig7_slowmanifold.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig7_slowmanifold.png"), dpi=200)

    # Fig 8 — summary: plateau reading for every real system vs its matched Gaussian
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    labs, real_v, match_v = [], [], []
    for lab, row in res["rows"].items():
        labs.append(lab.replace("_", " "))
        real_v.append(row["f0.2"]["plateau"])
        mg = res["matched_gauss"].get(lab)
        match_v.append(mg["f0.2"]["plateau"] if mg else np.nan)
    ypos = np.arange(len(labs))
    ax.plot(real_v, ypos, "o", color="black", label="measured FTLE (plateau)")
    ax.plot(match_v, ypos, "s", color="#d62728", mfc="none", label="matched Gaussian")
    ax.axvline(AEXP, color="#1f77b4", ls=":", lw=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labs, fontsize=7.5)
    ax.set_xlabel(r"cascade plateau $\alpha$ at $f{=}0.2$")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig8_summary.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "fig8_summary.png"), dpi=200)

    print(f"[addendum] DONE in {res['total_secs']}s", flush=True)


if __name__ == "__main__":
    main()
