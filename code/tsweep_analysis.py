#!/usr/bin/env python3
"""T-sweep: self-averaging analysis of the double pendulum FTLE (paper; reproducibility script).

Data: disc024_ftle_T{5..200}_N50000.npy (canonical dimensional DP,
G=9.81, dt=0.005, eps=1e-7, seed 42; 50k ICs per horizon).

Tests the paper's prediction (1): mixed-phase stickiness resists self-averaging.
- Normal CLT self-averaging: var(FTLE | chaotic) ~ T^-1 (beta = 1), kurtosis -> 0.
- Broken/anomalous: beta < 1, platykurtic shape persists; implied trapping exponent z = beta + 1
  (renewal picture) to compare against the universal recurrence decay ~1.57.
Also: classifier robustness in T (plateau vs the exponential separatrix at every T).
"""
import json
import os
import sys

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
import cascade  # noqa: E402

OUT = RESULTS
AEXP = 1 + 1 / (5 * float(exp1(np.log(5))))
TS = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]


def plateau(alphas, ns):
    good = [a for k, (a, n) in enumerate(zip(alphas, ns)) if k >= 2 and n >= 500 and np.isfinite(a)]
    if len(good) < 2:
        good = [a for k, a in enumerate(alphas) if k >= 1 and np.isfinite(a)]
    return float(np.median(good[-5:])) if good else float("nan")


rows = []
for T in TS:
    p = _data(f"disc024_ftle_T{T}_N50000.npy")
    x = np.load(p)
    ch = x[x > 0.01]
    m = {"T": T, "n_chaotic": int(len(ch)), "frac": float(len(ch) / len(x)),
         "mean": float(np.mean(ch)), "var": float(np.var(ch)),
         "cv": float(np.std(ch) / np.mean(ch)), "skew": float(st.skew(ch)),
         "exkurt": float(st.kurtosis(ch))}
    alphas, xmins, ns, ks, levels = cascade.build_power_law_tree(ch, tail_fraction=0.2)
    m["alpha0"] = float(alphas[0]) if alphas else float("nan")
    m["plateau"] = plateau(alphas, ns)
    m["above_separatrix"] = bool(m["plateau"] > AEXP)
    # matched-Gaussian null sign
    g = np.abs(np.random.default_rng(0x7EE7).normal(m["mean"], np.sqrt(m["var"]), len(ch)))
    ga, _, gns, _, _ = cascade.build_power_law_tree(g, tail_fraction=0.2)
    m["gaussnull_plateau"] = plateau(ga, gns)
    m["excess_vs_null"] = float(m["plateau"] - m["gaussnull_plateau"])
    rows.append(m)
    print(f"T={T:3d} frac={m['frac']:.4f} mean={m['mean']:.4f} var={m['var']:.5f} "
          f"cv={m['cv']:.3f} sk={m['skew']:+.2f} ku={m['exkurt']:+.2f} a0={m['alpha0']:.2f} "
          f"plat={m['plateau']:.3f} null={m['gaussnull_plateau']:.3f} "
          f"above={m['above_separatrix']} exc={m['excess_vs_null']:+.3f}", flush=True)

# ---- self-averaging exponent: var ~ T^-beta on the reliable window (T >= 15,
#      where the chaotic/regular separation is clean; T=5,10 flagged as floor-mixed)
sel = [r for r in rows if r["T"] >= 15]
lt = np.log([r["T"] for r in sel])
lv = np.log([r["var"] for r in sel])
beta, logC = np.polyfit(lt, lv, 1)
beta = -beta
resid = np.sqrt(np.mean((lv - (logC - beta * lt)) ** 2))
print(f"\n=== var(FTLE|chaotic) ~ T^-beta:  beta = {beta:.3f}  (fit RMS {resid:.3f}; "
      f"window T=15..200) ===")
print(f"    normal self-averaging beta=1; measured beta={beta:.3f}; implied renewal trapping "
      f"exponent z = beta+1 = {beta+1:.3f} (universal recurrence decay ~1.57)")
# also on the full window for transparency
beta_all, _ = np.polyfit(np.log([r['T'] for r in rows]), np.log([r['var'] for r in rows]), 1)
print(f"    (full window T=5..200: beta = {-beta_all:.3f})")

# kurtosis trajectory
print("\nT:      ", [r["T"] for r in rows])
print("exkurt: ", [round(r["exkurt"], 2) for r in rows])
print("cv:     ", [round(r["cv"], 3) for r in rows])
print("plateau:", [round(r["plateau"], 2) for r in rows])
print("excess: ", [round(r["excess_vs_null"], 2) for r in rows])

json.dump({"AEXP": AEXP, "rows": rows,
           "beta_T15_200": float(beta), "beta_full": float(-beta_all),
           "z_implied": float(beta + 1)},
          open(os.path.join(OUT, "tsweep_results.json"), "w"), indent=1)
print("\nsaved tsweep_results.json")
