#!/usr/bin/env python3
"""Energy-mixture decomposition of the double-pendulum FTLE ensemble.

The canonical DP ensemble releases from rest at uniform angles: each IC has E(th1,th2) =
-(m1+m2) g l1 cos(th1) - m2 g l2 cos(th2) (its own energy shell). Using the on-disk
100x100 FTLE grid (canonical T=15 protocol), test:
  (1) does FTLE collapse onto a 1-D lambda(E) curve?  eta^2 = Var(binned means)/Var(total)
  (2) variance decomposition: Var_total = Var(lambda(E)) + <within-shell var>
      -> is the T-frozen variance (~0.30) the mixture variance?
  (3) per-shell (E-band) shape: are narrow-E bands NARROW and non-platykurtic
      (i.e. the platykurtic broad bulk is the mixture, not the shell)?
  (4) mixture reconstruction: synthetic = lambda_bar(E_i) + N(0, sigma_band(E_i))
      -> does it reproduce the real ensemble's moments AND cascade plateau?
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
G = 9.81  # canonical dimensional convention, m1=m2=l1=l2=1


def plateau(alphas, ns):
    good = [a for k, (a, n) in enumerate(zip(alphas, ns)) if k >= 2 and n >= 500 and np.isfinite(a)]
    if len(good) < 2:
        good = [a for k, a in enumerate(alphas) if k >= 1 and np.isfinite(a)]
    return float(np.median(good[-5:])) if good else float("nan")


grid = np.load(_data("lyapunov_grid_100x100.npy"))
n1, n2 = grid.shape
th1 = np.linspace(-np.pi, np.pi, n1)
th2 = np.linspace(-np.pi, np.pi, n2)
T1, T2 = np.meshgrid(th1, th2, indexing="ij")
E = -2.0 * G * np.cos(T1) - G * np.cos(T2)          # potential energy at release (KE=0)

lam = grid.ravel()
Ev = E.ravel()
ch = lam > 0.01
lam_c, E_c = lam[ch], Ev[ch]
print(f"grid: {n1}x{n2}; chaotic {ch.sum()} ({ch.mean():.4f})   "
      f"[50k-array T=15 frac was 0.8816/0.9708 depending on array]")
print(f"chaotic FTLE: mean={lam_c.mean():.4f} var={lam_c.var():.5f} cv={lam_c.std()/lam_c.mean():.3f} "
      f"skew={st.skew(lam_c):+.2f} exkurt={st.kurtosis(lam_c):+.2f}")

# ---------------- (1)+(2) eta^2 and variance decomposition over E bins
nbins = 40
bins = np.quantile(E_c, np.linspace(0, 1, nbins + 1))
idx = np.clip(np.searchsorted(bins, E_c, side="right") - 1, 0, nbins - 1)
bmean = np.array([lam_c[idx == b].mean() for b in range(nbins)])
bvar = np.array([lam_c[idx == b].var() for b in range(nbins)])
bn = np.array([(idx == b).sum() for b in range(nbins)])
bE = np.array([E_c[idx == b].mean() for b in range(nbins)])
var_between = float(np.sum(bn * (bmean - lam_c.mean()) ** 2) / len(lam_c))
var_within = float(np.sum(bn * bvar) / len(lam_c))
eta2 = var_between / lam_c.var()
print(f"\n=== variance decomposition over {nbins} E-bins ===")
print(f"Var_total = {lam_c.var():.5f} = Var(lambda(E)) {var_between:.5f} + <within-shell> {var_within:.5f}")
print(f"eta^2 (E explains) = {eta2:.4f}   within-shell mean sigma = {np.sqrt(var_within):.4f}")

# finer bins to bound residual structure
for nb in (80, 160):
    b2 = np.quantile(E_c, np.linspace(0, 1, nb + 1))
    i2 = np.clip(np.searchsorted(b2, E_c, side="right") - 1, 0, nb - 1)
    m2 = np.array([lam_c[i2 == b].mean() for b in range(nb)])
    n2_ = np.array([(i2 == b).sum() for b in range(nb)])
    vb = float(np.sum(n2_ * (m2 - lam_c.mean()) ** 2) / len(lam_c))
    print(f"  nbins={nb}: eta^2 = {vb/lam_c.var():.4f}")

# ---------------- (3) per-shell shapes in representative bands
print("\n=== per-shell (E-band) shapes ===")
shell_rows = []
for lo_q, hi_q in [(0.10, 0.15), (0.30, 0.35), (0.50, 0.55), (0.70, 0.75), (0.90, 0.95)]:
    lo, hi = np.quantile(E_c, [lo_q, hi_q])
    sel = (E_c >= lo) & (E_c < hi)
    xs = lam_c[sel]
    r = {"E_mid": float((lo + hi) / 2), "n": int(sel.sum()), "mean": float(xs.mean()),
         "cv": float(xs.std() / xs.mean()), "skew": float(st.skew(xs)),
         "exkurt": float(st.kurtosis(xs))}
    shell_rows.append(r)
    print(f"E~{r['E_mid']:+7.2f}  n={r['n']:4d}  mean={r['mean']:.3f}  cv={r['cv']:.3f}  "
          f"sk={r['skew']:+.2f}  ku={r['exkurt']:+.2f}")

# ---------------- (4) mixture reconstruction -> moments + cascade
rng = np.random.default_rng(0xE0E)
synth = bmean[idx] + rng.normal(0, np.sqrt(np.maximum(bvar[idx], 0)))
synth = synth[synth > 0.01]
print("\n=== mixture reconstruction lambda_bar(E)+N(0,sigma(E)) vs real ===")
print(f"real : cv={lam_c.std()/lam_c.mean():.3f} sk={st.skew(lam_c):+.2f} ku={st.kurtosis(lam_c):+.2f}")
print(f"synth: cv={synth.std()/synth.mean():.3f} sk={st.skew(synth):+.2f} ku={st.kurtosis(synth):+.2f}")
a_r, _, ns_r, _, _ = cascade.build_power_law_tree(lam_c, tail_fraction=0.2)
a_s, _, ns_s, _, _ = cascade.build_power_law_tree(synth, tail_fraction=0.2)
print(f"cascade plateau: real={plateau(a_r, ns_r):.3f}  synth={plateau(a_s, ns_s):.3f}  "
      f"(separatrix {AEXP:.3f})   alpha0: real={a_r[0]:.2f} synth={a_s[0]:.2f}")

# pure-mixture limit (zero within-shell noise): the lambda(E) pushforward itself
push = bmean[idx]
a_p, _, ns_p, _, _ = cascade.build_power_law_tree(push[push > 0.01], tail_fraction=0.2)
print(f"pure pushforward lambda_bar(E): cv={push.std()/push.mean():.3f} ku={st.kurtosis(push):+.2f} "
      f"plateau={plateau(a_p, ns_p):.3f}")

json.dump({"eta2_40bins": eta2, "var_total": float(lam_c.var()),
           "var_between": var_between, "var_within": var_within,
           "lambda_of_E": {"E": bE.tolist(), "mean": bmean.tolist(),
                            "sigma": np.sqrt(bvar).tolist(), "n": bn.tolist()},
           "shell_rows": shell_rows,
           "real_plateau": plateau(a_r, ns_r), "synth_plateau": plateau(a_s, ns_s),
           "push_plateau": plateau(a_p, ns_p), "AEXP": AEXP},
          open(os.path.join(OUT, "energy_mixture_results.json"), "w"), indent=1)
print("\nsaved energy_mixture_results.json")
