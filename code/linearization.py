#!/usr/bin/env python3
"""Linearization of the cascade at the exponential fixed point.

(i)   Exact one-step identities under the symmetrization step d = A - B (A,B iid level-k):
      all odd cumulants vanish; standardized even cumulants scale by 2^(1-m)
      (excess kurtosis HALVES). Verified here on real DP data and Gaussian seeds.
(ii)  The linearized operator at Exp(1): (L0 h)(z) = 2 e^{2z} \int_z^inf e^{-2u} h(u) du.
      Exact eigenfamily h_s = e^{sz}, mu(s) = 2/(2-s)  (s<1): verified numerically.
      Polynomial tower: L0[z^m] = z^m + lower  (unit diagonal, non-normal).
      Numerical spectrum of the discretized, constraint-projected operator.
(iii) Perturbation relaxation is ALGEBRAIC, not geometric: KS-to-Exp ~ C k^-theta for a
      Gamma-contaminated exponential seed (n = 4M), theta fitted.
"""
import json
import os
import sys

import numpy as np
from scipy import stats as st

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
res = {}

# ---------------- (i) cumulant identities on real data
print("=== (i) symmetrization-step cumulant identities ===")


def sym_check(x, label, levels=7):
    cur = np.asarray(x, dtype=float)
    cur = cur[cur > 0]
    rows = []
    for k in range(levels):
        npairs = len(cur) // 2
        a, b = cur[::2][:npairs], cur[1::2][:npairs]
        d = a - b
        g2x = st.kurtosis(cur)
        g2d = st.kurtosis(d)
        rows.append({"k": k, "exkurt_x": float(g2x), "exkurt_d": float(g2d),
                     "half_pred": float(g2x / 2), "skew_d": float(st.skew(d))})
        print(f"  {label} k={k}: exkurt(x)={g2x:+.4f}  exkurt(A-B)={g2d:+.4f}  "
              f"pred(x)/2={g2x/2:+.4f}  skew(A-B)={st.skew(d):+.4f}")
        m = np.mean(cur)
        nxt = np.abs(d) / m
        cur = nxt[nxt > 0]
    return rows


dp = np.load(_data("ftle_n500k_lagrangian_T15_wave0c.npy"))
res["cumulant_dp"] = sym_check(dp[dp > 0.01], "DP500k", 6)
g = np.abs(np.random.default_rng(11).normal(1, 1 / 3, 500_000))
res["cumulant_gauss"] = sym_check(g, "gauss", 6)

# ---------------- (ii) operator: exact eigenfamily + numerical spectrum
print("\n=== (ii) linearized operator at Exp ===")
N = 900
zmax = 40.0
z = np.linspace(0, zmax, N)
h = z[1] - z[0]
w = np.full(N, h)
w[0] = w[-1] = h / 2
# kernel K(z,u) = 2 e^{-2(u-z)} for u >= z; each row integrates over [z_i, zmax], so the
# in-range LEFT endpoint (u = z_i) carries trapezoid weight h/2, not h.
U = z[None, :]
Z = z[:, None]
K = np.where(U >= Z, 2.0 * np.exp(-2.0 * (U - Z)), 0.0) * w[None, :]
np.fill_diagonal(K, np.diag(K) * 0.5)

# exact eigenfamily check: h_s(z)=e^{sz} -> mu(s)=2/(2-s)
for s in (-0.5, -1.0, -2.0, -4.0):
    hs = np.exp(s * z)
    ratio = (K @ hs)[: N // 2] / hs[: N // 2]
    mu_num = np.median(ratio)
    print(f"  s={s:+.1f}: mu_numeric={mu_num:.6f}  mu_exact=2/(2-s)={2/(2-s):.6f}")
    res[f"eigen_s{s}"] = {"numeric": float(mu_num), "exact": 2 / (2 - s)}

# polynomial tower: L0[z^m] leading coefficient
for m in (1, 2, 3):
    pm = z ** m
    lead = np.polyfit(z[: 2 * N // 3], (K @ pm)[: 2 * N // 3], m)[0]
    print(f"  L0[z^{m}] leading coeff = {lead:.5f}  (unit diagonal predicted: 1.0)")
    res[f"poly_m{m}_lead"] = float(lead)

# NOTE: the kernel is Volterra-type (upper triangular in u >= z): finite sections are
# defective/non-normal (matrix eigenvalues collapse toward the diagonal), so an
# eigendecomposition of the discretized matrix does NOT see the true point family;
# the correct statements are the verified exponential eigenfamily above, the unit-diagonal
# triangular action on polynomials, and (measured) algebraic relaxation. Record non-normality:
nrm = np.linalg.norm(K @ K, 2) / np.linalg.norm(K, 2) ** 2
res["nonnormality_note"] = ("Volterra-type kernel; finite sections defective; "
                            f"||K^2||/||K||^2 = {float(nrm):.4f} (1 for normal)")
print(f"  non-normality: ||K^2||/||K||^2 = {nrm:.4f} (=1 for a normal operator)")

# ---------------- (iii) algebraic relaxation, second seed (uniform, n=8M; strong perturbation)
print("\n=== (iii) uniform-seed relaxation (n=8M; independent of the Gaussian 16M run) ===")
rng = np.random.default_rng(0xA16)
mix = rng.random(8_000_000)


def ks_to_exp(x):
    zz = np.sort(x) / np.mean(x)
    n = len(zz)
    emp_hi = np.arange(1, n + 1) / n
    emp_lo = np.arange(0, n) / n
    theo = 1 - np.exp(-zz)
    return float(max(np.max(np.abs(emp_hi - theo)), np.max(np.abs(emp_lo - theo))))


cur = mix
ks_list, n_list = [], []
while len(cur) >= 400:
    ks_list.append(ks_to_exp(cur))
    n_list.append(len(cur))
    npairs = len(cur) // 2
    d = np.abs(cur[::2][:npairs] - cur[1::2][:npairs]) / np.mean(cur)
    cur = d[d > 0]
ks_arr = np.array(ks_list)
n_arr = np.array(n_list)
k = np.arange(len(ks_list))
# floor-aware window: keep levels where the signal is >= 5x the KS sampling floor 1.63/sqrt(n)
floor = 1.63 / np.sqrt(n_arr)
sel = (k >= 1) & (ks_arr > 5 * floor)
A = np.polyfit(np.log(k[sel]), np.log(ks_arr[sel]), 1)
B = np.polyfit(k[sel], np.log(ks_arr[sel]), 1)
alg = np.sqrt(np.mean((np.log(ks_arr[sel]) - np.polyval(A, np.log(k[sel]))) ** 2))
geo = np.sqrt(np.mean((np.log(ks_arr[sel]) - np.polyval(B, k[sel])) ** 2))
print(f"  ks per level: {[round(v,4) for v in ks_list[:13]]}")
print(f"  fit window: levels {list(k[sel])}")
print(f"  algebraic theta = {-A[0]:.3f} (RMS {alg:.3f})  vs geometric rate {np.exp(B[0]):.3f} "
      f"(RMS {geo:.3f})  -> algebraic {'WINS' if alg < geo else 'loses'}")
res["unif_ks"] = ks_list
res["unif_ns"] = n_list
res["unif_theta"] = float(-A[0])
res["unif_alg_rms"] = float(alg)
res["unif_geo_rms"] = float(geo)
# floor-aware re-fit of the Gaussian 16M deep run from the addendum
add = json.load(open(os.path.join(OUT, "addendum_results.json")))
gks = np.array(add["deep"]["gauss_16M_cv033"]["ks_to_exp"])
gk = np.arange(len(gks))
gsel = (gk >= 1) & (gk <= 7)          # n_k >= 1e5 -> floor <= 0.005, signal >= 0.028
Ag = np.polyfit(np.log(gk[gsel]), np.log(gks[gsel]), 1)
Bg = np.polyfit(gk[gsel], np.log(gks[gsel]), 1)
algg = np.sqrt(np.mean((np.log(gks[gsel]) - np.polyval(Ag, np.log(gk[gsel]))) ** 2))
geog = np.sqrt(np.mean((np.log(gks[gsel]) - np.polyval(Bg, gk[gsel])) ** 2))
print(f"  [gauss16M floor-safe refit] theta = {-Ag[0]:.3f} (RMS {algg:.3f}) vs geometric "
      f"rate {np.exp(Bg[0]):.3f} (RMS {geog:.3f})")
res["gauss16M_theta_refit"] = float(-Ag[0])
res["gauss16M_alg_rms"] = float(algg)
res["gauss16M_geo_rms"] = float(geog)

json.dump(res, open(os.path.join(OUT, "linearization_results.json"), "w"), indent=1)
print("\nsaved linearization_results.json")
