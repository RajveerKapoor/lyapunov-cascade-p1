#!/usr/bin/env python3
"""Figures 9 (frozen variance / ensemble structure) and 10 (Henon-Heiles sweep)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.path.join(_ROOT, "data")
RESULTS = os.path.join(_ROOT, "results")
FIGS = os.path.join(_ROOT, "paper", "figures")


def _data(rel):
    # resolve a legacy relative data path to the flat data/ directory
    return os.path.join(DATA, os.path.basename(rel))
CAL = RESULTS
FIG = FIGS

ts = json.load(open(os.path.join(CAL, "tsweep_results.json")))
em = json.load(open(os.path.join(CAL, "energy_mixture_results.json")))
hh = json.load(open(os.path.join(CAL, "hh_sweep_results.json")))
AEXP = ts["AEXP"]

# ---------------- Fig 9: frozen variance + shape saturation + lambda(E)
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
T = [r["T"] for r in ts["rows"]]
v = [r["var"] for r in ts["rows"]]
ax = axes[0]
ax.loglog(T, v, "o-", color="black", lw=1.5)
tgrid = np.array([15, 200.0])
ax.loglog(tgrid, v[2] * (tgrid / 15.0) ** (-1.0), "--", color="#d62728",
          label=r"CLT self-averaging $T^{-1}$")
ax.loglog(tgrid, v[2] * (tgrid / 15.0) ** (0.0), ":", color="#1f77b4",
          label=r"measured $\beta=-0.014$")
ax.set_xlabel("horizon $T$")
ax.set_ylabel(r"var(FTLE $\mid$ chaotic)")
ax.set_ylim(0.01, 1.0)
ax.legend(fontsize=7, frameon=False)
ax.set_title("frozen variance", fontsize=9)

ax = axes[1]
ax.semilogx(T, [r["exkurt"] for r in ts["rows"]], "s-", color="black", lw=1.4,
            label="excess kurtosis")
ax.semilogx(T, [r["excess_vs_null"] for r in ts["rows"]], "^-", color="#d62728", lw=1.4,
            label="matched-null excess")
ax.axhline(0, color="grey", lw=0.7)
ax.set_xlabel("horizon $T$")
ax.legend(fontsize=7, frameon=False)
ax.set_title("shape saturates; excess grows", fontsize=9)

ax = axes[2]
lam = em["lambda_of_E"]
E = np.array(lam["E"])
mu = np.array(lam["mean"])
sd = np.array(lam["sigma"])
ax.fill_between(E, mu - sd, mu + sd, color="#1f77b4", alpha=0.25,
                label=r"band mean $\pm$ band width")
ax.plot(E, mu, "-", color="#1f77b4", lw=1.6)
ax.set_xlabel(r"release energy $E(\theta_1,\theta_2)$")
ax.set_ylabel(r"FTLE")
ax.set_title(r"$\lambda(E)$ mixture skeleton ($\eta^2=0.41$)", fontsize=9)
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig9_ensemble.pdf"))
fig.savefig(os.path.join(FIG, "fig9_ensemble.png"), dpi=200)

# ---------------- Fig 10: HH sweep
fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
Es, fr, exc, ku = [], [], [], []
for k, row in sorted(hh["energies"].items(), key=lambda kv: float(kv[0])):
    Es.append(row["E"])
    fr.append(row["chaotic_fraction"])
    exc.append(row.get("excess_vs_null", np.nan))
    ku.append(row.get("moments", {}).get("exkurt", np.nan))
Es = np.array(Es); fr = np.array(fr); exc = np.array(exc); ku = np.array(ku)
ax1.plot(Es, fr, "o-", color="grey", lw=1.4, label="chaotic fraction")
ax1.set_xlabel("energy $E$")
ax1.set_ylabel("chaotic fraction", color="grey")
ax1.axvline(1 / 6, color="black", lw=0.6, ls=":")
ax1.text(1 / 6 - 0.002, 0.05, "escape\nenergy", fontsize=7, ha="right")
ax2 = ax1.twinx()
ok = np.isfinite(exc)
colors = ["#d62728" if e > 0 else "#1f77b4" for e in exc[ok]]
ax2.bar(Es[ok], exc[ok], width=0.006, color=colors, alpha=0.75,
        label="matched-null excess")
ax2.axhline(0, color="black", lw=0.7)
ax2.set_ylabel("matched-null excess (red $=$ mixed-phase signature)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig10_hh.pdf"))
fig.savefig(os.path.join(FIG, "fig10_hh.png"), dpi=200)
print("figs 9-10 written")
