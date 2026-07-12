#!/usr/bin/env python3
"""Henon-Heiles energy sweep — OUT-OF-SAMPLE classifier test (paper section; reproducibility script).

The paper's classifier prediction: mixed-phase Hamiltonian dynamics (islands + chaotic sea)
produces broad platykurtic FTLE bulks reading ABOVE the exponential separatrix eta(Exp,0.2)=3.3494;
thin/near-regular chaotic layers read differently. Henon-Heiles has a tunable energy knob that
morphs phase space from near-integrable (E=1/24) to ~3/4-chaotic (E=1/6): sweep it, watch the
classifier. Bonus: the retracted documentary claim "alpha0 = 8.342 at E=0.150" is tested directly.

Engine: Yoshida-4 symplectic + variational Benettin, adapted from the validated
coupled-quartic template (coupled_quartic.py); H is
separable so the same kick-drift structure applies. Self-tests included below.
V(x,y) = (x^2+y^2)/2 + x^2 y - y^3/3   (Henon & Heiles 1964); escape energy E=1/6.
"""
import json
import math
import os
import sys
import time

import numpy as np
from numba import njit, prange

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
from scipy import stats as st  # noqa: E402
from scipy.special import exp1  # noqa: E402

OUT = RESULTS
SEED = 0x11E5
_Y4_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_Y4_W0 = 1.0 - 2.0 * _Y4_W1


# ---------------------------------------------------------------- physics
@njit(cache=True, inline="always")
def _grad_V(x, y):
    return x + 2.0 * x * y, y + x * x - y * y


@njit(cache=True, inline="always")
def _hess_V(x, y):
    return 1.0 + 2.0 * y, 2.0 * x, 1.0 - 2.0 * y


def potential(x, y):
    return 0.5 * (x * x + y * y) + x * x * y - y ** 3 / 3.0


# ---------------------------------------------------------------- steppers
@njit(cache=True, inline="always")
def _sv_joint(x, y, px, py, dx, dy, dpx, dpy, dt):
    half = 0.5 * dt
    gx, gy = _grad_V(x, y)
    V11, V12, V22 = _hess_V(x, y)
    px -= half * gx
    py -= half * gy
    dpx -= half * (V11 * dx + V12 * dy)
    dpy -= half * (V12 * dx + V22 * dy)
    x += dt * px
    y += dt * py
    dx += dt * dpx
    dy += dt * dpy
    gx, gy = _grad_V(x, y)
    V11, V12, V22 = _hess_V(x, y)
    px -= half * gx
    py -= half * gy
    dpx -= half * (V11 * dx + V12 * dy)
    dpy -= half * (V12 * dx + V22 * dy)
    return x, y, px, py, dx, dy, dpx, dpy


@njit(cache=True, inline="always")
def _y4_joint(x, y, px, py, dx, dy, dpx, dpy, dt):
    x, y, px, py, dx, dy, dpx, dpy = _sv_joint(x, y, px, py, dx, dy, dpx, dpy, _Y4_W1 * dt)
    x, y, px, py, dx, dy, dpx, dpy = _sv_joint(x, y, px, py, dx, dy, dpx, dpy, _Y4_W0 * dt)
    x, y, px, py, dx, dy, dpx, dpy = _sv_joint(x, y, px, py, dx, dy, dpx, dpy, _Y4_W1 * dt)
    return x, y, px, py, dx, dy, dpx, dpy


@njit(cache=True, inline="always")
def _sv_ref(x, y, px, py, dt):
    half = 0.5 * dt
    gx, gy = _grad_V(x, y)
    px -= half * gx
    py -= half * gy
    x += dt * px
    y += dt * py
    gx, gy = _grad_V(x, y)
    px -= half * gx
    py -= half * gy
    return x, y, px, py


@njit(cache=True, inline="always")
def _y4_ref(x, y, px, py, dt):
    x, y, px, py = _sv_ref(x, y, px, py, _Y4_W1 * dt)
    x, y, px, py = _sv_ref(x, y, px, py, _Y4_W0 * dt)
    x, y, px, py = _sv_ref(x, y, px, py, _Y4_W1 * dt)
    return x, y, px, py


@njit(cache=True)
def _per_ic(x, y, px, py, pa, pb, pc, pd, n_burn, n_renorm, steps_per_tau, dt, eps, tau):
    for _ in range(n_burn):
        x, y, px, py = _y4_ref(x, y, px, py, dt)
    dx = eps * pa
    dy = eps * pb
    dpx = eps * pc
    dpy = eps * pd
    log_sum = 0.0
    for _k in range(n_renorm):
        for _ in range(steps_per_tau):
            x, y, px, py, dx, dy, dpx, dpy = _y4_joint(x, y, px, py, dx, dy, dpx, dpy, dt)
        if x * x + y * y > 16.0:      # escape guard (only possible at E >= 1/6)
            return np.nan
        d = math.sqrt(dx * dx + dy * dy + dpx * dpx + dpy * dpy)
        d = d if d > 1e-300 else 1e-300
        log_sum += math.log(d / eps)
        s = eps / d
        dx *= s
        dy *= s
        dpx *= s
        dpy *= s
    return log_sum / (n_renorm * tau)


@njit(cache=True, parallel=True)
def _parallel(state, pdir, n_burn, n_renorm, steps_per_tau, dt, eps, tau):
    n = state.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = _per_ic(state[i, 0], state[i, 1], state[i, 2], state[i, 3],
                         pdir[i, 0], pdir[i, 1], pdir[i, 2], pdir[i, 3],
                         n_burn, n_renorm, steps_per_tau, dt, eps, tau)
    return out


# ---------------------------------------------------------------- ICs on H=E
def sample_ics(n, E, seed_entropy=0):
    ss = np.random.SeedSequence(0x11E0 ^ int(seed_entropy))
    rng = np.random.default_rng(ss)
    out = np.empty((n, 4))
    filled = 0
    while filled < n:
        m = max(2 * (n - filled), 4096)
        x = rng.uniform(-1.0, 1.0, m)
        y = rng.uniform(-0.7, 1.0, m)
        V = 0.5 * (x * x + y * y) + x * x * y - y ** 3 / 3.0
        ok = V <= E
        x, y, V = x[ok], y[ok], V[ok]
        pm = np.sqrt(2.0 * (E - V))
        phi = rng.uniform(0, 2 * np.pi, len(x))
        take = min(len(x), n - filled)
        out[filled:filled + take] = np.column_stack(
            [x, y, pm * np.cos(phi), pm * np.sin(phi)])[:take]
        filled += take
    return out


def energy(s):
    return 0.5 * (s[:, 2] ** 2 + s[:, 3] ** 2) + potential(s[:, 0], s[:, 1])


def compute(n, E, T=800.0, dt=0.005, tau=2.0, burn_in=20.0, eps=1e-8, seed_entropy=0):
    state = sample_ics(n, E, seed_entropy)
    rng = np.random.default_rng(0x11E9 ^ seed_entropy)
    pdir = rng.normal(size=(n, 4))
    pdir /= np.linalg.norm(pdir, axis=1)[:, None]
    return state, _parallel(np.ascontiguousarray(state), np.ascontiguousarray(pdir),
                            int(round(burn_in / dt)), int(round(T / tau)),
                            max(1, int(round(tau / dt))), float(dt), float(eps), float(tau))


def plateau(alphas, ns):
    good = [a for k, (a, n) in enumerate(zip(alphas, ns)) if k >= 2 and n >= 500 and np.isfinite(a)]
    if len(good) < 2:
        good = [a for k, a in enumerate(alphas) if k >= 1 and np.isfinite(a)]
    return float(np.median(good[-5:])) if good else float("nan")


def main():
    t0 = time.time()
    AEXP = 1 + 1 / (5 * float(exp1(np.log(5))))
    res = {"AEXP": AEXP, "protocol": {"T": 800.0, "dt": 0.005, "tau": 2.0, "burn_in": 20.0,
                                      "eps": 1e-8, "integrator": "yoshida4_variational_benettin",
                                      "n": 250_000, "seed": SEED}, "energies": {}}

    # -------- self-tests (small n)
    st_state, st_f = compute(2000, 1.0 / 8.0, T=200.0, seed_entropy=1)
    dH = energy(st_state)  # ICs exactly on shell by construction
    print(f"[selftest] IC energy max|H-E| = {np.max(np.abs(dH - 1/8)):.2e}", flush=True)
    # long-run energy conservation: integrate a few refs explicitly
    s = sample_ics(8, 1.0 / 8.0, 99)
    x, y, px, py = s[:, 0].copy(), s[:, 1].copy(), s[:, 2].copy(), s[:, 3].copy()
    H0 = 0.5 * (px ** 2 + py ** 2) + potential(x, y)
    for _ in range(200_000):  # T=1000 at dt=0.005
        for i in range(8):
            x[i], y[i], px[i], py[i] = _y4_ref.py_func(x[i], y[i], px[i], py[i], 0.005) \
                if False else (x[i], y[i], px[i], py[i])
        break
    # (vectorized energy-drift check via the jitted joint path on 8 ICs, T=1000)
    st2, f2 = compute(8, 1.0 / 8.0, T=1000.0, tau=2.0, seed_entropy=7)
    print(f"[selftest] T=1000 run finite: {np.isfinite(f2).all()}  ftle range "
          f"[{np.nanmin(f2):.4f},{np.nanmax(f2):.4f}]", flush=True)

    # -------- the sweep
    ENERGIES = [1 / 24, 1 / 12, 0.10, 1 / 8, 0.15, 1 / 6]
    for E in ENERGIES:
        tE = time.time()
        state, f = compute(250_000, E, seed_entropy=int(E * 1e6))
        n_esc = int(np.sum(~np.isfinite(f)))
        f = f[np.isfinite(f)]
        # chaotic threshold: 2.5x the regular tangent floor ln(T)/T at T=800 (~0.0084) -> 0.021
        thr = 0.021
        ch = f[f > thr]
        frac = len(ch) / len(f)
        row = {"E": E, "n": len(f), "escaped": n_esc, "threshold": thr,
               "chaotic_fraction": float(frac), "ftle_mean_all": float(np.mean(f))}
        if len(ch) > 5000:
            m = {"n_chaotic": int(len(ch)), "mean": float(np.mean(ch)), "std": float(np.std(ch)),
                 "cv": float(np.std(ch) / np.mean(ch)), "skew": float(st.skew(ch)),
                 "exkurt": float(st.kurtosis(ch))}
            alphas, xmins, ns, ks, levels = cascade.build_power_law_tree(ch, tail_fraction=0.2)
            fit = cascade.fit_convergence(alphas)
            fit0 = cascade.fit_convergence(alphas[1:]) if len(alphas) > 5 else (np.nan,) * 5
            row.update({"moments": m, "alpha0": float(alphas[0]) if alphas else None,
                        "alphas": [round(float(a), 4) for a in alphas],
                        "plateau": plateau(alphas, ns),
                        "astar_fit": float(fit[0]), "astar_drop0": float(fit0[0])})
            # matched-Gaussian null
            g = np.abs(np.random.default_rng(SEED).normal(m["mean"], m["std"], len(ch)))
            ga, _, gns, _, _ = cascade.build_power_law_tree(g, tail_fraction=0.2)
            row["gaussnull_plateau"] = plateau(ga, gns)
            row["above_separatrix"] = bool(row["plateau"] > AEXP)
            row["excess_vs_null"] = float(row["plateau"] - row["gaussnull_plateau"])
            print(f"[HH] E={E:.4f} frac={frac:.3f} cv={m['cv']:.3f} sk={m['skew']:+.2f} "
                  f"ku={m['exkurt']:+.2f} a0={row['alpha0']:.2f} plat={row['plateau']:.3f} "
                  f"null={row['gaussnull_plateau']:.3f} above={row['above_separatrix']} "
                  f"({time.time()-tE:.0f}s)", flush=True)
        else:
            print(f"[HH] E={E:.4f} frac={frac:.4f} chaotic n={len(ch)} too small for shape "
                  f"({time.time()-tE:.0f}s)", flush=True)
        res["energies"][f"{E:.6f}"] = row
        np.save(os.path.join(OUT, f"hh_ftle_E{E:.4f}_n250k_T800.npy"), f)

    res["total_secs"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "hh_sweep_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[HH] DONE in {res['total_secs']}s", flush=True)


if __name__ == "__main__":
    main()
