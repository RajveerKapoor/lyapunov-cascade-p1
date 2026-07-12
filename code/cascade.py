#!/usr/bin/env python3
"""cascade.py — the finite-time-Lyapunov renormalization cascade operator.

This module is an importable, output-path-free reproduction of the published
cascade-renormalization operator used to produce
the original power-law analysis
(converged_value=3.97621612842499, decay_rate=0.4564706173194879).

The operator (T) — copied EXACTLY from
the original validation script:

  1. Start: current = chaotic FTLE values (FTLE > chaotic_threshold).
  2. At each level: alpha, xmin, n = fit_power_law_mle(current, tail_fraction=0.2)
       where xmin = the (1 - tail_fraction) quantile (80th percentile),
       tail = current[current >= xmin], alpha = 1 + n / sum(log(tail / xmin))
       (Clauset/Hill MLE). Also record KS statistic.
  3. Coarse-grain to next level (this is T):
       n_pairs = len(current)//2
       a = current[::2][:n_pairs]; b = current[1::2][:n_pairs]
       diffs = abs(a - b) / mean(current); current = diffs[diffs > 0]
     -> HALVES the sample each level (÷2 decimation).
  4. Stop when len(current) < 20.
  5. Fit convergence: alpha_k = alpha_inf + C * r**k via scipy curve_fit,
       bounds alpha_inf in [1.5,12], C in [-15,15], r in [0.05,0.99],
       p0 = [4.0, a[0]-4.0, 0.5], maxfev=20000.

Plus added machinery NOT in the original (clearly separated):
  - per-level Hill SE = alpha / sqrt(n_tail);
  - bootstrap CI machinery that resamples the BASE chaotic distribution and
    rebuilds the WHOLE tree each iterate (so CIs propagate through T);
  - the fixed-Hill estimator (test4_hill.json reproduction);
  - NaN-robust convergence fit.

Reference implementation accompanying the paper.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

# ===========================================================================
# CORE OPERATOR  (exact reproduction of monumental_validation.py)
# ===========================================================================

def fit_power_law_mle(data, xmin=None, tail_fraction=0.2):
    """Clauset et al. (2009) MLE of a power-law tail exponent.

    EXACT reproduction of monumental_validation.fit_power_law_mle.
    xmin defaults to the (1 - tail_fraction) quantile (the 80th percentile
    for tail_fraction=0.2), computed by index into the sorted array.

    Returns: (alpha, xmin, n_tail).  alpha is NaN if n_tail < 5 or log_sum<=0.
    """
    data = np.asarray(data)
    data = data[data > 0]
    data = np.sort(data)
    if len(data) == 0:
        return np.nan, np.nan, 0

    if xmin is None:
        tail_start = int((1 - tail_fraction) * len(data))
        xmin = data[tail_start]

    tail = data[data >= xmin]
    n = len(tail)
    if n < 5:
        return np.nan, xmin, n

    log_sum = np.sum(np.log(tail / xmin))
    if log_sum <= 0:
        return np.nan, xmin, n

    alpha = 1 + n / log_sum
    return alpha, xmin, n


def ks_power_law(data, alpha, xmin):
    """KS statistic for a power-law fit. EXACT repro of monumental_validation."""
    data = np.asarray(data)
    tail = np.sort(data[data >= xmin])
    n = len(tail)
    if n < 5:
        return np.nan
    empirical = np.arange(1, n + 1) / n
    theoretical = 1 - (tail / xmin) ** (-(alpha - 1))
    return float(np.max(np.abs(empirical - theoretical)))


def hill_se(alpha, n_tail):
    """Asymptotic Hill standard error of the tail exponent: alpha / sqrt(n_tail).

    NOTE: this is the std of the MLE *shape* alpha' = (alpha-1) under the
    Pareto MLE, which equals (alpha-1)/sqrt(n).  We specify
    SE = alpha / sqrt(n_tail) (a slightly conservative convention).  We report
    BOTH; this function returns the brief's alpha/sqrt(n) by default.
    """
    if n_tail is None or n_tail <= 0 or not np.isfinite(alpha):
        return float("nan")
    return float(alpha / np.sqrt(n_tail))


def hill_se_shape(alpha, n_tail):
    """Textbook Hill/MLE SE of the Pareto exponent: (alpha-1)/sqrt(n_tail)."""
    if n_tail is None or n_tail <= 0 or not np.isfinite(alpha):
        return float("nan")
    return float((alpha - 1.0) / np.sqrt(n_tail))


def coarse_grain(current):
    """The renormalization map T: pairwise normalized abs-differences.

    EXACT reproduction of the in-loop coarse-graining in
    monumental_validation.build_power_law_tree.  Halves the sample size.
    Returns the next-level array (strictly-positive diffs only).
    """
    current = np.asarray(current)
    if len(current) < 2:
        return np.array([], dtype=float)
    n_pairs = len(current) // 2
    a = current[::2][:n_pairs]
    b = current[1::2][:n_pairs]
    mean_val = np.mean(current)
    diffs = np.abs(a - b) / mean_val if mean_val > 0 else np.abs(a - b)
    return diffs[diffs > 0]


def build_power_law_tree(lyap_data, tail_fraction=0.2, max_levels=14):
    """Build the hierarchical power-law tree; track alpha at each level.

    EXACT reproduction of monumental_validation.build_power_law_tree, extended
    only with `max_levels` default 14 (orig 12; harmless — stop is len<20).

    Returns: alphas, xmins, ns, ks_stats, level_data  (level_data[i] is the
    input distribution AT level i, i.e. level_data[0] == initial chaotic set).
    """
    current = np.asarray(lyap_data)
    current = current[current > 0].copy()
    alphas, xmins, ns, ks_stats = [], [], [], []
    level_data = [current.copy()]

    for _level in range(max_levels):
        if len(current) < 20:
            break
        alpha, xmin, n = fit_power_law_mle(current, tail_fraction=tail_fraction)
        if np.isnan(alpha):
            break
        ks = ks_power_law(current, alpha, xmin)
        alphas.append(alpha)
        xmins.append(xmin)
        ns.append(n)
        ks_stats.append(ks)
        current = coarse_grain(current)
        level_data.append(current.copy())

    return alphas, xmins, ns, ks_stats, level_data


def fit_convergence(alphas, p0=None):
    """Fit alpha_k = alpha_inf + C * r**k.

    EXACT reproduction of monumental_validation.fit_convergence (which returns
    5-tuple alpha_inf, C, r, r2, stderr_of_alpha_inf).  NaN-robust: drops
    non-finite alphas before fitting (handles the Yoshida level-0 outlier that
    breaks curve_fit) and returns NaNs gracefully if <5 finite points remain.

    Returns: (alpha_inf, C, r, r2, stderr_alpha_inf).
    """
    a_all = np.asarray(alphas, dtype=float)
    finite = np.isfinite(a_all)
    a = a_all[finite]
    k = np.arange(len(a_all))[finite]
    if len(a) < 5:
        return np.nan, np.nan, np.nan, 0.0, np.nan

    def model(kk, a_inf, C, r):
        return a_inf + C * (r ** kk)

    if p0 is None:
        p0 = [4.0, a[0] - 4.0, 0.5]
    try:
        popt, pcov = curve_fit(
            model, k, a, p0=p0,
            bounds=([1.5, -15, 0.05], [12, 15, 0.99]),
            maxfev=20000,
        )
        pred = model(k, *popt)
        ss_res = np.sum((a - pred) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        stderr = np.sqrt(np.diag(pcov))
        return float(popt[0]), float(popt[1]), float(popt[2]), float(r2), float(stderr[0])
    except Exception:
        return np.nan, np.nan, np.nan, 0.0, np.nan


# ===========================================================================
# FIXED-HILL ESTIMATOR  (exact reproduction of monumental_validation TEST 4)
# ===========================================================================

def hill_estimator_curve(data, k_range=None):
    """Hill estimator curve: alpha_Hill(k) = k / sum_{i=1}^{k} log(x_{(i)}/x_{(k)})
    for the top-k order statistics (descending).  EXACT repro of TEST 4.

    NOTE: this convention returns the *shape* k/sum (NOT 1 + k/sum); the
    published test4_hill.json was produced with exactly this.
    """
    sorted_data = np.sort(data)[::-1]
    n = len(sorted_data)
    if k_range is None:
        k_range = np.arange(10, min(n // 2, 500))
    hill_vals = []
    for k in k_range:
        if k >= n:
            hill_vals.append(np.nan)
            continue
        log_sum = np.sum(np.log(sorted_data[:k] / sorted_data[k]))
        if log_sum <= 0:
            hill_vals.append(np.nan)
            continue
        hill_vals.append(k / log_sum)
    return np.array(k_range), np.array(hill_vals)


def build_hill_tree(chaotic_data, max_levels=8):
    """Reproduce TEST 4: fixed-Hill estimate per cascade level.

    At each level: k_range = arange(5, min(len(current)//3, 300)); the
    'stable' estimate is the median of the middle half of finite Hill values;
    coarse-grain with the same T.  Returns list of dicts {level, n, hill_stable,
    hill_std} (matches test4_hill.json schema).
    """
    current = np.asarray(chaotic_data).copy()
    out = []
    for level in range(max_levels):
        if len(current) < 50:
            break
        k_vals, hill_vals = hill_estimator_curve(
            current, k_range=np.arange(5, min(len(current) // 3, 300)))
        valid = np.isfinite(hill_vals)
        if np.sum(valid) > 10:
            v = hill_vals[valid]
            stable_region = v[len(v) // 4: 3 * len(v) // 4]
            out.append({
                "level": int(level), "n": int(len(current)),
                "hill_stable": float(np.median(stable_region)),
                "hill_std": float(np.std(stable_region)),
            })
        current = coarse_grain(current)
    return out


# ===========================================================================
# BOOTSTRAP CI MACHINERY  (NEW — not in monumental_validation)
# ===========================================================================

def cascade_once(chaotic_data, tail_fraction=0.2, max_levels=14):
    """Run the full cascade once; return a structured dict of per-level + fit."""
    alphas, xmins, ns, ks_stats, level_data = build_power_law_tree(
        chaotic_data, tail_fraction=tail_fraction, max_levels=max_levels)
    a_inf, C, r, r2, se_inf = fit_convergence(alphas)
    return {
        "alphas": [float(a) for a in alphas],
        "xmins": [float(x) for x in xmins],
        "ns": [int(n) for n in ns],
        "ks": [float(k) for k in ks_stats],
        "alpha_star": float(a_inf), "C": float(C), "r": float(r),
        "r2": float(r2), "alpha_star_se_curvefit": float(se_inf),
    }


def bootstrap_cascade(chaotic_data, tail_fraction=0.2, B=1000, seed=0xCA5CADE,
                      max_levels=14, max_level_for_ci=10):
    """Bootstrap the cascade by resampling the BASE chaotic distribution and
    rebuilding the WHOLE tree each iterate, so CIs propagate through T.

    Returns a dict with:
      - point: the point-estimate cascade (cascade_once on the original data),
      - level_alpha_ci95: list of [lo, hi] per level (up to max_level_for_ci),
      - alpha_star_ci95, alpha_star_se (bootstrap std), C_ci95, r_ci95, r_se,
      - lam_ci95, half_life_ci95,
      - B_effective (iterates that yielded a finite alpha_star).
    """
    rng = np.random.default_rng(seed)
    chaotic = np.asarray(chaotic_data)
    chaotic = chaotic[chaotic > 0]
    n = len(chaotic)

    point = cascade_once(chaotic, tail_fraction=tail_fraction, max_levels=max_levels)
    n_levels_point = len(point["alphas"])
    n_ci_levels = min(n_levels_point, max_level_for_ci)

    boot_level_alphas = [[] for _ in range(n_ci_levels)]
    boot_astar, boot_C, boot_r = [], [], []

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sample = chaotic[idx]
        alphas, _, _, _, _ = build_power_law_tree(
            sample, tail_fraction=tail_fraction, max_levels=max_levels)
        for lvl in range(min(len(alphas), n_ci_levels)):
            if np.isfinite(alphas[lvl]):
                boot_level_alphas[lvl].append(alphas[lvl])
        a_inf, C, r, _r2, _se = fit_convergence(alphas)
        if np.isfinite(a_inf):
            boot_astar.append(a_inf)
        if np.isfinite(C):
            boot_C.append(C)
        if np.isfinite(r):
            boot_r.append(r)

    def ci95(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            return [float("nan"), float("nan")]
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    def safe_std(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.std(arr, ddof=1)) if len(arr) >= 2 else float("nan")

    level_ci = [ci95(b) for b in boot_level_alphas]

    boot_r_arr = np.asarray(boot_r, dtype=float)
    boot_r_arr = boot_r_arr[np.isfinite(boot_r_arr) & (boot_r_arr > 0)]
    lam = -np.log(boot_r_arr) if len(boot_r_arr) else np.array([])
    half = np.log(2) / lam if len(lam) else np.array([])

    return {
        "point": point,
        "level_alpha_ci95": level_ci,
        "alpha_star_ci95": ci95(boot_astar),
        "alpha_star_se": safe_std(boot_astar),
        "C_ci95": ci95(boot_C),
        "r_ci95": ci95(boot_r),
        "r_se": safe_std(boot_r),
        "lam_ci95": ci95(lam.tolist()) if len(lam) else [float("nan")] * 2,
        "half_life_ci95": ci95(half.tolist()) if len(half) else [float("nan")] * 2,
        "B": int(B),
        "B_effective_astar": int(len(boot_astar)),
        "seed": int(seed),
    }


# ===========================================================================
# MONOTONICITY  (the monotonicity falsification rule)
# ===========================================================================

def monotone_within_1se(alphas, ns, se_fn=hill_se):
    """Rule: cascade is NON-monotone if some level k+1 alpha exceeds
    level k by more than 1 SE (SE = se_fn(alpha_k, n_k), default alpha/sqrt(n)).

    Returns (is_monotone, violations) where violations is a list of dicts.
    """
    alphas = list(alphas)
    ns = list(ns)
    violations = []
    mono = True
    for i in range(1, len(alphas)):
        se_prev = se_fn(alphas[i - 1], ns[i - 1])
        # "level k+1 exceeds level k by more than 1 SE": compare alpha[i] vs
        # alpha[i-1] + 1 SE.  Per this criterion the SE is the Hill SE.
        if np.isfinite(se_prev) and alphas[i] > alphas[i - 1] + se_prev:
            mono = False
            violations.append({
                "level_k": i - 1, "alpha_k": float(alphas[i - 1]),
                "level_kp1": i, "alpha_kp1": float(alphas[i]),
                "excess": float(alphas[i] - alphas[i - 1]),
                "se_k": float(se_prev),
            })
    return mono, violations


def derived_constants(r):
    """LAM = -ln(r); half-life = ln2 / (-ln r)."""
    if not np.isfinite(r) or r <= 0:
        return float("nan"), float("nan")
    lam = -np.log(r)
    half = np.log(2) / lam if lam != 0 else float("nan")
    return float(lam), float(half)
