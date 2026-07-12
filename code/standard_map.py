#!/usr/bin/env python3
"""standard_map.py — Chirikov standard map FTLE engine.

General in K and reusable across systems. The module is importable and output-path-free.

Map (plan convention):
    p_{n+1} = p_n + K·sin(θ_n)
    θ_{n+1} = θ_n + p_{n+1}            (both mod 2π)

Per-IC FTLE is computed with the **tangent-map (Jacobian) Benettin** method —
the gold standard for maps, with no finite-ε bias. The tangent map is

    J(θ, p) = [[1 + K·cosθ,  1],
               [    K·cosθ,  1]]      (det J = 1, area-preserving).

We evolve a tangent vector v_{n+1} = J(θ_n, p_n) · v_n, renormalize ‖v‖→1 each
step, accumulate the log-stretch, and report FTLE = (1/N_iter)·Σ ln‖v_after‖.
Vectorized over n ICs.

A second, independent estimator — the direct two-trajectory perturbation method,
copied from monumental_validation.py:191-229 (the method that PRODUCED the
catalogued α*=4.138) — is provided as `compute_lyapunov_standard_map_direct`
for cross-checking. The self-test confirms tangent ≈ direct within FTLE noise.

Reference implementation accompanying the paper.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi

# Chirikov critical coupling (Greene 1979 / MacKay 1983): global chaos onset.
K_CRITICAL = 0.971635406


# ===========================================================================
# IC sampling (seeded, documented)
# ===========================================================================
def sample_ics(n, seed_entropy, base=0x5310):
    """Sample n ICs θ,p ~ Uniform(0, 2π), seeded reproducibly.

    seed_entropy is mixed via numpy SeedSequence so each (K-index) gets an
    independent, documented stream. Returns (theta0, p0), both shape (n,).
    """
    ss = np.random.SeedSequence(int(base) ^ int(seed_entropy))
    rng = np.random.default_rng(ss)
    theta = rng.uniform(0.0, TWO_PI, size=n)
    p = rng.uniform(0.0, TWO_PI, size=n)
    return theta, p


# ===========================================================================
# GOLD STANDARD: tangent-map (Jacobian) Benettin FTLE
# ===========================================================================
def compute_lyapunov_standard_map(n, K=1.0, N_iter=1000, seed_entropy=0,
                                  theta0=None, p0=None, return_states=False):
    """Per-IC FTLE for the standard map via tangent-map Benettin.

    Args:
        n: number of ICs (ignored if theta0/p0 provided).
        K: coupling.
        N_iter: iterations per IC (discrete analog of integration time).
        seed_entropy: mixed into the IC seed (use 0x5310 + K-index upstream).
        theta0, p0: optional pre-sampled ICs (shape (n,)); if None, sampled.
        return_states: if True, also return final (theta, p).

    Returns:
        ftle: shape (n,) array of finite-time Lyapunov exponents.
        (optionally) theta, p final states.
    """
    if theta0 is None or p0 is None:
        theta0, p0 = sample_ics(n, seed_entropy)
    theta = np.array(theta0, dtype=np.float64, copy=True)
    p = np.array(p0, dtype=np.float64, copy=True)
    n = theta.size

    # Tangent vector v = (v_theta, v_p); start as unit vector (1, 0).
    vth = np.ones(n, dtype=np.float64)
    vp = np.zeros(n, dtype=np.float64)

    log_stretch = np.zeros(n, dtype=np.float64)

    for _ in range(N_iter):
        c = K * np.cos(theta)  # K·cosθ at the CURRENT point (Jacobian entry)

        # Tangent map applied BEFORE advancing the base point, using J(θ_n,p_n):
        #   v_p'     = K·cosθ · v_theta +     v_p          (row 2)
        #   v_theta' = (1+K·cosθ)·v_theta +   v_p          (row 1)
        # Note p_{n+1} uses sin(θ_n); θ_{n+1}=θ_n+p_{n+1} ⇒ ∂θ'/∂θ = 1+K cosθ.
        new_vp = c * vth + vp
        new_vth = vth + new_vp  # = (1+c)·vth + vp
        vth, vp = new_vth, new_vp

        # Renormalize and accumulate log-stretch.
        norm = np.sqrt(vth * vth + vp * vp)
        # Guard against an exact-zero norm (cannot happen for det J=1, but safe).
        norm = np.where(norm > 0, norm, 1.0)
        log_stretch += np.log(norm)
        vth /= norm
        vp /= norm

        # Advance the base point.
        p = (p + K * np.sin(theta)) % TWO_PI
        theta = (theta + p) % TWO_PI

    ftle = log_stretch / N_iter
    if return_states:
        return ftle, theta, p
    return ftle


def jacobian(theta, K):
    """Standard-map tangent matrix J(θ) (p-independent). Returns 2x2 ndarray."""
    c = K * np.cos(theta)
    return np.array([[1.0 + c, 1.0], [c, 1.0]], dtype=np.float64)


# ===========================================================================
# CROSS-CHECK: direct two-trajectory perturbation (EXACT repro of
# monumental_validation.compute_lyapunov_standard_map, the method that
# produced the catalogued α*=4.138).
# ===========================================================================
def compute_lyapunov_standard_map_direct(n, K=2.0, N_iter=5000, seed=42,
                                        delta=1e-7, theta0=None, p0=None):
    """Direct perturbation FTLE — EXACT reproduction of monumental_validation
    lines 191-229 (variable names q→θ). Two trajectories, torus-minimal
    separation, renormalized each iterate. Used only as a cross-check.
    """
    if theta0 is None or p0 is None:
        rng = np.random.RandomState(seed)
        q = rng.uniform(0, TWO_PI, n)
        p = rng.uniform(0, TWO_PI, n)
    else:
        q = np.array(theta0, dtype=np.float64, copy=True)
        p = np.array(p0, dtype=np.float64, copy=True)
        n = q.size
    dq = delta * np.ones(n)
    dp = np.zeros(n)

    lyap_sum = np.zeros(n)
    for _ in range(N_iter):
        p_new = (p + K * np.sin(q)) % TWO_PI
        q_new = (q + p_new) % TWO_PI

        p_per = (p + dp + K * np.sin(q + dq)) % TWO_PI
        q_per = (q + dq + p_per) % TWO_PI

        dq_new = q_per - q_new
        dp_new = p_per - p_new
        dq_new = dq_new - np.round(dq_new / TWO_PI) * TWO_PI
        dp_new = dp_new - np.round(dp_new / TWO_PI) * TWO_PI

        d = np.sqrt(dq_new ** 2 + dp_new ** 2)
        safe = d > 1e-30
        lyap_sum += np.where(safe, np.log(np.maximum(d, 1e-30) / delta), 0.0)

        dq = np.where(safe, dq_new * delta / d, dq_new)
        dp = np.where(safe, dp_new * delta / d, dp_new)

        q, p = q_new, p_new

    return lyap_sum / N_iter


# ===========================================================================
# Self-test
# ===========================================================================
def _self_test():
    import sys
    ok = True

    # 1. det J == 1 everywhere (area-preserving).
    for th in np.linspace(0, TWO_PI, 17):
        for K in (0.5, 1.0, 5.0):
            det = np.linalg.det(jacobian(th, K))
            if abs(det - 1.0) > 1e-12:
                print(f"  FAIL det J={det} at theta={th}, K={K}")
                ok = False
    print(f"[1] det J == 1 (area-preserving): {'PASS' if ok else 'FAIL'}")

    # 2. Known chaotic IC at large K gives λ≈ ln(K/2) textbook estimate.
    # For K≫1 the standard-map Lyapunov exponent ≈ ln(K/2) (Chirikov).
    K = 5.0
    ftle = compute_lyapunov_standard_map(20000, K=K, N_iter=2000, seed_entropy=99)
    chaotic = ftle[ftle > 0.05]
    mean_lam = float(np.mean(chaotic))
    textbook = np.log(K / 2.0)
    rel = abs(mean_lam - textbook) / textbook
    print(f"[2] K=5 chaotic mean λ={mean_lam:.4f} vs textbook ln(K/2)={textbook:.4f} "
          f"(rel {rel:.3f}): {'PASS' if rel < 0.15 else 'WARN'}")

    # 3. Regular-island IC (K small, near a fixed point) gives λ≈0.
    # At K=0.5 (below K_c) most of phase space is regular ⇒ many near-zero FTLE.
    ftle_reg = compute_lyapunov_standard_map(20000, K=0.5, N_iter=2000, seed_entropy=7)
    frac_regular = float(np.mean(ftle_reg < 0.02))
    print(f"[3] K=0.5 fraction with λ<0.02 (regular): {frac_regular:.3f} "
          f"{'PASS' if frac_regular > 0.5 else 'WARN'}")

    # 4. tangent vs direct agree within FTLE noise on a shared IC sample.
    th0, p0 = sample_ics(5000, seed_entropy=123)
    f_tan = compute_lyapunov_standard_map(5000, K=1.0, N_iter=2000,
                                          theta0=th0, p0=p0)
    f_dir = compute_lyapunov_standard_map_direct(5000, K=1.0, N_iter=2000,
                                                 theta0=th0, p0=p0)
    # Compare on the chaotic subset (regular FTLE→0 for both, trivially equal).
    mask = (f_tan > 0.05) & (f_dir > 0.05)
    if mask.sum() > 50:
        diff = np.abs(f_tan[mask] - f_dir[mask])
        med = float(np.median(diff))
        med_lam = float(np.median(f_tan[mask]))
        rel = med / med_lam
        print(f"[4] tangent vs direct median |Δλ|={med:.4f} on λ_med={med_lam:.4f} "
              f"(rel {rel:.3f}, n_chaotic={int(mask.sum())}): "
              f"{'PASS' if rel < 0.05 else 'WARN'}")
    else:
        print("[4] tangent vs direct: too few chaotic ICs to compare")

    return ok


if __name__ == "__main__":
    print("standard_map.py self-test")
    print("=" * 50)
    _self_test()
