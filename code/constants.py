"""Number-hunting library for the paper.

Builds:

1. KNOWN_CONSTANTS: dict[str, mpmath.mpf]
   ~80 named/algebraic/transcendental entries (transcendentals & basics, named
   constants, Riemann zeta values, Feigenbaum family, RMT constants, KAM /
   standard-map values, dynamical-system named values, Wave-1 ratio targets)
   plus a separately-handled set of half-integer rationals m/n with m, n <= 16.

2. match(value, threshold) -> list[Match]
   Returns matches against KNOWN_CONSTANTS (direct and reciprocal) sorted by
   relative error.

3. pairwise_ratio_match(values, threshold) -> list[RatioMatch]
   For every unordered pair (a, b) and every constant c, checks both a/b ~= c
   and b/a ~= c. Deduplicates by treating (a, b) ~ (b, a).

4. continued_fraction(value, n_terms) -> CFResult
   Simple continued fraction expansion via mpmath. Pads with zeros and sets
   `terminated=True` if rational.

5. pslq_relation(values, precision_decimals) -> tuple[int, ...] | None
   Wraps mpmath.pslq with maxcoeff=1e9 and bounded precision. Returns the
   coefficient tuple (c_1, ..., c_k) such that sum c_i * v_i ~= 0, or None.

6. fpr_calibrate_match(...) -> dict
   Stub. A calibration pass populates the real thresholds via Monte-Carlo on a null pool.

All constants are computed at module-import time using mpmath.mp.dps = 50.

Convention for relative_error:
    rel_err = abs(value - library_value) / abs(library_value)
i.e., normalized to the library (canonical) value. This is robust when
library_value is nonzero (all our entries are positive).

Sources (cited in the SOURCES dict):
- mpmath 1.3.0 built-in high-precision routines (mp.pi, mp.e, mp.euler,
  mp.zeta, mp.catalan, mp.khinchin, mp.glaisher) -- self-cited as
  "computed via mpmath.mp.{...} at dps=50".
- Levy's constant: gamma_tilde = exp(pi^2 / (12 ln 2)), computed from mpmath.
- Brun's constant B_2 = 1.902160583104 (Klyve 2013; Sebah-Gourdon numerical
  table). NOT a closed form -- stored as a high-precision literal.
- Feigenbaum constants delta, alpha, omega: Briggs 1991 "How to calculate
  the Feigenbaum constants" + Briggs 1989 thesis tables. Stored as literals.
- Tracy-Widom values F_beta(0): specification-provided values
  (F_1(0) ~= -1.27109, F_2(0) ~= -1.77109, F_4(0) ~= -2.30688). The latter
  two match Bornemann 2010 (arXiv:1004.4884) TW means to 5 decimals;
  TW_1 spec value differs from Bornemann TW_1 mean (-1.20652). Both are
  stored under separate keys so downstream consumers can choose.
- Greene R_c = 0.250888 and K_c = 0.971635406: MacKay 1983, Greene 1979.
- Sinai-billiard topological entropy h ~ 0.4434: spec value, citing
  Chernov-Markarian "Chaotic Billiards" (AMS 2006).
- Lorenz dimension D ~ 2.06: Sparrow 1982 "The Lorenz Equations".
- Chirikov diffusion D_0 ~ 0.2: Chirikov 1979 standard-map review.

DO NOT modify this file unless you are the calibration pass or have
documented reason. The library is used by the number-theoretic audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Union

import mpmath
from mpmath import mp, mpf

# ----------------------------------------------------------------------------
# Module-wide precision: 50 decimal digits.
# ----------------------------------------------------------------------------
mp.dps = 50

NumberLike = Union[int, float, mpf]

# ----------------------------------------------------------------------------
# 1. KNOWN_CONSTANTS dict (built once at import time).
# ----------------------------------------------------------------------------

KNOWN_CONSTANTS: dict[str, mpf] = {}
CATEGORIES: dict[str, str] = {}
SOURCES: dict[str, str] = {}


def _add(name: str, value: mpf, category: str, source: str) -> None:
    """Insert into KNOWN_CONSTANTS with category + source metadata."""
    if name in KNOWN_CONSTANTS:
        raise ValueError(f"Duplicate constant key: {name}")
    KNOWN_CONSTANTS[name] = mpf(value)
    CATEGORIES[name] = category
    SOURCES[name] = source


# --- 1a. Transcendentals & basics ------------------------------------------
_CAT = "transcendentals_basics"
_SRC_MP = "computed via mpmath at dps=50"

_add("pi",                mp.pi,                              _CAT, _SRC_MP)
_add("two_pi",            2 * mp.pi,                          _CAT, _SRC_MP)
_add("four_pi",           4 * mp.pi,                          _CAT, _SRC_MP)
_add("pi_squared",        mp.pi ** 2,                         _CAT, _SRC_MP)
_add("four_pi_squared",   4 * mp.pi ** 2,                     _CAT, _SRC_MP)
_add("pi_cubed",          mp.pi ** 3,                         _CAT, _SRC_MP)
_add("inv_pi",            1 / mp.pi,                          _CAT, _SRC_MP)
_add("inv_two_pi",        1 / (2 * mp.pi),                    _CAT, _SRC_MP)
_add("inv_four_pi_squared", 1 / (4 * mp.pi ** 2),             _CAT, _SRC_MP)
_add("e",                 mp.e,                               _CAT, _SRC_MP)
_add("e_squared",         mp.e ** 2,                          _CAT, _SRC_MP)
_add("inv_e",             1 / mp.e,                           _CAT, _SRC_MP)
_add("sqrt_2",            mpmath.sqrt(2),                     _CAT, _SRC_MP)
_add("sqrt_3",            mpmath.sqrt(3),                     _CAT, _SRC_MP)
_add("sqrt_5",            mpmath.sqrt(5),                     _CAT, _SRC_MP)
_add("sqrt_7",            mpmath.sqrt(7),                     _CAT, _SRC_MP)
_add("sqrt_3_over_10",    mpmath.sqrt(mpf(3) / 10),           _CAT, _SRC_MP)
_add("sqrt_7_over_10",    mpmath.sqrt(mpf(7) / 10),           _CAT, _SRC_MP)
_add("sqrt_10_over_3",    mpmath.sqrt(mpf(10) / 3),           _CAT, _SRC_MP)
_add("ln_2",              mpmath.log(2),                      _CAT, _SRC_MP)
_add("ln_3",              mpmath.log(3),                      _CAT, _SRC_MP)
_add("ln_pi",             mpmath.log(mp.pi),                  _CAT, _SRC_MP)
_add("ln_2pi",            mpmath.log(2 * mp.pi),              _CAT, _SRC_MP)
_add("log10_2",           mpmath.log10(2),                    _CAT, _SRC_MP)
_add("log10_pi",          mpmath.log10(mp.pi),                _CAT, _SRC_MP)

# --- 1b. Named constants ---------------------------------------------------
_CAT = "named_constants"
_PHI = (1 + mpmath.sqrt(5)) / 2  # golden ratio
_LEVY = mpmath.exp(mp.pi ** 2 / (12 * mpmath.log(2)))  # Levy's constant
_BRUN = mpf("1.902160583104")  # Klyve 2013 numerical estimate

_add("euler_mascheroni",     mp.euler,                        _CAT, _SRC_MP)
_add("phi_golden",           _PHI,                            _CAT, _SRC_MP)
_add("inv_phi",              1 / _PHI,                        _CAT, _SRC_MP)
_add("apery_zeta_3",         mpmath.zeta(3),                  _CAT, _SRC_MP)
_add("catalan",              mp.catalan,                      _CAT, _SRC_MP)
_add("khinchin",             mp.khinchin,                     _CAT, _SRC_MP)
_add("levy_gamma_tilde",     _LEVY,                           _CAT,
     "Levy's constant gamma_tilde = exp(pi^2/(12 ln 2)); computed via mpmath at dps=50")
_add("glaisher_kinkelin",    mp.glaisher,                     _CAT, _SRC_MP)
_add("brun_b2",              _BRUN,                           _CAT,
     "Brun's constant B_2: Klyve 2013 (rigorous bounds), Sebah-Gourdon table; numerical literal to 12 decimals")
_add("four_over_pi",         4 / mp.pi,                       _CAT, _SRC_MP)
_add("pi_over_four",         mp.pi / 4,                       _CAT, _SRC_MP)
# SPECULATIVE ALPHA* CANDIDATE  =====================================
_add("four_minus_one_over_4pi_squared",
     4 - 1 / (4 * mp.pi ** 2),                                _CAT,
     "speculative alpha* candidate: 4 - 1/(4 pi^2); computed via mpmath at dps=50")
# ===========================================================================

# Convenience: 2/pi appears in spec as both '2/pi' and '1/(pi/2)' for the
# match test feeding 0.6366. Expose under two_over_pi (canonical).
_add("two_over_pi",          2 / mp.pi,                       _CAT, _SRC_MP)

# --- 1c. Riemann zeta values ----------------------------------------------
_CAT = "riemann_zeta"
_add("zeta_2",  mp.pi ** 2 / 6,         _CAT, "zeta(2) = pi^2/6 (Euler 1734); computed via mpmath at dps=50")
_add("zeta_3",  mpmath.zeta(3),         _CAT, _SRC_MP)
_add("zeta_4",  mp.pi ** 4 / 90,        _CAT, "zeta(4) = pi^4/90 (Euler); computed via mpmath at dps=50")
_add("zeta_5",  mpmath.zeta(5),         _CAT, _SRC_MP)
_add("zeta_7",  mpmath.zeta(7),         _CAT, _SRC_MP)

# --- 1d. Feigenbaum family -------------------------------------------------
_CAT = "feigenbaum"
# Feigenbaum constants are universal but not in mpmath; use Briggs 1991 values
# to 25-30 digits.
_add("feigenbaum_delta",  mpf("4.6692016091029906718532038204662016172581855774757686327"),
     _CAT, "Briggs 1991 'How to calculate the Feigenbaum constants' (Math.Comp.); 25-30 digit numerical literal")
_add("feigenbaum_alpha",  mpf("2.5029078750958928222839028732182157863812713381532659"),
     _CAT, "Briggs 1991; 25-30 digit numerical literal")
_add("feigenbaum_omega",  mpf("0.7521211"),
     _CAT, "Feigenbaum subleading universal exponent omega ~ 0.7521 (spec value); 7-digit literal")

# --- 1e. RMT (Random Matrix Theory) ---------------------------------------
_CAT = "rmt"
_add("rmt_goe_wigner_normalization", mp.pi / 2,           _CAT,
     "GOE Wigner surmise prefactor pi/2; computed via mpmath at dps=50")
_add("rmt_gue_prefactor",            mpf(32) / (mp.pi ** 2), _CAT,
     "GUE coefficient 32/pi^2; computed via mpmath at dps=50")
_add("rmt_delta3_prefactor",         1 / mp.pi ** 2,      _CAT,
     "Delta_3 statistic prefactor 1/pi^2 (Dyson-Mehta); computed via mpmath at dps=50")
# Tracy-Widom F_beta(0) values. Spec provides specific numerical values; we
# cite the spec and cross-check against Bornemann 2010 in the notes.
_add("rmt_tw_f1_at_zero",  mpf("-1.27109"), _CAT,
     "Tracy-Widom F_1(0) per spec; spec-provided 5-decimal value, NOTE: differs from Bornemann 2010 TW_1 mean -1.20652. See WAVE_0b_SUB2_NOTES.md.")
_add("rmt_tw_f2_at_zero",  mpf("-1.77109"), _CAT,
     "Tracy-Widom F_2(0) per spec; spec-provided 5-decimal value, matches Bornemann 2010 TW_2 mean -1.77108680 to 5 decimals")
_add("rmt_tw_f4_at_zero",  mpf("-2.30688"), _CAT,
     "Tracy-Widom F_4(0) per spec; spec-provided 5-decimal value, matches Bornemann 2010 TW_4 mean -2.30688081 to 5 decimals")
# Cross-reference TW_1 mean from Bornemann (used by some downstream consumers)
_add("rmt_tw_f1_bornemann_mean", mpf("-1.20652574"), _CAT,
     "Tracy-Widom TW_1 distribution mean from Bornemann 2010 (arXiv:1004.4884); alternative to spec value")

# --- 1f. KAM / standard-map / Chirikov ------------------------------------
_CAT = "kam_standard_map"
_add("greene_residue_critical",
     mpf("0.250888"),
     _CAT, "Greene residue critical R_c = 0.250888; MacKay 1983, Greene 1979; spec-stated 6-decimal numerical literal")
_add("chirikov_k_critical",
     mpf("0.971635406"),
     _CAT, "Chirikov standard-map critical K_c = 0.971635406; Greene 1979, MacKay 1983; 9-decimal numerical literal")
_add("golden_rotation",
     1 - 1 / _PHI,
     _CAT, "Golden rotation 1 - 1/phi (irrationality measure 1); computed via mpmath at dps=50")
_add("silver_rotation",
     mpmath.sqrt(2) - 1,
     _CAT, "Silver rotation sqrt(2) - 1; computed via mpmath at dps=50")
_add("chirikov_diffusion_d0",
     mpf("0.2"),
     _CAT, "Chirikov diffusion coefficient D_0 ~ 0.2 in standard map at K > K_c; Chirikov 1979 review; 1-decimal coarse value")

# --- 1g. Dynamical-system named values ------------------------------------
_CAT = "dynamical_named"
_add("sinai_billiard_entropy",
     mpf("0.4434"),
     _CAT, "Sinai-billiard topological entropy h ~ 0.4434; Chernov-Markarian 2006 'Chaotic Billiards' (AMS); 4-decimal literal")
_add("baker_map_entropy",
     mpmath.log(2),
     _CAT, "Baker-map entropy h = ln 2; computed via mpmath at dps=50")
_add("lorenz_attractor_dimension",
     mpf("2.06"),
     _CAT, "Lorenz attractor information dimension D ~ 2.06; Sparrow 1982 'The Lorenz Equations' Kaplan-Yorke; 2-decimal literal")

# --- 1h. ratio targets (lambda*/Lambda* ~ 5.496) ------------------------
_CAT = "ratio_targets"
_add("sqrt_30_21",     mpmath.sqrt(mpf("30.21")),         _CAT,
     "ratio candidate sqrt(30.21); computed via mpmath at dps=50")
_add("pi_sqrt_3",      mp.pi * mpmath.sqrt(3),            _CAT,
     "ratio candidate sqrt(pi^2 * 3) = pi * sqrt(3); computed via mpmath at dps=50")
_add("sqrt_10_pi",     mpmath.sqrt(10 * mp.pi),           _CAT,
     "ratio candidate sqrt(10 pi); computed via mpmath at dps=50")
# Auxiliary radicands so pairwise-ratio match has them
_add("aux_30_21",      mpf("30.21"),                      _CAT,
     "Auxiliary: radicand 30.21 from sqrt(30.21) candidate")
_add("aux_pi_squared_times_3", mp.pi ** 2 * 3,           _CAT,
     "Auxiliary: radicand pi^2 * 3 from sqrt candidate")
_add("aux_ten_pi",     10 * mp.pi,                       _CAT,
     "Auxiliary: radicand 10 * pi from sqrt candidate")


# ----------------------------------------------------------------------------
# 2. Half-integer rationals m/n with m, n <= 16 (reduced; handled separately
#    from KNOWN_CONSTANTS to keep the named library clean). pairwise_ratio_match
#    and match both consult this dict.
# ----------------------------------------------------------------------------

def _build_rationals(max_mn: int = 16) -> dict[str, mpf]:
    """Generate reduced fractions m/n with 1 <= m,n <= max_mn (m != n allowed,
    integer matches up to max_mn^2 separate). Skip integer cases (n=1) and the
    integer m=n cases since they reduce to 1 and pure integers (handled
    separately)."""
    out: dict[str, mpf] = {}
    for m in range(1, max_mn + 1):
        for n in range(1, max_mn + 1):
            if n == 1 and m == 1:
                continue
            frac = Fraction(m, n)
            # Use the reduced form as the canonical key
            key = f"frac_{frac.numerator}_over_{frac.denominator}"
            if key in out:
                continue
            out[key] = mpf(frac.numerator) / mpf(frac.denominator)
    return out


RATIONALS_M_N: dict[str, mpf] = _build_rationals(16)


# ----------------------------------------------------------------------------
# 3. Match dataclasses.
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Match:
    """A single match between an input value and a library entry."""
    constant_name: str
    library_value: float            # cast to float for downstream JSON-safety
    ratio: float                    # value / library_value
    relative_error: float           # |value - library_value| / |library_value|
    library_value_str: str          # string repr of full-precision mpf

    def to_dict(self) -> dict:
        return {
            "constant_name": self.constant_name,
            "library_value": self.library_value,
            "ratio": self.ratio,
            "relative_error": self.relative_error,
            "library_value_str": self.library_value_str,
        }


@dataclass(frozen=True)
class RatioMatch:
    """A pair (a, b) whose ratio matches some library constant."""
    a: float
    b: float
    ratio: float                    # the one tested (a/b or b/a)
    direction: str                  # "a_over_b" or "b_over_a"
    constant_name: str
    library_value: float
    relative_error: float

    def to_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "ratio": self.ratio,
            "direction": self.direction,
            "constant_name": self.constant_name,
            "library_value": self.library_value,
            "relative_error": self.relative_error,
        }


@dataclass(frozen=True)
class CFResult:
    """Result of continued_fraction()."""
    terms: list[int]
    terminated: bool                # True iff the expansion terminated (rational)
    n_requested: int

    def to_dict(self) -> dict:
        return {
            "terms": list(self.terms),
            "terminated": self.terminated,
            "n_requested": self.n_requested,
        }


# ----------------------------------------------------------------------------
# 4. Match function.
# ----------------------------------------------------------------------------

def _rel_err(value: mpf, library_value: mpf) -> mpf:
    """Relative error normalized to the library (canonical) value."""
    if library_value == 0:
        return mpf("inf") if value != 0 else mpf(0)
    return abs(value - library_value) / abs(library_value)


def match(value: NumberLike,
          threshold: float = 0.001,
          include_rationals: bool = True,
          include_small_integers: bool = True,
          max_small_integer: int = 1024) -> list[Match]:
    """Find library constants close to `value`.

    For each library entry c, compute relative_error(value, c). Keep entries
    with rel_err < threshold. Also test the reciprocal: if 1/value is close
    to some constant c, emit a match with name `<c>_reciprocal`.

    For pure integer values (or near-integer with rel_err < threshold to an
    integer <= max_small_integer): emit a `integer_<N>` match. This is a small
    special case since pure integers cause noisy pairwise-ratio matches.

    Args:
        value: input number (int, float, or mpmath.mpf).
        threshold: maximum relative error to emit.
        include_rationals: also test against RATIONALS_M_N.
        include_small_integers: also test against integers 1..max_small_integer.
        max_small_integer: upper bound for integer-match special case.

    Returns:
        list[Match], sorted ascending by relative_error.
    """
    v = mpf(value)
    abs_v = abs(v)
    threshold_mp = mpf(threshold)

    matches: list[Match] = []

    # --- Direct matches against KNOWN_CONSTANTS ---
    for name, libv in KNOWN_CONSTANTS.items():
        err = _rel_err(v, libv)
        if err < threshold_mp:
            ratio = float(v / libv) if libv != 0 else float("nan")
            matches.append(Match(
                constant_name=name,
                library_value=float(libv),
                ratio=ratio,
                relative_error=float(err),
                library_value_str=mpmath.nstr(libv, 30),
            ))

    # --- Reciprocal matches: 1/v against KNOWN_CONSTANTS ---
    if abs_v > 0:
        v_inv = 1 / v
        for name, libv in KNOWN_CONSTANTS.items():
            err = _rel_err(v_inv, libv)
            if err < threshold_mp:
                ratio = float(v_inv / libv) if libv != 0 else float("nan")
                matches.append(Match(
                    constant_name=f"{name}_reciprocal",
                    library_value=float(libv),
                    ratio=ratio,
                    relative_error=float(err),
                    library_value_str=mpmath.nstr(libv, 30),
                ))

    # --- Rational matches m/n with m,n <= 16 ---
    if include_rationals:
        for name, libv in RATIONALS_M_N.items():
            err = _rel_err(v, libv)
            if err < threshold_mp:
                ratio = float(v / libv) if libv != 0 else float("nan")
                matches.append(Match(
                    constant_name=name,
                    library_value=float(libv),
                    ratio=ratio,
                    relative_error=float(err),
                    library_value_str=mpmath.nstr(libv, 30),
                ))

    # --- Small-integer matches ---
    if include_small_integers and abs_v > 0:
        # nearest positive integer to |v|
        rounded = int(mpmath.nint(abs_v))
        if 1 <= rounded <= max_small_integer:
            libv = mpf(rounded)
            err = _rel_err(abs_v, libv)
            if err < threshold_mp:
                matches.append(Match(
                    constant_name=f"integer_{rounded}",
                    library_value=float(libv),
                    ratio=float(abs_v / libv),
                    relative_error=float(err),
                    library_value_str=str(rounded),
                ))

    # Sort ascending by relative_error.
    matches.sort(key=lambda m: m.relative_error)
    return matches


# ----------------------------------------------------------------------------
# 5. Pairwise-ratio match.
# ----------------------------------------------------------------------------

def pairwise_ratio_match(values: list[NumberLike],
                         threshold: float = 0.001) -> list[RatioMatch]:
    """For all unordered pairs (a, b) with a, b in `values` and a != b position,
    test a/b and b/a against every constant in KNOWN_CONSTANTS plus
    RATIONALS_M_N. Returns matches sorted by relative_error.

    The pair (i, j) is considered as "i < j" to avoid double-counting.
    Both directions (a/b, b/a) of each pair are tested, but each direction is
    only emitted once.

    Args:
        values: list of input numbers.
        threshold: maximum relative error to emit.

    Returns:
        list[RatioMatch], sorted ascending by relative_error.
    """
    n = len(values)
    if n < 2:
        return []

    vs_mp = [mpf(v) for v in values]
    threshold_mp = mpf(threshold)

    combined_library: dict[str, mpf] = {**KNOWN_CONSTANTS, **RATIONALS_M_N}

    results: list[RatioMatch] = []

    for i in range(n):
        for j in range(i + 1, n):
            a, b = vs_mp[i], vs_mp[j]
            if a == 0 or b == 0:
                continue
            r_ab = a / b
            r_ba = b / a
            for name, libv in combined_library.items():
                err_ab = _rel_err(r_ab, libv)
                if err_ab < threshold_mp:
                    results.append(RatioMatch(
                        a=float(a), b=float(b),
                        ratio=float(r_ab),
                        direction="a_over_b",
                        constant_name=name,
                        library_value=float(libv),
                        relative_error=float(err_ab),
                    ))
                err_ba = _rel_err(r_ba, libv)
                if err_ba < threshold_mp:
                    results.append(RatioMatch(
                        a=float(a), b=float(b),
                        ratio=float(r_ba),
                        direction="b_over_a",
                        constant_name=name,
                        library_value=float(libv),
                        relative_error=float(err_ba),
                    ))

    results.sort(key=lambda r: r.relative_error)
    return results


# ----------------------------------------------------------------------------
# 6. Continued fraction.
# ----------------------------------------------------------------------------

def continued_fraction(value: NumberLike, n_terms: int = 10) -> CFResult:
    """Simple continued fraction expansion of `value` via the standard
    Euclidean-style algorithm at module precision.

    Returns CFResult(terms, terminated, n_requested) where:
      - terms is a length-n_terms list of int. If the expansion terminates
        early (i.e., value is rational at working precision), the remaining
        slots are 0-padded and `terminated=True`.
      - terminated indicates whether the fractional part hit zero (rational).

    Termination criterion: if the fractional part falls below 10^-(dps - 5),
    treat as terminated (avoids spurious terms from near-rational floats).
    """
    if n_terms < 1:
        raise ValueError("n_terms must be >= 1")

    # Raise precision a bit to give the algorithm headroom.
    with mp.workdps(mp.dps + 10):
        v = mpf(value)
        terms: list[int] = []
        terminated = False
        # Termination threshold relative to working precision.
        eps = mpf(10) ** (-(mp.dps - 5))

        for _ in range(n_terms):
            a = int(mpmath.floor(v))
            terms.append(a)
            frac = v - a
            if abs(frac) < eps:
                terminated = True
                break
            v = 1 / frac

    # Pad to n_terms if terminated early.
    while len(terms) < n_terms:
        terms.append(0)

    return CFResult(terms=terms, terminated=terminated, n_requested=n_terms)


# ----------------------------------------------------------------------------
# 7. PSLQ integer relation finder.
# ----------------------------------------------------------------------------

def pslq_relation(values: list[NumberLike],
                  precision_decimals: int = 30,
                  maxcoeff: int = 10 ** 9) -> Optional[tuple[int, ...]]:
    """Find integer coefficients (c_1, ..., c_k) such that
    sum_i c_i * values[i] ~= 0 to the working precision, or None.

    Wraps mpmath.pslq with:
      - mp.dps temporarily raised to `precision_decimals` if higher than
        current (we never lower, to avoid breaking module state).
      - maxcoeff bound (default 1e9).

    Args:
        values: list of input numbers (at least 2 entries).
        precision_decimals: working precision in decimal digits.
        maxcoeff: upper bound on |c_i|.

    Returns:
        tuple[int, ...] of coefficients (one per input), or None if no
        relation was found within (precision, maxcoeff).
    """
    if len(values) < 2:
        raise ValueError("Need at least 2 values for an integer relation.")

    target_dps = max(precision_decimals, 30)

    with mp.workdps(target_dps):
        vs_mp = [mpf(v) for v in values]
        result = mpmath.pslq(vs_mp, maxcoeff=maxcoeff)

    if result is None:
        return None
    return tuple(int(c) for c in result)


# ----------------------------------------------------------------------------
# 8. FPR calibration stub (the calibration pass fills this in).
# ----------------------------------------------------------------------------

def fpr_calibrate_match(values: list[NumberLike] | None = None,
                        n_replicates: int = 1000,
                        thresholds: Optional[list[float]] = None) -> dict:
    """Placeholder stub. False-positive-rate calibration is performed in
    a calibration pass over a null pool of ~1000
    junk numbers and measuring empirical match rates. Until then, the
    audit uses the default thresholds tau_1=0.003, tau_2=0.0003,
    tau_3=0.00001 (1e-5).

    Returns:
        {
          "calibrated": False,
          "thresholds_used_as_default": [0.003, 0.0003, 1e-5],
          "wave_4a_will_overwrite": True
        }
    """
    if thresholds is None:
        thresholds = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    return {
        "calibrated": False,
        "thresholds_used_as_default": [0.003, 0.0003, 1e-5],
        "wave_4a_will_overwrite": True,
        "n_replicates_if_run": n_replicates,
        "thresholds_to_probe": list(thresholds),
        "n_values_supplied": 0 if values is None else len(values),
    }


# ----------------------------------------------------------------------------
# 9. Inspection helpers.
# ----------------------------------------------------------------------------

def library_size() -> int:
    """Total size of KNOWN_CONSTANTS (named entries only — excludes rationals)."""
    return len(KNOWN_CONSTANTS)


def library_size_breakdown() -> dict:
    """Per-category counts for KNOWN_CONSTANTS plus the rational generator."""
    counts: dict[str, int] = {}
    for name, cat in CATEGORIES.items():
        counts[cat] = counts.get(cat, 0) + 1
    counts["rationals_m_n_leq_16"] = len(RATIONALS_M_N)
    return counts


if __name__ == "__main__":
    # Self-print summary when run directly.
    bd = library_size_breakdown()
    print(f"KNOWN_CONSTANTS size (named): {library_size()}")
    print(f"RATIONALS_M_N size:            {len(RATIONALS_M_N)}")
    print("Breakdown:")
    for k, v in bd.items():
        print(f"  {k:35s} {v:4d}")
