# Reproduction bundle — *An exact exponential fixed point and metastable shape manifolds in a renormalization cascade of finite-time Lyapunov distributions*

Code, data, and result files that reproduce every number and figure in the paper

> R. Kapoor, *An exact exponential fixed point and metastable shape manifolds in a renormalization
> cascade of finite-time Lyapunov distributions: a two-class split of chaotic dynamics* (2026).

The manuscript (source + PDF) is included under [`paper/`](paper/).

## What the paper is about

The distribution of finite-time Lyapunov exponents (FTLEs) is compressed, by a **renormalization
cascade** `T` (mean-normalized pair-differencing + a maximum-likelihood tail reading), onto an
**exact exponential fixed point** with a closed-form reading `α*_Exp(0.2) = 3.3494`. The cascade is
solved exactly (unique fixed point; one-step Gaussian collapse; cumulant-halving that derives the
empirical decay constant `r ≈ 1/2`; a gapless non-normal spectrum giving algebraic relaxation).
"`α* ≈ 4`" is shown to be a **metastable plateau**, not a constant. What survives is a **two-class
split**: mixed-phase chaos is platykurtic and reads *above* the fixed point; fully chaotic /
dissipative / near-regular dynamics is leptokurtic and reads *below* it — sign-perfect across 11
ensembles from 5 systems, with an out-of-sample Hénon–Heiles confirmation.

## Layout

```
code/       the cascade operator + the standard-map engine + all analysis scripts
data/       raw FTLE ensembles (.npy) analysed by the scripts
results/    the JSON outputs the scripts produce (checked in for convenience)
paper/      manuscript source (main.tex), compiled PDF, and figures/
```

## Running it

Requires Python 3.10+ with `numpy`, `scipy`, `numba`, `matplotlib`, and `mpmath`. From `code/`:

```bash
cd code
MPLBACKEND=Agg python3 synthetic_battery.py      # calibration + two-class split
MPLBACKEND=Agg python3 addendum_mechanism.py     # matched-null, metastable slow manifold
MPLBACKEND=Agg python3 linearization.py          # exact cumulant flow + eigenfamily
MPLBACKEND=Agg python3 tsweep_analysis.py        # frozen-variance horizon sweep
MPLBACKEND=Agg python3 energy_mixture.py         # energy-shell mixture decomposition
MPLBACKEND=Agg python3 hh_sweep.py               # Hénon–Heiles out-of-sample sweep (slow)
MPLBACKEND=Agg python3 make_figs_9_10.py         # Figures 9 and 10
python3 constants.py                             # the number-theoretic audit library
```

Each analysis script reads its inputs from `../data`, writes JSON to `../results`, and figures to
`../paper/figures`. Paths are resolved relative to the script, so the layout above is all that is
needed.

## Script → paper map

| Script | Paper section | Produces |
|--------|---------------|----------|
| `cascade.py` | §2 (operator), used by all | the cascade operator `T` and the tail reading `η_f` |
| `synthetic_battery.py` | §4 calibration, §5 two-class split | Figs 1, 2, 3, 5; synthetic battery + real-system readings |
| `addendum_mechanism.py` | §5 matched null, §6 metastability | Figs 6, 7, 8; matched-Gaussian nulls, deep-cascade slow manifold |
| `linearization.py` | §2.7 | cumulant-halving identity, eigenfamily `μ(s)=2/(2−s)`, polynomial tower |
| `tsweep_analysis.py` | §7 | frozen chaotic-subset variance across horizons |
| `energy_mixture.py` | §7 | energy-shell mixture `η²` of the FTLE variance |
| `hh_sweep.py` | §8 | Hénon–Heiles energy sweep (Yoshida-4 + variational Benettin) |
| `make_figs_9_10.py` | §7, §8 | Figs 9, 10 |
| `constants.py` | §10 | the 67-constant + rational + PSLQ audit library |
| `standard_map.py` | §3 | the Chirikov standard-map FTLE engine |

## Data

`data/` holds the raw FTLE ensembles (mostly `n = 5×10^5` samples each; one `5×10^6` double-pendulum
run; the Hénon–Heiles per-energy arrays at `n = 2.5×10^5`). File names encode the system and horizon,
e.g. `standard_map_K5_ftle_n500k.npy`, `lorenz_ftle_n500k_T100.npy`,
`disc024_ftle_T{5..200}_N50000.npy` (the horizon sweep).

## License

Code under the MIT License (see [`LICENSE`](LICENSE)). The manuscript and data files are released
under CC BY 4.0, consistent with the preprint.
