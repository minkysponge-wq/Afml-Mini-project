# Learning to Initialise Viscosity Estimation from Sparse PDE Measurements

Research code for a controlled, cost-accounted study: does a small supervised network that predicts
log-viscosity reduce end-to-end cost of estimating Burgers viscosity from sparse sensors, compared with
strong classical alternatives (direct bounded scalar search, coarse grid, nearest neighbour)?

**Status (2026-10-23): Phase A (simulator validation + classical recovery pilot) only.** No ML model has
been trained yet. See `RESEARCH_LOG.md` for what has and has not been verified.

## Setup
```
python -m pip install -r requirements.txt     # Python 3.10
```

## Commands
| What | Command | Approx. time (laptop CPU) |
|---|---|---|
| Unit tests (simulator correctness) | `python -m pytest -q` | ~10 s |
| Smoke test (3 inverse problems) | `python -m blinit.experiments.phase_a --quick` | ~2 min |
| Phase A pilot (20 problems + grid study) | `python -m blinit.experiments.phase_a` | ~50–60 min |
| Phase A figures/summary from saved results | `python -m blinit.experiments.phase_a_report --dir results/phaseA` | ~10 s |

## Layout
```
configs/        versioned JSON configs (simulator_v1, phaseA_v1)
blinit/         simulator.py (FV Burgers solver), observe.py (sensors/masks), inverse.py (estimators),
                exact.py (Cole–Hopf), experiments/ (phase scripts)
tests/          pytest suite
results/        machine-readable outputs (runs.csv, problems.npz, meta.json, summary.json, figures/)
docs/           literature_audit.md, bibliography.md
RESEARCH_LOG.md decisions, commands, results, limitations
```

## Conventions (see `configs/simulator_v1.json`)
PDEBench Burgers convention: `u_t + (u^2/2)_x = (nu/pi) u_xx` on [0,1), periodic. `nu ∈ [0.01, 0.1]` is
PDEBench's `epsilon`, so the effective diffusivity is `nu/pi`. The solver is an independent NumPy
re-implementation of PDEBench's scheme (PDEBench's licence restricts redistribution of derivatives,
so no PDEBench code is included). Software developed by NEC Laboratories Europe (PDEBench) was used
as the reference for the numerical method.
