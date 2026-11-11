# Learning to Initialise Viscosity Estimation from Sparse PDE Measurements

Research code for a controlled, cost-accounted study: does a small supervised network that predicts
log-viscosity reduce end-to-end cost of estimating Burgers viscosity from sparse sensors, compared with
strong classical alternatives (direct bounded scalar search, coarse grid, nearest neighbour)?

**Status (2026-11-11):** Phases A–D done, including the pre-registered pilot test. **Result: negative.** A saving over
Brent's direct search is not established: the MLP start saves ~1 CPU-s/query when LSQ stops loosely (xtol 1e-3),
but nothing when LSQ is at least as precise as Brent (xtol 1e-4). The 2x target is not met. The learned start does
make the LSQ refiner itself ~25% cheaper than a midpoint start. A separately pre-registered follow-up, which uses the
MLP to set a ±0.5 bracket for Brent, gives a small but consistent saving at matched precision: 8–10% (≈0.7–0.9
CPU-s/query on average), all 3 seeds, break-even after ~2,400–3,300 queries. The report draft is in
`docs/report_draft.md`. Every claim is traced in `RESEARCH_LOG.md`.

## Setup
```
python -m pip install -r requirements.txt     # Python 3.10 (pinned versions used for all logged runs)
python -m blinit.experiments.build_dataset --config configs/phaseB_v2.json --workers 8   # ~4 min; data/ is not in git
```
The dataset regenerates bit-identically from its config. Rebuilding overwrites `data/phaseB_v2/integrity.json`,
whose generation timings feed the break-even numbers, so restore it afterwards with
`git checkout -- data/phaseB_v2/integrity.json`. The pickled MLPs in `models/` were saved with scikit-learn 1.7.2.
Newer versions warn on load, but predictions were verified identical (to 1e-15) with scikit-learn 1.8.0.

## Commands
| What | Command | Approx. time (laptop CPU) |
|---|---|---|
| Unit tests | `python -m pytest -q` | ~10–60 s |
| Smoke test (3 inverse problems) | `python -m blinit.experiments.phase_a --quick` | ~2 min |
| Phase A pilot (20 problems + grid study) | `python -m blinit.experiments.phase_a` | ~50–60 min |
| Phase A figures | `python -m blinit.experiments.phase_a_report --dir results/phaseA` | ~10 s |
| Init-quality sweep (diagnostic) | `python -m blinit.experiments.init_sweep` | not logged (sequential, long) |
| Phase A′ noise / sensor pilot | `python -m blinit.experiments.phase_a_noise --workers 10` | not logged (parallel) |
| Phase B dataset | see Setup | ~4 min |
| Phase C MLP (tuning + 3 seeds) | `python -m blinit.experiments.train_mlp --config configs/phaseC_v2.json` | ~10 min |
| Phase D comparison (val or test) | `python -m blinit.experiments.evaluate --config configs/phaseD_v1_test.json --split test --workers 12` | ~40 min |
| Phase D analysis | `python -m blinit.experiments.analyze_d --dir results/phaseD_v1_3seeds_test` | ~1 min |
| Follow-up: learned bracket for Brent | `python -m blinit.experiments.followup_bracket run --config configs/followup_bracket_test.json --split test` then `... analyze --dir results/followup_bracket_w0.5_3seeds_test --brent-dir results/phaseD_v1_3seeds_test` | ~20 min |
| Report figures + tables | `python -m blinit.experiments.report_figures` | ~10 s |

## Layout
```
configs/        versioned JSON configs (simulator, phase A–D, follow-up)
blinit/         simulator.py (FV Burgers solver), exact.py (Cole–Hopf), observe.py (sensors/masks),
                inverse.py (Problem + Brent, LSQ, grid, learned-bracket Brent), data.py (dataset, integrity checks),
                model.py (MLP), knn.py (kNN initialiser), experiments/ (one script per phase)
tests/          pytest suite (simulator, data integrity, model, bracket search)
models/         trained MLPs (phaseC_v1, phaseC_v2) with tuning logs and training curves
results/        runs.csv / meta.json / analysis.json per experiment; results/report/ = report figures and tables
docs/           WORKFLOW.md, preregistration_*.md, report_draft.md, literature_audit.md, bibliography.md
RESEARCH_LOG.md decisions, commands, results, limitations
```

## Conventions (see `configs/simulator_v1.json`)
PDEBench Burgers convention: `u_t + (u^2/2)_x = (nu/pi) u_xx` on [0,1), periodic. `nu ∈ [0.01, 0.1]` is
PDEBench's `epsilon`, so the effective diffusivity is `nu/pi`. The solver is an independent NumPy
re-implementation of PDEBench's scheme (PDEBench's licence restricts redistribution of derivatives,
so no PDEBench code is included). Software developed by NEC Laboratories Europe (PDEBench) was used
as the reference for the numerical method.

Cost is reported as solver steps x 144.3 µs/step (the sequential Phase A median on the original machine), i.e.
CPU-s per query, plus forward-solve counts. Parallel wall times are never used for claims.
