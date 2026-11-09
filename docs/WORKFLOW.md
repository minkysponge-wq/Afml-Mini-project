# Project workflow: status and phase plan

*Last updated 2026-11-09 (evening). **PAUSED** — see `RESEARCH_LOG.md` → "PAUSED by user" for resume commands. Each phase lists tasks, outputs and an **exit gate**: a concrete check
that must pass before the next phase starts. "Verified" means the result exists on disk and was
checked; anything else is a plan.*

## 1. Where we are

| Area | Status | Evidence |
|---|---|---|
| Proposal read, scope fixed | Done | `RESEARCH_LOG.md` |
| PDEBench Burgers convention checked (`nu/pi` diffusion) | Done, verified | log, `configs/simulator_v1.json` |
| Own NumPy solver (PDEBench scheme, float64) | Done, verified | `blinit/simulator.py` |
| Simulator tests (Cole–Hopf exact solution, heat limit, periodicity, conservation, convergence) | 11/11 pass | `tests/test_simulator.py` |
| Classical methods: Brent, midpoint+LSQ, coarse grid+LSQ, oracle diagnostic | Done | `blinit/inverse.py` |
| Phase A pilot (20 problems, 16 sensors, noise-free) | Done, verified | `results/phaseA/` |
| Grid convergence at sensors | Done, verified | `results/phaseA/meta.json`, figure |
| Accuracy-vs-cost curves | Done, verified | `results/phaseA/figures/accuracy_vs_solves.png` |
| Starting-point quality sweep (how good must an init be?) | **Re-running** (first attempt killed at 10/20) | `blinit/experiments/init_sweep.py` |
| Literature audit | First pass only (6 verified sources) | `docs/literature_audit.md` |
| Phase A′ noise / 4–8 sensors + starting-point sweep | Done, verified | `results/phaseA_noise/`, `results/phaseA/init_sweep.csv` |
| Phase B dataset (v2: 3000/50/50, 9 integrity checks pass) | Done, verified | `data/phaseB_v2/integrity.json` |
| Phase C MLP (3 seeds; val median error 8–12%) | Done, verified | `models/phaseC_v2/` |
| Phase D comparison on validation | Done, verified (MLP vs Brent: −1.49 CPU-s/query, CI [−1.84, −1.16]; 1.17–1.35×) | `results/phaseD_v1_val/` |
| Accuracy-matched validation check | **Paused** (not saved; rerun) | `configs/phaseD_v1_tightlsq.json` |
| Pilot test (pre-registered, 3 seeds × 2 tolerances) | **Paused** (not saved; rerun) | `docs/preregistration_pilot_test.md` |
| Go/no-go (E), scale (F), report (G), submission (H) | Not started | — |

### Verified Phase A findings
- **Recovery:** every method recovers ν on all 20 noise-free problems (worst relative error 1.2e-5), with no failures.
- **Objective shape:** the mismatch-vs-ν curve has one minimum in all 20 problems.
- **Cost to reach 0.1% error (median simulator runs):**

  | Method | Runs |
  |---|---|
  | Brent | 6.5 |
  | Midpoint + LSQ | 7 |
  | Coarse grid + LSQ | 10 |

  Brent is the strongest classical baseline.
- **Floor:** a perfect starting point still costs 2 runs under the LSQ stopping rule. A 2× saving is therefore possible only if the learned initialiser is quite accurate.
- **Discretisation error:** at ν = 0.01, the 256-cell grid error is about the size of a 1% change in ν. Expect a 1–2% bias in the finer-grid test at low ν.
- **Cost per run:** about 1.0 s on average, varying about 10× with ν. Laptop sleep corrupted some timings; these are flagged automatically.

## 2. Phase plan

### Phase A′: finish the pilot (≈1–2 days)
- [ ] Re-run the starting-point sweep. Output: runs needed vs initial error (1%, 3%, 10%, 35%) and Brent at matched tolerances.
- [ ] Repeat the pilot with **1% and 5% noise** and **4 / 8 sensors**. Check for flat or multimodal objectives and say whether hard cases come from too little information, optimiser failure or numerical error.
- [ ] Define the matched-accuracy criterion from this development data: a common stopping tolerance tied to the noise level, the same for every method.
- **Gate A:** classical methods behave sensibly with noise, and the required initialiser accuracy is known (e.g. "an init within X% saves Y runs vs Brent"). **If even a ~3%-accurate start cannot beat Brent by a meaningful margin, stop and discuss scope before any ML.**

### Literature audit, part 2 (in parallel, ≈1–2 days)
- [ ] Read the full texts of Sambharya et al. (2024) and Zhou et al. (2023): baselines, cost accounting, whether training cost is counted.
- [ ] Search ODE/PDE calibration warm starts (systems biology, reservoir history matching) and work on missing sensors and identifiability.
- [ ] Verify or drop every "[found]" lead. Update `bibliography.md`.
- **Gate:** each cited source is verified, and the novelty-risk paragraph is rewritten from the evidence.

### Phase B: dataset (≈2 days)
- [ ] `blinit/data.py`:
  - Generate trajectories separately from observations.
  - Split **by initial-condition seed**: 300 train / 50 val / 50 test (pilot).
  - Use a log-uniform ν distribution.
  - Store ν, IC coefficients and seed, sensor x/t, observations, masks, noise settings, solver config and version.
- [ ] Use fixed sensor slots with masks. Standardise with training statistics only, and use one documented fill value for missing readings.
- [ ] `tests/test_data.py`: no seed crosses splits, noise/dropout variants stay in their parent's split, normalisation statistics come only from train, and the reading→(x, t) mapping is preserved.
- **Gate B:** integrity tests pass, and the dataset regenerates bit-identically from its config.

### Phase C: MLP initialiser (≈2 days)
- [ ] MLP with two hidden layers of 64 units.
  - **Inputs:** observations, mask, IC coefficients (cos/sin form).
  - **Target:** log ν. **Loss:** MSE.
  - **Early stopping** on the validation set.
  - **Library:** scikit-learn is already installed, so no new dependency is needed.
- [ ] Small documented tuning budget (e.g. ≤ 6 configs on validation only). Save checkpoints, normalisation statistics, configs and training curves.
- [ ] Clip predictions to the bounds. Report neural-only error.
- **Gate C:** neural-only validation error is better than the midpoint/kNN guess, and its size compared with the Gate A threshold tells us whether savings are even possible.

### Phase D: full baseline comparison (≈2–3 days)
- [ ] **kNN initialiser:** same features and masks as the MLP, with a library built from training data only. Count library construction and retrieval cost.
- [ ] Compare these 5 methods on the identical pilot test set, with identical refinement settings and bounds:
  1. Midpoint + LSQ
  2. Coarse grid + LSQ
  3. kNN + LSQ
  4. MLP + LSQ
  5. Brent
- [ ] Log every simulator run, failure and timeout. Time on AC power with the lid open, with the cold-start run separated.
- **Gate D:** comparison tables plus accuracy-vs-cost curves, using paired per-problem differences.

### Phase E: go/no-go checkpoint (≈½ day; **team decision**)
- [ ] Answer from evidence:
  - Does the starting point matter?
  - Does the MLP beat Brent and kNN at matched accuracy?
  - By how much, with a confidence interval?
  - Is accuracy preserved?
- [ ] Compute break-even = (training + data + tuning cost) / (online saving per query). If the saving is ≤ 0, say there is no finite break-even.
- **Go:** proceed to Phase F. **No-go:** report the negative result honestly. Optionally pursue at most two narrow alternatives (e.g. a cheaper refinement such as Brent in a learned bracket, or harder regimes such as 4 sensors with 5% noise). Any scope change needs approval first.

### Phase F: scale and robustness (≈4–6 days, needs compute approval)
- [ ] 3,000 / 500 / 500 trajectories.
  - **Grid:** 4/8/16 sensors × 0/1/5% noise.
  - **Robustness:** seen and unseen missing-sensor patterns, a held-out higher-frequency IC set, and finer-grid test observations.
  - **Seeds:** 3 training seeds.
- [ ] Ablations: neural-only vs refined, dropout augmentation vs none, IC-coefficient inputs vs none.
- [ ] Statistics: paired comparisons, bootstrap CIs resampled by trajectory, and ambiguity/failure criteria fixed **before** the test set is touched.
- **Compute budget (measured ~1 s/solve):**
  - The full grid is roughly 500 test trajectories × 9 conditions × ~7 method runs × ~10 solves ≈ 300k solves ≈ **85 CPU-hours** (about 6–7 h on 14 parallel workers).
  - **Options:** (a) parallel processes for accuracy and solve counts, plus a small sequential subset for clean timing; (b) a numba speed-up, if profiling justifies it; (c) a smaller full-grid test set (e.g. 200) with the full 500 for the main condition.
  - Decide before launching.
- **Gate F:** every result regenerates from saved files, with no retraining needed for figures.

### Phase G: analysis and write-up (≈3–4 days)
- [ ] Main figures:
  - accuracy-vs-cost curves;
  - runs and time per method (median and 90th percentile);
  - error vs number of sensors and noise level;
  - shift tests;
  - break-even curve.
- [ ] Report sections: problem, related work (verified only), methods, a fair-comparison protocol, results incl. negative/ambiguous cases, limitations.
- [ ] Every claim maps to a result file. Missing results are marked as missing.
- [ ] README with one smoke-test command and one command per experiment. Run `pytest` green before submission.

### Phase H: submission package (≈1 day)
- [ ] Report PDF, slides, a short demo (smoke test + one figure), the repo zip/link and the research log.
- [ ] Dry run of the presentation, so each team member can explain their part plus the fairness protocol.

## 3. Suggested team split (4 members)
| Member | Owns |
|---|---|
| A | Simulator, tests, Phase A′ noise study |
| B | Dataset and integrity checks (B), kNN baseline (D) |
| C | MLP training, tuning, ablations (C, F) |
| D | Literature audit, evaluation statistics, figures, report assembly (G, H) |

Everyone reviews the Phase E decision together.

## 4. Decisions that need the team's approval
1. Phase F compute plan: which option (a/b/c) to use.
2. Any scope change after a no-go at Phase E.
3. Adding any dependency beyond NumPy/SciPy/matplotlib/scikit-learn.

## 5. What graders are likely to reward (based on the one-pager's own deliverables, not a rubric)
The one-pager promises three deliverables: a trained initialiser, reproducible classical baselines, and an accuracy-vs-cost study "showing where learning helps and where it does not". A clean, honest version of these, including a negative result if that is what the data shows, meets the proposal. Strong baselines, reproducibility and clear limitations generally count for more than a forced speed-up. **Check the course's actual rubric and adjust.**
