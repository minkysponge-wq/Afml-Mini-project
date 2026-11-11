# Research log

Legend: **[PLAN]** intended, **[RUN]** executed, **[RESULT]** verified output on disk, **[DECISION]**, **[LIMITATION]**.

## 2026-10-23 — Session 1: setup, simulator, Phase A pilot

### Environment [RUN]
Windows 11, Intel Core Ultra 9 185H (16 cores), 31.6 GB RAM, NVIDIA RTX 4050 Laptop 6 GB (unused so
far). Python 3.10.11, numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.8, scikit-learn 1.7.2, pytest 8.4.2.
No torch/jax/numba/yaml/h5py installed. Project folder was empty at start (nothing to preserve).

### Proposal read [RUN]
Extracted text from `AFML_Learned_Initialization_OnePager.pdf` (ReportLab PDF; parsed with a stdlib
script since no PDF library was installed). Scope: 1-D Burgers, known 3-mode IC, periodic, nu in
0.01–0.1, 4/8/16 sensors at 4 times, 0/1/5% noise; MLP -> log nu -> bounded least-squares
refinement; baselines midpoint / coarse grid / kNN / direct bounded scalar search; target 2x median
online time at no worse error (hypothesis, not result).

### PDEBench inspection [RUN]
Downloaded (to scratch, not repo) `burgers_multi_solution_Hydra.py`, `utils.py`, `config/multi/1e-2.yaml`
from github.com/pdebench/PDEBench (main branch, 2026-10-23). Findings:
- Equation: flux 0.5 u^2; diffusive flux `-epsilon * (1/pi) * du/dx` -> **u_t + (u^2/2)_x = (epsilon/pi) u_xx**.
  So the proposal's "viscosity 0.01–0.1" is interpreted as PDEBench `epsilon`; effective D in [0.00318, 0.0318].
- Domain [0,1], nx=1024 cells (cell centres), periodic ghost cells, t in [0,2], dt_save 0.01, CFL 0.25.
- Space: MUSCL with `VLlimiter` (alpha=2), Rusanov/Murman flux with speed |(uL+uR)/2|.
- Time: 2-stage midpoint (predictor dt/2 with flux(u), corrector dt with flux(u_half)); dt = CFL*min(dx/max|u|, 0.5dx^2/D).
- IC (`init_multi`): picks 2 of k_tot modes (k_tot=4 for Burgers), amp U(0,1), phase U(0,2pi), 10% abs, 10% window, random sign;
  wavenumber uses `xc[-1]-xc[0]` (= 1-dx), i.e. **IC slightly non-periodic** in PDEBench.
- Default float32 (x64 commented out).
- **Licence**: NEC non-commercial academic licence; derivatives owned by NEC; redistribution restricted.

**[DECISION]** Re-implement the scheme in NumPy (no copied code), float64, exact periodic wavenumbers,
fixed 3-mode IC (k=1,2,3, amp U(0,1), phase U(0,2pi)) per proposal; nx=256 default (to be justified
by grid study); observation times {0.25,0.5,0.75,1.0}; 16 sensor slots at x=(s+0.5)/16; observation
operator = periodic linear interpolation. Config: `configs/simulator_v1.json`. Dependency-free JSON configs.

### Simulator tests [RUN] [RESULT]
`python -m pytest -q` -> **11 passed** (9.7 s). Tests: config values; Cole–Hopf exact solution reduces to IC;
constant state preserved; mass conservation to 1e-13 + finite output; periodic shift-equivariance
(exact to 1e-12); small-amplitude heat limit matches decay with D=nu/pi (and rejects D=nu);
Cole–Hopf grid convergence for nu in {0.02, 0.1} (errors decrease 64->128->256, ratio >2.5, err<1e-2 at 256);
call counter; interpolation; nested sensor masks.

### Cost observation [RESULT]
Per-solve cost depends strongly on nu because the explicit diffusive time-step limit scales as 1/nu:
nx=256, t_end=1: nu=0.01 -> 1,686 steps, 0.08 s; nu=0.1 -> 16,692 steps, 0.84 s.
**Implication:** method runtime depends on *which* nu values are evaluated, not just how many.
Report both solve counts and wall time; solver steps are also logged.

### Finite-difference step probe (exploratory dev case, rng seed 999, not in any split) [RESULT]
FD gradient of objective in theta=log nu is stable from h=1e-8 to 1e-4 (e.g. 0.105824 vs 0.105825),
so step-count discontinuities are not visible at these scales. **[DECISION]** relative diff_step=1e-6.

### Inverse methods implemented (`blinit/inverse.py`)
Parameterisation theta = log nu, bounds [log 0.01, log 0.1] for all methods. Objective 0.5||r||^2 over
available sensors x 4 times. Shared `Problem` counts every forward solve (incl. FD Jacobian solves);
memoises exact repeats for all methods. Methods: `brent` (scipy bounded, xatol 1e-4), `midpoint+lsq`,
`grid5+lsq` (5 log-spaced points then refine), `ORACLE+lsq(diagnostic)` (true nu start; diagnostic only).
Refinement: scipy `least_squares` TRF, 2-point FD, xtol 1e-4, ftol 1e-10, gtol 1e-12, max_nfev 100.

### Phase A smoke [RUN] `python -m blinit.experiments.phase_a --quick`
3 problems: all methods recover nu (rel. err <= 1.3e-5). Solves: brent 9–12, midpoint+lsq 6–18,
grid5+lsq 10–14, oracle 2. **[LIMITATION]** accuracies not matched (brent ~1e-5 vs lsq ~1e-10): need
accuracy-vs-cost sweeps. Noise-free oracle is degenerate (residual exactly 0 at start).

### Phase A full run [RUN, in progress]
`python -m blinit.experiments.phase_a > results/phaseA/stdout.log` (config `phaseA_v1.json`, seed 20261023).

### Phase A run 1 [RUN 11:01–19:00] -> archived as `results/phaseA_run1_sleep_contaminated/`
**[RESULT] accuracy (noise-free, 16 sensors, 20 problems, inverse crime):** all methods recover nu;
max rel. error brent 1.2e-5, midpoint+lsq 1.0e-6, grid5+lsq 9e-8. No simulator failures, no
optimiser non-success. **Solves (median [range]):** brent 10 [8–12], midpoint+lsq 11 [6–18],
grid5+lsq 13 [10–16], oracle 2 (degenerate, zero residual at start).
**[RESULT] objective landscape:** 41-point log-scan has exactly one local minimum for all 20 problems
(at the true nu). No multimodality detected at this resolution, noise-free, 16 sensors.
**[RESULT] grid convergence at sensors (vs nx=1024, 3 ICs):** ~2nd order (error ratio 4–5 per halving).
At nx=256: RMS sensor error 5–8e-4 (nu=0.01), 1.8–2.8e-4 (nu=0.0316), 1.6–3.0e-5 (nu=0.1). The
observation change for a +1% nu perturbation is 4–5e-4 (nu=0.01) and 0.8–1.4e-3 otherwise.
**Implication:** at low nu, nx=256 discretisation error ≈ the signal of a 1% nu change -> expect
~1–2% bias at low nu in the finer-grid mismatch test; negligible at high nu. Small relative to the
statistical error expected at 1–5% noise. [DECISION] keep nx=256 for the pilot; quantify in finer-grid test.
**[LIMITATION] timing contamination:** laptop entered Modern Standby (lid closed) 12:26–18:48 (Windows
Kernel-Power events 506/507). Four runs show 400 s, 313 s, 1000 s and 23,653 s wall time for 2–12 solves
(per-step cost 6,500–1,388,000 us vs ~200 us typical). Solve and step counts are unaffected. Timing
protocol changes: (1) keep-awake requested for long runs; (2) AC/battery state logged in meta.json;
(3) automatic flag for runs with us/step > 3x median (catches exactly these 4); (4) solver steps
reported as a deterministic cost measure alongside wall time.
**[DECISION]** Added per-solve traces (theta, objective, cumulative steps) so accuracy-vs-cost curves
can be computed post hoc; best-so-far estimate is selected by objective only (no ground truth); truth
used only to score. Rerun: `python -m blinit.experiments.phase_a --reuse-grid` (grid study reused; deterministic).

### Phase A run 2 [RUN 2026-10-23 19:02 – 23:59, AC power] -> `results/phaseA/` (verified)
Command: `python -m blinit.experiments.phase_a --reuse-grid`; report: `python -m blinit.experiments.phase_a_report`.
Accuracy, solve counts and objective scans are bit-identical in kind to run 1 (deterministic).
**[LIMITATION]** The laptop again entered standby (~19:45–23:06; keep-awake prevents idle sleep only,
not lid close). One run flagged (problem 9, midpoint+lsq: 13,611 s for 8 solves); excluded from
timing statistics, kept for accuracy/solves. Median cost 144 us/step on unflagged runs.
**[RESULT] summary (20 problems, noise-free, 16 sensors):**

| method | rel err median / max | solves median / p90 | solves to reach 1e-2 / 1e-3 / 1e-4 (median) |
|---|---|---|---|
| brent (xatol 1e-4) | 3.2e-6 / 1.2e-5 | 10 / 11.1 | 5.5 / 6.5 / 7 |
| midpoint+lsq | 7.3e-10 / 1.0e-6 | 11 / 16.2 | 6 / 7 / 8 |
| grid5+lsq | 4.1e-11 / 9.0e-8 | 13 / 14 | 9 / 10 / 10.5 |
| ORACLE+lsq (diagnostic) | 0 / 0 | 2 / 2 | 1 / 1 / 1 |

"Solves to reach" uses ground truth only to score the best-so-far (objective-selected) estimate — it is
an anytime diagnostic, not a deployable stopping rule. Scans: all 20 unimodal.
**Interpretation (provisional):** Brent is the strongest classical baseline at practically relevant
accuracy (1e-2–1e-3). The coarse grid's 5 up-front solves are not recovered by the refinement.
Headroom for any initialiser is bounded: even a perfect start costs >= 2 solves under the LSQ stopping
rule, vs ~6.5–10 for Brent. Next: init-quality diagnostic sweep (`blinit.experiments.init_sweep`).

## 2026-11-09 — Session 2: Phase A', B, C, D setup

### Housekeeping
- `file (afml).txt` (12.7 MB, the word "file" repeated ~2.66M times) appeared in the project folder; not created by
  this pipeline, not project content; git-ignored, untouched.
- Repo pushed (on user request) to github.com/minkysponge-wq/AFML-Mini-project, branch `main`.
- First init-sweep attempt was killed after 10/20 problems (no traceback; process ended with the session). Re-run completed.

### Init-quality sweep [RESULT] (`python -m blinit.experiments.init_sweep`, noise-free, 20 Phase A problems)
DIAGNOSTIC (truth-perturbed starts). Median solves / max rel err:
brent xatol 1e-2: 8 / 1.5e-3; 1e-3: 9 / 1.7e-4; 1e-4: 10 / 1.2e-5.
lsq xtol 1e-3 from |delta| = 0.01/0.03/0.1/0.3 (log-nu): 6/6/8/8 solves, max err <= 8.6e-5.
lsq xtol 1e-2: 4/4/6/8 solves but max err up to 1e-2 (too loose to be matched).
**Reading:** at matched accuracy (~1e-4), even a 1%-accurate start saves ~3 solves vs Brent (6 vs 9);
a 10%-accurate start saves ~1 (8 vs 9). Solve-count savings are bounded at ~1.1–1.5x.

### Phase A' noise/sensor pilot [RESULT] (`python -m blinit.experiments.phase_a_noise --workers 10`)
20 problems x {4,8,16 sensors} x {0,1,5% noise}; 25-pt scans; parallel (no timing reported).
- Scans: 0/160 with >1 local minimum. No multimodality detected.
- All methods reach the same optimum (objective gaps negligible; max relative gap ~6e-6 from xtol stop).
  -> no optimiser failures; difficult cases are information-limited.
- Information-limited error (best-found optimum vs truth, median/p90): s16 n1%: 0.26%/0.8%; s16 n5%: 0.46%/2.2%;
  s8 n5%: 1.9%/6%; s4 n5%: 2.6%/14% (problem 19 at s4 n5%: 22% for every method).
- Solves (median): brent 9–10, midpoint+lsq 10–11, grid5+lsq 13–14, ORACLE+lsq 6 (noisy; 8 at s4 n5%).
  **With noise, even a perfect start needs ~6 solves (xtol 1e-4) -> best-case ~1.5x vs Brent in solves.**

### Phase B [RESULT]
`python -m blinit.experiments.build_dataset` (phaseB_v1: 300/50/50) and `--config configs/phaseB_v2.json`
(train enlarged to 3000; first 300 train, all val and test bit-identical to v1 - verified). 9 variants per
trajectory (3 sensor counts x 3 noise levels). All 9 integrity checks pass for both. Generation wall time v2:
train 230 s, val 7 s, test 6 s on 8 workers (~0.6 CPU-h total; to be counted in break-even).
Tests: `tests/test_data.py` (4) incl. deliberate-leak detection.

### Phase C [RESULT] (validation only; test split not loaded)
v1 (300 traj): selected lr 1e-3, alpha 1e-3; neural-only val rel. err median/p90 = 18.7%/65% (seed 0),
22%/68% (s1), 21%/66% (s2). Strong overfitting (train MSE ~0.01 vs val ~0.3, standardised).
Learning curve (val median, 2 seeds): 75 traj ~39%, 150 ~32%, 300 ~20% -> data-limited.
kNN (tuned on val: k=5, IC weight 2) neural-only: 39% median, p90 155% -> much worse than MLP.
**[DECISION]** enlarge train to 3000 (within proposal scope) -> phaseB_v2 / phaseC_v2.
v2 (3000 traj): selected lr 1e-3, alpha 1e-3; val median/p90 = 8.2%/33% (s0), 9.8%/29% (s1), 11.7%/33% (s2).
Tuning 4 configs: 313 s; per-seed training 25–110 s (CPU, sklearn). Midpoint guess: 57% median.

### Phase D setup [DECISION]
Matched tolerances from development data: brent xatol 1e-3, lsq xtol 1e-3 (`configs/phaseD_v1.json`).
Smoke run (4 val variants): midpoint+lsq used 6 solves vs mlp+lsq 8 on some cases - initialisation may matter
little for Gauss-Newton on this smooth 1-D objective. Full val comparison running:
`python -m blinit.experiments.evaluate --split val --workers 12`.

### Phase D comparison on VALIDATION [RESULT] (`python -m blinit.experiments.evaluate --split val --workers 12`;
analysis `python -m blinit.experiments.analyze_d --dir results/phaseD_v1_val`)
450 variants (50 val trajectories x 3 sensor counts x 3 noise levels), 1404 s wall on 12 workers. 0 simulator failures.
Cost = solver steps x 144.3 us/step (sequential Phase A median) -> CPU-s; parallel wall times not used.
kNN re-tuned on val with the 3000-traj library: k=10, IC weight 1.0.

| method | rel err median / p90 | solves median / mean | CPU-s mean / median / p90 |
|---|---|---|---|
| brent | 2.47e-3 / 2.93e-2 | 9 / 9.74 | 9.05 / 7.41 / 17.44 |
| midpoint+lsq | 2.44e-3 / 2.93e-2 | 12 / 12.28 | 11.86 / 6.73 / 28.30 |
| grid5+lsq | 2.45e-3 / 2.94e-2 | 12 / 11.79 | 12.29 / 9.89 / 19.86 |
| knn+lsq | 2.48e-3 / 2.93e-2 | 10 / 10.28 | 10.13 / 7.01 / 22.98 |
| mlp_s0+lsq | 2.45e-3 / 2.93e-2 | 8 / 8.04 | 7.72 / 5.47 / 14.19 |

Paired (trajectory-level bootstrap, 4000 resamples), median difference in CPU-s per query:
mlp - brent: -1.49 [95% CI -1.84, -1.16]; mlp - midpoint+lsq: -2.82 [-3.51, -1.98] (negative in every
sensor/noise cell); knn - brent: +0.56 [-0.19, +1.27] (no evidence of saving).
Ratios (brent/mlp): mean CPU 1.17x, median CPU 1.35x, median solves 9/8. **2x target not met on val.**
Break-even vs brent (strongest classical by mean CPU): offline 2254 CPU-s (train data generation, upper bound
wall x workers, + 4 tuning runs + seed-0 training) / 1.33 CPU-s saving per query (incl. 1.2e-4 s single-query
MLP inference) = **~1,700 queries**. kNN: saving <= 0 -> no finite break-even.
Smoke-run impression (midpoint+lsq cheap) did NOT hold: on 450 variants midpoint+lsq is the costliest-but-one.
Accuracy matching check: errors vs truth are identical across methods at median and p90 (information-limited).
Objective-gap flags (rel. gap > 1e-3 vs best-found) occur almost only in noise-free cases where the optimum is ~0
(relative gap meaningless); in nu, max deviation from the best-found estimate is 0.06% (brent) vs 0.25–0.39%
(LSQ methods). **[LIMITATION]** LSQ xtol 1e-3 is looser than brent xatol 1e-3 in the worst case; part of the
saving may come from that. Next: accuracy-matched check with LSQ xtol 1e-4 (or brent 1e-2) before final claims.
**[LIMITATION]** Validation was used for MLP early stopping / tuning and kNN tuning -> these numbers are
optimistic for learned methods. The pilot TEST split has not been touched.

### Repo / attribution (2026-11-09)
Repo renamed by the user to github.com/minkysponge-wq/Afml-Mini-project (same GitHub repo id; it was empty when
checked). History rewritten locally to remove Claude co-author trailers (file trees bit-identical); pushed; GitHub
contributors API shows only `minkysponge-wq` (7 commits). No co-author trailers on future commits.

### Pre-registration (2026-11-09)
`docs/preregistration_pilot_test.md` + `configs/phaseD_v1_test.json`, `configs/phaseD_v1_tightlsq_test.json`
committed (a9de91a) BEFORE any test evaluation.

### PAUSED by user (2026-11-09 ~17:55) — runs stopped, nothing saved from them
- Accuracy-matched VAL run (`configs/phaseD_v1_tightlsq.json`): stopped at 400/450 variants.
- Pilot TEST primary (`configs/phaseD_v1_test.json`): stopped at 200/450 variants.
- Pilot TEST tight (`configs/phaseD_v1_tightlsq_test.json`): not started.
`evaluate.py` writes results only on completion, so no partial results exist; empty output folders deleted.
**Test-integrity note:** test variants were simulated, but no test result was written, printed or viewed
(stdout shows only progress counts). The pre-registration stands.
**To resume** (≈30 min + ≈2 × 40 min on 12 workers; keep the laptop on AC with the lid open):
```
python -m blinit.experiments.evaluate --config configs/phaseD_v1_tightlsq.json --split val --workers 12
python -m blinit.experiments.analyze_d --dir results/phaseD_v1_tightlsq_val
python -m blinit.experiments.evaluate --config configs/phaseD_v1_test.json --split test --workers 12
python -m blinit.experiments.evaluate --config configs/phaseD_v1_tightlsq_test.json --split test --workers 12
python -m blinit.experiments.analyze_d --dir results/phaseD_v1_3seeds_test
python -m blinit.experiments.analyze_d --dir results/phaseD_v1_tightlsq_3seeds_test
```
Possible improvement before resuming (infrastructure only, not a protocol change): make `evaluate.py` append
rows per variant and skip completed variants on restart, so future pauses lose nothing.

### Accuracy-matched VALIDATION check [RESULT] (2026-11-10; `configs/phaseD_v1_tightlsq.json`: lsq xtol 1e-4, brent xatol 1e-3)
Checkpointed `evaluate.py` used (resume verified bit-identical on smoke run). 0 simulator failures.
CPU-s mean / median / p90: brent 9.05 / 7.41 / 17.44; mlp_s0+lsq 9.02 / 6.24 / 17.54; knn+lsq 11.52 / 7.85 / 24.96;
midpoint+lsq 13.36 / 7.34 / 32.49; grid5+lsq 13.21 / 11.00 / 22.91. Errors vs truth identical (median 2.45e-3).
Paired mlp - brent: median -0.65 CPU-s [95% CI -0.84, -0.22], **mean -0.035 CPU-s** -> break-even ~64,600 queries.
CI includes 0 in 5 of 9 sensor/noise cells (s4n1%, s8n5%, s16 all noise levels).
kNN - brent: +1.26 [+0.58, +2.38] (kNN worse). mlp - midpoint+lsq: -2.84 [-3.58, -1.89] (learned init still
clearly helps LSQ itself).
**Interpretation:** at the looser LSQ tolerance (xtol 1e-3) the MLP saved 1.49 CPU-s/query vs Brent; at the tighter
one (xtol 1e-4, now more precise than Brent) the saving is ~0 on average and present only at the median. A large
part of the earlier saving came from LSQ stopping looser in the tail. The learned initialiser reliably helps the
LSQ refinement (vs midpoint) but is **not** reliably cheaper than direct Brent search at equal-or-better precision.
Neither tolerance pair is exactly matched (xtol 1e-3: LSQ looser in worst case; 1e-4: LSQ tighter) -> the honest
summary is an accuracy-vs-cost curve across tolerances, not a single speedup number. The pre-registered test
runs both settings, so this will be reported from test too.

### Pilot TEST runs completed (2026-11-10) — NOT yet analyzed
Both pre-registered runs finished on Sampreeth's laptop (resumed with checkpointed `evaluate.py`):
`results/phaseD_v1_3seeds_test/` and `results/phaseD_v1_tightlsq_3seeds_test/`, each 450 variants x 7 methods
= 3,150 rows. Only completion markers and row counts were checked; no test metric has been computed, printed or
viewed. Analysis and the go/no-go verdict are handover task T1 (see `docs/preregistration_pilot_test.md`).

### T1 — Pilot TEST analysis and go/no-go [RESULT] (2026-11-11)
Commands: `python -m blinit.experiments.analyze_d --dir results/phaseD_v1_3seeds_test` and
`... --dir results/phaseD_v1_tightlsq_3seeds_test`. No model, tolerance or code was changed. 450 test variants
(50 trajectories x 3 sensor counts x 3 noise levels), 7 methods, 0 simulator failures in both runs.

Accuracy is the same for every method in both runs: median relative error 2.2e-3, p90 2.6e-2.

| setting | method | CPU-s mean / median / p90 | solves median / mean |
|---|---|---|---|
| primary (lsq xtol 1e-3) | brent | 9.01 / 8.12 / 14.72 | 9 / 9.20 |
| | midpoint+lsq | 11.06 / 7.37 / 22.05 | 10 / 11.25 |
| | grid5+lsq | 12.79 / 11.75 / 20.77 | 12 / 11.98 |
| | knn+lsq | 9.27 / 7.65 / 15.61 | 8 / 9.31 |
| | mlp_s0 / s1 / s2 +lsq | 8.20 / 6.75 / 15.02; 8.16 / 6.65 / 14.35; 8.52 / 7.21 / 15.05 | 8 / 8.08; 8 / 8.11; 8 / 8.39 |
| matched (lsq xtol 1e-4) | brent | 9.01 / 8.12 / 14.72 | 9 / 9.20 |
| | midpoint+lsq | 12.37 / 9.32 / 22.20 | 12 / 12.58 |
| | grid5+lsq | 13.88 / 13.32 / 23.67 | 13 / 12.96 |
| | knn+lsq | 10.53 / 8.76 / 17.62 | 10 / 10.58 |
| | mlp_s0 / s1 / s2 +lsq | 9.30 / 8.06 / 15.72; 9.22 / 7.77 / 15.62; 9.57 / 7.77 / 17.28 | 8 / 9.18; 8 / 9.20; 8 / 9.43 |

Paired median difference in CPU-s per query, MLP - brent, all 450 variants (95% trajectory bootstrap CI):

| seed | primary | matched |
|---|---|---|
| 0 | -1.073 [-1.396, -0.834] (mean -0.808) | -0.118 [-0.547, +0.267] (mean +0.288) |
| 1 | -0.947 [-1.348, -0.626] (mean -0.853) | -0.098 [-0.417, +0.334] (mean +0.208) |
| 2 | -0.795 [-1.177, -0.423] (mean -0.489) | -0.033 [-0.322, +0.236] (mean +0.564) |

Per sensor/noise cell: primary, the CI is below 0 in 9/9 (seed 0), 8/9 (seed 1), 8/9 (seed 2) cells; matched, the CI
includes 0 in all 9 cells for all 3 seeds.
MLP - midpoint+lsq: -1.87 to -1.97 (primary) and -2.06 to -2.48 (matched), every CI below 0.
kNN - brent: -0.203 [-0.771, +0.621] (primary); +0.942 [+0.439, +1.652] (matched, kNN costs more).
Median CPU ratio brent / MLP: 1.20, 1.22, 1.13 (primary); 1.01, 1.05, 1.05 (matched). Mean ratio: 1.06-1.10 (primary);
0.94-0.98 (matched).
Break-even vs brent: 2,790 / 2,625 / 4,455 queries (primary, seeds 0/1/2); no finite break-even at the matched
tolerance (mean saving <= 0 for all seeds). kNN: no finite break-even in either setting.

**Verdict under the pre-registered rule: NOT positive.** The rule needs the CI to exclude 0 under both tolerance
settings for at least 2 of 3 seeds; it does so under the primary setting for 3/3 seeds and under the matched setting
for 0/3. **The 2x target is not met** (largest median ratio 1.22).
**Go/no-go: go to Phase F with the main finding stated as:** the learned initialiser makes the least-squares
refinement cheaper than a midpoint start in every setting, but it is not cheaper than Brent's bounded search once
the refinement is run to equal-or-better precision. Phase F explains this (T2 tolerance curves, T3 Cramér–Rao bound)
and tests whether it holds under shift (T4), with ablations (T5).
Caveats: 50 test trajectories; neither tolerance pair is exactly precision-matched (T2 addresses this); cost is solver
steps x 144.3 us, not wall time.

### T0 — fresh-clone reproduction (2026-11-11)
Clone at 2581dc0: 16/16 tests pass; `analyze_d --dir results/phaseD_v1_val` prints `mlp_s0+lsq - brent | all`
median -1.491, CI [-1.844, -1.161] (matches). `build_dataset --config configs/phaseB_v2.json --workers 8` rebuilt the
dataset (train 193 s wall): all 9 integrity checks true. The committed `data/phaseB_v2/integrity.json` was kept, so
break-even numbers still use the original generation wall time.

### T3 — Cramér–Rao bound [RESULT] (2026-11-11)
Code: `blinit/crb.py`, `blinit/experiments/crb.py`, `tests/test_crb.py` (3 tests). Commands:
`python -m blinit.experiments.crb --split val` and `--split test`. Outputs: `results/phaseF_crb/`.
300 noisy variants per split (50 trajectories x 3 sensor counts x 1% and 5% noise); sensitivity by central difference
in log nu, h = 1e-3 (h = 1e-4 agrees within 1%, tested).
z = (theta_hat - theta_true) / sqrt(CRB), all noisy variants: validation std(z) 0.965 (brent) and 0.966 (mlp_s0+lsq),
mean z +0.06; test std(z) 0.947 and 0.944, mean z -0.002. Fraction with |z| > 2: 3.3-4.7% (5% expected for a
unit normal). Per condition std(z) lies in 0.77-1.07; RMSE / sqrt(mean CRB) lies in 0.50-1.15 (validation) and
0.70-1.12 (test).
**In one sentence: both Brent and MLP+LSQ reach the Cramér–Rao bound within sampling error, so the remaining error is
set by the data and no method can be more accurate; methods can differ only in cost.**
Caveats: the bound assumes an unbiased estimator and Gaussian noise and ignores the parameter bounds; noise-free
variants are excluded (bound 0); one noise draw per case, so only the aggregate is meaningful; the per-condition
ratio uses sqrt(mean CRB), which a few weakly identified trajectories dominate (this is why some ratios are below 1).

### Phase F code and pre-registration (2026-11-11, commit bde5604)
`docs/preregistration_phaseF.md` and all T2/T4/T5 code were committed before any robustness set was generated.
26 tests pass. The three sets were then built with `python -m blinit.experiments.build_shift_set --config
configs/phaseF_<set>.json --workers 6`: highfreq (450 variants), unseenmask (900), finegrid (450, 147 s wall); all 9
integrity checks true for each, including IC seeds disjoint from the 3,100 seeds of phaseB_v2.

### PAUSED by user (2026-11-11) — state of the runs
- T2 tolerance runs (validation): `phaseF_tol_anchor` complete (450/450); `phaseF_tol_brent1e-2` stopped at 444/450
  rows (checkpointed, resumes); `brent1e-4`, `lsq1e-2`, `lsq1e-5` not started. No tolerance-curve result exists yet.
- T5: `models/phaseC_v2_noaug` trained (3 seeds); `phaseC_v2_noic` stopped during tuning (rerun from scratch, ~8 min).
  `ablations.py` not yet run.
- T4: evaluation of `phaseF_highfreq` stopped before any variant was written; no robustness result has been
  computed, printed or viewed. `phaseF_finegrid` and `phaseF_unseenmask` not started.
- T7: `docs/literature_audit.md` and `docs/bibliography.md` updated (11 works opened).
To resume:
```
for t in "brent1e-2 brent" "brent1e-4 brent" "lsq1e-2 midpoint,mlp" "lsq1e-5 midpoint,mlp"; do set -- $t; python -m blinit.experiments.evaluate --config configs/phaseF_tol_$1.json --split val --methods $2 --workers 12; done
python -m blinit.experiments.tolerance_curves
python -m blinit.experiments.train_mlp --config configs/phaseC_v2_noic.json && python -m blinit.experiments.ablations
for s in highfreq finegrid unseenmask; do python -m blinit.experiments.evaluate --config configs/phaseD_v1_test.json --split test --test-dir data/phaseF_$s --workers 12; python -m blinit.experiments.analyze_d --dir results/phaseD_v1_3seeds_test_phaseF_$s; done
```

### T5 — Ablations [RESULT] (2026-11-11, validation only, no new simulator calls)
Commands: `python -m blinit.experiments.train_mlp --config configs/phaseC_v2_noaug.json`, `... phaseC_v2_noic.json`,
`python -m blinit.experiments.ablations`. Outputs: `results/phaseF_ablations/ablations.{json,md}`, models
`models/phaseC_v2_noaug/`, `models/phaseC_v2_noic/`. Same tuning protocol (4 configs, validation loss) and 3 seeds.
Selected: noaug lr 3e-3, alpha 1e-4 (9,000 train variants); noic lr 1e-3, alpha 1e-3 (27,000).

(a) MLP alone vs MLP + refinement (seed 0, 450 val variants): median relative error 8.2% (p90 33%) alone vs 0.245%
(p90 2.9%) after refinement; median refinement cost 5.47 CPU-s. With noise-free data the refined error is ~1e-7 to
1e-6. **The MLP alone is about 33 times less accurate than the data allow; refinement is necessary and the MLP's
only job is to make it cheaper.**

(b) Sensor-dropout augmentation (train on all 4/8/16-sensor variants) vs 16-sensor training only; neural-only
median error, mean of 3 seeds [min, max]: all 9.9% [8.2, 11.6] vs 27.9% [25.9, 31.2]; 4 sensors 10-13% vs 55%
[52, 59]; 8 sensors 9-11% vs 36% [32, 38]; 16 sensors 9-11% vs 8-11% (no difference).
**Without the augmentation the network is no better than the midpoint guess (57%) at 4 sensors; with it, missing
sensors cost little. At 16 sensors the augmentation neither helps nor hurts.**

(c) Known IC coefficients as inputs vs hidden: all 9.9% [8.2, 11.6] vs 10.9% [10.1, 11.4]; 4 sensors 10.2-12.6% vs
13.9-16.6% (ranges of the three noise levels' seed means); 8 and 16 sensors: no difference beyond seed spread.
**The network uses the known starting state only when sensors are few; with 8 or 16 sensors the readings alone
carry the same information. The proposal's claim that the IC inputs matter is supported at 4 sensors only.**
Caveat: validation was also used for early stopping and tuning of every model compared here, equally for all.
