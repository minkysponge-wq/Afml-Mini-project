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
