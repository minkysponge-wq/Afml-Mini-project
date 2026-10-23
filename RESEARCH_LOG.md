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
