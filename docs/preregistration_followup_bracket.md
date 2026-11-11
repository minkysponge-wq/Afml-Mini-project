# Pre-registration: follow-up "learned bracket for Brent" on the pilot TEST split

Written and committed **before** the follow-up method is run on the pilot test split. The test split was already used
once for the pre-registered Phase D comparison (`docs/preregistration_pilot_test.md`). This is a second, separately
declared use, for a narrow follow-up allowed by `docs/WORKFLOW.md` Phase E after a no-go. It is labelled as such in
every report.

## Method
`mlp_s{k}_w0.5+brent_bracket`: bounded Brent on [θ_MLP − 0.5, θ_MLP + 0.5] ∩ [log 0.01, log 0.1], with xatol 1e-3 (the
baseline's own stopping rule). If the estimate lies within 2·xatol of an interior bracket edge, Brent re-runs on the full
bounds, and every solve is counted. Code: `blinit/inverse.py::brent_bracket`. Config: `configs/followup_bracket_test.json`.

## What is fixed in advance
- **Models.** `models/phaseC_v2`, seeds 0, 1 and 2 (unchanged).
- **Width.** w = 0.5, selected on validation only by lowest mean CPU-s.
- **Comparator.** Brent from the committed test run `results/phaseD_v1_3seeds_test`. Brent is deterministic, and
  re-runs were verified bit-identical. Both methods use the same stopping rule, so there is only one tolerance setting.
- **Cost metric.** Solver steps × 144.3 µs/step (CPU-s per query); solve counts are also reported.
- **Primary statistic.** Median paired difference against Brent in CPU-s, with a 95% bootstrap CI that resamples
  test trajectories (4,000 resamples, seed 0). Mean difference, all 9 sensor/noise cells, fallback rate and accuracy
  (median and p90 relative error; ν deviation from Brent's estimate) are also reported.
- **Success.** A positive saving is claimed only if the CI excludes 0, in favour of the bracket method, for ≥ 2 of 3
  seeds **and** the mean difference is < 0 for those seeds, so that a finite break-even exists.
- **Break-even.** Same offline cost as Phase D (data generation + tuning + training for that seed) divided by the mean
  per-query saving.
- **2× target.** Reported as the median CPU ratio Brent / bracket method.
- **Single use.** Run once. Nothing (width, fallback rule, models, analysis) is changed after seeing the results.
