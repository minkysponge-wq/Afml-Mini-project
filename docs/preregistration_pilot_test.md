# Pre-registration: pilot TEST evaluation

Written and committed **before** the pilot test split (`data/phaseB_v2`, test: 50 trajectories,
IC seeds 2,000,000–2,000,049, 450 observation variants) was used for any evaluation.

## What is fixed in advance
- **Models.** MLP `models/phaseC_v2` seeds 0, 1 and 2, all trained before this document. kNN is
  re-tuned on validation only, inside `evaluate.py`, exactly as in the validation run.
- **Methods.** brent, midpoint+lsq, grid5+lsq, knn+lsq, mlp_s0/s1/s2+lsq.
- **Two tolerance settings, both run and both reported.** Neither is chosen after seeing test results.
  - *Primary* (`configs/phaseD_v1_test.json`): brent xatol 1e-3, lsq xtol 1e-3.
  - *Accuracy-matched* (`configs/phaseD_v1_tightlsq_test.json`): brent xatol 1e-3, lsq xtol 1e-4.
- **Cost metric.** Solver steps × 144.3 µs/step, reported as CPU-s per query. Forward-solve counts are
  also reported. Parallel wall time is not used for claims.
- **Primary comparison.** Each MLP seed against brent, the strongest classical baseline on validation.
  Statistic: median paired difference in CPU-s, with a 95% bootstrap CI that resamples test trajectories.
  Secondary comparisons are against midpoint+lsq and kNN. All 9 sensor/noise cells are reported; none are dropped.
- **Accuracy.** Median and p90 relative error against the true ν; deviation of each method's estimate
  in ν from the best-found optimum; simulator failures; objective gaps.
- **Break-even.** Offline CPU (training-data generation, taken as wall × workers as an upper bound, plus
  tuning and training) divided by the mean per-query CPU saving against brent. If the saving is ≤ 0, the
  report says there is no finite break-even.
- **What counts as success.**
  - A saving is reported as positive only if the 95% CI excludes 0 under **both** tolerance settings,
    for at least 2 of the 3 seeds.
  - The proposal's 2× target is evaluated on the median CPU ratio brent/MLP, and is stated as met or not met.
- **Single use.** The test split is evaluated once per setting. Nothing (models, tolerances, kNN grid,
  analysis code) is changed in response to test results. Bugs found afterwards are reported as such,
  with both the original and the corrected numbers.
