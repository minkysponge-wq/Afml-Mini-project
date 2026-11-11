# Pre-registration: Phase F robustness test sets (T4)

Written and committed **before** any of the three sets below was generated or evaluated. The pilot test
(`docs/preregistration_pilot_test.md`) has been analysed (T1, `RESEARCH_LOG.md`, 2026-11-11): under its rule the
saving of MLP+LSQ against Brent was *not positive* (CI excludes 0 at the primary tolerance for 3/3 seeds, at the
accuracy-matched tolerance for 0/3). These sets ask whether that picture changes when the data differ from training.

## The three sets
Each is a new held-out test set of 50 fresh trajectories. Train, validation, normaliser, kNN library and models are
unchanged (`data/phaseB_v2`, `models/phaseC_v2`). Inversion always uses the 256-cell simulator.

| Set | What changes | IC seed offset | Config |
|---|---|---|---|
| `phaseF_highfreq` | IC uses sine modes 2, 3, 4 instead of 1, 2, 3 (the inverse problem is told the modes; the learned initialisers were trained on modes 1, 2, 3) | 3,000,000 | `configs/phaseF_highfreq.json` |
| `phaseF_finegrid` | data simulated on 1,024 cells, inverted on 256 | 4,000,000 | `configs/phaseF_finegrid.json` |
| `phaseF_unseenmask` | sensor-availability patterns never seen in training | 5,000,000 | `configs/phaseF_unseenmask.json` |

High-frequency and fine-grid sets use the usual 9 conditions (4/8/16 nested sensors x 0/1/5% noise): 450 variants each.
The unseen-mask set uses six named masks x 3 noise levels: 900 variants. Four masks are unseen: `half_left`
(sensors 0-7 off), `random8` (8 sensors from `default_rng(5_000_001)`), `random4` (4 sensors from
`default_rng(5_000_002)`), `every4_shift` (every 4th sensor starting at index 1). Two are the nested 8- and 4-sensor
patterns seen in training (`seen_nested8`, `seen_nested4`), included as same-trajectory references. The slot lists
are written in the config.

## Fixed in advance
- **Methods:** brent, midpoint+lsq, grid5+lsq, knn+lsq, mlp_s0/s1/s2+lsq. For unseen masks, kNN falls back to the
  all-16-sensor library group and measures distance over the available readings plus IC features (implemented and
  tested before the sets were built).
- **Tolerance:** the primary pilot setting only, `configs/phaseD_v1_test.json` (brent xatol 1e-3, lsq xtol 1e-3).
  The accuracy-matched setting is not rerun on these sets (compute); T2 covers the tolerance question on validation.
  Consequently the two-tolerance success rule cannot be met here by construction; what is reported per set is whether
  the 95% CI of `mlp_sK+lsq - brent | all` excludes 0 at the primary tolerance for at least 2 of 3 seeds, labelled as
  the **primary-tolerance criterion**, next to the same statistic from the pilot test.
- **Cost metric, statistics, accuracy measures:** exactly as in the pilot pre-registration (solver steps x 144.3 us;
  median paired difference in CPU-s with a 95% bootstrap CI resampling trajectories; median and p90 relative error
  against the true nu; failures logged, none dropped).
- **Also reported:** the MLP's neural-only error (`init_rel_err`) on each set against the original test set; for the
  unseen-mask set, every statistic separately for each mask and for seen vs unseen masks.
- **Single use.** Each set is evaluated once. No model, tolerance, kNN setting or analysis choice is changed in
  response to these results. Bugs found afterwards are reported with the original and corrected numbers.

## Expected, stated before the runs
- High-frequency ICs carry less information about nu at high nu (mode 4 decays about 150-fold by the first reading at
  nu = 0.1), so larger errors for every method there are a property of the data, not a failure.
- The MLP's neural-only error should rise on shifted inputs; the refinement should absorb a worse start at extra cost.
- Fine-grid data add a model-mismatch error that no method can remove; accuracy should degrade equally for all.
