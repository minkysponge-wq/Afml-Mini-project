# Literature audit (first pass — INCOMPLETE)

Status: 2026-10-23, first targeted pass (≈10 web searches + source checks). This is **not** a
systematic review. No "first"/"unexplored" claim may be made from it.

Verification levels: **[V-meta]** bibliographic metadata verified against an authoritative record
(Crossref / journal page / arXiv abstract page); **[V-abs]** abstract read; **[V-src]** source code
read; **[found]** surfaced by search only — not yet verified, do not cite in a paper yet.

## Closest works

| Work | Task | Learned component | Baselines | Cost accounting | Overlap with us |
|---|---|---|---|---|---|
| Sambharya, Hall, Amos, Stellato (2024) JMLR 25(166) [V-meta, V-abs via search summary] | Parametric fixed-point / convex optimisation (control, statistics, signal processing) | NN maps problem parameters -> warm start, trained through k solver steps (fixed-point-residual or regression loss) | Cold start, nearest-neighbour warm start, others (to verify in full text) | Reports iterations and solve time; training cost treatment to verify | **High conceptually**: learned warm start + generic solver. Differs: convex fixed-point solvers, not PDE inverse problems with expensive simulators. |
| Zhou, Han, Rachh, Borges (2023) J. Comput. Phys. 488:112243, arXiv:2212.08736 [V-meta, V-abs] | Inverse acoustic obstacle scattering (shape) | NN predicts initial guess, refined by Gauss–Newton | GN from unit circle; GN from linear-sampling-method init | Abstract does not report run-time/training accounting (full text to check) | **High**: neural init + classical refinement for a PDE inverse problem. Differs: motivation is avoiding local minima (high frequency), not cost; high-dimensional shape vs our scalar. |
| Amos (2023) Found. Trends ML 16(5):592–732, arXiv:2202.00665 [V-meta] | Survey of amortised optimisation | Fully vs semi-amortised models; regression- vs objective-based learning | — | Discusses amortisation trade-offs | Frames our method: **semi-amortised, regression-based** initialiser. Our contribution cannot be the idea itself. |
| Long, Khaliq, Furati (2025) Sci. Rep. 15:33828, doi:10.1038/s41598-025-02410-3 [V-meta via Crossref] | PDE parameter identification from sparse interior data (Burgers, Allen–Cahn, NLS per abstract summary) | GRU + implicit numerical scheme trained per problem | To verify in full text | To verify | Same problem class (sparse-data PDE parameter ID incl. Burgers); **different** approach (per-instance hybrid training, not amortised init). |
| Takamoto et al. (2022) PDEBench, NeurIPS Datasets & Benchmarks, arXiv:2210.07182 [V-meta, V-src] | Benchmark; includes Burgers data generator | — | Includes a gradient-based inverse baseline | — | Simulator source and convention (nu/pi). |
| Cranmer, Brehmer, Louppe (2020) PNAS 117(48):30055, doi:10.1073/pnas.1912789117 [V-meta] | Simulation-based inference review | Amortised neural posterior / likelihood estimators | — | — | Neural-only amortised estimators are an established alternative to our neural-only predictor; relevant for the "neural-only vs refinement" ablation. |

Other leads surfaced by search, **[found] — not verified, not to be cited yet**: neural-operator warm
starts for steady CFD (J. Comput. Phys. 2025, "super-fidelity"); "Spectrally safe neural operator
warm-starts for Newton solvers" (arXiv 2606.21828); learned starting models for full-waveform
inversion (several, 2021–2025); DeepONet-based Burgers viscosity inference (arXiv 2412.05133);
conditional-diffusion bifidelity parameter estimation (arXiv 2504.01894).

## Separation of claims

1. **Established techniques we reuse.** Supervised regression to amortise an inverse map; warm-starting
   a local solver from a learned prediction (Sambharya et al.; Zhou et al.); bounded least squares
   (TRF) and Brent's bounded scalar minimisation; nearest-neighbour warm starts; PDEBench's FV scheme.
2. **Contribution we currently hypothesise** (from the proposal): a *controlled cost-accounted study*
   of when learned initialisation of a scalar PDE-parameter fit saves end-to-end computation versus
   strong classical alternatives (incl. direct scalar search and kNN), under missing sensors and IC
   shift, with break-even analysis.
3. **Evidence that would be required.** (a) Positive, statistically supported per-query savings
   versus the *strongest* classical baseline at matched accuracy, across tolerance sweeps; (b)
   robustness under sensor dropout / noise / IC shift / discretisation mismatch; (c) honest break-even;
   (d) for anything beyond a workshop-level benchmark: more than one PDE/parameter dimension, and a
   result that is not trivially explained by "a 1-D search needs ~10 evaluations anyway".

## Novelty-risk assessment (candid)

**High.** The mechanism (learned warm start + classical refinement) is established in both the
optimisation-learning literature and PDE inverse problems. A single-scalar, single-PDE study is a
narrow setting where a few-evaluation 1-D search is already cheap, so the attainable saving is
bounded (Phase A quantifies this ceiling via the oracle diagnostic). A defensible contribution would
be a careful *negative or conditional* result plus a reusable, cost-honest benchmark protocol —
plausible as a course project or workshop paper, not as a main-track novelty claim on current evidence.

## Still to do in the audit
Full-text read of Sambharya et al. (baselines incl. NN warm start, cost accounting) and Zhou et al.;
search for learned warm starts in ODE/PDE *calibration* (systems biology, reservoir history matching);
classical robustness to missing sensors (optimal sensor placement / identifiability literature);
verify the [found] items or drop them.
