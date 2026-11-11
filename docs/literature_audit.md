# Literature audit (second pass, 2026-11-11)

This is a targeted audit, **not** a systematic review. No "first" or "unexplored" claim may be made from it.

Verification levels: **[full]** full text read for the points recorded here; **[abs]** abstract page opened and
read; **[meta]** bibliographic record checked only; **[src]** source code read. Every work below was opened at the
link given in `bibliography.md`.

## Closest works

| Work | Task | Learned part | Baselines | Cost accounting | Overlap with us |
|---|---|---|---|---|---|
| Sambharya, Hall, Amos, Stellato (2024), JMLR [full, first 100k characters of the arXiv HTML; later SCS examples not read] | Parametric convex problems solved by fixed-point methods (gradient descent, proximal gradient, OSQP, SCS): QP, lasso, quadcopter MPC, image deblurring, robust Kalman filtering | Network maps problem parameters to a warm start, trained through k solver steps | Cold start; **nearest-neighbour warm start**; previous-solution warm start (MPC only) | Reports iterations to a fixed-point residual and solve times (ms). Training time is stated (trained on an HPC cluster) but is **not** set against the savings; no break-even count | High conceptually: learned warm start in front of an unchanged solver, with a nearest-neighbour baseline. Different setting: convex fixed-point iterations, cheap iterations, no PDE simulator. |
| Zhou, Han, Rachh, Borges (2023), J. Comput. Phys. [full] | Inverse acoustic obstacle scattering (obstacle shape from scattered field) | CNN predicts Fourier coefficients of the boundary; Gauss–Newton refines | Gauss–Newton from a unit disk; linear sampling method alone and refined; network alone | Reports average running time per obstacle (their Table 3) and training time (their Table 1 caption). Training time is **excluded** from the comparison; cost of generating training data is not reported | High: neural initial guess plus classical refinement for a PDE inverse problem. Their motive is avoiding local minima at high frequency; ours is cost on a problem with a single minimum. Their unknown is a shape (many coefficients); ours is one scalar. |
| Borisyak, Born, Neubauer, Cruz-Bournazou (2023), arXiv:2312.03166 [abs] | Parameters of ODE bioprocess growth models | Network predicts parameters from data | Gradient-based fitting alone; network alone; network followed by fitting | Abstract says inference is orders of magnitude faster than fitting and that training needs computational resources; no break-even stated in the abstract | Same pattern (prediction, optionally refined) for ODE calibration. |
| Oh, Lee, Darbon, Karniadakis (2026), arXiv:2606.21828 [abs] | Newton solvers for nonlinear PDEs (hyperelasticity) | Neural-operator warm start, fine-tuned so the starting Jacobian stays positive definite | Standard Newton | Reports up to 5.4x wall-clock speed-up on a 3-D problem | Learned warm start for a forward PDE solve; shows low prediction error does not guarantee a usable start. |
| Amos (2023), Found. Trends ML [abs] | Tutorial on amortised optimisation | Fully and semi-amortised models | — | Discusses amortisation in general | Places our method: a semi-amortised, regression-trained initialiser. The idea itself is not ours. |
| Cranmer, Brehmer, Louppe (2020), PNAS [meta] | Review of simulation-based inference | Amortised posterior / likelihood estimators | — | — | Neural-only amortised estimation is the established alternative to prediction-plus-refinement. |
| Tatsuoka, Yang, Xiu, Zhang (2025), arXiv:2504.01894 [abs] | Bayesian parameter estimation with expensive forward models | Low-fidelity conditional diffusion model, then a high-fidelity model on a focused sample | — | Motivated by the cost of repeated forward runs | Amortised parameter estimation with a cost motive; Bayesian, not a point-estimate warm start. |
| Roy Sarkar, Kag, Pal, Goswami (2024), arXiv:2412.05133 [abs] | Hidden-physics discovery and parameter identification from sparse, noisy data (Burgers among the benchmarks) | Pretrained DeepONet used inside a physics-informed inverse step | — (abstract) | Not stated in the abstract | Same problem family (sparse-data parameter identification incl. Burgers); a surrogate replaces the simulator, whereas we keep the simulator and learn only the start. |
| Long, Khaliq, Furati (2025), Sci. Rep. [meta; full text not opened, publisher page redirected to a sign-in] | PDE parameter identification from sparse interior data | GRU with an implicit numerical scheme | not verified | not verified | Same problem class; trained per problem, not amortised. Cite for the problem class only. |
| Neitzel, Pieper, Vexler, Walter (2019), Numer. Math. [abs] | Optimal sensor placement for PDE parameter estimation | none (classical optimal design) | — | — | Judges sensor layouts by the covariance of the least-squares estimator, the same Fisher-information view as our Cramér–Rao check (T3). |
| Takamoto et al. (2022), PDEBench, NeurIPS D&B [abs, src] | Benchmark with a Burgers generator | — | includes a gradient-based inverse baseline | — | Source of the numerical scheme and the nu/pi convention; our solver is an independent re-implementation. |

Leads from the first pass that were **dropped** because they were not opened: neural-operator warm starts for steady
CFD ("super-fidelity", J. Comput. Phys. 2025) and learned starting models for full-waveform inversion.

## New searches in this pass (2026-11-11)
1. *Neural warm starts for ODE/PDE model calibration.* Found Borisyak et al. (above). A preprint on neural-network
   and 1-nearest-neighbour initial parameters for DEB model calibration appeared only on an aggregator page and was
   not opened; not cited.
2. *Break-even / amortisation cost accounting for learned solvers.* No paper in the learned-warm-start line was found
   that reports a break-even query count; the two closest works above report training time separately and leave it
   out of the comparison. Break-even formulas of the form (offline cost) / (per-query saving) appear in unrelated
   areas (learned query optimisers, LLM agents); none of those was opened in full, none is cited.
3. *Identifiability and sensor placement for diffusion-type parameters.* Found Neitzel et al. (above). Further
   Fisher-information design papers surfaced in the search (interface identification, PINN-based placement) but
   were not opened; not cited.

## What we can and cannot claim
1. **Established, reused here.** Supervised regression to amortise an inverse map; warm-starting a local solver from
   a prediction (Sambharya et al.; Zhou et al.; Borisyak et al.); nearest-neighbour warm starts as a baseline
   (Sambharya et al.); bounded least squares and Brent-type bounded scalar search (SciPy); the Cramér–Rao bound;
   the PDEBench finite-volume scheme.
2. **What this project adds.** A controlled, cost-counted comparison on one scalar PDE inverse problem in which
   (a) the learned start is compared with a direct bounded scalar search and with a nearest-neighbour start, not
   only with a cold start; (b) the offline cost (training-data generation, tuning, training) is divided by the
   per-query saving to give a break-even count, which the two closest works do not report; (c) the comparison is
   repeated across stopping tolerances, because a single tolerance pair gave a saving that vanished when precision
   was matched; (d) the test evaluation was pre-registered and the negative outcome is reported.
3. **Limits of that contribution.** One PDE, one scalar parameter, synthetic data, a few-evaluation 1-D search as the
   competitor. The mechanism is not new, and the result is specific to a problem whose objective has one minimum.

## Novelty risk (candid)
High. Learned warm starts are established in optimisation and in PDE inverse problems. The defensible framing is a
careful, cost-honest evaluation with a strong scalar-search baseline and a reported negative result.
