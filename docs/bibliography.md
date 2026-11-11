# Verified bibliography

Every entry was opened at the link given (see `literature_audit.md` for how far each was read).

- Amos, B. (2023). Tutorial on Amortized Optimization. *Foundations and Trends in Machine Learning*, 16(5),
  592–732. doi:10.1561/2200000102. https://arxiv.org/abs/2202.00665
- Borisyak, M., Born, S., Neubauer, P., Cruz-Bournazou, M. N. (2023). Deep Learning for Fast Inference of
  Mechanistic Models' Parameters. arXiv:2312.03166. https://arxiv.org/abs/2312.03166
- Cranmer, K., Brehmer, J., Louppe, G. (2020). The frontier of simulation-based inference. *PNAS*, 117(48),
  30055–30062. doi:10.1073/pnas.1912789117.
- Long, J., Khaliq, A., Furati, K. M. (2025). Parameter identification for PDEs using sparse interior data and a
  recurrent neural network. *Scientific Reports*, 15, 33828. doi:10.1038/s41598-025-02410-3 (metadata verified via
  Crossref; full text not opened).
- Neitzel, I., Pieper, K., Vexler, B., Walter, D. (2019). A sparse control approach to optimal sensor placement in
  PDE-constrained parameter estimation problems. *Numerische Mathematik*, 143, 943–984.
  https://arxiv.org/abs/1905.01696
- Oh, J., Lee, Y., Darbon, J., Karniadakis, G. E. (2026). Spectrally Safe Neural Operator Warm-Starts for
  Large-Scale Newton Solvers. arXiv:2606.21828. https://arxiv.org/abs/2606.21828
- Roy Sarkar, D., Kag, V., Pal, B., Goswami, S. (2024). Learning Hidden Physics and System Parameters with Deep
  Operator Networks. arXiv:2412.05133. https://arxiv.org/abs/2412.05133
- Sambharya, R., Hall, G., Amos, B., Stellato, B. (2024). Learning to Warm-Start Fixed-Point Optimization
  Algorithms. *JMLR*, 25(166), 1–46. https://arxiv.org/abs/2309.07835 Code: github.com/stellatogrp/l2ws.
- Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., Niepert, M. (2022).
  PDEBench: An Extensive Benchmark for Scientific Machine Learning. NeurIPS Datasets and Benchmarks Track.
  https://arxiv.org/abs/2210.07182 Code: https://github.com/pdebench/PDEBench (Burgers generator read at
  `pdebench/data_gen/data_gen_NLE/BurgersEq/burgers_multi_solution_Hydra.py`, `utils.py`).
- Tatsuoka, C., Yang, M., Xiu, D., Zhang, G. (2025). Bifidelity Parameter Estimation Using Conditional Diffusion
  Models. arXiv:2504.01894. https://arxiv.org/abs/2504.01894
- Zhou, M., Han, J., Rachh, M., Borges, C. (2023). A neural network warm-start approach for the inverse acoustic
  obstacle scattering problem. *Journal of Computational Physics*, 488, 112243. https://arxiv.org/abs/2212.08736

Software: SciPy (`least_squares`, trust-region reflective; `minimize_scalar(method="bounded")`, Brent-type bounded
search); scikit-learn (`MLPRegressor`); NumPy; Matplotlib.
