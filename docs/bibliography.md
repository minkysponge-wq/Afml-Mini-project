# Verified bibliography

Only entries whose metadata was checked against an authoritative record are listed. See
`literature_audit.md` for verification levels and unverified leads.

- Amos, B. (2023). Tutorial on Amortized Optimization. *Foundations and Trends in Machine Learning*,
  16(5), 592–732. doi:10.1561/2200000102. arXiv:2202.00665.
- Cranmer, K., Brehmer, J., Louppe, G. (2020). The frontier of simulation-based inference. *PNAS*,
  117(48), 30055–30062. doi:10.1073/pnas.1912789117.
- Long, J., Khaliq, A., Furati, K. M. (2025). Parameter identification for PDEs using sparse interior
  data and a recurrent neural network. *Scientific Reports*, 15, 33828. doi:10.1038/s41598-025-02410-3
  (metadata verified via Crossref).
- Sambharya, R., Hall, G., Amos, B., Stellato, B. (2024). Learning to Warm-Start Fixed-Point
  Optimization Algorithms. *JMLR*, 25(166), 1–46. arXiv:2309.07835. Code: github.com/stellatogrp/l2ws.
- Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., Niepert, M.
  (2022). PDEBench: An Extensive Benchmark for Scientific Machine Learning. NeurIPS Datasets and
  Benchmarks Track. arXiv:2210.07182. Code: github.com/pdebench/PDEBench (Burgers generator read at
  `pdebench/data_gen/data_gen_NLE/BurgersEq/burgers_multi_solution_Hydra.py`, `utils.py`).
- Zhou, M., Han, J., Rachh, M., Borges, C. (2023). A neural network warm-start approach for the
  inverse acoustic obstacle scattering problem. *Journal of Computational Physics*, 488, 112243.
  arXiv:2212.08736.

Software: SciPy (`least_squares` TRF; `minimize_scalar(method="bounded")`, Brent-type bounded search).
