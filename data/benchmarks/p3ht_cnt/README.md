# P3HT/CNT blend conductivity

`P3HT_dataset.csv` is copied unchanged from [PV-Lab/Benchmarking](https://github.com/PV-Lab/Benchmarking)
(`datasets/P3HT_dataset.csv`), released under the MIT License in `LICENSE`.

- Benchmark: Q. Liang et al., "Benchmarking the performance of Bayesian optimization across multiple
  experimental materials science domains", *npj Computational Materials* 7, 188 (2021),
  doi:[10.1038/s41524-021-00656-9](https://doi.org/10.1038/s41524-021-00656-9).
- Original experiments: D. Bash et al., "Multi-fidelity high-throughput optimization of electrical
  conductivity in P3HT-CNT composites", *Advanced Functional Materials* (2021).

Each row is one drop-cast film: the percentage of P3HT polymer and of four carbon-nanotube types
(D1, D2, D6, D8), and its measured conductivity in S/cm. `mapping.yaml` imports it into Lodestone;
`uv run lodestone bench --published` runs the comparison.
