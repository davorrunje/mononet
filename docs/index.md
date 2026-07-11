---
hide-toc: false
---

# mononet

**Unconstrained monotonic neural networks** with first-class support for
**PyTorch**, **JAX** (Flax NNX), and **Keras 3**.

Reference implementation of:

> Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic
> Neural Networks.* ICML 2023. [arXiv:2205.11775](https://arxiv.org/abs/2205.11775)

## Install

```
pip install "mononet[torch]"      # or [jax], [keras], [all]
```

See [Installation](installation.md) for the full extras reference, GPU extras,
and the CPU-torch (uv vs pip) caveat.

## Citation

If you use `mononet` in academic work, please cite the reference paper:

```bibtex
@inproceedings{runje2023constrained,
  title     = {Constrained Monotonic Neural Networks},
  author    = {Runje, Davor and Shankaranarayana, Sharath M.},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  year      = {2023},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v202/runje23a.html},
  eprint    = {2205.11775},
  archivePrefix = {arXiv}
}
```

> Note: confirm the exact BibTeX entry against the PMLR proceedings page
> before the first PyPI release — venue, volume, and URL fields are
> sensitive to typos.

```{toctree}
:hidden:

installation
guides/index
concepts/index
benchmarks/index
reference
releasing
about/index
```
