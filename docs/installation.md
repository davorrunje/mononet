# Installation

`mononet` ships **layers only** — pick the backend(s) you use via extras.

```
pip install "mononet[torch]"      # PyTorch
pip install "mononet[jax]"        # JAX + Flax NNX
pip install "mononet[keras]"      # Keras 3 (JAX backend)
pip install "mononet[all]"        # all three
```

## Extras reference

| Extra | Pulls | Notes |
|---|---|---|
| `torch` | `torch>=2.4` | Default PyPI wheel — CUDA build on linux-x86_64, CPU wheel on macOS/Windows. |
| `jax` | `jax`, `flax` | CPU. |
| `keras` | `keras`, `jax` | Keras 3 on the JAX backend, CPU. |
| `all` | `torch`, `jax`, `keras` | All three backends. Installs on linux/macOS/Windows. |
| `torch-gpu` | `torch>=2.12` | CUDA 13 wheel (bundles its own toolkit; linux). |
| `jax-gpu` | `jax[cuda12]`, `flax` | CUDA 12 (**linux-only**). |
| `keras-gpu` | `keras`, `jax[cuda12]` | Keras 3 on JAX GPU (**linux-only**). |
| `torch-cpu` | `torch>=2.4` | CPU wheel — **uv only** (see caveat). |
| `all-cpu` | `torch-cpu`, `jax`, `keras` | All three, guaranteed CPU under uv. |

## Why `all` is CUDA-heavy on linux (and why there is no `all-gpu`)

torch's default PyPI wheel *is* the CUDA build on linux-x86_64, and it ships
under the same name+version as the macOS/Windows CPU wheels — so `torch>=2.4`
resolves everywhere and opportunistically carries CUDA on linux. `all` inherits
that; it is the "all backends, portable" install, not a GPU install.

jax is packaged differently: `jax[cuda12]` pulls linux-only plugin wheels, so a
symmetric "all backends on GPU" extra could not install on macOS/Windows. That
is why `all` keeps plain (CPU) `jax`, and why there is no `all-gpu` — GPU work
uses the single-backend `*-gpu` extras (see the GPU devcontainer flavors).

## CPU-only torch: the uv-vs-pip caveat

There is no way to express "CPU torch" in extras metadata that **pip** honors —
plain `torch` on PyPI is the CUDA wheel. `mononet` provides `torch-cpu` /
`all-cpu`, which redirect torch to the [PyTorch CPU wheel
index](https://download.pytorch.org/whl/cpu). This redirect is a **uv-only**
mechanism (`[tool.uv.sources]`), so:

- Under **uv** (`uv sync --extra all-cpu`) you get CPU torch and zero
  `nvidia-*` wheels.
- Under plain **pip**, `mononet[torch-cpu]` resolves to the same default wheel
  as `mononet[torch]` (CUDA on linux). To force CPU torch with pip, install it
  explicitly from the CPU index:

  ```
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install "mononet[jax,keras]"
  ```

## Devcontainer flavors

| Flavor | Extra synced |
|---|---|
| `default` (CPU) | `all-cpu` |
| `gpu-torch` | `torch-gpu` |
| `gpu-jax` | `jax-gpu` |
| `gpu-keras` | `keras-gpu` |
