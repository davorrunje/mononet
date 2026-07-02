# Modern CUDA base, kept consistent across the GPU flavors. Note: the
# Keras JAX backend brings its own CUDA via the jax[cuda12] pip wheels
# (CUDA 12.9), so the base CUDA version does not gate Blackwell (sm_120)
# support — a recent jaxlib does. Driver on the host must support the GPU.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs gnupg zsh sudo \
 && rm -rf /var/lib/apt/lists/*

# Create the non-root 'vscode' user (uid/gid 1000) with passwordless
# sudo, matching the standard devcontainer convention and the
# mcr.microsoft.com/devcontainers base used by the CPU and proofs
# flavors. updateRemoteUserUID (default true) remaps this uid to the
# host user at create time if it differs, keeping bind-mounted files
# correctly owned.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
 && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
 && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
 && chmod 0440 /etc/sudoers.d/$USERNAME

# Everything below runs as the non-root user; uv and its managed Python
# install under /home/vscode and are owned by that user.
USER $USERNAME
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.13 via uv (the base image does not ship a Python).
RUN uv python install 3.13

WORKDIR /workspaces/mononet

CMD ["sleep", "infinity"]
