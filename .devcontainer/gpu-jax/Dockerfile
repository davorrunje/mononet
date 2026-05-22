FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs gnupg zsh sudo \
 && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.13 via uv (the base image does not ship a Python).
RUN /root/.local/bin/uv python install 3.13

WORKDIR /workspaces/mononet

# Dev containers run as root by design (matches devcontainer.json
# remoteUser=root); the missing-USER warning does not apply.
# nosemgrep: dockerfile.security.missing-user.missing-user
CMD ["sleep", "infinity"]
