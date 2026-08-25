#!/usr/bin/env bash
# Runs on the HOST (not the container) via devcontainer.json
# `initializeCommand`, for the GPU flavors only.
#
# Unlike host-init.sh, failure here SHOULD block container start: the GPU
# flavors pass `--gpus=all`, so without a working host driver the container
# cannot start at all. Docker's own failure mode is an opaque OCI message
#
#   error running prestart hook #0: exit status 1 …
#   nvidia-container-cli: initialization error: nvml error: driver not loaded
#
# which says nothing about the actual cause. This script fails earlier with an
# actionable diagnosis instead.
#
# Set MONONET_SKIP_GPU_PREFLIGHT=1 to bypass (e.g. a remote Docker context
# whose GPUs are not visible from this machine).
set -u

readonly NVIDIA_PCI_VENDOR='0x10de'

log() { printf '%s\n' "$*" >&2; }

die() {
    log ""
    log "=============================================================="
    log " mononet GPU devcontainer preflight FAILED"
    log "=============================================================="
    local line
    for line in "$@"; do log "  ${line}"; done
    log ""
    log "  Use the 'default' (CPU) flavor if you don't need the GPU, or set"
    log "  MONONET_SKIP_GPU_PREFLIGHT=1 to bypass this check."
    log "=============================================================="
    exit 1
}

if [ -n "${MONONET_SKIP_GPU_PREFLIGHT:-}" ] \
    && [ "${MONONET_SKIP_GPU_PREFLIGHT}" != "0" ]; then
    log "gpu-preflight: skipped (MONONET_SKIP_GPU_PREFLIGHT is set)."
    exit 0
fi

# Only Linux hosts run the nvidia-container-toolkit path this script knows how
# to diagnose. Elsewhere (Docker Desktop, WSL passthrough, remote contexts)
# stay out of the way rather than guess.
if [ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]; then
    log "gpu-preflight: non-Linux host; skipping GPU checks."
    exit 0
fi

# --- Is there an NVIDIA GPU at all? -----------------------------------------
# Read PCI vendor IDs from sysfs so this needs no lspci/pciutils.
has_nvidia_gpu() {
    local vendor_file
    for vendor_file in /sys/bus/pci/devices/*/vendor; do
        [ -r "${vendor_file}" ] || continue
        if [ "$(cat "${vendor_file}")" = "${NVIDIA_PCI_VENDOR}" ]; then
            return 0
        fi
    done
    return 1
}

if ! has_nvidia_gpu; then
    die "No NVIDIA GPU found on this host (no PCI device with vendor" \
        "${NVIDIA_PCI_VENDOR}). The GPU flavors pass --gpus=all and cannot" \
        "start here."
fi

# --- Driver branch + package-name hints -------------------------------------
# Best-effort: used only to make the remedy concrete on Debian/Ubuntu.
driver_branch() {
    command -v dpkg-query >/dev/null 2>&1 || return 1
    dpkg-query -W -f='${Package}\n' 'nvidia-driver-*' 2>/dev/null \
        | sed -n 's/^nvidia-driver-\([0-9]\+\).*/\1/p' \
        | sort -rn | head -1
}

# Fills the global REMEDY array. An array (not a $(...) capture) so that each
# line survives as one argument to die() instead of being word-split.
REMEDY=()
set_remedy() {
    local branch="${1:-}" kernel="${2:-}"
    if [ -n "${branch}" ]; then
        REMEDY=(
            "sudo apt update"
            "sudo apt install -y linux-modules-nvidia-${branch}-open-generic-hwe-24.04 nvidia-driver-${branch}-open"
            "sudo modprobe nvidia_uvm && nvidia-smi"
        )
    else
        REMEDY=(
            "Install the NVIDIA driver and the kernel modules matching"
            "kernel ${kernel}, then reload them (or reboot)."
        )
    fi
}

# --- Driver present and loaded? ---------------------------------------------
kernel="$(uname -r)"
branch="$(driver_branch || true)"
set_remedy "${branch}" "${kernel}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "An NVIDIA GPU is present but 'nvidia-smi' is not installed, so no" \
        "driver is available to the container runtime." \
        "" \
        "Fix:" \
        "${REMEDY[@]}"
fi

if ! nvidia_smi_out="$(nvidia-smi -L 2>&1)"; then
    # The classic case: the kernel was upgraded but the NVIDIA kernel modules
    # were not rebuilt/installed for the running kernel, so NVML is dead.
    detail=""
    if [ -z "$(find "/lib/modules/${kernel}" -name 'nvidia.ko*' -print -quit \
        2>/dev/null)" ]; then
        detail="No nvidia.ko for the RUNNING kernel (${kernel}) — the kernel was upgraded without a matching NVIDIA module build."
    else
        detail="nvidia.ko exists for kernel ${kernel} but the driver is not loaded (try: sudo modprobe nvidia_uvm)."
    fi
    die "'nvidia-smi' failed: the NVIDIA driver is not loaded on the host." \
        "" \
        "  ${detail}" \
        "" \
        "nvidia-smi said:" \
        "  ${nvidia_smi_out}" \
        "" \
        "Fix:" \
        "${REMEDY[@]}"
fi

# --- Container runtime hook present? ----------------------------------------
# This is the binary Docker invokes as the prestart hook for --gpus=all.
if ! command -v nvidia-container-cli >/dev/null 2>&1; then
    die "The host driver works, but 'nvidia-container-cli' is missing, so" \
        "Docker cannot expose GPUs to the container (--gpus=all will fail)." \
        "" \
        "Fix: install the NVIDIA Container Toolkit, then restart Docker:" \
        "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" \
        "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
fi

gpu_count="$(printf '%s\n' "${nvidia_smi_out}" | grep -c '^GPU ' || true)"
log "gpu-preflight: OK — ${gpu_count} GPU(s), kernel ${kernel}."
exit 0
