#!/usr/bin/env bash
umask 0077
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$REPO_ROOT/.local/cache/uv"
export UV_PYTHON_INSTALL_DIR="$REPO_ROOT/.local/python"
export XDG_CACHE_HOME="$REPO_ROOT/.local/cache"
export HF_HOME="$REPO_ROOT/.local/cache/huggingface"
export TMPDIR="$REPO_ROOT/.local/tmp"
export WANDB_MODE=offline
export WANDB_DIR="$REPO_ROOT/runs"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export NCCL_TIMEOUT=120
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120
mkdir -p "$TMPDIR" "$WANDB_DIR" "$XDG_CACHE_HOME"
# Prefer the matching CUDA wheel libraries over older system CUDA installations.
for cuda_lib in "$REPO_ROOT"/.venv/lib/python*/site-packages/nvidia/*/lib; do
    if [ -d "$cuda_lib" ]; then export LD_LIBRARY_PATH="$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
done
export WARP_CACHE_PATH="$REPO_ROOT/.local/cache/warp"
export CUDA_CACHE_PATH="$REPO_ROOT/.local/cache/cuda"
export JAX_COMPILATION_CACHE_DIR="$REPO_ROOT/.local/cache/jax"
mkdir -p "$CUDA_CACHE_PATH" "$WARP_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR/xla_gpu_per_fusion_autotune_cache_dir"
export MPLCONFIGDIR="$REPO_ROOT/.local/cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
