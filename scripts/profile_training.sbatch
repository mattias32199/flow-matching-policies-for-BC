#!/usr/bin/env bash
#SBATCH --job-name=ocbench-profile
#SBATCH --constraint=A6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
set -euo pipefail
# Submit from the checkout, or use private_slurm.py which sets the process cwd.
source scripts/env.sh
PROFILE_OUTPUT="$REPO_ROOT/.local/profile/$SLURM_JOB_ID"
mkdir -p "$PROFILE_OUTPUT"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv > "$PROFILE_OUTPUT/gpu.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,memory.used --format=csv -l 1 > "$PROFILE_OUTPUT/telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT
.venv/bin/python -u scripts/profile_training.py --output "$PROFILE_OUTPUT"
.venv/bin/python -u scripts/benchmark_speed.py sim > "$PROFILE_OUTPUT/simulation.log" 2>&1
touch "$PROFILE_OUTPUT/DONE"
