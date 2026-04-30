#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/data/lhy_data/checkpoints_wyy/SoftMatch}"
cd "$ROOT_DIR"

GPU_IDS=(5 6)
SEEDS=(2025 2026 2027)
DATASETS=(20acdc 10acdc 5acdc)
EXP_PREFIX="${EXP_PREFIX:-SoftMatch}"
LOG_DIR="${LOG_DIR:-$DATA_DIR/logs/softmatch}"

mkdir -p "$LOG_DIR"

run_job() {
    local gpu_id="$1"
    local dataset="$2"
    local seed="$3"
    local exp_name="${EXP_PREFIX}_${dataset}_seed${seed}"
    local log_file="$LOG_DIR/${exp_name}.log"

    echo "[$(date '+%F %T')] START gpu=${gpu_id} dataset=${dataset} seed=${seed} log=${log_file}"
    CUDA_VISIBLE_DEVICES="$gpu_id" python comparsion/softmatch.py \
        --cli_dataset "$dataset" \
        --seed "$seed" \
        --exp "$exp_name" \
        --device "cuda:$gpu_id" \
    2>&1 | tee "$log_file"
    echo "[$(date '+%F %T')] DONE  gpu=${gpu_id} dataset=${dataset} seed=${seed}"
}

JOBS=()
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        JOBS+=("${dataset}:${seed}")
    done
done

worker() {
    local worker_idx="$1"
    local gpu_id="$2"
    local total_jobs="${#JOBS[@]}"
    local total_workers="${#GPU_IDS[@]}"
    local job_spec
    local dataset
    local seed

    for ((idx=worker_idx; idx<total_jobs; idx+=total_workers)); do
        job_spec="${JOBS[$idx]}"
        dataset="${job_spec%%:*}"
        seed="${job_spec##*:}"
        run_job "$gpu_id" "$dataset" "$seed"
    done
}

for worker_idx in "${!GPU_IDS[@]}"; do
    worker "$worker_idx" "${GPU_IDS[$worker_idx]}" &
done

wait
echo "All SoftMatch experiments completed."
