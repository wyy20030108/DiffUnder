#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASET="20acdc"
SEED=2026
EXP_PREFIX="AHE-ablation"

GPU0_ID=4
GPU1_ID=5

A_MODES_GPU0=(A0 A2 A4)
A_MODES_GPU1=(A1 A3)

run_ablation_group() {
    local gpu_id="$1"
    shift
    local ablation_modes=("$@")

    for mode in "${ablation_modes[@]}"; do
        echo "[$(date '+%F %T')] Start ${mode} on GPU ${gpu_id}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" python ablations/AHE.py \
            --cli_dataset "${DATASET}" \
            --seed "${SEED}" \
            --exp "${EXP_PREFIX}-${mode}" \
            --ablation_mode "${mode}" \
            --device "cuda:${gpu_id}"
        echo "[$(date '+%F %T')] Finish ${mode} on GPU ${gpu_id}"
    done
}

cleanup() {
    jobs -pr | xargs -r kill
}

trap cleanup INT TERM

run_ablation_group "${GPU0_ID}" "${A_MODES_GPU0[@]}" &
PID_GPU0=$!

run_ablation_group "${GPU1_ID}" "${A_MODES_GPU1[@]}" &
PID_GPU1=$!

status=0
wait "${PID_GPU0}" || status=$?
wait "${PID_GPU1}" || status=$?

if [ "${status}" -ne 0 ]; then
    echo "AHE ablation runs finished with errors."
    exit "${status}"
fi

echo "All A-series ablation experiments completed successfully."
