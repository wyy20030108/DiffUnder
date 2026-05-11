#!/bin/bash

#!/bin/bash

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

SEEDS=(2027)
LABELS=(10acdc)

# 默认使用 6 和 7 号卡（如果不传参数的话）
DEFAULT_GPUS=(6 7)
EXP_PREFIX="AgScore"
LOG_ROOT="$ROOT_DIR/logs/agscore_$(date +%Y%m%d_%H%M%S)"

if [[ $# -gt 0 ]]; then
  GPU_IDS=("$@")
else
  GPU_IDS=("${DEFAULT_GPUS[@]}")
fi

# 【修改点】：将 -ne 2 改为 -eq 0，允许任意数量的 GPU
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "Usage: $0 [GPU_ID_1 GPU_ID_2 ...]"
  echo "Example: $0 0 1 2 3"
  exit 1
fi

mkdir -p "$LOG_ROOT"

run_experiment() {
  local gpu_id="$1"
  local seed="$2"
  local label="$3"
  local log_file="$4"
  local exp_name="${EXP_PREFIX}_${label}_seed${seed}"

  echo "[$(date '+%F %T')] START gpu=${gpu_id} seed=${seed} label=${label}" | tee "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu_id" python comparsion/agscore.py \
    --cli_dataset "$label" \
    --seed "$seed" \
    --exp "$exp_name" \
    --device "cuda:${gpu_id}" 2>&1 | tee -a "$log_file"
}

PIDS=("" "")
TASK_DESC=("" "")
FAILED_TASKS=()
slot_count=${#GPU_IDS[@]}
task_index=0
total_tasks=$((${#SEEDS[@]} * ${#LABELS[@]}))

for label in "${LABELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    slot=$((task_index % slot_count))
    gpu_id="${GPU_IDS[$slot]}"
    task_desc="seed=${seed}, label=${label}, gpu=${gpu_id}"
    log_file="$LOG_ROOT/agscore_${label}_seed${seed}_gpu${gpu_id}.log"

    if [[ -n "${PIDS[$slot]}" ]]; then
      wait "${PIDS[$slot]}"
      exit_code=$?
      if [[ $exit_code -ne 0 ]]; then
        echo "[$(date '+%F %T')] FAILED ${TASK_DESC[$slot]} (exit_code=${exit_code})"
        FAILED_TASKS+=("${TASK_DESC[$slot]} (exit_code=${exit_code})")
      else
        echo "[$(date '+%F %T')] DONE ${TASK_DESC[$slot]}"
      fi
    fi

    echo "[$(date '+%F %T')] QUEUE $((task_index + 1))/${total_tasks}: ${task_desc}"
    run_experiment "$gpu_id" "$seed" "$label" "$log_file" &
    PIDS[$slot]=$!
    TASK_DESC[$slot]="$task_desc"
    task_index=$((task_index + 1))
  done
done

for slot in "${!PIDS[@]}"; do
  if [[ -n "${PIDS[$slot]}" ]]; then
    wait "${PIDS[$slot]}"
    exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
      echo "[$(date '+%F %T')] FAILED ${TASK_DESC[$slot]} (exit_code=${exit_code})"
      FAILED_TASKS+=("${TASK_DESC[$slot]} (exit_code=${exit_code})")
    else
      echo "[$(date '+%F %T')] DONE ${TASK_DESC[$slot]}"
    fi
  fi
done

echo "Logs saved to: $LOG_ROOT"
if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
  echo "Failed tasks:"
  for failed_task in "${FAILED_TASKS[@]}"; do
    echo "  - $failed_task"
  done
  exit 1
fi

echo "All AgScore experiments finished successfully."
