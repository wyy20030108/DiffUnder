#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_PY="${TEST_PY:-$ROOT_DIR/test.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"

METHOD_NAME="${METHOD_NAME:-SoftMatch}"
DATASET="${DATASET:-ACDC}"
BASE_DIR="${BASE_DIR:-/data/lhy_data/ACDC}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/lhy_data/checkpoints_wyy}"
METHOD_DIR="${METHOD_DIR:-$CHECKPOINT_ROOT/$METHOD_NAME}"

GPU_IDS_STR="${GPU_IDS:-0}"
IFS=', ' read -r -a GPU_ID_LIST <<< "$GPU_IDS_STR"
if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
    echo "No GPU ids provided. Set GPU_IDS, for example: GPU_IDS=\"0\" or GPU_IDS=\"0,1\"."
    exit 1
fi

README_METHOD_VALUE="${README_METHOD:-$METHOD_NAME}"
SKIP_README_UPDATE="${SKIP_README_UPDATE:-1}"
README_PATH_VALUE="${README_PATH:-$ROOT_DIR/readme.md}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/test_logs/${METHOD_NAME,,}}"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

if [[ ! -f "$TEST_PY" ]]; then
    echo "test.py not found: $TEST_PY"
    exit 1
fi

if [[ ! -d "$METHOD_DIR" ]]; then
    echo "Method checkpoint directory not found: $METHOD_DIR"
    exit 1
fi

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

parse_metric_triplet() {
    local ckpt_name="$1"
    "$PYTHON_BIN" - "$ckpt_name" <<'PY'
import re
import sys

name = sys.argv[1]
match = re.search(r'_m([0-9]+(?:\.[0-9]+)?)_ema([0-9]+(?:\.[0-9]+)?)\.pth$', name)
if match is None:
    raise SystemExit(1)
student = float(match.group(1))
teacher = float(match.group(2))
best_role = 'teacher' if teacher > student else 'student'
best_score = teacher if teacher > student else student
print(f"{student}|{teacher}|{best_role}|{best_score}")
PY
}

discover_jobs() {
    local exp_dir
    local setting_dir
    local ckpt_path
    local ckpt_name
    local parsed
    local parsed_student
    local parsed_teacher
    local parsed_role
    local parsed_score
    local best_ckpt
    local best_role
    local best_score
    local current_score
    local found_any=0

    while IFS= read -r exp_dir; do
        found_any=1
        while IFS= read -r setting_dir; do
            best_ckpt=""
            best_role=""
            best_score="-1"

            while IFS= read -r ckpt_path; do
                ckpt_name="$(basename "$ckpt_path")"
                if ! parsed="$(parse_metric_triplet "$ckpt_name")"; then
                    echo "Skip invalid checkpoint name: $ckpt_path" >&2
                    continue
                fi

                IFS='|' read -r parsed_student parsed_teacher parsed_role parsed_score <<< "$parsed"
                current_score="$parsed_score"

                if [[ -z "$best_ckpt" ]] || python - "$current_score" "$best_score" <<'PY'
import sys
current = float(sys.argv[1])
best = float(sys.argv[2])
raise SystemExit(0 if current > best else 1)
PY
                then
                    best_ckpt="$ckpt_path"
                    best_role="$parsed_role"
                    best_score="$parsed_score"
                fi
            done < <(find "$setting_dir" -maxdepth 1 -type f -name 'ep*_m*_ema*.pth' | sort -V)

            if [[ -n "$best_ckpt" ]]; then
                printf '%s|%s|%s\n' "$best_ckpt" "$best_role" "$best_score"
            fi
        done < <(find "$exp_dir" -mindepth 1 -maxdepth 1 -type d | sort -V)
    done < <(find "$METHOD_DIR" -mindepth 1 -maxdepth 1 -type d -name "*_${DATASET}_seed*_label*" | sort -V)

    if [[ "$found_any" -eq 0 ]]; then
        return 1
    fi
}

run_inference() {
    local gpu_id="$1"
    local ckpt_path="$2"
    local best_role="$3"
    local best_score="$4"
    local exp_dir
    local setting_dir
    local exp_name
    local setting_name
    local log_file
    local use_ema_flag="False"

    exp_dir="$(basename "$(dirname "$(dirname "$ckpt_path")")")"
    setting_dir="$(basename "$(dirname "$ckpt_path")")"
    exp_name="$(trim "$exp_dir")"
    setting_name="$(trim "$setting_dir")"
    log_file="$LOG_DIR/${exp_name}__${setting_name}__${best_role}.log"

    if [[ "$best_role" == "teacher" ]]; then
        use_ema_flag="True"
    fi

    echo "[$(date '+%F %T')] START gpu=${gpu_id} role=${best_role} score=${best_score} ckpt=${ckpt_path}"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" - "$TEST_PY" "$BASE_DIR" "$ckpt_path" "$README_METHOD_VALUE" "$README_PATH_VALUE" "$use_ema_flag" "$SKIP_README_UPDATE" >"$log_file" 2>&1 <<'PY'
import importlib.util
import os
import sys

test_py, base_dir, save_model_path, readme_method, readme_path, use_ema_text, skip_readme_text = sys.argv[1:]
use_ema = use_ema_text == 'True'
skip_readme_update = skip_readme_text == '1'

spec = importlib.util.spec_from_file_location('diffunder_test_module', test_py)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

args = module.get_parser().parse_args([
    '--dataset', 'ACDC',
    '--base_dir', base_dir,
    '--device', 'cuda:0',
    '--save_model_path', save_model_path,
    '--readme_method', readme_method,
    '--readme_path', readme_path,
] + (['--skip_readme_update'] if skip_readme_update else []))

module.apply_checkpoint_metadata(args)
args.use_ema = use_ema

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
module.torch.manual_seed(args.seed)
module.torch.cuda.manual_seed(args.seed)
module.torch.cuda.manual_seed_all(args.seed)
module.np.random.seed(args.seed)

cfg = module.build_fixed_cfg(args)
avg_metrics, mean_metrics = module.test_acdc(args, cfg)

if not args.skip_readme_update:
    resolved_method = module.infer_readme_method(args.save_model_path, args.readme_method)
    metadata = module.parse_checkpoint_metadata(args.save_model_path)
    module.update_readme_results(
        readme_path=args.readme_path,
        dataset=metadata['dataset'] or args.dataset,
        seed=metadata['seed'] or args.seed,
        labelnum=metadata['labelnum'] or args.labelnum,
        method_name=resolved_method,
        avg_metrics=avg_metrics,
        mean_metrics=mean_metrics,
    )
    print(f'Updated {args.readme_path} -> {resolved_method}')
PY
    echo "[$(date '+%F %T')] DONE  gpu=${gpu_id} role=${best_role} score=${best_score} log=${log_file}"
}

mapfile -t JOBS < <(discover_jobs)
if [[ "${#JOBS[@]}" -eq 0 ]]; then
    echo "No checkpoints found under: $METHOD_DIR"
    exit 1
fi

echo "Discovered ${#JOBS[@]} jobs under $METHOD_DIR"

worker() {
    local worker_idx="$1"
    local gpu_id="$2"
    local total_jobs="${#JOBS[@]}"
    local total_workers="${#GPU_ID_LIST[@]}"
    local job_spec
    local ckpt_path
    local best_role
    local best_score

    for ((idx=worker_idx; idx<total_jobs; idx+=total_workers)); do
        job_spec="${JOBS[$idx]}"
        IFS='|' read -r ckpt_path best_role best_score <<< "$job_spec"
        run_inference "$gpu_id" "$ckpt_path" "$best_role" "$best_score"
    done
}

for worker_idx in "${!GPU_ID_LIST[@]}"; do
    worker "$worker_idx" "${GPU_ID_LIST[$worker_idx]}" &
done

wait
echo "All ${METHOD_NAME} inference jobs completed."
