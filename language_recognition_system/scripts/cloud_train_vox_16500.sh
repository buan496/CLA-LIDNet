#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
WORK_ROOT="${WORK_ROOT:-$HOME/cla_lidnet_runs}"
DATASET_ROOT="${DATASET_ROOT:-${WORK_ROOT}/dataset_vox_16500}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
RUN_NAME="${RUN_NAME:-run_vox_16500_b32_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"
REPORT_PATH="${RUN_DIR}/vox_16500_report.json"
LOG_PATH="${RUN_DIR}/train.log"

LANGUAGES="${LANGUAGES:-en ja zh}"
MAX_SHARDS_PER_LANGUAGE="${MAX_SHARDS_PER_LANGUAGE:-11}"
MAX_FILES_PER_LANGUAGE="${MAX_FILES_PER_LANGUAGE:-5500}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_REPEAT_FACTOR="${TRAIN_REPEAT_FACTOR:-3}"
MIN_EPOCHS="${MIN_EPOCHS:-15}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
NUM_WORKERS="${NUM_WORKERS:--1}"
DEVICE="${DEVICE:-cuda}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

mkdir -p "${WORK_ROOT}" "${CHECKPOINT_ROOT}" "${RUN_DIR}"

echo "[INFO] Checking environment"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/cloud_check_environment.py" \
  --workspace-root "${PROJECT_ROOT}"

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  echo "[INFO] Downloading VoxLingua107 subset into ${DATASET_ROOT}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/download_voxlingua_subset.py" \
    --output-root "${DATASET_ROOT}" \
    --languages ${LANGUAGES} \
    --max-shards-per-language "${MAX_SHARDS_PER_LANGUAGE}" \
    --max-files-per-language "${MAX_FILES_PER_LANGUAGE}" \
    --report-path "${REPORT_PATH}"
else
  echo "[INFO] Skipping dataset download because SKIP_DOWNLOAD=1"
fi

echo "[INFO] Checking training readiness"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/check_training_ready.py" \
  --dataset-root "${DATASET_ROOT}" \
  --device "${DEVICE}"

echo "[INFO] Starting training run ${RUN_NAME}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${RUN_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --train-repeat-factor "${TRAIN_REPEAT_FACTOR}" \
  --num-workers "${NUM_WORKERS}" \
  --min-epochs "${MIN_EPOCHS}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE}" | tee "${LOG_PATH}"

echo "[INFO] Training finished. Outputs:"
echo "  Run directory: ${RUN_DIR}"
echo "  Summary JSON: ${RUN_DIR}/train_summary.json"
echo "  Best model:   ${RUN_DIR}/best_model.pt"
echo "  Train log:    ${LOG_PATH}"
