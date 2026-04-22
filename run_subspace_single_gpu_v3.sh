#!/usr/bin/env bash

set -euo pipefail

echo "=============================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================="

# Default configuration. You can edit these values directly in this file.
DEFAULT_CONDA_ENV_NAME="tu_name"
DEFAULT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_FOLDER_PATH="${DEFAULT_PROJECT_ROOT}/test_data"
DEFAULT_OUTPUT_DIR="${DEFAULT_PROJECT_ROOT}/output_subspace_v3"
DEFAULT_SCRIPT_PATH="${DEFAULT_PROJECT_ROOT}/run_subspace_single_gpu_v3"
DEFAULT_MODEL_NAME="/path/to/Meta-Llama-3-8B-Instruct"

DEFAULT_PROMPT_ORIG="multipivot"
DEFAULT_PROMPT_NEW="direct"
DEFAULT_MAX_EXAMPLES=""
DEFAULT_BATCH_SIZE="1"
DEFAULT_IDENTIFICATION_BATCH_SIZE="1"
DEFAULT_RANK="1"
DEFAULT_TOP_K="20"
DEFAULT_DEVICE="cuda:0"
DEFAULT_DTYPE="auto"
DEFAULT_FREEZE_MLPS="false"
DEFAULT_HAVE_INTERNAL_INTERACTIONS="false"
DEFAULT_PER_SOURCE_FILE="true"
DEFAULT_SKIP_PER_SOURCE_FILE="false"
DEFAULT_CUDA_VISIBLE_DEVICES="0"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-$DEFAULT_CONDA_ENV_NAME}"
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
FOLDER_PATH="${FOLDER_PATH:-$DEFAULT_FOLDER_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
SCRIPT_PATH="${SCRIPT_PATH:-$DEFAULT_SCRIPT_PATH}"
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"

PROMPT_ORIG="${PROMPT_ORIG:-$DEFAULT_PROMPT_ORIG}"
PROMPT_NEW="${PROMPT_NEW:-$DEFAULT_PROMPT_NEW}"
MAX_EXAMPLES="${MAX_EXAMPLES:-$DEFAULT_MAX_EXAMPLES}"
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
IDENTIFICATION_BATCH_SIZE="${IDENTIFICATION_BATCH_SIZE:-$DEFAULT_IDENTIFICATION_BATCH_SIZE}"
RANK="${RANK:-$DEFAULT_RANK}"
TOP_K="${TOP_K:-$DEFAULT_TOP_K}"
DEVICE="${DEVICE:-$DEFAULT_DEVICE}"
DTYPE="${DTYPE:-$DEFAULT_DTYPE}"
FREEZE_MLPS="${FREEZE_MLPS:-$DEFAULT_FREEZE_MLPS}"
HAVE_INTERNAL_INTERACTIONS="${HAVE_INTERNAL_INTERACTIONS:-$DEFAULT_HAVE_INTERNAL_INTERACTIONS}"
PER_SOURCE_FILE="${PER_SOURCE_FILE:-$DEFAULT_PER_SOURCE_FILE}"
SKIP_PER_SOURCE_FILE="${SKIP_PER_SOURCE_FILE:-$DEFAULT_SKIP_PER_SOURCE_FILE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Optional environment activation.
if [[ -n "${CONDA_ENV_NAME}" ]]; then
  source "${HOME}/.bashrc"
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

mkdir -p "${OUTPUT_DIR}"

echo "Project root:   ${PROJECT_ROOT}"
echo "Input dir:      ${FOLDER_PATH}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Script path:    ${SCRIPT_PATH}"
echo "Model:          ${MODEL_NAME}"
echo "Prompt orig:    ${PROMPT_ORIG}"
echo "Prompt new:     ${PROMPT_NEW}"
echo "Batch size:     ${BATCH_SIZE}"
echo "ID batch:       ${IDENTIFICATION_BATCH_SIZE}"
echo "Rank:           ${RANK}"
echo "Top-k:          ${TOP_K}"
echo "Device:         ${DEVICE}"
echo "Dtype:          ${DTYPE}"
echo "Max examples:   ${MAX_EXAMPLES:-all}"
echo "Freeze MLPs:    ${FREEZE_MLPS}"
echo "Interactions:   ${HAVE_INTERNAL_INTERACTIONS}"
echo "Per source:     ${PER_SOURCE_FILE}"
echo "Skip per-file:  ${SKIP_PER_SOURCE_FILE}"

cd "${PROJECT_ROOT}"

CMD=(
  python "${SCRIPT_PATH}" compare_prompt_styles_with_subspace
  --folder_path="${FOLDER_PATH}"
  --model_name="${MODEL_NAME}"
  --prompt_orig="${PROMPT_ORIG}"
  --prompt_new="${PROMPT_NEW}"
  --batch_size="${BATCH_SIZE}"
  --identification_batch_size="${IDENTIFICATION_BATCH_SIZE}"
  --rank="${RANK}"
  --top_k="${TOP_K}"
  --output_dir="${OUTPUT_DIR}"
  --device="${DEVICE}"
  --dtype="${DTYPE}"
  --freeze_mlps="${FREEZE_MLPS}"
  --have_internal_interactions="${HAVE_INTERNAL_INTERACTIONS}"
  --per_source_file="${PER_SOURCE_FILE}"
  --skip_per_source_file="${SKIP_PER_SOURCE_FILE}"
)

if [[ -n "${MAX_EXAMPLES}" ]]; then
  CMD+=(--max_examples="${MAX_EXAMPLES}")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"

echo "=============================="
echo "End time: $(date)"
echo "=============================="
