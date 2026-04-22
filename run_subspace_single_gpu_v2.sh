#!/bin/bash
#SBATCH --job-name=subspace_single_gpu_v2
#SBATCH --partition=crtai2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=component_logs/subspace_single_gpu_v2_out_%j.log
#SBATCH --error=component_logs/subspace_single_gpu_v2_err_%j.log

set -euo pipefail

echo "=============================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================="

# ===== activate environment =====
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate tu_name

export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:${LD_PRELOAD:-}"

# ===== CUDA / PyTorch =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
export TOKENIZERS_PARALLELISM="false"

export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# ===== Paths =====
PROJECT_ROOT="${PROJECT_ROOT:-/storage/crtai-fast/crtai5/ttran/nmt/Activation_steering}"
FOLDER_PATH="${FOLDER_PATH:-$PROJECT_ROOT/test_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output_subspace_v2}"
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/run_subspace_component_analysis_single_gpu _v2.py}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"

mkdir -p "$OUTPUT_DIR"
mkdir -p component_logs

# ===== Experiment settings =====
PROMPT_ORIG="${PROMPT_ORIG:-multipivot}"
PROMPT_NEW="${PROMPT_NEW:-direct}"

# Let the Python script auto-scale these for >=40GB GPUs.
# In _v2, batch_size=1 and identification_batch_size=1 on a 48GB GPU
# are automatically promoted to 16 and 32 respectively.
BATCH_SIZE="${BATCH_SIZE:-1}"
IDENTIFICATION_BATCH_SIZE="${IDENTIFICATION_BATCH_SIZE:-1}"

RANK="${RANK:-1}"
TOP_K="${TOP_K:-20}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-auto}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
FREEZE_MLPS="${FREEZE_MLPS:-false}"
HAVE_INTERNAL_INTERACTIONS="${HAVE_INTERNAL_INTERACTIONS:-false}"

echo "Input dir:     $FOLDER_PATH"
echo "Output dir:    $OUTPUT_DIR"
echo "Script:        $SCRIPT_PATH"
echo "Model:         $MODEL_NAME"
echo "Prompt orig:   $PROMPT_ORIG"
echo "Prompt new:    $PROMPT_NEW"
echo "Batch size:    $BATCH_SIZE"
echo "ID batch:      $IDENTIFICATION_BATCH_SIZE"
echo "Rank:          $RANK"
echo "Top-k:         $TOP_K"
echo "Device:        $DEVICE"
echo "Dtype:         $DTYPE"
echo "Max examples:  ${MAX_EXAMPLES:-all}"
echo "Freeze MLPs:   $FREEZE_MLPS"
echo "Interactions:  $HAVE_INTERNAL_INTERACTIONS"

CMD=(
  python "$SCRIPT_PATH" compare_prompt_styles_with_subspace
  --folder_path="$FOLDER_PATH"
  --model_name="$MODEL_NAME"
  --prompt_orig="$PROMPT_ORIG"
  --prompt_new="$PROMPT_NEW"
  --batch_size="$BATCH_SIZE"
  --identification_batch_size="$IDENTIFICATION_BATCH_SIZE"
  --rank="$RANK"
  --top_k="$TOP_K"
  --output_dir="$OUTPUT_DIR"
  --device="$DEVICE"
  --dtype="$DTYPE"
  --freeze_mlps="$FREEZE_MLPS"
  --have_internal_interactions="$HAVE_INTERNAL_INTERACTIONS"
)

if [[ -n "$MAX_EXAMPLES" ]]; then
  CMD+=(--max_examples="$MAX_EXAMPLES")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"

echo "=============================="
echo "End time: $(date)"
echo "=============================="
