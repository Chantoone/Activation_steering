#!/bin/bash
#SBATCH --job-name=subspace_single_gpu
#SBATCH --partition=crtai2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=component_logs/subspace_single_gpu_out_%j.log
#SBATCH --error=component_logs/subspace_single_gpu_err_%j.log

set -euo pipefail

echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================="

# ===== activate environment =====
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate tu_name

export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:${LD_PRELOAD:-}"

# ===== CUDA =====
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# ===== Paths =====
PROJECT_ROOT="/storage/crtai-fast/crtai5/ttran/nmt/Activation_steering"
FOLDER_PATH="$PROJECT_ROOT/test_data"
OUTPUT_DIR="$PROJECT_ROOT/output_100sample"
SCRIPT_PATH="$PROJECT_ROOT/run_subspace_component_analysis_single_gpu.py"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

mkdir -p "$OUTPUT_DIR"
mkdir -p component_logs

echo "Input dir:   $FOLDER_PATH"
echo "Output dir:  $OUTPUT_DIR"
echo "Model:       $MODEL_NAME"
echo "Method:      subspace_intervened_path_patching"
echo "Implementation: single_gpu_memory_optimized"
echo "Subspace:    generated span"

PROMPT_ORIG="multipivot"
PROMPT_NEW="direct"
BATCH_SIZE=4
IDENTIFICATION_BATCH_SIZE=4
RANK=1
TOP_K=20
DEVICE="cuda:0"
DTYPE="auto"

export TOKENIZERS_PARALLELISM="false"

echo "Prompt orig: $PROMPT_ORIG"
echo "Prompt new:  $PROMPT_NEW"
echo "Batch size:  $BATCH_SIZE"
echo "ID batch:    $IDENTIFICATION_BATCH_SIZE"
echo "Rank:        $RANK"
echo "Device:      $DEVICE"
echo "Dtype:       $DTYPE"

# ===== Run =====
python "$SCRIPT_PATH" compare_prompt_styles_with_subspace \
  --folder_path="$FOLDER_PATH" \
  --model_name="$MODEL_NAME" \
  --prompt_orig="$PROMPT_ORIG" \
  --prompt_new="$PROMPT_NEW" \
  --batch_size="$BATCH_SIZE" \
  --identification_batch_size="$IDENTIFICATION_BATCH_SIZE" \
  --rank="$RANK" \
  --top_k="$TOP_K" \
  --output_dir="$OUTPUT_DIR" \
  --device="$DEVICE" \
  --dtype="$DTYPE"

echo "=============================="
echo "End time: $(date)"
echo "=============================="
