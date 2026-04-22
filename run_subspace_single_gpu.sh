#!/bin/bash
#SBATCH --job-name=subspace_single_gpu_v2
#SBATCH --partition=crtai2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=component_logs/subspace_single_gpu_v2_out_%j.log
#SBATCH --error=component_logs/subspace_single_gpu_v2_err_%j.log

set -euo pipefail

echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================="

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate tu_name

export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:${LD_PRELOAD:-}"

# ===== CUDA device =====
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM="false"

PROJECT_ROOT="/storage/crtai-fast/crtai5/ttran/nmt/Activation_steering"
DATA_DIR="/storage/crtai-fast/crtai5/ttran/nmt/Activation_steering/test_data"
OUTPUT_DIR="/storage/crtai-fast/crtai5/ttran/nmt/Activation_steering/output_subspace_v2_overall"
MODEL_PATH="/storage/crtai-fast/crtai5/ttran/nmt/Meta-Llama-3-8B-Instruct"
SCRIPT_PATH="/storage/crtai-fast/crtai5/ttran/nmt/Activation_steering/run_subspace_component_analysis_single_gpu _v2.py"

mkdir -p "$OUTPUT_DIR"
mkdir -p "/storage/crtai-fast/crtai5/ttran/nmt/Activation_steering/component_logs"

cd "$PROJECT_ROOT"

PROMPT_ORIG="multipivot"
PROMPT_NEW="direct"
BATCH_SIZE=1
IDENTIFICATION_BATCH_SIZE=1
RANK=1
TOP_K=20
DTYPE="auto"
FREEZE_MLPS=false
HAVE_INTERNAL_INTERACTIONS=false
PER_SOURCE_FILE=false
SKIP_PER_SOURCE_FILE=true

echo "Input dir:         $DATA_DIR"
echo "Output dir:        $OUTPUT_DIR"
echo "Model path:        $MODEL_PATH"
echo "Script path:       $SCRIPT_PATH"
echo "Prompt orig:       $PROMPT_ORIG"
echo "Prompt new:        $PROMPT_NEW"
echo "Batch size:        $BATCH_SIZE"
echo "ID batch:          $IDENTIFICATION_BATCH_SIZE"
echo "Rank:              $RANK"
echo "Top-k:             $TOP_K"
echo "Device:            cuda:0"
echo "Dtype:             $DTYPE"
echo "Freeze MLPs:       $FREEZE_MLPS"
echo "Interactions:      $HAVE_INTERNAL_INTERACTIONS"
echo "Per source file:   $PER_SOURCE_FILE"
echo "Skip per-file:     $SKIP_PER_SOURCE_FILE"
echo "Mode:              overall data only"

python "$SCRIPT_PATH" compare_prompt_styles_with_subspace \
  --folder_path="$DATA_DIR" \
  --model_name="$MODEL_PATH" \
  --prompt_orig="$PROMPT_ORIG" \
  --prompt_new="$PROMPT_NEW" \
  --batch_size="$BATCH_SIZE" \
  --identification_batch_size="$IDENTIFICATION_BATCH_SIZE" \
  --rank="$RANK" \
  --top_k="$TOP_K" \
  --output_dir="$OUTPUT_DIR" \
  --device="cuda:0" \
  --dtype="$DTYPE" \
  --freeze_mlps="$FREEZE_MLPS" \
  --have_internal_interactions="$HAVE_INTERNAL_INTERACTIONS" \
  --per_source_file="$PER_SOURCE_FILE" \
  --skip_per_source_file="$SKIP_PER_SOURCE_FILE"

echo "=============================="
echo "End time: $(date)"
echo "=============================="
