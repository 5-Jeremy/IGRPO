#!/bin/bash
#SBATCH --job-name=math-coder
#SBATCH --account=prj-02-llm-reasoning-kalathil
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/user/sushil22_tamu.edu/projects/IGRPO/data/runs/slurm-%j.out

# Submit: sbatch --export=ALL,METHOD=igrpo steps/run_math.sbatch
# The math environment runs Python locally and does not start a retriever.
set -eo pipefail

REPO=/scratch/user/sushil22_tamu.edu/projects/IGRPO
DATA=$REPO/data
CONDA_BASE=/scratch/user/sushil22_tamu.edu/miniconda3
export DATA
export PROJECT_NAME=${PROJECT_NAME:-MATH}
METHOD=${METHOD:-treehca}
case "$METHOD" in
    treehca|igrpo|igpo|gigpo) TRAIN_SCRIPT="examples/${METHOD}_trainer/run_math.sh" ;;
    *) echo "Unknown METHOD: $METHOD" >&2; exit 2 ;;
esac
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-$METHOD}
export MODEL_PATH=${MODEL_PATH:-$DATA/Base_models/Qwen2.5-Coder-3B-Instruct}
export RUN_DIR=${RUN_DIR:-$DATA/runs/$PROJECT_NAME/$EXPERIMENT_NAME}
export MASTER_PORT=${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-$(( ${SLURM_CPUS_PER_TASK:-96} - 8 ))}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONUNBUFFERED=1

test -f "$DATA/math/train.parquet"
test -f "$DATA/math/val.parquet"
test -f "$MODEL_PATH/config.json"
mkdir -p "$RUN_DIR"
ulimit -u 200000
ulimit -n 131072
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate igrpo
cd "$REPO"
bash "$TRAIN_SCRIPT" "$@" > "$RUN_DIR/train_${SLURM_JOB_ID}.log" 2>&1
