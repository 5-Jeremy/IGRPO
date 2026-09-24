#!/bin/bash
#SBATCH --job-name=trajectory-audit7b
#SBATCH --account=prj-02-llm-reasoning-kalathil
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/sushil22_tamu.edu/projects/IGRPO/data/runs/ICLR/slurm-%j.out

# Matched diagnostic evaluation: 509 questions, five step-200 checkpoints.
# The existing evaluator starts one retriever and loads checkpoints sequentially.
set -euo pipefail
REPO=/scratch/user/sushil22_tamu.edu/projects/IGRPO
DATA=$REPO/data
cd "$REPO"
export EVAL_DATA=${EVAL_DATA:-$DATA/analysis/trajectory_audit_20260921/eval_509.parquet}
export EVAL_BATCH=128
export RUNS="gigpo-7B igrpo-7B treehca-7B treehca-qh-infoval-7B treehca-qh-logratio-7B"
export PROJECT_NAME=trajectory-audit-20260921
export STEP=200
export TRAJECTORY_AUDIT_ROOT=${TRAJECTORY_AUDIT_ROOT:-$DATA/analysis/trajectory_audit_20260921/evaluation_${SLURM_JOB_ID}}
test -f "$EVAL_DATA"
mkdir -p "$TRAJECTORY_AUDIT_ROOT"
bash steps/run_eval_7b.sbatch
