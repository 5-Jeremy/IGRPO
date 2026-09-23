#!/bin/bash
#SBATCH --job-name=iclr-eval3b
#SBATCH --account=prj-02-llm-reasoning-shakkottai
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:5
#SBATCH --mem=320G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/project/prj-02-llm-reasoning-shakkottai/debajoy/IGRPO/runs/ICLR-eval/slurm-%j.out

# Evaluate the step-200 3B checkpoints on the whole Search-R1 test split: all
# 51,713 questions across both the single-hop sets (nq, triviaqa, popqa) and the
# multi-hop ones (hotpotqa, 2wikimultihopqa, musique, bamboogle). _validate keys
# its metrics on data_source, so one pass gives the per-dataset breakdown.
#
# Every checkpoint is scored in one job against a single retriever. Loading the
# 64 GB faiss index costs about half an hour, so paying that once instead of
# once per checkpoint is most of the reason this is a loop rather than N jobs.
#
# Budget about 3 h per checkpoint for the full split; the multi-hop-only subset
# (eval_multihop_full.parquet, 22,523 rows) took ~77 min.
#
# GPU layout follows the 3B training runs exactly: retriever on GPU 4, actor on
# GPUs 0-3. That is not a preference -- the checkpoints are FSDP shards written
# as model_world_size_4_rank_*.pt and only reload on a world size of 4.
#
# Submit with:
#   sbatch steps/run_eval_3b.sbatch
# Override the runs, the step, or the test set with:
#   sbatch --export=ALL,RUNS="igrpo-allval-3B",STEP=150 steps/run_eval_3b.sbatch
#   sbatch --export=ALL,EVAL_DATA=$DATA/searchR1_processed_direct/eval_multihop_full.parquet steps/run_eval_3b.sbatch

set -eo pipefail

REPO=/scratch/user/debajoym98_tamu.edu/IGRPO
DATA=/scratch/project/prj-02-llm-reasoning-shakkottai/debajoy/IGRPO
CONDA_BASE=/sw/eb/sw/Miniforge3/25.3.0-3

RUNS=${RUNS:-"igrpo-allval-3B igpo-allval-3B gigpo-allval-3B"}
STEP=${STEP:-200}
TRAIN_PROJECT=${TRAIN_PROJECT:-ICLR}
export PROJECT_NAME=${PROJECT_NAME:-ICLR-eval}
export DATA

# The whole test split, single-hop and multi-hop together.
export EVAL_DATA=${EVAL_DATA:-$DATA/searchR1_processed_direct/test.parquet}

OUT_DIR=$DATA/runs/$PROJECT_NAME
mkdir -p "$OUT_DIR"

# Distinct port per job so two runs landing on one node cannot collide.
export SEARCH_PORT=$((20000 + SLURM_JOB_ID % 20000))

# Ray defaults to every CPU on the host (224 here) and then hangs forever
# because the Slurm cgroup only permits --cpus-per-task.
export RAY_NUM_CPUS=${SLURM_CPUS_PER_TASK:-32}
export PYTHONUNBUFFERED=1

# Slurm hands out nproc=4096 / nofile=1024 by default, which Ray + vLLM + faiss
# exhaust immediately ("thread: Resource temporarily unavailable").
ulimit -u 200000
ulimit -n 131072
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

echo "=== eval step $STEP | host $(hostname) | job ${SLURM_JOB_ID} | $(date) ==="
echo "runs: $RUNS"
echo "eval_data=$EVAL_DATA"
echo "port=$SEARCH_PORT out=$OUT_DIR"

# Fail before the retriever load rather than an hour into the job.
for run in $RUNS; do
    ckpt="$DATA/runs/$TRAIN_PROJECT/$run/checkpoints/global_step_${STEP}"
    if [ ! -d "$ckpt/actor" ]; then
        echo "!!! missing checkpoint: $ckpt/actor"
        exit 1
    fi
    echo "  found $ckpt"
done

source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# retrieval server on the 5th allocated GPU
# ---------------------------------------------------------------------------
cd "$REPO"
conda activate retriever
RETRIEVER_LOG="$OUT_DIR/retrieval_server_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=4 python examples/search/retriever/retrieval_server.py \
    --index_path "$DATA/searchR1/e5_Flat.index" \
    --corpus_path "$DATA/searchR1/wiki-18.jsonl" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model "$DATA/Base_models/e5-base-v2" \
    --faiss_gpu \
    --port "$SEARCH_PORT" \
    > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!

cleanup() {
    echo "=== stopping retrieval server (pid $RETRIEVER_PID) ==="
    kill "$RETRIEVER_PID" 2>/dev/null || true
    wait "$RETRIEVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== waiting for retriever on port $SEARCH_PORT ==="
for i in $(seq 1 240); do
    if ! kill -0 "$RETRIEVER_PID" 2>/dev/null; then
        echo "!!! retrieval server died during startup, last 50 lines:"
        tail -50 "$RETRIEVER_LOG"
        exit 1
    fi
    if curl -sf -m 10 -X POST "http://127.0.0.1:${SEARCH_PORT}/retrieve" \
         -H 'Content-Type: application/json' \
         -d '{"query":"who wrote hamlet","topk":1,"return_scores":false}' > /dev/null; then
        echo "retriever ready after ~$((i * 15))s"
        break
    fi
    if [ "$i" -eq 240 ]; then
        echo "!!! retriever did not come up within 60 minutes"
        tail -50 "$RETRIEVER_LOG"
        exit 1
    fi
    sleep 15
done

# ---------------------------------------------------------------------------
# score each checkpoint on GPUs 0-3
# ---------------------------------------------------------------------------
conda deactivate
conda activate igrpo
cd "$REPO"      # required: igpo/gigpo/igrpo/treehca resolve as namespace pkgs from cwd

export CUDA_VISIBLE_DEVICES="0,1,2,3"
port=29600
failed=""

for run in $RUNS; do
    export CKPT_PATH="$DATA/runs/$TRAIN_PROJECT/$run/checkpoints/global_step_${STEP}"
    export EXPERIMENT_NAME="${run}-eval${STEP}"
    # A fresh port per run; a previous Ray head can leave the old one in TIME_WAIT.
    export MASTER_PORT=$((port++))

    echo ""
    echo "=================================================================="
    echo "=== $EXPERIMENT_NAME ($(date)) ==="
    echo "=================================================================="

    if bash examples/eval/run_eval_search.sh 2>&1 | tee "$OUT_DIR/${EXPERIMENT_NAME}_${SLURM_JOB_ID}.log"; then
        echo "=== $EXPERIMENT_NAME done ($(date)) ==="
    else
        echo "!!! $EXPERIMENT_NAME failed, continuing with the rest"
        failed="$failed $run"
    fi

    # Leave nothing behind for the next iteration to trip over.
    ray stop --force > /dev/null 2>&1 || true
    sleep 20
done

echo ""
echo "=== all evals finished: $(date) ==="
[ -n "$failed" ] && echo "!!! failed runs:$failed" && exit 1
echo "results: grep 'val/' $OUT_DIR/*_${SLURM_JOB_ID}.log"
exit 0
