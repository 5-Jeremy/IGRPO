#!/bin/bash
#SBATCH --job-name=iclr-eval7b
#SBATCH --account=prj-02-llm-reasoning-shakkottai
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/project/prj-02-llm-reasoning-shakkottai/debajoy/IGRPO/runs/ICLR-eval/slurm-%j.out

# Evaluate the step-200 7B checkpoints on the whole Search-R1 test split: all
# 51,713 questions across the single-hop sets (nq, triviaqa, popqa) and the
# multi-hop ones (hotpotqa, 2wikimultihopqa, musique, bamboogle).
#
# Same idea as run_eval_3b.sbatch, but with the 7B GPU layout: the actor wants
# all 8 cards, so the retriever cannot have one to itself. faiss shards the
# index (fp16) across every GPU instead, exactly as in run_baseline_7b.sbatch.
# The 7B checkpoints are FSDP shards written as model_world_size_8_rank_*.pt and
# only reload on a world size of 8, so this is forced, not a preference.
#
# Ordering matters: faiss must claim its slice before vLLM sizes its KV cache.
# The health check gate below enforces that.
#
# Submit with:
#   sbatch steps/run_eval_7b.sbatch
# Override the runs, the step, or the test set with:
#   sbatch --export=ALL,RUNS="igrpo-allval-7B",STEP=150 steps/run_eval_7b.sbatch
#   sbatch --export=ALL,EVAL_DATA=$DATA/searchR1_processed_direct/eval_multihop_full.parquet steps/run_eval_7b.sbatch

set -eo pipefail

REPO=/scratch/user/debajoym98_tamu.edu/IGRPO
DATA=/scratch/project/prj-02-llm-reasoning-shakkottai/debajoy/IGRPO
CONDA_BASE=/sw/eb/sw/Miniforge3/25.3.0-3

RUNS=${RUNS:-"igrpo-allval-7B igpo-allval-7B gigpo-allval-7B"}
STEP=${STEP:-200}
TRAIN_PROJECT=${TRAIN_PROJECT:-ICLR}
export PROJECT_NAME=${PROJECT_NAME:-ICLR-eval}
export DATA

# The whole test split, single-hop and multi-hop together.
export EVAL_DATA=${EVAL_DATA:-$DATA/searchR1_processed_direct/test.parquet}

# 7B weights, not 3B. Only the config and tokenizer come from here; the
# checkpoint overwrites the parameters.
export MODEL_PATH=$DATA/Base_models/Qwen2.5-7B-Instruct

# Match the memory envelope the 7B training runs used on these same 8 GPUs.
export N_GPUS=8
export GPU_MEM_UTIL=0.55
export PPO_MICRO_BSZ=8
export LOGPROB_MICRO_BSZ=16

OUT_DIR=$DATA/runs/$PROJECT_NAME
mkdir -p "$OUT_DIR"

# Distinct port per job so two runs landing on one node cannot collide.
export SEARCH_PORT=$((20000 + SLURM_JOB_ID % 20000))

# Ray defaults to every CPU on the host (224 here) and then hangs forever
# because the Slurm cgroup only permits --cpus-per-task. Leave a slice for the
# retriever's uvicorn workers and faiss.
export RAY_NUM_CPUS=$(( ${SLURM_CPUS_PER_TASK:-96} - 16 ))
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
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Fail before the retriever load rather than an hour into the job.
for run in $RUNS; do
    ckpt="$DATA/runs/$TRAIN_PROJECT/$run/checkpoints/global_step_${STEP}"
    if [ ! -d "$ckpt/actor" ]; then
        echo "!!! missing checkpoint: $ckpt/actor"
        exit 1
    fi
    shards=$(ls "$ckpt/actor"/model_world_size_"${N_GPUS}"_rank_*.pt 2>/dev/null | wc -l)
    if [ "$shards" -ne "$N_GPUS" ]; then
        echo "!!! $ckpt/actor has $shards shards for world size $N_GPUS, expected $N_GPUS"
        exit 1
    fi
    echo "  found $ckpt ($shards shards)"
done

source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# retrieval server, index sharded across all 8 GPUs
# ---------------------------------------------------------------------------
cd "$REPO"
conda activate retriever
RETRIEVER_LOG="$OUT_DIR/retrieval_server_${SLURM_JOB_ID}.log"
python examples/search/retriever/retrieval_server.py \
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

# Record what faiss actually took, so a later OOM can be attributed.
echo "=== GPU memory after retriever load ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# ---------------------------------------------------------------------------
# score each checkpoint on all 8 GPUs, alongside the faiss shards
# ---------------------------------------------------------------------------
conda deactivate
conda activate igrpo
cd "$REPO"      # required: igpo/gigpo/igrpo/treehca resolve as namespace pkgs from cwd

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
port=29700
failed=""

for run in $RUNS; do
    export CKPT_PATH="$DATA/runs/$TRAIN_PROJECT/$run/checkpoints/global_step_${STEP}"
    export EXPERIMENT_NAME="${run}-eval${STEP}"
    if [ -n "${TRAJECTORY_AUDIT_ROOT:-}" ]; then
        export VALIDATION_DATA_DIR="$TRAJECTORY_AUDIT_ROOT/$EXPERIMENT_NAME"
    fi
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
