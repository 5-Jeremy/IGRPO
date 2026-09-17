#!/bin/bash
#SBATCH --job-name=iclr7b
#SBATCH --account=prj-02-llm-reasoning-shakkottai
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/project/prj-02-llm-reasoning-shakkottai/debajoy/IGRPO/runs/ICLR/slurm-%j.out

# One 7B search baseline end to end on a single node, with the retriever and the
# trainer sharing all 8 GPUs.
#
# The 3B recipe gave the retriever a private GPU and trained on the other four.
# That does not scale: a 7B actor wants all 8 cards. Instead we let faiss shard
# the index across every GPU. retrieval_server.py already clones the index with
# useFloat16 + shard, so the 64 GB fp32 index becomes ~32 GB of fp16 split eight
# ways, about 4 GB a card, and the trainer keeps the rest.
#
# The ordering below matters: faiss must claim its slice before vLLM sizes its
# KV cache, otherwise vLLM takes gpu_memory_utilization of a still-empty card and
# faiss has nowhere to land. The health check gate enforces that.
#
# Submit with:
#   sbatch --export=ALL,TRAIN_SCRIPT=...,EXPERIMENT_NAME=... steps/run_baseline_7b.sbatch

set -eo pipefail

REPO=/scratch/user/debajoym98_tamu.edu/IGRPO
DATA=/scratch/project/prj-02-llm-reasoning-shakkottai/debajoy/IGRPO
CONDA_BASE=/sw/eb/sw/Miniforge3/25.3.0-3

: "${TRAIN_SCRIPT:?must be set, e.g. examples/igrpo_trainer/run_search_7b.sh}"
: "${EXPERIMENT_NAME:?must be set, e.g. igrpo-7B}"
export PROJECT_NAME=${PROJECT_NAME:-ICLR}
export DATA EXPERIMENT_NAME

RUN_DIR=$DATA/runs/$PROJECT_NAME/$EXPERIMENT_NAME
export RUN_DIR
mkdir -p "$RUN_DIR"

# Distinct port per job so two runs landing on one node cannot collide.
export SEARCH_PORT=$((20000 + SLURM_JOB_ID % 20000))

# Ray defaults to every CPU on the host (224 here) and then hangs forever
# because the Slurm cgroup only permits --cpus-per-task. verl's own config
# warns about exactly this; the training scripts pass it as ray_init.num_cpus.
# Leave a slice for the retriever's uvicorn workers and faiss.
export RAY_NUM_CPUS=$(( ${SLURM_CPUS_PER_TASK:-96} - 16 ))
export PYTHONUNBUFFERED=1

# Slurm hands out nproc=4096 / nofile=1024 by default, which Ray + vLLM + faiss
# exhaust immediately ("thread: Resource temporarily unavailable"). The hard
# limits are far higher, so raise the soft ones.
ulimit -u 200000
ulimit -n 131072
# Keep OpenMP from spawning a thread per core (224) in every worker.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
echo "limits: nproc=$(ulimit -u) nofile=$(ulimit -n) omp=$OMP_NUM_THREADS ray_cpus=$RAY_NUM_CPUS"

echo "=== $EXPERIMENT_NAME | host $(hostname) | job ${SLURM_JOB_ID} | $(date) ==="
echo "script=$TRAIN_SCRIPT  port=$SEARCH_PORT  run_dir=$RUN_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# retrieval server, index sharded across all 8 GPUs
# ---------------------------------------------------------------------------
cd "$REPO"
conda activate retriever
python examples/search/retriever/retrieval_server.py \
    --index_path "$DATA/searchR1/e5_Flat.index" \
    --corpus_path "$DATA/searchR1/wiki-18.jsonl" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model "$DATA/Base_models/e5-base-v2" \
    --faiss_gpu \
    --port "$SEARCH_PORT" \
    > "$RUN_DIR/retrieval_server_${SLURM_JOB_ID}.log" 2>&1 &
RETRIEVER_PID=$!

cleanup() {
    echo "=== stopping retrieval server (pid $RETRIEVER_PID) ==="
    kill "$RETRIEVER_PID" 2>/dev/null || true
    wait "$RETRIEVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Loading a 64 GB faiss index and sharding it onto the GPUs plus the 14 GB
# corpus takes a while.
RETRIEVER_LOG="$RUN_DIR/retrieval_server_${SLURM_JOB_ID}.log"
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
# training on all 8 GPUs, alongside the faiss shards
# ---------------------------------------------------------------------------
conda deactivate
conda activate igrpo
cd "$REPO"      # required: igpo/gigpo/igrpo/treehca resolve as namespace pkgs from cwd

echo "=== launching $TRAIN_SCRIPT ($(date)) ==="
bash "$TRAIN_SCRIPT"

echo "=== finished: $(date) ==="
