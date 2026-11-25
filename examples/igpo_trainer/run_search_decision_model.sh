#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if [[ -f .env ]]; then
    set -a
    source ./.env
    set +a
fi

engine=vllm
if [[ $# -gt 0 && $1 != *=* && $1 != -* ]]; then
    engine=$1
    shift
fi

use_decision_model="${USE_DECISION_MODEL:-True}"
decision_model_ip="${DECISION_MODEL_IP:-127.0.0.1}"
retrieval_server_ip="${RETRIEVAL_SERVER_IP:-127.0.0.1}"
log_dir="${LOG_DIR:-$repo_root/logs/search/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
rollout_log_dir="${ROLLOUT_LOG_DIR:-$log_dir/rollouts}"
checkpoint_dir="$(dirname -- "$rollout_log_dir")"

exec bash examples/igpo_trainer/run_search.sh "$engine" \
    algorithm.reasoning_advantage.enabled="$use_decision_model" \
    env.search.search_url="http://${retrieval_server_ip}:8008/retrieve" \
    trainer.decision_model.enabled="$use_decision_model" \
    trainer.decision_model.url="http://${decision_model_ip}:8009/v1/models" \
    trainer.decision_model.selective_requests=false \
    trainer.decision_model.request_timeout=600.0 \
    trainer.decision_model.max_concurrent_requests=128 \
    "trainer.rollout_data_dir='$rollout_log_dir'" \
    trainer.experiment_name=igpo-3B-decision-model \
    "trainer.default_local_dir='$checkpoint_dir'" \
    "$@"
