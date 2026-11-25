set -x

ENGINE=vllm
if [[ $# -gt 0 && $1 != *=* && $1 != -* ]]; then
    ENGINE=$1
    shift
fi

rollout_overrides=()
if [[ $ENGINE == vllm ]]; then
    rollout_overrides=(
        '+actor_rollout_ref.rollout.stop=["</search>","</answer>"]'
        '+actor_rollout_ref.rollout.detokenize=True'
        '+actor_rollout_ref.rollout.include_stop_str_in_output=True'
    )
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
log_dir="${LOG_DIR:-$repo_root/logs/search/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
rollout_log_dir="${ROLLOUT_LOG_DIR:-$log_dir/rollouts}"
checkpoint_dir="$(dirname -- "$rollout_log_dir")"

train_data_size=128
val_data_size=1024
group_size=5
log_prob_micro_batch_size_per_gpu=32

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3}"
export MASTER_PORT=29510

TRAIN_DATA="data/searchR1_processed_direct/train.parquet"
VAL_DATA="data/searchR1_processed_direct/val_subset.parquet"

MODEL_PATH="HuggingFaceTB/SmolLM3-3B"
PROJECT_NAME="search_adv_shaping"
EXPERIMENT_NAME="SmolLM3-3B-igpo"
DEBUG_DIR="tmp/debug_batches"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=igpo \
    algorithm.gamma=1.0 \
    algorithm.igpo.prob_diff_mode=True \
    algorithm.igpo.use_think=False \
    algorithm.unfinished_thinking_penalty=-0.1 \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    "trainer.rollout_data_dir='$rollout_log_dir'" \
    "trainer.default_local_dir='$checkpoint_dir'" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.debug_freq=-1 \
    trainer.debug_dir=$DEBUG_DIR \
    trainer.total_epochs=1 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.resume_mode=auto \
    trainer.val_before_train=False \
    hydra.run.dir='./output/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
    hydra.output_subdir=null \
    "${rollout_overrides[@]}" \
    "$@"
