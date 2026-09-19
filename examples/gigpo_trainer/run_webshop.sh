set -x
set -o pipefail
ENGINE=vllm
SEED=${1:-0}
ulimit -u 65536
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export RAY_DEBUG_POST_MORTEM=1
export RAY_TMPDIR=$PWD/ray_tmp
mkdir -p "$RAY_TMPDIR"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export MASTER_PORT=29511

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16
val_data_size=128
val_batch_size=$val_data_size
group_size=8

MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
PROJECT_NAME=${PROJECT_NAME:-ICLR}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gigpo-webshop}
RUN_DIR=${RUN_DIR:-runs/$PROJECT_NAME/$EXPERIMENT_NAME}
DEBUG_DIR=${DEBUG_DIR:-$RUN_DIR/debug_batches}
mkdir -p "$RUN_DIR/logs/$run_timestamp"

# mode="mean_norm" # "mean_norm" or "mean_std_norm"
run_timestamp=$(date -u +'%Y%m%dT%H%M%S.%N')-$$
rollout_data_dir=${ROLLOUT_DATA_DIR:-"$RUN_DIR/rollouts"}/$run_timestamp
validation_data_dir=${VALIDATION_DATA_DIR:-"$RUN_DIR/validation"}/$run_timestamp

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --local_dir "agent_system/environments/env_package/webshop/train_data/" \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=agent_system/environments/env_package/webshop/train_data/text/train.parquet \
    data.val_files=agent_system/environments/env_package/webshop/train_data/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=6144 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='middle' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode="mean_norm" \
    env.env_name=Webshop \
    env.webshop.use_small=False \
    env.webshop.human_goals=True \
    env.seed=$SEED \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    ray_init.num_cpus=${RAY_NUM_CPUS:-48} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir="${RUN_DIR}/checkpoints" \
    trainer.rollout_data_dir="$rollout_data_dir" \
    trainer.validation_data_dir="$validation_data_dir" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=75 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    trainer.resume_mode=disable \
    hydra.output_subdir=null \
    trainer.val_before_train=True \
    trainer.val_only=False \
    2>&1 | tee "$RUN_DIR/logs/$run_timestamp/train.log"
