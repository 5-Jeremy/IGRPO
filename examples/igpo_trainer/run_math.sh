set -x

ENGINE=${1:-vllm}

train_data_size=256
val_data_size=${VAL_DATA_SIZE:-100}
group_size=5

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export MASTER_PORT=${MASTER_PORT:-29515}

DATA=${DATA:-/scratch/user/sushil22_tamu.edu/projects/IGRPO/data}

TRAIN_DATA=${TRAIN_DATA:-"$DATA/math/train.parquet"}
VAL_DATA=${VAL_DATA:-"$DATA/math/val.parquet"}

MODEL_PATH=${MODEL_PATH:-$DATA/Base_models/Qwen2.5-Coder-3B-Instruct}
PROJECT_NAME=${PROJECT_NAME:-MATH}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-igpo}
# Checkpoints go to the project dir; /scratch/user is quota'd at 1 TB.
RUN_DIR=${RUN_DIR:-$DATA/runs/$PROJECT_NAME/$EXPERIMENT_NAME}
DEBUG_DIR=${DEBUG_DIR:-$RUN_DIR/debug_batches}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=igpo \
    algorithm.gamma=1.0 \
    algorithm.igpo.prob_diff_mode=True \
    algorithm.igpo.use_think=False \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    env.env_name=math \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.python.timeout=${PYTHON_TIMEOUT:-10} \
    ray_init.num_cpus=${RAY_NUM_CPUS:-64} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir="${RUN_DIR}/checkpoints" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.debug_freq=10 \
    trainer.debug_dir=$DEBUG_DIR \
    trainer.total_epochs=6 \
    trainer.total_training_steps=100 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.resume_mode=auto \
    trainer.val_before_train=False \
    hydra.run.dir='./output/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
    hydra.output_subdir=null \
    $@
