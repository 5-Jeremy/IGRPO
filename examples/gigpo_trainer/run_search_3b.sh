set -x

# GiGPO on search with Qwen2.5-3B-Instruct, configured as a matched baseline
# against examples/igrpo_trainer/run_search.sh and examples/igpo_trainer/run_search.sh:
# same model, data, group size, ppo batch sizes, GPU count and retriever port.
# Only the advantage estimator and its own hyperparameters differ.
#
# run_search.sh in this directory is the upstream 7B recipe and is left alone.

ENGINE=${1:-vllm}

train_data_size=256
val_data_size=1024
group_size=5

# GiGPO config
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"
enable_similarity=True # enable similarity-based GiGPO
similarity_thresh=0.9 # similarity threshold for GiGPO

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export MASTER_PORT=29512

DATA=${DATA:-/scratch/project/prj-02-llm-reasoning-shakkottai/debajoy/IGRPO}

TRAIN_DATA="$DATA/searchR1_processed_direct/train.parquet"
# 1024 HotpotQA dev examples, not the full 51,713-row test split; see
# examples/data_preprocess/make_val_subset.py
VAL_DATA="$DATA/searchR1_processed_direct/val_subset.parquet"

MODEL_PATH="$DATA/Base_models/Qwen2.5-3B-Instruct"
PROJECT_NAME=${PROJECT_NAME:-ICLR}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gigpo-3B}
SEARCH_PORT=${SEARCH_PORT:-8008}
# Checkpoints go to the project dir; /scratch/user is quota'd at 1 TB.
RUN_DIR=${RUN_DIR:-$DATA/runs/$PROJECT_NAME/$EXPERIMENT_NAME}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.gigpo.enable_similarity=$enable_similarity \
    algorithm.gigpo.similarity_thresh=$similarity_thresh \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url="http://127.0.0.1:${SEARCH_PORT}/retrieve" \
    ray_init.num_cpus=${RAY_NUM_CPUS:-32} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir="${RUN_DIR}/checkpoints" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    trainer.resume_mode=auto \
    trainer.val_before_train=False \
    hydra.run.dir='./output/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
    hydra.output_subdir=null \
    $@
