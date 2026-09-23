set -x
set -o pipefail
ENGINE=vllm
SEED=${1:-0}
# Small-catalog runs use the original seeded first-500 validation pool. Set this
# to fixed_human only with env.webshop.use_small=False and the full catalog.
WEBSHOP_VALIDATION_MODE=${WEBSHOP_VALIDATION_MODE:-legacy}
ulimit -u 65536
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export RAY_DEBUG_POST_MORTEM=1
export RAY_TMPDIR=$PWD/ray_tmp
mkdir -p "$RAY_TMPDIR"

export CUDA_VISIBLE_DEVICES="0,1"
export MASTER_PORT=29511

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16
val_data_size=128
val_batch_size=$val_data_size
group_size=8

MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
val_only=False
if [ "$#" -ge 2 ]; then
    MODEL_PATH=$2
    val_only=True
    val_data_size=500
    val_batch_size=50
fi

PROJECT_NAME=${PROJECT_NAME:-ICLR}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-treehca-webshop}
RUN_DIR=${RUN_DIR:-runs/$PROJECT_NAME/$EXPERIMENT_NAME}
run_timestamp=$(date -u +'%Y%m%dT%H%M%S.%N')-$$
DEBUG_DIR=${DEBUG_DIR:-$RUN_DIR/$run_timestamp/debug_batches}

mkdir -p "$RUN_DIR/$run_timestamp/logs"
mkdir -p "${RUN_DIR}/$run_timestamp/checkpoints"

rollout_data_dir=${ROLLOUT_DATA_DIR:-"$RUN_DIR/$run_timestamp/rollouts"}
validation_data_dir=${VALIDATION_DATA_DIR:-"$RUN_DIR/$run_timestamp/validation"}

# info_val: (p_gt + info_gain) / 2, the original score. log_ratio: log p_gt(child) -
# log p_gt(parent), which the softmax can actually separate -- on the saved debug
# batches info_val prunes at rank 0.49 (random) and log_ratio at 0.37.
EXPAND_SCORE=${EXPAND_SCORE:-log_ratio}
# snis: SNIS backup over children. q_hindsight: 70% GRPO advantage + 30% of
# Q * (1 - 1/h), h = p_gt(node) / p_gt(parent).
TREEHCA_CREDIT=${TREEHCA_CREDIT:-q_hindsight}

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --local_dir "agent_system/environments/env_package/webshop/train_data/" \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# TreeHCA reuses the IGRPO branching scheme (algorithm.igrpo.*) and replaces only
# the credit assignment (algorithm.treehca.*). reward_mode must stay avg/max so the
# batch keeps one row per tree node instead of unrolled root-to-leaf chains.
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=treehca \
    algorithm.igrpo.prob_diff_mode=True \
    algorithm.igrpo.gamma=1.0 \
    algorithm.igrpo.expand_mode='full' \
    algorithm.igrpo.max_traj_to_expand_per_node=2 \
    algorithm.igrpo.reduce_expand_num_per_steps_num=-1 \
    algorithm.igrpo.reward_mode='max' \
    algorithm.igrpo.expand_score=$EXPAND_SCORE \
    algorithm.treehca.credit=$TREEHCA_CREDIT \
    algorithm.treehca.max_inv_ratio=2.0 \
    algorithm.treehca.q_weight=1.0 \
    algorithm.treehca.grpo_weight=1.0 \
    algorithm.treehca.aux_mode=td \
    algorithm.treehca.prob_floor=1e-6 \
    algorithm.treehca.weight_temp=1.0 \
    algorithm.treehca.max_weight_ratio=-1.0 \
    algorithm.treehca.subtree_size_weight=True \
    algorithm.treehca.leaf_baseline='group' \
    algorithm.treehca.norm_adv_by_std=True \
    algorithm.treehca.no_progress_advantage_cap=False \
    algorithm.treehca.webshop_probe_batch_size=128 \
    actor_rollout_ref.rollout.info_gain_compute_log_prob_micro_batch_size_per_gpu=32 \
    reward_model.reward_manager='tree_structure' \
    data.train_files=agent_system/environments/env_package/webshop/train_data/text/train.parquet \
    data.val_files=agent_system/environments/env_package/webshop/train_data/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=6144 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='middle' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
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
    env.env_name=Webshop \
    env.webshop.use_small=True \
    env.webshop.human_goals=False \
    env.webshop.validation.mode=$WEBSHOP_VALIDATION_MODE \
    env.seed=$SEED \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    ray_init.num_cpus=${RAY_NUM_CPUS:-48} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir="${RUN_DIR}/$run_timestamp/checkpoints" \
    trainer.rollout_data_dir="$rollout_data_dir" \
    trainer.validation_data_dir="$validation_data_dir" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=75 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    trainer.resume_mode=disable \
    hydra.output_subdir=null \
    trainer.val_before_train=True \
    trainer.val_only=$val_only \
    2>&1 | tee "$RUN_DIR/$run_timestamp/logs/train.log"
