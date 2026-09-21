set -x

# Evaluate one saved checkpoint on the full multi-hop test sets.
#
# There is no standalone agent evaluator in this repo: verl/trainer/main_eval.py
# only rescores an existing generations parquet and never touches the retriever.
# The agent loop lives in the trainer, so evaluation is main_ppo with
# trainer.val_only=True, which loads the checkpoint, runs _validate() and returns
# before the first training step.
#
# _validate reports val/{data_source}/test_score, so one pass over a parquet
# holding several data_sources yields a per-dataset breakdown for free.
#
# The algorithm knobs are deliberately absent. Evaluation is identical for every
# method in this repo:
#   - rollout_loop gates tree branching on `and is_train`, so IGRPO and TreeHCA
#     roll out exactly like IGPO and GiGPO here
#   - main_ppo always builds the validation reward with EpisodeRewardManager,
#     whatever reward_model.reward_manager says
# so a single adv_estimator=grpo / reward_manager=episode config evaluates all of
# them without changing a number.
#
# Usage:
#   CKPT_PATH=.../global_step_200 EXPERIMENT_NAME=igrpo-3B-eval bash examples/eval/run_eval_search.sh

ENGINE=${ENGINE:-vllm}

DATA=${DATA:-/scratch/project/prj-02-llm-reasoning-shakkottai/debajoy/IGRPO}

: "${CKPT_PATH:?must be set, e.g. \$DATA/runs/ICLR/igrpo-3B/checkpoints/global_step_200}"
: "${EXPERIMENT_NAME:?must be set, e.g. igrpo-3B-eval200}"

# Base weights only supply the config and tokenizer; CKPT_PATH overwrites them.
MODEL_PATH=${MODEL_PATH:-$DATA/Base_models/Qwen2.5-3B-Instruct}

# hotpotqa + 2wikimultihopqa + bamboogle + musique, all 22,523 rows, built by
# examples/data_preprocess/make_val_subset.py --n 0
EVAL_DATA=${EVAL_DATA:-$DATA/searchR1_processed_direct/eval_multihop_full.parquet}
TRAIN_DATA="$DATA/searchR1_processed_direct/train.parquet"

PROJECT_NAME=${PROJECT_NAME:-ICLR-eval}
SEARCH_PORT=${SEARCH_PORT:-8008}

# Number of search environments held open, and the dataloader batch. The envs pad
# and mask a short final batch, so this need not divide the dataset.
EVAL_BATCH=${EVAL_BATCH:-1024}

# Checkpoints are FSDP shards named model_world_size_<N>_rank_*.pt, so the actor
# has to be rebuilt on exactly the world size that wrote them: 4 for the 3B runs,
# 8 for the 7B ones.
N_GPUS=${N_GPUS:-4}

# A 7B actor plus the sharded faiss index leaves vLLM less room than a 3B one.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
PPO_MICRO_BSZ=${PPO_MICRO_BSZ:-8}
LOGPROB_MICRO_BSZ=${LOGPROB_MICRO_BSZ:-32}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager='episode' \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.train_batch_size=256 \
    data.val_batch_size=$EVAL_BATCH \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BSZ \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BSZ \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=5 \
    env.history_length=4 \
    env.search.search_url="${SEARCH_URL:-http://127.0.0.1:${SEARCH_PORT}/retrieve}" \
    ray_init.num_cpus=${RAY_NUM_CPUS:-32} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=$CKPT_PATH \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.total_epochs=1 \
    hydra.run.dir='./output/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
    hydra.output_subdir=null
