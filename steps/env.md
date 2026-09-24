# IGRPO on the TAMU `vision` cluster

Everything needed to reproduce the setup, plus the seven things that had to change
to make the repo actually run here. All of it is verified: three baselines are
training and logging to the wandb project **ICLR**.

| | |
|---|---|
| Conda env | `igrpo` @ `/scratch/user/sushil22_tamu.edu/miniconda3/envs/igrpo` — Python 3.12, 12 GB |
| Retriever env | `retriever` @ `/scratch/user/sushil22_tamu.edu/miniconda3/envs/retriever` — Python 3.10, pre-existing |
| Repo | `/scratch/user/sushil22_tamu.edu/projects/IGRPO` |
| Data, models, checkpoints | `/scratch/user/sushil22_tamu.edu/projects/IGRPO/data` |
| Cluster | partition `def`, account `prj-02-llm-reasoning-kalathil`, 8×H200 per node |

---

## Quick start

```bash
cd /scratch/user/sushil22_tamu.edu/projects/IGRPO

sbatch --job-name=iclr-igrpo \
  --export=ALL,PROJECT_NAME=ICLR,EXPERIMENT_NAME=igrpo-3B,\
TRAIN_SCRIPT=examples/igrpo_trainer/run_search.sh steps/run_baseline.sbatch
```

Swap `TRAIN_SCRIPT`/`EXPERIMENT_NAME` for the other two:

| Baseline | Script |
|---|---|
| IGRPO | `examples/igrpo_trainer/run_search.sh` |
| IGPO | `examples/igpo_trainer/run_search.sh` |
| GiGPO | `examples/gigpo_trainer/run_search_3b.sh` |

`steps/run_baseline.sbatch` does the whole run: starts a private retrieval server on
the 5th GPU, waits for it to answer, then trains on GPUs 0-3. Logs and checkpoints
land in `$DATA/runs/$PROJECT_NAME/$EXPERIMENT_NAME/`.

`examples/gigpo_trainer/run_search.sh` is the **upstream 7B recipe, left untouched**.
`run_search_3b.sh` is the matched 3B baseline.

---

## The seven changes that were required

| # | Problem | Fix |
|---|---|---|
| 1 | README's `flash-attn==2.7.4.post1` has no torch 2.8 wheel → multi-hour source build | install the prebuilt **2.8.3** wheel by URL |
| 2 | Every online snippet says `cxx11abiFALSE`; torch 2.8 needs **TRUE** | derive the ABI from the installed torch |
| 3 | Ray grabs all 224 host CPUs, Slurm cgroup allows 32 → **silent hang forever** | `ray_init.num_cpus=$SLURM_CPUS_PER_TASK` |
| 4 | Slurm defaults `nproc=4096`/`nofile=1024` → **SIGABRT** in Ray | `ulimit -u 200000`, `ulimit -n 131072` |
| 5 | 256 concurrent queries into a non-thread-safe faiss GPU index → **assertion crash** | lock in `retrieval_server.py` |
| 6 | `/scratch/user` is quota'd at 1 TB and nearly full | data + checkpoints in the project dir |
| 7 | Validating on all 51,713 test rows cost as much as training | 1,024-example HotpotQA subset |

### 1-2. flash-attn

The README asks for `vllm==0.11.0` *and* `flash-attn==2.7.4.post1`. These contradict:
vllm 0.11.0 pins `torch==2.8.0`, and flash-attention 2.7.4.post1 published no torch 2.8
wheel, so pip silently falls back to compiling from source. 2.8.3 is the first release
with a `cu12torch2.8` wheel. Never hardcode the ABI — on this cluster it resolves to
`flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`, and the
`cxx11abiFALSE` wheel everyone copy-pastes installs fine then dies at import.

### 3. Ray CPU detection — the nastiest one

verl's own config warns about it:

```303:304:verl/trainer/config/ppo_trainer.yaml
ray_init:
  num_cpus: null # `None` means using all CPUs, which might cause hang if limited in systems like SLURM. Please set to a number allowed then.
```

Symptom is brutal to debug: Ray's GCS and dashboard start, **no worker is ever spawned**,
no error is printed, and the driver blocks indefinitely. All three search scripts now pass
`ray_init.num_cpus=${RAY_NUM_CPUS:-32}`.

### 4. Process and thread limits

Slurm hands out `nproc=4096` and `nofile=1024`; Ray + vLLM + faiss exhaust them and Ray
aborts with `Unhandled exception ... what(): thread: Resource temporarily unavailable`.
The hard limits are ~8.2M and 131072, so the job script just raises the soft ones. It also
sets `OMP_NUM_THREADS=8` to stop every worker spawning 224 OpenMP threads.

### 5. faiss GPU is not thread-safe

`agent_system/.../search/envs.py` fires up to 256 concurrent retrieval requests, and FastAPI
serves the sync `/retrieve` endpoint from a threadpool. Concurrent searches free faiss's
temp-memory stack out of order and kill the server with
`Faiss assertion 'p + size == head_' failed`. Training then runs on with every retrieval
failing. Fixed with a `threading.Lock` around `retriever.search` in `retrieval_server.py`.
**This is a repo bug, not a cluster quirk.**

### 6. Storage

`df -h /scratch` reports filesystem capacity, which may differ from your quota.
The wiki-18 assets need about 81 GB steady and 150 GB peak. Check your quota before
preparing data or starting runs; this setup keeps data and checkpoints under the
local repo's `data/` directory.

### 7. Validation cost

`data.val_batch_size` is **not** a subsample — `_validate()` iterates the whole dataloader.
Shipped config evaluated all 51,713 test rows every 50 steps: ~724k multi-turn episodes
against ~847k training episodes. Now 1,024 HotpotQA dev examples every 10 steps (~68k
episodes, <10% overhead, denser curve). `save_freq=50` is a multiple of `test_freq`, so
every checkpoint sits on a validated step.

---

## Install

Nothing compiles, so this runs anywhere; only the final check needs a GPU.

```bash
source /scratch/user/sushil22_tamu.edu/miniconda3/etc/profile.d/conda.sh
conda create -y -n igrpo python=3.12 && conda activate igrpo
python -m pip install --upgrade pip setuptools wheel packaging

cat > /tmp/c.txt <<'EOF'
vllm==0.11.0
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
EOF

pip install vllm==0.11.0                      # brings torch 2.8.0+cu128

TORCH_MM=$(python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")
ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
PYTAG="cp$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")"
pip install einops
pip install --no-deps --no-build-isolation \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch${TORCH_MM}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"

cd /scratch/user/sushil22_tamu.edu/projects/IGRPO
pip install -e . -c /tmp/c.txt
pip install -c /tmp/c.txt liger-kernel torchdata uvicorn fastapi hf_transfer
cd agent_system/environments/env_package/search/third_party && pip install -e . -c /tmp/c.txt
cd - && pip install gym==0.26.2 -c /tmp/c.txt
```

**Use `setup.py`, never `requirements.txt`** — the latter pins `transformers==4.51.1`,
incompatible with vllm 0.11.0 (needs >=4.55.2). Resulting stack: torch 2.8.0+cu128,
vllm 0.11.0, flash-attn 2.8.3, transformers 4.57.3, ray 2.50.0, tensordict 0.10.0,
numpy 2.2.6, gym 0.26.2.

Verify from the repo root on a GPU node:

```python
import torch; from flash_attn import flash_attn_func
import verl.trainer.ppo.ray_trainer            # pulls in gigpo + igpo
import igrpo.core_igrpo, igpo.core_igpo, gigpo.core_gigpo
import agent_system.environments.env_package.search.envs
q = torch.randn(1, 64, 4, 64, dtype=torch.bfloat16, device="cuda")
flash_attn_func(q, q, q, causal=True)          # -> (1, 64, 4, 64)
```

> **One env, three algorithms — not optional.** `verl/trainer/ppo/ray_trainer.py:63-64`
> imports `gigpo` and `igpo` at module level, so the trainer will not import unless all
> three are present. And because `igpo/`, `gigpo/`, `igrpo/` have no `__init__.py` they
> are namespace packages resolved from the cwd: **always launch from the repo root.**

---

## Data and models

`DATA=/scratch/user/sushil22_tamu.edu/projects/IGRPO/data` — 81 GB.

| Path | Size |
|---|---|
| `Base_models/Qwen2.5-3B-Instruct` | 6.2 GB |
| `Base_models/e5-base-v2` | 0.5 GB |
| `searchR1/e5_Flat.index` | 61 GB |
| `searchR1/wiki-18.jsonl` | 14 GB |
| `searchR1_processed_direct/train.parquet` | 169,615 rows |
| `searchR1_processed_direct/test.parquet` | 51,713 rows — final eval only |
| `searchR1_processed_direct/val_subset.parquet` | 1,024 rows — in-training validation |

Search-R1's `PeterJinGo/nq_hotpotqa_train`: questions and answers only, no gold paragraphs.
Train is HotpotQA 90,447 + NQ 79,168 (both complete official splits); test is a seven-way
mixture (PopQA 14,267, 2Wiki 12,576, TriviaQA 11,313, HotpotQA 7,405, NQ 3,610,
MuSiQue 2,417, Bamboogle 125).

```bash
export DATA=/scratch/user/sushil22_tamu.edu/projects/IGRPO/data
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen2.5-3B-Instruct --local-dir "$DATA/Base_models/Qwen2.5-3B-Instruct"
hf download intfloat/e5-base-v2 --local-dir "$DATA/Base_models/e5-base-v2" \
    --exclude "onnx/*" "openvino/*" "*.bin" "*.h5" "*.ot" "*.msgpack"

python examples/search/searchr1_download.py --local_dir "$DATA/searchR1"
cat "$DATA/searchR1"/part_* > "$DATA/searchR1/e5_Flat.index"
rm -f "$DATA/searchR1"/part_a[ab]          # reclaims 65 GB; the cat peaks at ~130 GB
gzip -d "$DATA/searchR1/wiki-18.jsonl.gz"

python examples/data_preprocess/preprocess_search_r1_dataset.py --local_dir "$DATA/searchR1_processed_direct"
python examples/data_preprocess/make_val_subset.py     # 1024 HotpotQA, seeded
```

---

## Run layout

**5 GPUs per run**: 4 for training (`n_gpus_per_node=4`, TP=1) + 1 for the retrieval server,
which holds the fp16 index at ~34 GB. One node covers it.

662 steps (169,615 / 256, one epoch). Validation every 10 steps, checkpoint every 50
(13 saves, `max_actor_ckpt_to_keep=3` prunes to the last three, ~35 GB each).

```
$DATA/runs/ICLR/
├── igrpo-3B/{checkpoints,debug_batches,retrieval_server_<jid>.log}
├── igpo-3B/...
├── gigpo-3B/...
└── slurm-<jid>.out
```

Scripts read `DATA`, `PROJECT_NAME`, `EXPERIMENT_NAME`, `SEARCH_PORT`, `RUN_DIR` from the
environment, so nothing needs editing to relocate a run. The job script derives
`SEARCH_PORT` from the job id so two runs on one node cannot collide.

### Edits made to the shipped scripts

All hardcoded `$HOME/data/...`, where nothing exists.

| File | Change |
|---|---|
| `examples/igrpo_trainer/run_search.sh` | `DATA` paths, val subset, `val_data_size` 512→1024, `test_freq` 50→10, `save_freq` 25→50, `ray_init.num_cpus`, checkpoint dir, `CUDA_VISIBLE_DEVICES="0, 1, 2, 3"`→`"0,1,2,3"` (spaces make CUDA stop parsing and expose only GPU 0) |
| `examples/igpo_trainer/run_search.sh` | same |
| `examples/gigpo_trainer/run_search_3b.sh` | **new** — 3B baseline matched to the above |
| `examples/search/retriever/retrieval_launch.sh` | `save_path`, local e5 path |
| `examples/search/retriever/retrieval_server.py` | thread lock (change #5) |
| `examples/data_preprocess/make_val_subset.py` | **new** |

Untouched: `gigpo_trainer/run_search.sh` (upstream 7B) and the ALFWorld / WebShop /
Sokoban / GSM8K scripts, whose environments are not installed.

---

## Retriever environment

Pre-existing, not rebuilt: Python 3.10.20, `libfaiss 1.8.0 cuda120` from conda-forge
(GPU symbols present, `--faiss_gpu` works), torch 2.4.0+cu121, pyserini 1.2.0, fastapi.

It carries `transformers 5.3.0`, a major version ahead of what `retrieval_server.py`
targets — this turned out to be fine, the e5 BertModel loads. If a future change breaks it,
`conda activate retriever && pip install "transformers<5"` fixes it without touching `igrpo`.

---

## Math training

`math-verify==0.9.0` is installed in the `igrpo` conda environment and is
used to score mathematically equivalent boxed answers. The `math` optional
dependency in `setup.py` pins the same version. To install it in a fresh
environment, run `python -m pip install 'math-verify==0.9.0'` after activating
`igrpo`.

`examples/data_preprocess/preprocess_math.py` downloads the public QA-code
parquet files, keeps math questions including multiple choice questions, and
writes `data/math/train.parquet` (4,999 rows) and `data/math/val.parquet`
(100 rows). The default model is
`data/Base_models/Qwen2.5-Coder-3B-Instruct`.
Use `--exclude-multiple-choice` to match TreeHCA's filtered math split; that
makes the validation split 99 rows, so set `VAL_DATA_SIZE=99` for those runs.
One training row with an empty target is always excluded because it cannot be
scored reliably.

```bash
cd /scratch/user/sushil22_tamu.edu/projects/IGRPO
python examples/data_preprocess/preprocess_math.py
sbatch --export=ALL,METHOD=treehca,PROJECT_NAME=MATH,EXPERIMENT_NAME=treehca steps/run_math.sbatch
sbatch --export=ALL,METHOD=igrpo,PROJECT_NAME=MATH,EXPERIMENT_NAME=igrpo steps/run_math.sbatch
sbatch --export=ALL,METHOD=igpo,PROJECT_NAME=MATH,EXPERIMENT_NAME=igpo steps/run_math.sbatch
sbatch --export=ALL,METHOD=gigpo,PROJECT_NAME=MATH,EXPERIMENT_NAME=gigpo steps/run_math.sbatch
```

Each method's `run_math.sh` keeps the training settings from its
`run_search_7b.sh` recipe. The math environment exposes the Python tool and
runs without a retriever. The terminal reward is 0.1 after a Python call plus
0.9 for a correct boxed answer; a trajectory with no Python call scores zero.
The Python tool limits execution time, memory, code length, and output, but its
subprocess is not isolated from local files or the network.
