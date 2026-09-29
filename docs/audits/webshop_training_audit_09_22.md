# WebShop training audit — 2026-09-22

The audit did **not find a full-catalog/human-goal regression in catalog cleaning,
goal construction, ordinary search, or purchase rewards that explains the large
performance gap**. It did find important changes to the experimental conditions,
an evaluation coverage bug, and evidence that failed option selection accounts
for much of the observed failure in one completed run.

This is an audit, not a training fix. No environment, trainer, launch script,
dataset, index, or installed dependency was changed. The evidence and standalone
diagnostic scripts are in [audits/webshop_20260922](audits/webshop_20260922).

The comparison base is `65707c6`, the last commit before Jeremy Carleton's changes.
The WebShop files were also compared with `750a444`; their baseline environment
behavior is identical. The audited checkout is `2fa3b9a`, including the existing
uncommitted changes. Data and search indexes are not tracked by Git, so their
historical contents cannot be established from the fork history.

1. **The old and new results do not use the same benchmark protocol.**

   The original GiGPO launch script inherits `use_small=True` and
   `human_goals=False`. The public author's [GiGPO script](https://raw.githubusercontent.com/langfengQ/verl-agent/master/examples/gigpo_trainer/run_webshop.sh)
   and [configuration](https://raw.githubusercontent.com/langfengQ/verl-agent/master/verl/trainer/config/ppo_trainer.yaml)
   also currently have these defaults. The [paper](https://arxiv.org/html/2505.10978v2#A4.SS1)
   specifies 150 iterations, 16 groups of 8, a 15-turn horizon, and binary success
   rewards. The paper describes the full WebShop benchmark, but that description
   and the public script's defaults do not establish which data overrides were
   actually used for its reported results. Full/human should not be assumed to
   reproduce the released default experiment without resolving that discrepancy.

   On this machine, the small synthetic pool has **6,910 goals targeting only
   415 distinct products**. The full human pool has **12,087 goals targeting
   10,014 distinct products**, among 1,181,430 loaded products. Synthetic prompts
   explicitly enumerate option names and values; human prompts express them in
   natural language. This changes retrieval difficulty, language interpretation,
   and opportunities to revisit the same target during 150 updates.

   Commit `657678d` also changes validation from worker-specific seeded shuffles
   to the canonical seed-233 human holdout. Previously each training worker
   excluded positions 0–499 of its own shuffle, while validation workers used
   different shuffles with seeds starting at training seed + 1000. These are not
   disjoint goal identities. The new split excludes the same 500 human identities
   for every training seed. Lower held-out performance after this change need
   not indicate broken training. Reverting the split would change the question
   being measured.

   Code: `env_manager.py:695`, `webshop/envs.py:156`,
   `web_agent_site/engine/goal.py:22`, `catalog_data.py:94`.

2. **The evaluation-only launcher does not evaluate all 500 held-out goals.**

   `examples/gigpo_trainer/run_webshop.sh:23` sets 500 dataset rows but a batch
   size of 50 for evaluation. Every call to `WebshopMultiProcessEnv.reset()`
   independently samples without replacement *within that batch*, not across
   the evaluation. Ten batches therefore revisit goals and omit others. Using
   the actual seed-233 NumPy sampler, the first such evaluation covers **326
   distinct goals out of 500**. Ordinary training validation uses a changing
   128-goal sample from the pool, not all 500.

   This affects coverage and variance; it does not demonstrate a systematic
   downward bias or explain the training failure by itself. A benchmark evaluation
   needs either one 500-environment batch or explicit goal-index enumeration
   across smaller batches. The latter avoids starting 500 workers.

   The launch scripts also set `trainer.validation_data_dir`, but the current
   `RayPPOTrainer._validate()` never writes to it. Its generation dump call is
   only in the training loop. This explains why the saved validation metrics
   cannot be followed back to complete evaluation trajectories in those paths.

   Code: `webshop/envs.py:205`, `ray_trainer.py:800`,
   `examples/gigpo_trainer/run_webshop.sh:26`.

3. **The small-catalog search behavior changed substantially; the full path did not.**

   Before `657678d`, `make_envs` passed `num_products=None` even when loading
   the small catalog. This searches the full Lucene index and then discards hits
   absent from the 1,000-product dictionary. The new code correctly selects the
   small index when `use_small=True`.

   Reproducing both paths against the current files gives:

   | Query | Old path: retained hits | Current small index: hits |
   | --- | ---: | ---: |
   | `shoes` | 0 | 39 |
   | `jade roller` | 0 | 5 |
   | `men dress shirts` | 0 | 50 |
   | `cotton shirt` | 0 | 50 |
   | `wireless headphones` | 1 | 50 |

   This is an improvement to small-catalog behavior, but it makes small runs
   before and after the change incomparable. Full-catalog runs still use the
   full index. Code: `env_manager.py:689`, `engine.py:169`.

4. **The loader and option-cleaning audit rules out widespread target corruption.**

   The loader reads 1,181,436 raw rows and retains 1,181,430 unique valid products.
   All 10,136 ASINs present in the human annotation file are loaded. Goal
   construction skips 164 annotations with empty attributes, as the original
   code did, and produces 12,087 goals. All human goal option containers are
   lists, so the inherited dictionary-option handling is not used here.

   The actual cleaning changes are an extra whitespace trim after slash
   replacement and removal of cross-group duplicate option values. The native
   text interface already used a value-only dictionary in which the last group
   owns a repeated value. The audit reconstructs those old accessible choices
   and compares their optimal reward with current choices.

   **625 products have changed option dictionaries; 41 human target products,
   covering 147 human goals, are affected. No audited human target has a lower
   or higher maximum reward after these changes.**

   For each human goal, a coverage search finds the best combination with at
   most one accessible option per group, then calls the production reward
   function on the annotated target:

   | Split | Goals | Annotated target can earn 1.0 | Mean best target score |
   | --- | ---: | ---: | ---: |
   | Train | 11,587 | 11,572 | 0.999718 |
   | Validation | 500 | 498 | 0.998833 |

   All target attributes and type checks pass. The 17 exceptions are option
   annotation/interface inconsistencies already present before the cleaning.
   Examples include requests for either of two colors whose annotation demands
   both, or two sizes that occupy the same option group. The validation exceptions
   are `B07GCLX7KN` and `B01LR0OOBC`. Details are in
   [target_anomalies.json](audits/webshop_20260922/target_anomalies.json).

   These are **target-product** checks, not claims of global impossibility:
   another product could satisfy the reward. They also do not establish that an
   LLM can retrieve or recognize the target. They do establish that cleaning has
   not made a substantial portion of the annotated tasks unrewardable.

5. **The installed full search index covers every human target.**

   The full index contains 1,181,370 documents. Every one of the 10,014 constructed
   human target ASINs is retrievable directly by document ID, and its indexed
   title matches the loaded catalog. No human target is missing from the index.
   Direct document presence is not a guarantee of top-50 keyword retrieval.

   There is a 60-document cardinality difference relative to the loaded catalog,
   which deserves index provenance/coverage checks but does not remove any human
   target in this audit. The service currently only checks that the index is
   nonempty and decodable; its content fingerprint covers the data JSON files,
   not the Lucene index contents.

   There is also an inherited rebuilding trap: `convert_product_file_format.py`
   uses `utils.DEFAULT_FILE_PATH` and `DEFAULT_ATTR_PATH`, currently the small
   files, even when writing the directory called `resources`. `setup.sh -d all`
   does not switch those constants; its download section is now commented out.
   Rebuilding without explicitly selecting full inputs could replace the full
   index with a small one that passes the startup probe. **That has not happened
   to the currently installed index.**

6. **The centralized backend matched the legacy backend under full/human settings.**

   All **27 existing catalog-service tests passed**. A direct full-catalog
   comparison checks all prices and all goals for training seeds 0 and 7 and
   validation seed 233. It compares 24 episodes, covering 192 transitions through
   search, product selection, description/features, option clicks, and purchase.
   Observations, nonterminal action sets, rewards, and done flags match exactly.

   The test shares loaded catalog records to avoid loading duplicate 5 GB JSON
   inputs, but price/goal construction and both transition implementations remain
   separate. This is an in-process backend test; it is not a full Ray/GPU training
   replay. Existing tests cover RPC contracts, clone isolation, cache eviction,
   validation splits, and small-catalog deterministic trajectories.

   Static review found no ordinary-search or reward-equation change. Remote
   terminal metadata captures the purchased episode before auto-reset, whereas
   legacy action metadata is read after auto-reset; this does not change the
   terminal reward. Special random-search RNG differs deliberately, and malformed
   RPC inputs now fail explicitly. Neither is evidence of silent degradation in
   the completed ordinary-search run.

7. **Saved runs show a real gap, but the available completed pair is confounded.**

   | Saved W&B run | Algorithm/settings | Step | Validation success | Task score |
   | --- | --- | ---: | ---: | ---: |
   | `20260918_215226-qsf1fqv3` | GiGPO, small/synthetic, legacy split, batch 8 | 150 | 46.875% | 0.538914 |
   | `20260921_100946-jtcyfc59` | TreeHCA `q_hindsight`, full/human, fixed split, batch 16 | 150 | 37.5% | 0.658073 |
   | `20260922_044228-jebpfly2` | TreeHCA `snis` + no-progress cap, small/synthetic, legacy split, batch 16 | 150 | 62.5% | 0.806281 |

   All three use Qwen2.5-1.5B-Instruct. The full TreeHCA run uses four GPUs/TP=1;
   the recent small run uses two GPUs/TP=2. The credit assignment and no-progress
   cap also differ. I did not locate a completed full/human **GiGPO** run in the
   saved configurations. The directories under `runs` and the configurations
   under `wandb` were cross-referenced; a directory name alone is not sufficient
   to identify the experiment. More completed runs are recorded in
   [run_summary.json](audits/webshop_20260922/run_summary.json).

   The final full/human training batch contains 950 unique tree nodes and 430
   leaves: 168 successes, 131 failed purchases, 124 pruned leaves, and 7 turn
   limits. Padding duplicates were removed by node UUID. Thus the reported
   TreeHCA training success is 168/430, not simply successes/purchases.

   Reconstructing the 299 purchases from their ancestor actions and applying the
   production reward function reproduces **296 scores exactly**. The other three
   have price ranges straddling the goal limit; their worker-seeded price was not
   reconstructed by this lightweight replay. Among the 131 failures:

   | Failed component | Purchases |
   | --- | ---: |
   | Options | 105 |
   | Type | 51 |
   | Attributes | 47 |
   | Price | At least 40 |

   Counts overlap. Of the 105 option failures, 48 purchased without selecting any
   option. Only two failed purchases bought their annotated target. This points
   toward retrieval/selection and option completion, rather than a broadly
   corrupted target catalog. It is one training batch from TreeHCA, not a causal
   explanation of GiGPO performance or a validation-wide diagnosis.

8. **Other changes and inherited limitations were checked.**

   - Removing the Reviews button changes the action set for both backends. Reviews
     were already empty in the loader, so this does not remove review information
     that could previously satisfy human goals.
   - Binary training rewards remain 10 only for exact task score 1.0 and 0 otherwise.
     This is inherited and agrees with the paper's described reward. Partial
     attribute/option matches do not provide task reward. Harder human tasks can
     therefore produce many all-failure groups, but that is not a new reward bug.
   - Prompt templates, task extraction, action projection, the 13,000-character
     history fallback, type/attribute/option normalization, and the GiGPO credit
     code are unchanged by the user's fork changes. TreeHCA additions in the
     rollout/trainer paths are gated by estimator selection. Model-worker
     temperature overrides are limited to explicitly tagged TreeHCA probes.
   - Full-run logs show prompt clipping in some batches, but the mean logged
     clipping fraction over 150 training updates is about 0.001073; the final
     batch has none. This does not establish truncation as the main bottleneck.
   - The fixed split retains 11,587 training goals. The original WebShop baseline's
     three-way partition also excluded a 1,000-goal development set, leaving
     10,587. The new split matches the canonical 500 test positions, but is not
     the complete original train/dev/test protocol.
   - Default `use_small=True` plus default `validation.mode=fixed_human` requests
     500 goals from only 13 small-catalog human goals and fails intentionally.
     Small launchers need an explicit legacy validation setting or a reduced
     smoke-test holdout; this does not affect full/human startup.

The next experiment should hold the algorithm and evaluation protocol constant:
run GiGPO with full/human goals and the fixed holdout, recording goal IDs and
reward components, then compare against full/synthetic training evaluated on
that **same human holdout**. Separately, a small/synthetic legacy run can test
reproduction of the public default script. Changing catalog size, goal language,
split, algorithm, and hardware together prevents attribution. Before calling an
evaluation “the 500-goal test,” fix enumeration across batches.

To reproduce the CPU diagnostics, use the existing `webshop` conda environment:

```bash
PYTHONPATH="$PWD" conda run -n webshop python -m pytest tests/webshop/test_catalog_service.py -q
conda run --no-capture-output -n webshop python docs/audits/webshop_20260922/full_catalog.py
conda run --no-capture-output -n webshop python docs/audits/webshop_20260922/backend_parity.py
conda run --no-capture-output -n webshop python docs/audits/webshop_20260922/purchase_replay.py
```

The full-catalog diagnostics require substantial CPU memory. The purchase replay
uses the saved step-150 rollout path embedded in the diagnostic. No packages are
installed by these commands. Audit scripts write compact results next to
themselves and detailed intermediate records under `/tmp`.
