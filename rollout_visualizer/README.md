# Search rollout visualizer

From the repository root, point the viewer at the directory supplied as `trainer.rollout_data_dir`. The server uses the Python standard library and serves the UI locally.

```bash
python -m rollout_visualizer logs/search/rollouts
```

Open **http://127.0.0.1:8050**. For remote training, forward port 8050 through your IDE or SSH. Use `--port` to select another port. Refresh files picks up newly written training steps.

Select a step and training group, then a node to inspect its prompt, model output, search question, reasoning steps, action history, or full JSON record. Drag to pan, scroll to zoom, and select **Fit graph** to reset the view. Double-click a detail field to move it to the top.

**Decision model results** shows the chosen search or answer and each reasoning step’s support, contradiction, and no-claim probabilities. The panel includes the original and shaped advantage and optional reasoning and answer entropy in nats. Highlight steps whose selected probability exceeds a threshold. Missing measurements appear as a dash.

Color nodes by structure, score, environment turn reward, answer entropy, IGPO or GiGPO turn advantage, or mean shaped reasoning advantage. IGPO records include `igpo_step_reward` (the reward supplied to the estimator, including any action penalty) and `igpo_turn_advantage` (after normalization and discounting, before shaping). GiGPO records include the episode, turn, and weighted turn components.

The trainer saves one JSONL file per step. Set `trainer.validation_data_dir` to also save evaluation turns. Those files use the suffix `.validation.jsonl` and contain the generated traces without training-time judgments. Records carry the training group ID, turn ID and ordered ancestor IDs, and trajectory identity when available. Padding copies are omitted. In IGRPO full-path mode, a shared generated node can receive different credit on different terminal paths. The graph merges the node by its generation ID and keeps each path’s record selectable. A node’s color averages its available records.

Older files without IDs can still be opened. The viewer groups identical prompts and shows independent turns when the file lacks ancestry. Missing parents, inconsistent ancestry, and malformed lines are reported in the UI.
