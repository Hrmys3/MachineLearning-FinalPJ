# Lift PH20 baseline

- Training: 20 PH demonstrations, seed 42, 2,000 optimizer updates.
- Evaluation: 20 validation demonstration initial states, 100 steps at 20 Hz.
- Prediction chunk: 20; execute 5 actions before replanning.
- Any success: 19/20; success for 10 consecutive steps: 19/20.
- Final success: 3/20. Successful episodes' mean first success: 41.9474 steps.
- This is validation, not held-out testing. One training seed only.
- The checkpoint is the last update, not validation-selected best; no optimizer state.

`config.json` records key settings; training code/checkpoint contains further settings.
`results/` and `process_status.json` are copies of actual evaluation records.
Full CSV/GIF evidence remains in ignored `outputs/act_valid100/`.
`artifact_manifest.json` records the original baseline commit and model hash;
the repository reorganization was committed as `47a4b16`. It did not retrain the
model or change these results. Model weights and full evaluation attachments have
not been published as downloadable resources.
