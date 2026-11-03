# Repository layout

Run scripts from the repository root using their new paths. Each runtime script
also resolves the project root from its own location, so data/output paths remain
unchanged. Do not rename or move `data/`, `outputs/` or `third_party/` independently.

| Directory | Role |
| --- | --- |
| scripts/checks | Physics, camera and model smoke checks |
| scripts/data | Source replay, inspection, splitting, caching, fixed random initial states |
| scripts/train | Smoke and PH20 candidate training |
| scripts/eval | Single rollout, logs, batch development evaluation, GIF grid |
| configs/lift.yaml | Shared editable training, model and evaluation parameters |
| experiments/random_dev100 | Small manifest of the frozen random development set |
| scripts/experts | Scripted and aligned experts, historical self-collection |
| scripts/tools | ACT adaptation, speed benchmarks and hardware monitoring |
| scripts/performance.py | Shared wall/CUDA timing and optional hardware sampling |
| experiments/lift20_baseline | Small, version-controlled result snapshot |
| env/snapshots | Historical dependency records |
| docs | Repository structure documentation |
| archive | Historical helpers; local backups may be ignored |
| outputs/logs | Existing top-level logs, moved without changing their contents |

Old root-level Python entry points have moved; no compatibility wrappers are added.
The batch validation runner calls `scripts/eval/eval_act_one.py`.
Evaluation defaults to the frozen 100-state random development set;
`--initial-states valid` selects the original 20 demonstration states.
Generate random states with `scripts/data/prepare_random_lift_states.py`.
Training and evaluation both accept `--config` (default `configs/lift.yaml`).
Single and batch evaluation now require `--output` with a new experiment directory.
`eval_act_logged.py` shares the single-rollout implementation; `lift_protocol.py`
contains the optional `--policy act_hold` controller and first-lift hold metric.
The optional controller uses simulator object-height truth to trigger holding;
its results must be reported separately from pure ACT. Batch `metrics.json` and
`process_status.json` summarize all scheduled initial states, including failures.
The batch runner automatically produces `grid_001.gif`, etc., grouping every
20 scheduled episodes into a 4-by-5 grid. Missing rollouts retain their slots.
`make_eval_grid.py --input <experiment>` also works independently.
See the root README for commands and the precise metric definition.
Checkpoint/data/result directories have not moved. No new training or evaluation
is performed by this restructuring. Current outputs still correspond to the
pre-reorganization baseline; the manifest records that source commit.

## What an ordinary clone contains

Source code, the adapted ACT implementation, fixed splits/statistics, and the small
baseline records under `experiments/lift20_baseline/` are tracked. Downloaded
demonstrations, episode caches, checkpoints, logs, CSV traces and GIFs are ignored.
The original `third_party/act/` checkout is optional and ignored; the adapted
`third_party/act_lift/` copy is tracked. Model assets have not been published.

The upstream ACT README is retained for attribution and reference; its installation
commands and task examples are not this project's Lift instructions. Start with
the root README and `env/README.md`.
