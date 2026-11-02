# Repository layout

Run scripts from the repository root using their new paths. Each runtime script
also resolves the project root from its own location, so data/output paths remain
unchanged. Do not rename or move `data/`, `outputs/` or `third_party/` independently.

| Directory | Role |
| --- | --- |
| scripts/checks | Physics, camera and model smoke checks |
| scripts/data | Source replay, inspection, splitting, caching |
| scripts/train | Smoke and PH20 candidate training |
| scripts/eval | Single rollout, logs, batch validation, GIF grid |
| scripts/experts | Scripted and aligned experts, historical self-collection |
| scripts/tools | ACT source adaptation |
| experiments/lift20_baseline | Small, version-controlled result snapshot |
| env/snapshots | Historical dependency records |
| docs | Project documentation |
| archive | Historical helpers and earlier documents |
| outputs/logs | Existing top-level logs, moved without changing their contents |

Old root-level Python entry points have moved; no compatibility wrappers are added.
The batch validation runner calls `scripts/eval/eval_act_one.py`.
Checkpoint/data/result directories have not moved. No new training or evaluation
is performed by this restructuring. Current outputs still correspond to the
pre-reorganization baseline; the manifest records that source commit.
