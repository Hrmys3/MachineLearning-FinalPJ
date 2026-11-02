# Environment records

The files in `snapshots/` are historical pip snapshots, not a tested portable lockfile.
They may contain machine-local paths and omit packages installed later.

Validated core: Python 3.11; robosuite 1.5.1; mujoco 3.3.0; mink 0.0.13.
Observed training stack: torch 2.14.1+cu130, torchvision 0.29.1+cu130, numpy 2.4.6.
Use the existing `ml` Conda environment on the original machine. A fresh-install
workflow has not been validated end to end. Do not blindly install all snapshots.
