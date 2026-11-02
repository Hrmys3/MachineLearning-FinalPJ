# Third-party sources

## ACT

Adapted from https://github.com/tonyzhaozh/act at commit
`742c753c0d4a5d87076c8f69e5628c79a8cc5488`.
The adapted copy is under `third_party/act_lift/`. Retain its MIT license and
the DETR license at `third_party/act_lift/detr/LICENSE`.
Local changes include 9D state / 7D action projections and torchvision weights API.

## Demonstration data

Training uses RoboMimic Lift proficient-human demonstrations, obtained through
`chaoqi-liu/robomimic_mt4_ph` (community conversion) and official source metadata
from `robomimic/robomimic_datasets`, `v1.5/lift/ph/demo_v15.hdf5`.
See upstream dataset cards for attribution and licensing. Original training
videos and demonstration datasets are not included in the ordinary Git repository.

## Visual initialization

ResNet18 ImageNet initialization is provided by torchvision, downloaded from
https://download.pytorch.org/models/resnet18-f37072fd.pth .
The locally saved checkpoint contains the resulting trained model parameters.
No checkpoint download has been published by this project.

## Simulation dependencies

MuJoCo and robosuite are installed dependencies, not relicensed by this project.
No new license for original project code is selected by this documentation.
