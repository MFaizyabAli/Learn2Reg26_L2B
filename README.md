[![Static Badge](https://img.shields.io/badge/MICCAI-SIG_BIR-%2337677e?style=flat&labelColor=%23ececec&link=https%3A%2F%2Fmiccai.org%2Findex.php%2Fspecial-interest-groups%2Fbir%2F)](https://miccai.org/index.php/special-interest-groups/bir/) ![Static Badge](https://img.shields.io/badge/MICCAI-Learn2Reg-%23214f5f?labelColor=%23ececec&link=https%3A%2F%2Flearn2reg.grand-challenge.org%2F) <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

This repository houses the Docker submission template and baseline method for the
**Learn2Breath** task at Learn2Reg 2026: registering expiration (EXP, moving) to
inspiration (INSP, fixed) lung CT.

Please visit [***learn2reg.grand-challenge.org***](https://learn2reg.grand-challenge.org/) for more information.

## Contents

- [`Test_Phase_Docker_Example/`](./Test_Phase_Docker_Example) — the Docker submission
  template and full submission instructions (build, run, export, validate).
- [`convexadam_l2b_baseline/`](./convexadam_l2b_baseline) — a standalone (non-Docker)
  batch runner for the same ANTs Affine + ConvexAdam-MIND (SVF) baseline, useful for
  generating baseline predictions over a full dataset manifest.
