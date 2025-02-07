# Contrastive RL

Fast and simple implementation of RL algorithms with contrastive learning, designed to run fully on GPU.
This code is an evolution of [rsl_rl](https://github.com/leggedrobotics/rsl_rl) provided with Robotic Systems Lab.

Only PPO with contrastive learning based on barlow twins method is implemented for now. Contributions are welcome.

I only tested the performance difference between `contrastive_rl` and `rsl_rl` in a quadruped robot reinforcement learning task on flat terrain, and the results showed that `contrastive_rl` was trained slightly better than `rsl_rl`, and there is no significant decrease in training speed.

## Setup

Following are the instructions to setup the repository for your workspace:

```bash
git clone https://github.com/xliu0105/contrastive_rl.git
cd contrastive_rl
pip install -e .
```
The usage of this package is exactly the same as the `rsl_rl`. You only need to replace `rsl_rl` with `contrastive_rl` in `from rsl_rl.runners import OnPolicyRunner`.

This package refers to the paper: 
- [Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response](https://arxiv.org/abs/2312.11460), DOI: 10.48550/ARXIV.2312.11460
- [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230), DOI: 10.48550/ARXIV.2103.03230


The framework supports the following logging frameworks which can be configured through `logger`:

* Tensorboard: https://www.tensorflow.org/tensorboard/
* Weights & Biases: https://wandb.ai/site
* Neptune: https://docs.neptune.ai/

For a demo configuration of the PPOContrastive, please check: [dummy_config.yaml](config/dummy_config.yaml) file.
