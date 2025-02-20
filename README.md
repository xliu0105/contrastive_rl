# Contrastive RL

Fast and simple implementation of RL algorithms with contrastive learning, designed to run fully on GPU.
This code is an evolution of `rsl_rl` provided with Robotic Systems Lab.

Only PPO with contrastive learning based on barlow twins method is implemented for now.
Contributions are welcome.

## Setup

Following are the instructions to setup the repository for your workspace:

```bash
git clone https://github.com/xliu0105/contrastive_rl.git
cd contrastive_rl
pip install -e .
```
The usage of this package is exactly the same as the `rsl_rl`. You only need to replace `rsl_rl` with `contrastive_rl` in `from rsl_rl.runners import OnPolicyRunner`.

This package offers a config file in `cli_args` folder, 

This package refers to the paper: 
- Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response, DOI: 10.48550/ARXIV.2312.11460
- Barlow Twins: Self-Supervised Learning via Redundancy Reduction, DOI: 10.48550/ARXIV.2103.03230


The framework supports the following logging frameworks which can be configured through `logger`:

* Tensorboard: https://www.tensorflow.org/tensorboard/
* Weights & Biases: https://wandb.ai/site
* Neptune: https://docs.neptune.ai/

For a demo configuration of the PPOContrastive, please check: [dummy_config.yaml](config/dummy_config.yaml) file.
