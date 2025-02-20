# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

# must install isaaclab first
from isaaclab.utils import configclass

@configclass
class ContrastiveRlPpoActorCriticCfg:
  """Configuration for the PPO_contrastive actor-critic networks."""

  class_name: str = "ActorCriticContrastive"
  """The policy class name. Default is ActorCriticContrastive."""

  init_noise_std: float = MISSING
  """The initial noise standard deviation for the policy."""

  actor_hidden_dims: list[int] = MISSING
  """The hidden dimensions of the actor network."""

  critic_hidden_dims: list[int] = MISSING
  """The hidden dimensions of the critic network."""

  encoder_hidden_dims: list[int] = [256, 256, 256]
  """The hidden dimensions of the encoder network. The encoder input the observation of actor and output a vector."""

  env_vector_dims: int = 100
  """The output dimensions of the encoder network."""

  activation: str = MISSING
  """The activation function for the actor and critic networks."""


@configclass
class ContrastiveRlPpoAlgorithmCfg:
  """Configuration for the PPO_contrastive algorithm."""

  class_name: str = "PPOContrastive"
  """The algorithm class name. Default and only support PPOContrastive."""

  value_loss_coef: float = MISSING
  """The coefficient for the value loss."""

  barlowtwins_loss_coef: float = 2.0
  """The coefficient for the barlowtwins loss, default is 2.0."""

  barlowtwins_lambda: float = 5e-3
  """The lambda parameter for the barlowtwins loss, default is 5e-3."""

  encoder_output_loss_coef: float = 5e-5
  """The coefficient for the encoder output loss, default is 5e-5."""

  use_clipped_value_loss: bool = MISSING
  """Whether to use clipped value loss."""

  clip_param: float = MISSING
  """The clipping parameter for the policy."""

  entropy_coef: float = MISSING
  """The coefficient for the entropy loss."""

  num_learning_epochs: int = MISSING
  """The number of learning epochs per update."""

  num_mini_batches: int = MISSING
  """The number of mini-batches per update."""

  learning_rate: float = MISSING
  """The learning rate for the policy."""

  schedule: str = MISSING
  """The learning rate schedule."""

  gamma: float = MISSING
  """The discount factor."""

  lam: float = MISSING
  """The lambda parameter for Generalized Advantage Estimation (GAE)."""

  desired_kl: float = MISSING
  """The desired KL divergence."""

  max_grad_norm: float = MISSING
  """The maximum gradient norm."""


@configclass
class ContrastiveRlOnPolicyRunnerCfg:
  """Configuration of the runner for on-policy algorithms."""

  seed: int = 42
  """The seed for the experiment. Default is 42."""

  device: str = "cuda:0"
  """The device for the rl-agent. Default is cuda:0."""

  num_steps_per_env: int = MISSING
  """The number of steps per environment per update."""

  max_iterations: int = MISSING
  """The maximum number of iterations."""

  empirical_normalization: bool = MISSING
  """Whether to use empirical normalization."""

  policy: ContrastiveRlPpoActorCriticCfg = MISSING
  """The policy configuration."""

  algorithm: ContrastiveRlPpoAlgorithmCfg = MISSING
  """The algorithm configuration."""

  ##
  # Checkpointing parameters
  ##

  save_interval: int = MISSING
  """The number of iterations between saves."""

  experiment_name: str = MISSING
  """The experiment name."""

  run_name: str = ""
  """The run name. Default is empty string.

  The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
  then it is appended to the run directory's name, i.e. the logging directory's name will become
  ``{time-stamp}_{run_name}``.
  """

  ##
  # Logging parameters
  ##

  logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
  """The logger to use. Default is tensorboard."""

  neptune_project: str = "isaaclab"
  """The neptune project name. Default is "isaaclab"."""

  wandb_project: str = "isaaclab"
  """The wandb project name. Default is "isaaclab"."""

  ##
  # Loading parameters
  ##

  resume: bool = False
  """Whether to resume. Default is False."""

  load_run: str = ".*"
  """The run directory to load. Default is ".*" (all).

  If regex expression, the latest (alphabetical order) matching run will be loaded.
  """

  load_checkpoint: str = "model_.*.pt"
  """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

  If regex expression, the latest (alphabetical order) matching file will be loaded.
  """