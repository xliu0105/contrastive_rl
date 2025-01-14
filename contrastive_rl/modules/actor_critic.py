#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()  # 要通过nn.Module的构造函数初始化，需要先调用父类的构造函数
        activation = get_activation(activation)  # 根据参数获取激活类型函数

        mlp_input_dim_a = num_actor_obs  # 根据参数获取network的输入和输出维度
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)  # 使用*操作符将list解包成参数列表传给nn.Sequential
        # NOTE: 由于定义actor和critic的时候都是用的nn.Sequential，所以可以不用显式的定义forward函数，在要进行前向计算的时候直接调用这个类的实例的actor和critic即可

        # Value function
        # critic输出的应该是V(s)，而不是Q(s,a)
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        # nn.Parameter是用来定义在训练过程中被优化的变量的，这些变量是模型的一部分。所以在训练过程中，self.std也会被优化
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):  # 计算概率分布的熵。当熵越大，说明概率分布比较均匀，意味着概率分布的不确定性高；而熵越小，说明概率分布比较集中，意味着概率分布的不确定性低
        return self.distribution.entropy().sum(dim=-1)  # 最后一个dim是action的维度，对所有的action的分布的熵求和

    def update_distribution(self, observations): 
        mean = self.actor(observations)  # 根据传入的观测计算actor的实际输出
        self.distribution = Normal(mean, mean * 0.0 + self.std) # 用actor的输出和std来初始化一个Normal分布，注意由于self.std是一个nn.Parameter，所以这个分布的std是可以被优化的

    def act(self, observations, **kwargs):  # actor网络会输出一个mean，然后根据mean和std生成一个分布，再从这个分布中采样一个动作。注意std也是一个被优化的参数
        self.update_distribution(observations)  # 传入actor的观测给update_distribution函数
        return self.distribution.sample()  # 从update_distribution计算出的分布中采样一个动作

    def get_actions_log_prob(self, actions):
        # 计算actions在当前分布下的对数概率，并对最后一个维度求和。由于是对数概率，所以对所有的对数概率求和就是对所有的概率求积再取对数
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):  # 计算不添加噪音的actor输出的动作，是用于在inference(训练结束后测试)时使用的
        actions_mean = self.actor(observations)
        return actions_mean  # 只返回actor网络的输出，即mean值，不使用std
        # 补充一点，在训练过程中，actor的输出是mean+std的一个正态分布的采样值，并且在训练过程中，鼓励分布的熵越大越好，因为这样可以增加探索性

    def evaluate(self, critic_observations, **kwargs):  # 根据传入的critic观测计算critic的输出
        value = self.critic(critic_observations)
        return value


def get_activation(act_name): # 根据参数返回激活函数类型
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
