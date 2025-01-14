#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from contrastive_rl.modules import ActorCriticContrastive
from contrastive_rl.storage import RolloutStorageContrastive


class PPOContrastive:
    actor_critic: ActorCriticContrastive  # PPO_Contrastive算法必须使用ActorCriticContrastive模型

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        barlowtwins_loss_coef=2.0,
        barlowtwins_lambda=5e-3,
        encoder_output_loss_coef=5e-5,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later，这个storage很重要，是用来存储多步的数据的
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorageContrastive.Transition()  # Transition的意思是过渡，这里应该是用来存储一个step的数据的

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.barlowtwins_loss_coef = barlowtwins_loss_coef
        self.barlowtwins_lambda = barlowtwins_lambda
        self.encoder_output_loss_coef = encoder_output_loss_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    # 这里的num_transitions_per_env应该是每次迭代需要采样的样本数量，我理解的是multi-step的数量，注意应该还需要考虑decimation的大小
    # RolloutStorage类应该是作为replay buffer的作用吧(或者说是用来储存multi-step过程中的数据的？也许并不是replay-buffer，因为这个的PPO的名字就叫on_policy_runner)，但添加了很多其他额外的功能，如计算advantage等
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorageContrastive(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.test()  # TODO：这个test()方法是什么意思？是设置模型为测试模式吗？我记着应该是eval()方法啊

    def train_mode(self):
        self.actor_critic.train()  # 继承于nn.Module的类有train()方法，用来设置模型为训练模式

    def act(self, obs, critic_obs):  # 计算actor的action和critic的value，同时计算其他的一些信息，如action的log_prob等
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, new_obs, rewards, dones, infos):
        self.transition.rewards = rewards.clone()  # .clone()是创建一个张量的副本，原数据不变
        self.transition.next_observations = new_obs  # 这里的next_observations是下一个step的obs
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:  # time_outs信息是在infos中的
            self.transition.rewards += self.gamma * torch.squeeze(  # 这里确实发生更严重的自举了，因为是用的当前action的reward + 当前critic的value，而不是下一个obs的value
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):  # 这里的last_critic_obs是episode的step长度执行后，获取的后面的一个critic_obs
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam) # 计算一个episode中各个step的returns和advantages

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:  # 如果使用循环网络的actor-critic，获取生成器对象，这两个函数体内部都用了yield关键字，所以是生成器
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            nextobs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,  # hid_states_batch是一个元组，包含了actor和critic的hidden states
            masks_batch,  # 这里的hid_states_batch和masks_batch都是对循环网络的actor-critic用的，非循环网络的返回的都是None
        ) in generator:  # 使用for循环获取生成器的值
            # masks和hidden_states都是用在循环网络的actor-critic中的
            # NOTE: 注意后面三次计算都是计算了梯度的，而收集数据的时候是不计算梯度的
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])  # 使用最新的actor网络计算actions
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)  # 使用最新的actor网络计算actions的log_prob
            value_batch = self.actor_critic.evaluate(  # 使用最新的critic网络计算values
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL，这里使用KL散度是用来自适应调整学习率的，如果KL散度太大，说明两个分布的差距太大，需要减小学习率；如果KL散度小，说明两个分布的差距小，可以增大学习率
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(  # 两个高斯分布的KL散度计算公式，应该是从标准的KL散度公式推导出来的
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,  # 这里应该用dim=-1吧，axis应该不是torch的参数，但在代码运行过程中并没有报错
                    )
                    kl_mean = torch.mean(kl)

                    # KL散度越大，说明两个分布的差距越大，因此要减小学习率；KL散度越小，说明两个分布的差距越小，因此可以增大学习率
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            # 这里用的PPO2，也就是PPO-clip方法
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            # 使用value clip方法，限制critic相对于在收集数据时计算的target value的变化幅度。如果某个value的偏差特别大，我们认为critic更新的太大了，因此clip critic的输出
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )  # 这里的value_clipped是用来计算，更新后的critic计算的value和在收集数据时计算target_value的差距不能太大，如果太大就要clip，然后计算出clip后的当前的value
                value_losses = (value_batch - returns_batch).pow(2)  # 计算没有clip的value的loss
                value_losses_clipped = (value_clipped - returns_batch).pow(2)  # 计算clip后的value的loss
                value_loss = torch.max(value_losses, value_losses_clipped).mean()  # 选择较大的loss作为最终的value loss
                # 如何理解这里的value clip能够提高critic的稳定性呢？这里的value clip是用来限制critic的输出，使得critic的输出不能差距太大，这样可以减小critic的更新幅度，从而提高稳定性
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            ''' 
                计算Barlow Twins的loss
                1、首先计算encoder的对obs_batch和nextobs_batch的输出，Z_obs和Z_nextobs
                2、按照Barlow Twins论文中的公式(1)和(2)计算Barlow Twins的损失L_BT，注意公式(2)要减去均值，即对Z_obs和Z_nextobs标准化
                3、将L_BT加入到总的loss中，乘上barlowtwins_loss_coef
                4、设置限制encoder输出的损失，要让encoder输出的数据的数量级不要太大，
            '''
            # 求encoder的输出
            Z_obs = self.actor_critic.source_encoder_forward(obs_batch)
            Z_nextobs = self.actor_critic.target_encoder_forward(nextobs_batch)
            # 求Z_obs和Z_nextobs的均值
            Z_obs_mean = torch.mean(Z_obs, dim=0, keepdim=True)
            Z_nextobs_mean = torch.mean(Z_nextobs, dim=0, keepdim=True)
            # 对Z_obs和Z_nextobs减去均值
            Z_obs_zeromean = Z_obs - Z_obs_mean
            Z_nextobs_zeromean = Z_nextobs - Z_nextobs_mean
            # 求Z_obs和Z_nextobs的互相关矩阵
            Z_obs_sumsquare = torch.sum(Z_obs**2, dim=0, keepdim=True)
            Z_nextobs_sumsquare = torch.sum(Z_nextobs**2, dim=0, keepdim=True)
            Z_obs_norm = Z_obs_zeromean / torch.sqrt(Z_obs_sumsquare + 1e-5)
            Z_nextobs_norm = Z_nextobs_zeromean / torch.sqrt(Z_nextobs_sumsquare + 1e-5)
            cross_correlation_matrix = torch.matmul(Z_obs_norm.T, Z_nextobs_norm)
            # 设置mask，用来方便求Barlow Twins的loss
            mask = torch.eye(cross_correlation_matrix.size(0), dtype = torch.bool, device=cross_correlation_matrix.device)
            # 计算Barlow Twins的loss
            loss_barlowtwins = (torch.sum((torch.eye(cross_correlation_matrix.size(0),device=cross_correlation_matrix.device)-cross_correlation_matrix[mask])**2)\
                + self.barlowtwins_lambda * torch.sum(cross_correlation_matrix[~mask]**2)) / cross_correlation_matrix.size(0)  # 计算Barlow Twins的loss
            
            # 设置一个损失函数，如果encoder输出的数据超过[-1,1]范围，就会有loss，用来限制encoder输出幅度的损失
            # 后面除了方差，是因为我认为方差能够体现数据的幅度，如果方差很大，说明数据的幅度也很大，因此为了保证这个loss的数量级不要太大，除以方差
            loss_barlowtwinsencoder_output = (torch.sum(torch.nn.functional.relu(torch.abs(Z_obs) - 1)) + \
                        torch.sum(torch.nn.functional.relu(torch.abs(Z_nextobs) - 1)))/(torch.sqrt(torch.sum(Z_obs_sumsquare+Z_nextobs_sumsquare)))

            # IMPORTANT: 这里的loss是整个PPO的总loss，包括了actor的loss和critic的loss一起计算，并且在之后一起反向传播
            # 最后添加了熵的loss函数，期望能够获得一个比较大的熵，以增加actor输出的概率分布的不确定性(随机性，相当于正态分布的方差比较大)，以增加探索性
            # 在训练结束后的play时，actor的输出是actor的输出的mean值，不再使用std。这样可以减少噪音，提高性能
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + \
                    self.barlowtwins_loss_coef * loss_barlowtwins + self.encoder_output_loss_coef * loss_barlowtwinsencoder_output

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            # 如果梯度的范数(默认是2范数，可以通过参数指定)超过了max_grad_norm，就对梯度进行裁剪
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear() # NOTE: 执行clear操作，即将storage中的step置为0。不要忘了这一步

        return mean_value_loss, mean_surrogate_loss
