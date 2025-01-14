#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import contrastive_rl
from contrastive_rl.algorithms import PPOContrastive
from contrastive_rl.env import VecEnv # IMPORTANT: rsl-rl使用的是自己定义的VecEnv类格式，必须要按照VecEnv的接口来实现自己的环境类
from contrastive_rl.modules import ActorCriticContrastive, EmpiricalNormalization
from contrastive_rl.utils import store_code_state
from rich import print as rprint


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""
    # 根据后面对train_cfg的取值方法，train_cfg是一个dict吧，里面包含了训练的一些参数
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        obs, extras = self.env.get_observations() # 获取环境的观测值，obs是用来给actor用的，extras中如果有"critic"这个key，那么critic的值是给critic用的，否则critic的观测和actor一样
        num_obs = obs.shape[1] # 获取观测值的维度
        if "critic" in extras["observations"]: # 如果在extras中的observations中有"critic"这个key，那么critic的观测就是这个的取值，否则就和actor的观测一样
            num_critic_obs = extras["observations"]["critic"].shape[1] # 获取critic的观测维度
        else:
            num_critic_obs = num_obs
        # eval接收一个字符串，并执行这个字符串代表的表达式，如eval("ActorCritic")就是要生成一个ActorCritic类的执行对象
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic: ActorCriticContrastive = actor_critic_class(
            num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device) # 注意最后的to(self.device)是将actor_critic的参数放到device上，默认是cpu，但一般是cuda
        
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO，eval("PPO")就是要生成一个PPO类的执行对象
        self.alg: PPOContrastive = alg_class(actor_critic, device=self.device, **self.alg_cfg) # 创建PPO算法的实例
        self.num_steps_per_env = self.cfg["num_steps_per_env"] # 从配置参数中获取每个环境的步数，指的是每次迭代需要执行的环境步数
        self.save_interval = self.cfg["save_interval"] # 从配置参数中获取保存模型的间隔
        self.empirical_normalization = self.cfg["empirical_normalization"] # 从配置参数中获取是否使用经验归一化
        if self.empirical_normalization: # 如果要使用经验归一化，就会调用在modules/normalizer.py中定义的EmpiricalNormalization类
            # 这里所谓的经验归一化，就是根据训练过程中已有的数据，计算观测值的均值和方差，在训练过程中使用这个均值和方差来归一化观测值。均值和方差是在训练过程中不断更新的
            # 这个until参数是用来控制在什么时候停止更新均值和方差的，如果until是None，那么就会一直更新，否则当更新的样本数量超过until的时候就会停止更新
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            # torch.nn.Identity()是不对传递给他的数据进行任何处理，直接返回传递给他的数据
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        # init storage and model
        self.alg.init_storage( # 要注意调用init_storage函数，这个函数需要手动初始化，应该是用来储存replay_buffer的还是用来储存multi-step过程中的数据的？(大概率后者)
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir # 输出路径
        self.writer = None # 先给writer赋值为None
        self.tot_timesteps = 0 # 初始化总的执行步数
        self.tot_time = 0 # 初始化总的训练时间
        self.current_learning_iteration = 0 # 初始化当前的学习迭代次数
        self.git_status_repos = [contrastive_rl.__file__]

        # Terminal Output
        rprint("[green]=======================================================[/green]")
        rprint("[purple]You are using contrastive_rl, happy training.[/purple]")
        rprint("[green]=======================================================[/green]")

    # MODIFIED: 这个learn函数我做了修改，添加了only_positive_rewards参数，如果设置为True，那么只有正的reward才会被记录，负的reward会被clip成0
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False, only_positive_rewards: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None: # 如果log_dir不是None，且writer是None，那么就初始化writer
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard") # 字典的get方法，如果字典中有logger这个key，那么就返回这个key的值，否则返回"tensorboard"
            self.logger_type = self.logger_type.lower() # 将logger_type转换为小写

            if self.logger_type == "neptune":
                from contrastive_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from contrastive_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard": # 调用tensorboard的SummaryWriter作为信息的输出
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len: # 如果设置了init_at_random_ep_len为true，那么就会随机设置episode的长度
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs) # critic的obs试图从extras中获取，如果没有就和actor的obs一样
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device) # 将obs和critic_obs放到device上
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter): # 建立一个for循环，从start_iter到tot_iter
            start = time.time()
            # Rollout
            with torch.inference_mode(): # 设置只进行模型的前向传播，不进行梯度的计算
                for i in range(self.num_steps_per_env): # 在每个环境中执行num_steps_per_env次step
                    actions = self.alg.act(obs, critic_obs) # 虽然这里只获取了actions，但是在act函数中还会计算critic的value等其他各种值
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device)) # 在环境中执行一次step，返回的是obs, rewards, dones, infos
                    # 其实这里还可以参考其他论文对rewards的处理，设计一个exp函数来重塑reward的分布
                    if only_positive_rewards: # MODIFIED: 这里做了修改，如果only_positive_rewards为True，那么负的reward会被clip成0
                        rewards = torch.clamp(rewards, min=0.0)
                    # move to the right device
                    obs, critic_obs, rewards, dones = ( # 将obs, critic_obs, rewards, dones放到device上
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # perform normalization
                    obs = self.obs_normalizer(obs) # 对obs进行归一化
                    if "critic" in infos["observations"]: # 这里应该是用来处理非对称的observation的，有些时候critic可以观测到更多actor看不到的信息，以为critic只在训练的时候用到
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"]) # 对critic_obs进行归一化
                    else:
                        critic_obs = obs
                    # process the step
                    self.alg.process_env_step(obs, rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs) # 主要是为了计算return和advantage

            # algorithm类的update函数，包含了对loss的计算，以及对模型参数的更新
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals()) # NOTE: locals()函数会以字典类型返回当前作用域的全部局部变量，注意，类的属性(self关键字定义的属性)不会在这个字典中
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None): # 这里的保存模型是将所有需要保存的信息保存到一个字典中，然后调用torch.save保存到指定的路径(一个.pt文件)，这里使用的是保存为state_dict的形式
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(), # 注意，optimizer也是保存为state_dict的形式
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization: # 如果使用了经验归一化，那么还需要保存归一化模型的各类参数
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True): # 加载模型的函数
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None): # 获取推理模式的policy，这个policy是用来在测试的时候使用的，不会计算梯度，不会使用dropout等
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy  # 注意这里返回的是一个函数，并不是一个计算值

    def train_mode(self): # 将所有pytorch的网络模型都设置为train模式，这样在训练过程中就会使用如dropout层等
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self): # 与train_mode相反，将所有的网络模型设置为eval模式，这样在测试过程中就不会使用dropout层等
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
