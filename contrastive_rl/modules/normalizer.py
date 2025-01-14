#  Copyright (c) 2020 Preferred Networks, Inc.
#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability. eps是一个很小的数, 用于防止除0错误
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.count = 0

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0] # 这个是获取的batch size的大小，反映在rsl-rl的强化学习结构中，应该是环境的数量
        self.count += count_x # 更新总样本的数量
        rate = count_x / self.count # 计算当前批次的样本数占总处理的样本数的比例，用于更新mean和var时的权重

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True) # 沿着第0维度计算方差，unbiased=False表示计算的是有偏方差(除以n)，keepdim=True表示保持维度
        mean_x = torch.mean(x, dim=0, keepdim=True) # 沿着第0维度计算均值，keepdim=True表示保持维度
        delta_mean = mean_x - self._mean # 计算新均值和当前均值的差
        self._mean += rate * delta_mean # 利用计算出的rate更新mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        # NOTE: 这里的均值和方差的更新公式是根据Welford's online algorithm来的
        self._std = torch.sqrt(self._var) # 标准差是方差的开方

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean
