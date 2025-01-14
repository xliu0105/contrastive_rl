#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup

setup(
    name="contrastive_rl",
    version="0.1.0",
    packages=find_packages(),
    author="",
    maintainer="",
    maintainer_email="",
    url="https://github.com/xliu0105/contrastive_rl.git",
    license="BSD-3",
    description="Fast and simple RL algorithms with contrastive learning module implemented in pytorch.",
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4",
        "GitPython",
        "onnx",
        "rich",
    ],
)
