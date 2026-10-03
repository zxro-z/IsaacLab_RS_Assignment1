# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning-rate variants of the Ant RSL-RL PPO configuration."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AntPPOLowLRRunnerCfg(AntPPORunnerCfg):
    """Ant PPO configuration with a learning rate of 1e-4."""

    def __post_init__(self):
        self.algorithm.learning_rate = 1.0e-4


@configclass
class AntPPOHighLRRunnerCfg(AntPPORunnerCfg):
    """Ant PPO configuration with a learning rate of 1e-3."""

    def __post_init__(self):
        self.algorithm.learning_rate = 1.0e-3
