# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Fresh contact-observation experiment; inherit every baseline PPO parameter."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AntTerrainHeightScanContactPPORunnerCfg(AntPPORunnerCfg):
    experiment_name = "ant_terrainrandom_heightscan_contact_127d"
    run_name = "contact_127d"
    resume = False
