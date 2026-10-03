# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation-only ablations on one fixed Ant training terrain configuration.

All four scenes keep the same sensors, contact reporting and USD cloning. A
sensor's presence does not expose it to the policy: only observation terms vary.
The independent native terrain RNG is fixed, even if the policy/reset seed varies.
Existing tasks and historical training configurations are not modified.
"""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .ant_env_cfg import ObservationsCfg
from .ant_terrain_heightscan_env_cfg import (
    AntFootContactObservationsCfg,
    AntHeightScanContactObservationsCfg,
    AntHeightScanObservationsCfg,
    AntTerrainHeightScanContactSceneCfg,
    AntTerrainRandomHeightScanContactEnvCfg,
    _training_terrain_cfg,
)

TRAINING_TERRAIN_SEED = 42


def ablation_training_terrain_cfg():
    """Reuse the historical distribution, with an explicit shared geometry seed."""
    terrain = _training_terrain_cfg()
    terrain.terrain_generator.seed = TRAINING_TERRAIN_SEED
    return terrain


@configclass
class AntTerrainAblationSceneCfg(AntTerrainHeightScanContactSceneCfg):
    terrain = ablation_training_terrain_cfg()


@configclass
class AntTerrainAblationBaseEnvCfg(AntTerrainRandomHeightScanContactEnvCfg):
    """60-D proprioceptive policy; common sensors are not policy inputs."""

    scene: InteractiveSceneCfg = AntTerrainAblationSceneCfg(
        num_envs=4096, env_spacing=5.0, clone_in_fabric=False
    )
    observations: ObservationsCfg = ObservationsCfg()


@configclass
class AntTerrainAblationHeightScanEnvCfg(AntTerrainAblationBaseEnvCfg):
    """123-D: original 60 inputs plus the unchanged 63-ray height scan."""

    observations: AntHeightScanObservationsCfg = AntHeightScanObservationsCfg()


@configclass
class AntTerrainAblationContactEnvCfg(AntTerrainAblationBaseEnvCfg):
    """64-D: original 60 inputs plus four ordered binary foot contacts."""

    observations: AntFootContactObservationsCfg = AntFootContactObservationsCfg()


@configclass
class AntTerrainAblationHeightScanContactEnvCfg(AntTerrainAblationBaseEnvCfg):
    """127-D: original inputs, height scan, then four binary foot contacts."""

    observations: AntHeightScanContactObservationsCfg = AntHeightScanContactObservationsCfg()
