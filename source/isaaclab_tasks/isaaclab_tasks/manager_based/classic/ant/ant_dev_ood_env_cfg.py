# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable evaluation-only Development OOD terrains for Ant sensor studies.

These fixed scenes are for repeated development comparison, not policy training.
Final unseen evaluation scenes are intentionally not defined here.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .ant_contact_observations import configure_foot_contacts
from .ant_ood_terrain_env_cfg import (
    AntOODTerrainEnvCfg,
    AntOODTerrainGenerator,
    AntOODTerrainSceneCfg,
    _terrain_cfg,
)


def _dev_ood_terrain_cfg(sub_terrain, seed: int):
    """Use one fixed realization per task, shared across all observation variants."""
    terrain = _terrain_cfg(sub_terrain)
    terrain.terrain_generator = TerrainGeneratorCfg(
        class_type=AntOODTerrainGenerator,
        size=(8.0, 8.0),
        num_rows=10,
        num_cols=10,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.6, 0.6),
        use_cache=False,
        seed=seed,
        sub_terrains={"dev_ood": sub_terrain},
    )
    return terrain


@configclass
class AntDevOODUnevenBlocksSceneCfg(AntOODTerrainSceneCfg):
    """Irregular positive/negative cuboid obstacles; central spawn remains flat."""

    terrain = _dev_ood_terrain_cfg(
        terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.35, 0.80),
            obstacle_height_range=(0.04, 0.12),
            num_obstacles=14,
            platform_width=2.0,
        ),
        seed=2401,
    )


@configclass
class AntDevOODSteppingStonesSceneCfg(AntOODTerrainSceneCfg):
    """Separated, mildly height-varying footholds with shallow gaps."""

    terrain = _dev_ood_terrain_cfg(
        terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=1.0,
            stone_height_max=0.06,
            stone_width_range=(0.65, 0.90),
            stone_distance_range=(0.08, 0.16),
            holes_depth=-0.30,
            platform_width=2.0,
        ),
        seed=2402,
    )


@configclass
class AntDevOODGapPathSceneCfg(AntOODTerrainSceneCfg):
    """A moderate perimeter trench around a broad central platform."""

    terrain = _dev_ood_terrain_cfg(
        terrain_gen.MeshGapTerrainCfg(
            proportion=1.0,
            gap_width_range=(0.35, 0.35),
            platform_width=2.4,
        ),
        seed=2403,
    )


@configclass
class AntDevOODBaseEnvCfg(AntOODTerrainEnvCfg):
    """Shared stock Ant reward, terrain-relative fall term, and fixed scene physics."""

    scene: InteractiveSceneCfg = AntDevOODUnevenBlocksSceneCfg(
        num_envs=4096, env_spacing=5.0, clone_in_fabric=False
    )

    def __post_init__(self):
        super().__post_init__()
        configure_foot_contacts(self.scene)


@configclass
class AntDevOODUnevenBlocksEnvCfg(AntDevOODBaseEnvCfg):
    """Development OOD task: native irregular blocks, evaluation only."""


@configclass
class AntDevOODSteppingStonesEnvCfg(AntDevOODBaseEnvCfg):
    """Development OOD task: native stepping stones, evaluation only."""

    scene: InteractiveSceneCfg = AntDevOODSteppingStonesSceneCfg(
        num_envs=4096, env_spacing=5.0, clone_in_fabric=False
    )


@configclass
class AntDevOODGapPathEnvCfg(AntDevOODBaseEnvCfg):
    """Development OOD task: native perimeter gap, evaluation only."""

    scene: InteractiveSceneCfg = AntDevOODGapPathSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
