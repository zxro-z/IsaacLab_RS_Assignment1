"""Evaluation-only transfer test on the teammate v3 mixed terrain."""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from .ant_final_unseen_env_cfg import AntFinalUnseenEnvCfg, AntFinalUnseenSceneCfg


# Terrain parameters reconstructed from /home/zxro/teammate_ant_rl/source/ant/ant_env_cfg.py
# and logs/rsl_rl/ant/v3_depth/params/env.yaml in that repository. Only geometry and
# distribution are transferred; reward, termination, and robot remain RS definitions.
TEAMMATE_V3_TERRAIN_CFG = ROUGH_TERRAINS_CFG.copy()
TEAMMATE_V3_TERRAIN_CFG.size = (10.0, 10.0)
TEAMMATE_V3_TERRAIN_CFG.num_rows = 20
TEAMMATE_V3_TERRAIN_CFG.num_cols = 10
TEAMMATE_V3_TERRAIN_CFG.border_width = 2.0
TEAMMATE_V3_TERRAIN_CFG.seed = 42
TEAMMATE_V3_TERRAIN_CFG.curriculum = False
TEAMMATE_V3_TERRAIN_CFG.difficulty_range = (0.0, 1.0)
TEAMMATE_V3_TERRAIN_CFG.horizontal_scale = 0.1
TEAMMATE_V3_TERRAIN_CFG.vertical_scale = 0.005
TEAMMATE_V3_TERRAIN_CFG.slope_threshold = 0.75

for name in ("pyramid_stairs", "pyramid_stairs_inv", "boxes", "hf_pyramid_slope", "hf_pyramid_slope_inv"):
    TEAMMATE_V3_TERRAIN_CFG.sub_terrains[name].proportion = 0.2
    TEAMMATE_V3_TERRAIN_CFG.sub_terrains[name].platform_width = 1.0
TEAMMATE_V3_TERRAIN_CFG.sub_terrains["random_rough"].proportion = 0.0

for name in ("pyramid_stairs", "pyramid_stairs_inv"):
    sub_terrain = TEAMMATE_V3_TERRAIN_CFG.sub_terrains[name]
    sub_terrain.step_height_range = (0.03, 0.07)
    sub_terrain.step_width = 0.3
    sub_terrain.border_width = 0.0

boxes = TEAMMATE_V3_TERRAIN_CFG.sub_terrains["boxes"]
boxes.grid_width = 0.45
boxes.grid_height_range = (0.02, 0.10)

for name in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
    sub_terrain = TEAMMATE_V3_TERRAIN_CFG.sub_terrains[name]
    sub_terrain.slope_range = (0.0, 0.20)
    sub_terrain.border_width = 0.0


@configclass
class AntTeammateTerrainTransferSceneCfg(AntFinalUnseenSceneCfg):
    """Retain RS Final sensors, robot, and material; replace only the terrain mesh."""

    terrain = AntFinalUnseenSceneCfg().terrain.copy()
    terrain.class_type = TerrainImporter
    terrain.terrain_generator = TEAMMATE_V3_TERRAIN_CFG


@configclass
class AntTeammateTerrainTransferEnvCfg(AntFinalUnseenEnvCfg):
    """RS Final evaluation semantics on the teammate v3 terrain distribution."""

    scene: InteractiveSceneCfg = AntTeammateTerrainTransferSceneCfg(
        num_envs=4096, env_spacing=5.0, clone_in_fabric=False
    )
