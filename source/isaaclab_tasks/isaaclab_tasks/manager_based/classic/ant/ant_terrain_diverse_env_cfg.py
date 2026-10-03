"""Terrain-distribution-only experiment with the unchanged 127-D policy inputs.

Random uniform height fields avoid the native discrete-obstacle, stepping-stone,
and gap primitives reserved for Dev OOD. This is interpolated roughness, not a
replica of separated footholds. Only the central spawn platform is flattened.
"""

import math

from isaaclab.terrains import HfRandomUniformTerrainCfg
from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from .ant_terrain_ablation_env_cfg import (
    AntTerrainAblationHeightScanContactEnvCfg,
    ablation_training_terrain_cfg,
)


@height_field_to_mesh
def random_uniform_with_spawn_platform(difficulty, cfg):
    """Use native height sampling/interpolation and mesh conversion unchanged.

    Native random_uniform_terrain has no platform option. Flatten the centered
    2 m square before native meshing; leave the surrounding random field intact.
    The original AntTrainingTerrainGenerator retains spawn-origin normalization.
    """
    heights = random_uniform_terrain.__wrapped__(difficulty, cfg)
    half = math.ceil(cfg.platform_width / (2 * cfg.horizontal_scale))
    cx, cy = heights.shape[0] // 2, heights.shape[1] // 2
    heights[cx - half : cx + half + 1, cy - half : cy + half + 1] = 0
    return heights


@configclass
class AntSpawnRandomUniformTerrainCfg(HfRandomUniformTerrainCfg):
    function = random_uniform_with_spawn_platform
    platform_width: float = 2.0


def diverse_training_terrain_cfg():
    terrain = ablation_training_terrain_cfg()
    primitives = terrain.terrain_generator.sub_terrains
    # Preserve 3:2:1 easy/medium/harder weighting exactly within the 50% slopes.
    proportions = {
        "flat": 0.20,
        "up_easy": 0.125, "down_easy": 0.125,
        "up_medium": 1 / 12, "down_medium": 1 / 12,
        "up_harder": 1 / 24, "down_harder": 1 / 24,
        "low_up_stairs": 0.10,
    }
    for name, proportion in proportions.items():
        primitives[name].proportion = proportion
    primitives["random_uniform"] = AntSpawnRandomUniformTerrainCfg(
        proportion=0.20, noise_range=(-0.05, 0.05), noise_step=0.025,
        downsampled_scale=0.4, platform_width=2.0,
    )
    assert math.isclose(sum(cfg.proportion for cfg in primitives.values()), 1.0, abs_tol=1e-15)
    return terrain


@configclass
class AntTerrainDiverseHeightScanContactEnvCfg(AntTerrainAblationHeightScanContactEnvCfg):
    """Same observation, robot, resets, reward, termination and PPO; new terrain mix."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = diverse_training_terrain_cfg()
