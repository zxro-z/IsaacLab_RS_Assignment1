# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Spatial height-scan ablations on the unchanged common Ant training environment."""

import torch

from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass

from .ant_terrain_ablation_env_cfg import AntTerrainAblationHeightScanEnvCfg


def ant_height_scan_grid_pattern(cfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Native RayCaster pattern interface; X varies fastest, as in GridPatternCfg's xy order.

    Coordinates are sensor-local, before the unchanged (0.8, 0, 20) offset.
    Explicit axes allow different X/Y resolutions and nonuniform forward spacing.
    No terrain values or observation preprocessing are handled by this function.
    """
    x = torch.tensor(cfg.x_coordinates, dtype=torch.float32, device=device)
    y = torch.tensor(cfg.y_coordinates, dtype=torch.float32, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    starts = torch.zeros((grid_x.numel(), 3), dtype=torch.float32, device=device)
    starts[:, 0] = grid_x.flatten()
    starts[:, 1] = grid_y.flatten()
    directions = torch.zeros_like(starts)
    directions[:] = torch.tensor(cfg.direction, dtype=starts.dtype, device=device)
    return starts, directions


@configclass
class AntHeightScanGridPatternCfg(PatternBaseCfg):
    """Cartesian product of explicit ordered axes, with downward parallel rays."""

    func = ant_height_scan_grid_pattern
    x_coordinates: tuple[float, ...] = ()
    y_coordinates: tuple[float, ...] = ()
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)


def dense_height_scan_pattern_cfg() -> AntHeightScanGridPatternCfg:
    """13 x 9 = 117 rays; identical 1.6 x 1.2 m footprint, dx=1.6/12, dy=0.15 m.

    Native GridPatternCfg has one isotropic resolution; explicit axes retain both
    original footprint endpoints with this compact anisotropic uniform grid.
    """
    return AntHeightScanGridPatternCfg(
        x_coordinates=tuple(-0.8 + i * (1.6 / 12) for i in range(13)),
        y_coordinates=tuple(-0.6 + i * 0.15 for i in range(9)),
    )


def forward_dense_height_scan_pattern_cfg() -> AntHeightScanGridPatternCfg:
    """13 x 7 = 91 rays; X=0..0.8 every 0.1 m, then 1.0..1.6 every 0.2 m.

    Lateral positions remain the original seven positions, 0.2 m apart.
    The X=0.8 seam is present once, with no duplicated rays.
    """
    forward_x = tuple(i * 0.1 for i in range(9)) + (1.0, 1.2, 1.4, 1.6)
    return AntHeightScanGridPatternCfg(
        x_coordinates=tuple(x - 0.8 for x in forward_x),
        y_coordinates=tuple(-0.6 + i * 0.2 for i in range(7)),
    )


@configclass
class AntTerrainAblationHeightScanDenseEnvCfg(AntTerrainAblationHeightScanEnvCfg):
    """177-D policy: original 60 terms plus 117 uniform-grid height samples."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.pattern_cfg = dense_height_scan_pattern_cfg()


@configclass
class AntTerrainAblationHeightScanForwardDenseEnvCfg(AntTerrainAblationHeightScanEnvCfg):
    """151-D policy: original 60 terms plus 91 forward-focused height samples."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.pattern_cfg = forward_dense_height_scan_pattern_cfg()
