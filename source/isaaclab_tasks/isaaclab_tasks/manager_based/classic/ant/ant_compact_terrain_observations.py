"""Deterministic 18-D descriptors of the unchanged Original 9x7 height scan.

Native GridPatternCfg(ordering="xy") flattens X fastest: flat[iy * 9 + ix].
X groups: Near=0:3, Mid=3:6, Far=6:9. In the torso yaw frame, +X is
forward and +Y is left: Left=5:7, Center=2:5, Right=0:2.
Each sector emits (mean_height, edge_strength), Near/Mid/Far then
Left/Center/Right. Edges include ONLY pairs with both endpoints in the sector;
cross-sector boundaries are deliberately excluded. Sector ray counts are 6/9/6.
Means use the native clipped heights [-1, 1]; edge magnitudes range [0, 2].
No extra sensor, randomness, host transfer, or hidden state is used.
"""

import torch

X_GROUPS = (("near", 0, 3), ("mid", 3, 6), ("far", 6, 9))
Y_GROUPS = (("left", 5, 7), ("center", 2, 5), ("right", 0, 2))
DESCRIPTOR_ORDER = tuple(
    f"{x}_{y}_{feature}"
    for x, _, _ in X_GROUPS
    for y, _, _ in Y_GROUPS
    for feature in ("mean", "edge")
)


def compact_from_height_scan(heights: torch.Tensor) -> torch.Tensor:
    """Reduce already normalized/clipped (N, 63) heights to (N, 18).

    Grid indices are [env, iy, ix], NOT [env, ix, iy]. Input dtype and device
    are preserved. Finite inputs produce finite outputs without sanitization.
    """
    if heights.ndim != 2 or heights.shape[1] != 63:
        raise ValueError(f"Expected Original (N, 63) scan, got {tuple(heights.shape)}")
    grid = heights.reshape(-1, 7, 9)
    features = []
    for _, x0, x1 in X_GROUPS:
        for _, y0, y1 in Y_GROUPS:
            sector = grid[:, y0:y1, x0:x1]
            dx = (sector[:, :, 1:] - sector[:, :, :-1]).abs().flatten(1)
            dy = (sector[:, 1:, :] - sector[:, :-1, :]).abs().flatten(1)
            features.extend((sector.mean(dim=(1, 2)), torch.cat((dx, dy), dim=1).amax(dim=1)))
    return torch.stack(features, dim=1)


def compact_terrain_descriptor(env, sensor_cfg, offset: float = 0.5) -> torch.Tensor:
    """Reuse native height_scan and its Original observation clipping convention."""
    from isaaclab.envs.mdp import height_scan

    return compact_from_height_scan(height_scan(env, sensor_cfg, offset).clamp(-1.0, 1.0))
