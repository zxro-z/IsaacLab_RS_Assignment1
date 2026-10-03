"""Observation-only compact terrain ablations; all physical/PPO settings unchanged."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .ant_compact_terrain_observations import compact_terrain_descriptor
from .ant_env_cfg import ObservationsCfg
from .ant_terrain_ablation_env_cfg import AntTerrainAblationHeightScanEnvCfg
from .ant_terrain_heightscan_env_cfg import AntHeightScanObservationsCfg


def compact_term():
    # Means already use clipped native heights. Do not clip edge strengths to 1.
    return ObsTerm(
        func=compact_terrain_descriptor,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
    )


@configclass
class AntCompactObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        compact_terrain = compact_term()

    policy: PolicyCfg = PolicyCfg()


@configclass
class AntHeightScanCompactObservationsCfg(AntHeightScanObservationsCfg):
    @configclass
    class PolicyCfg(AntHeightScanObservationsCfg.PolicyCfg):
        compact_terrain = compact_term()

    policy: PolicyCfg = PolicyCfg()


@configclass
class AntTerrainAblationCompactEnvCfg(AntTerrainAblationHeightScanEnvCfg):
    """60 base + 18 compact inputs; Original scanner remains internal, not policy input."""

    observations: AntCompactObservationsCfg = AntCompactObservationsCfg()


@configclass
class AntTerrainAblationHeightScanCompactEnvCfg(AntTerrainAblationHeightScanEnvCfg):
    """60 base + unchanged 63 raw inputs + 18 compact inputs; exactly one look-ahead scan."""

    observations: AntHeightScanCompactObservationsCfg = AntHeightScanCompactObservationsCfg()
