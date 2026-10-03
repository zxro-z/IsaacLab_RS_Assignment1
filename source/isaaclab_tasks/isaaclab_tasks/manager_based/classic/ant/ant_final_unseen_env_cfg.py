"""Frozen evaluation-only composite course; never train/tune against this task.

Deterministic analytic height geometry, converted with the repository-native HF
mesher. Not a copy of Dev obstacles/stones/gap or sampled uniform training noise.
All parameters are preregistered before the first mesh/environment is generated.
"""
import numpy as np

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from .ant_dev_ood_env_cfg import AntDevOODBaseEnvCfg
from .ant_ood_terrain_env_cfg import AntOODTerrainGenerator, AntOODTerrainSceneCfg, _terrain_cfg
from .ant_terrain_heightscan_env_cfg import height_scanner_cfg


COURSE_DESIGN = {
    "size_m": [40.0, 8.0], "spawn_local_xyz": [2.0, 4.0, 0.0],
    "horizontal_scale": 0.1, "vertical_scale": 0.005, "slope_threshold": 0.75,
    "layout": [10, 10], "terrain_seed": 2404, "evaluation_seed": 24,
    "spawn_flat_x": [0.0, 4.0],
    "ridge": {"x": [4.0, 10.0], "center_height": 0.02, "cross_amplitude": 0.04, "cross_scale": 1.5},
    "terraces": [
        {"x": [10.0, 12.2], "height": 0.04, "y_offset": -0.3},
        {"x": [12.2, 15.6], "height": 0.075, "y_offset": 0.4},
        {"x": [15.6, 18.0], "height": 0.025, "y_offset": -0.2},
    ],
    "terrace_lateral_half_extent": 3.6, "terrace_lateral_taper": 0.6,
    "basin": {"x": [18.0, 25.0], "depth": 0.07, "lateral_scale": 1.7},
    "landing": {"x": [25.0, 40.0], "amplitudes": [0.025, 0.015],
                "x_wavelengths": [4.0, 7.0], "y_wavelength": 5.0, "phase": float(np.pi / 3), "end_taper_m": 2.0},
    "lateral_envelope": "cos(pi*y/width)^2, y measured from centerline",
    "friction": {"static": 1.0, "dynamic": 1.0, "restitution": 0.0, "combine": "average"},
    "termination": {"torso_height": "root_z-local_terrain_z < 0.31 or missing terrain hit", "time_out": "original 16 seconds"},
    "novelty": "analytic asymmetric ridge + irregular connected offset terraces + smooth basin + deterministic Fourier landing",
    "no": ["isolated stepping stones", "explicit gaps/trenches", "discrete obstacle generator", "uniform noise generator", "curriculum"],
}


@height_field_to_mesh
def _course_height_field(difficulty, cfg):
    # The native decorator adds one border pixel on each side. Coordinates here
    # include that 0.1 m offset so analytic segment boundaries stay in tile space.
    d = COURSE_DESIGN
    x = (np.arange(round(cfg.size[0] / cfg.horizontal_scale)) + 1)[:, None] * cfg.horizontal_scale
    y = (np.arange(round(cfg.size[1] / cfg.horizontal_scale)) + 1)[None, :] * cfg.horizontal_scale - d["size_m"][1] / 2
    z = np.zeros((len(x), y.shape[1]))
    side = np.cos(np.pi * y / d["size_m"][1]) ** 2
    ridge = d["ridge"]
    a, b = ridge["x"]
    mask = (x >= a) & (x < b)
    z += np.where(mask, np.sin(np.pi * (x-a)/(b-a)) ** 2, 0) * (
        ridge["center_height"] + ridge["cross_amplitude"] * np.tanh(y / ridge["cross_scale"])) * side
    for terrace in d["terraces"]:
        a, b = terrace["x"]
        lateral = np.clip((d["terrace_lateral_half_extent"] - np.abs(y-terrace["y_offset"])) / d["terrace_lateral_taper"], 0, 1)
        z += ((x >= a) & (x < b)) * terrace["height"] * lateral
    basin = d["basin"]
    a, b = basin["x"]
    z -= np.where((x >= a) & (x < b), basin["depth"] * np.sin(np.pi*(x-a)/(b-a)) ** 2, 0) * np.exp(-(y / basin["lateral_scale"]) ** 2) * side
    landing = d["landing"]
    a, b = landing["x"]
    taper = np.minimum(np.clip((x-a)/landing["end_taper_m"], 0, 1), np.clip((b-x)/landing["end_taper_m"], 0, 1))
    z += taper * side * (
        landing["amplitudes"][0] * np.sin(2*np.pi*(x-a)/landing["x_wavelengths"][0])
        + landing["amplitudes"][1] * np.cos(2*np.pi*(x-a)/landing["x_wavelengths"][1] + landing["phase"])
        * np.sin(2*np.pi*y/landing["y_wavelength"]))
    return np.rint(z / cfg.vertical_scale).astype(np.int16)


def final_course_terrain(difficulty, cfg):
    meshes, _ = _course_height_field(difficulty, cfg)
    return meshes, np.array(COURSE_DESIGN["spawn_local_xyz"], dtype=float)


@configclass
class AntFinalCourseTerrainCfg(HfTerrainBaseCfg):
    function = final_course_terrain


@configclass
class AntFinalUnseenSceneCfg(AntOODTerrainSceneCfg):
    # Sensors exist for all candidates; only observation terms differ at runtime.
    height_scanner = height_scanner_cfg()
    terrain = _terrain_cfg(AntFinalCourseTerrainCfg(proportion=1.0, size=(40.0, 8.0)))
    terrain.terrain_generator = TerrainGeneratorCfg(
        class_type=AntOODTerrainGenerator, size=(40.0, 8.0), num_rows=10, num_cols=10,
        border_width=20.0, horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
        seed=2404, curriculum=False, difficulty_range=(1.0, 1.0), use_cache=False,
        sub_terrains={"final_composite": AntFinalCourseTerrainCfg(proportion=1.0, size=(40.0, 8.0))},
    )


@configclass
class AntFinalUnseenEnvCfg(AntDevOODBaseEnvCfg):
    """One physical scene for all policies; inherited stock reward/local fall test."""
    scene = AntFinalUnseenSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
