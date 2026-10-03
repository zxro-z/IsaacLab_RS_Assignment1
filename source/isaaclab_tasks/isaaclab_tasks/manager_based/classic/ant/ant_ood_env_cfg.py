# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation-only out-of-distribution Ant environment configurations.

These configurations are intended for evaluation only, not training or model selection.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .ant_env_cfg import AntEnvCfg, EventCfg, mdp


@configclass
class AntOODLowFrictionEventCfg(EventCfg):
    """Baseline events plus evaluation-only low-friction randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.10, 0.25),
            "dynamic_friction_range": (0.05, 0.20),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


@configclass
class AntOODHighFrictionEventCfg(EventCfg):
    """Baseline events plus evaluation-only high-friction randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.10, 1.40),
            "dynamic_friction_range": (0.90, 1.10),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


@configclass
class AntOODMassPushEventCfg(EventCfg):
    """Baseline events plus evaluation-only mass and push perturbations."""

    mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    push = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-1.0, 1.0),
        },
    )


@configclass
class AntOODLowFrictionEnvCfg(AntEnvCfg):
    """Evaluation-only Ant environment with friction below the training distribution."""

    events: AntOODLowFrictionEventCfg = AntOODLowFrictionEventCfg()


@configclass
class AntOODHighFrictionEnvCfg(AntEnvCfg):
    """Evaluation-only Ant environment with friction above the training distribution."""

    events: AntOODHighFrictionEventCfg = AntOODHighFrictionEventCfg()


@configclass
class AntOODMassPushEnvCfg(AntEnvCfg):
    """Evaluation-only Ant environment with mass scaling and intermittent torso pushes."""

    events: AntOODMassPushEventCfg = AntOODMassPushEventCfg()
