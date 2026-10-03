# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (similar to OpenAI Gym Ant-v2).
"""

import gymnasium as gym

# Frozen final evaluation only. Never train or select hyperparameters on this task.
gym.register(
    id="Isaac-Ant-Final-Unseen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.classic.ant.ant_final_unseen_env_cfg:AntFinalUnseenEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

# Evaluation only: teammate v3 terrain geometry with RS Final dynamics and metrics.
gym.register(
    id="Isaac-Ant-Teammate-Terrain-Transfer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.classic.ant.ant_teammate_terrain_transfer_env_cfg:AntTeammateTerrainTransferEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

from . import agents

# Fair observation-only terrain ablations; every task uses the original Ant PPO.
for variant, env_class in (
    ("Base", "AntTerrainAblationBaseEnvCfg"),
    ("HeightScan", "AntTerrainAblationHeightScanEnvCfg"),
    ("Contact", "AntTerrainAblationContactEnvCfg"),
    ("HeightScan-Contact", "AntTerrainAblationHeightScanContactEnvCfg"),
):
    gym.register(
        id=f"Isaac-Ant-Terrain-Ablation-{variant}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_terrain_ablation_env_cfg:{env_class}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )

# New training ablation; preserve every existing task and the original PPO config.
gym.register(
    id="Isaac-Ant-TerrainRandom-HeightScan-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_terrain_heightscan_env_cfg:AntTerrainRandomHeightScanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

##
# Register Gym environments.
##

# Dedicated fresh 127-D run; keep the original 123-D task/config available.
gym.register(
    id="Isaac-Ant-TerrainRandom-HeightScan-Contact-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_terrain_heightscan_env_cfg:AntTerrainRandomHeightScanContactEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_contact_cfg:AntTerrainHeightScanContactPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:AntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Ant-LR-Low-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:AntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_lr_cfg:AntPPOLowLRRunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-LR-High-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:AntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_lr_cfg:AntPPOHighLRRunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-FrictionRandom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_friction_random_env_cfg:AntFrictionRandomEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

# Evaluation-only OOD tasks. Do not use these task IDs for training or model selection.
gym.register(
    id="Isaac-Ant-OOD-LowFriction-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_ood_env_cfg:AntOODLowFrictionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-OOD-HighFriction-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_ood_env_cfg:AntOODHighFrictionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-OOD-MassPush-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_ood_env_cfg:AntOODMassPushEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

# Evaluation-only terrain OOD tasks. Do not use these task IDs for training or model selection.
gym.register(
    id="Isaac-Ant-OOD-RampUp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_ood_terrain_env_cfg:AntOODRampUpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-OOD-RampDown-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_ood_terrain_env_cfg:AntOODRampDownEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-OOD-Stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_ood_terrain_env_cfg:AntOODStairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

# Fixed Development OOD scenes for repeated evaluation/model comparison only.
for task_id, env_class in (
    ("Isaac-Ant-DevOOD-UnevenBlocks-v0", "AntDevOODUnevenBlocksEnvCfg"),
    ("Isaac-Ant-DevOOD-SteppingStones-v0", "AntDevOODSteppingStonesEnvCfg"),
    ("Isaac-Ant-DevOOD-GapPath-v0", "AntDevOODGapPathEnvCfg"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_dev_ood_env_cfg:{env_class}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )

# Spatial height-scan ablations: only the look-ahead ray pattern changes.
for variant, env_class in (
    ("Dense", "AntTerrainAblationHeightScanDenseEnvCfg"),
    ("ForwardDense", "AntTerrainAblationHeightScanForwardDenseEnvCfg"),
):
    gym.register(
        id=f"Isaac-Ant-Terrain-Ablation-HeightScan-{variant}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_heightscan_variants_env_cfg:{env_class}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )

# Terrain-diversity experiment: retain the existing HeightScan+Contact observation.
gym.register(
    id="Isaac-Ant-Terrain-Diverse-HeightScan-Contact-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_terrain_diverse_env_cfg:AntTerrainDiverseHeightScanContactEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

# Compact observation ablations reuse the Original scanner and physical scene.
for variant, env_class in (
    ("Compact", "AntTerrainAblationCompactEnvCfg"),
    ("HeightScan-Compact", "AntTerrainAblationHeightScanCompactEnvCfg"),
):
    gym.register(
        id=f"Isaac-Ant-Terrain-Ablation-{variant}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_compact_terrain_env_cfg:{env_class}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )
