"""Audit config equality and smoke-test one Ant terrain ablation; never trains.

Run each task in a fresh process using scripts/assignment/run.py. The final JSON
record includes geometry/origin hashes for cross-task reproducibility checks.
"""

import argparse
import hashlib
import json

from isaaclab.app import AppLauncher

TASK_DIMS = {
    "Isaac-Ant-Terrain-Ablation-Base-v0": 60,
    "Isaac-Ant-Terrain-Ablation-HeightScan-v0": 123,
    "Isaac-Ant-Terrain-Ablation-Contact-v0": 64,
    "Isaac-Ant-Terrain-Ablation-HeightScan-Contact-v0": 127,
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=TASK_DIMS, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 1:
    parser.error("--num_envs and --steps must be positive")
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from pxr import UsdGeom

import isaaclab_tasks  # noqa: F401
from isaaclab.envs.mdp import height_scan
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.classic.ant.ant_contact_observations import (
    CONTACT_FORCE_THRESHOLD,
    FOOT_BODY_NAMES,
    foot_contact_state,
)
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main():
    configs = {task: parse_env_cfg(task, device=args.device, num_envs=args.num_envs) for task in TASK_DIMS}
    agents = {task: load_cfg_from_registry(task, "rsl_rl_cfg_entry_point") for task in TASK_DIMS}
    baseline = AntEnvCfg()
    reference = None
    reference_policy = baseline.observations.policy.to_dict()
    reference_agent = None
    for task, cfg in configs.items():
        cfg.seed = args.seed
        whole = cfg.to_dict()
        observation = whole.pop("observations")
        if reference is None:
            reference = whole
            reference_agent = agents[task].to_dict()
        assert whole == reference, f"Non-observation config mismatch: {task}"
        assert agents[task].to_dict() == reference_agent, f"PPO mismatch: {task}"
        assert cfg.rewards.to_dict() == baseline.rewards.to_dict(), task
        assert cfg.actions.to_dict() == baseline.actions.to_dict(), task
        for name, value in reference_policy.items():
            assert observation["policy"][name] == value, (task, name)
        additions = set(observation["policy"]) - set(reference_policy)
        expected = set()
        if TASK_DIMS[task] in (123, 127):
            expected.add("height_scan")
        if TASK_DIMS[task] in (64, 127):
            expected.add("foot_contacts")
        assert additions == expected, (task, additions)
        assert gym.spec(task).kwargs["rsl_rl_cfg_entry_point"].endswith(":AntPPORunnerCfg")
    print("ABLATION_AUDIT non_observation_configs_identical=True PPO_identical=True", flush=True)
    print("REWARD_AUDIT_RESULT: MATCHES_BASIC_ANT", flush=True)

    cfg = configs[args.task]
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
    try:
        base = env.unwrapped
        obs = env.get_observations()["policy"]
        assert obs.shape == (args.num_envs, TASK_DIMS[args.task])
        assert env.num_actions == 8 and torch.isfinite(obs).all()
        for name in ("height_scanner", "feet_contacts", "torso_ground_ray"):
            sensor = base.scene[name]
            assert sensor._num_envs == args.num_envs, name
            assert sensor._timestamp_last_update.shape == (args.num_envs,), name
        ground = base.scene["torso_ground_ray"]
        clearance = base.scene["robot"].data.root_pos_w[:, 2] - ground.data.ray_hits_w[:, 0, 2]
        assert torch.allclose(clearance, torch.full_like(clearance, 0.5), atol=0.01)
        mesh = UsdGeom.Mesh(base.scene.stage.GetPrimAtPath(cfg.scene.terrain.prim_path + "/terrain/mesh"))
        geometry = hashlib.sha256()
        geometry.update(np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32).tobytes())
        geometry.update(np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).tobytes())
        origin_hash = hashlib.sha256(base.scene.env_origins.cpu().numpy().tobytes()).hexdigest()
        material_hash = hashlib.sha256(
            base.scene["robot"].root_physx_view.get_material_properties().cpu().numpy().tobytes()
        ).hexdigest()
        feet = SceneEntityCfg("feet_contacts", body_names=FOOT_BODY_NAMES.copy(), preserve_order=True)
        feet.resolve(base.scene)
        seen_contact = torch.zeros(4, dtype=torch.bool, device=env.device)
        seen_air = torch.zeros_like(seen_contact)
        rewards = []
        for _ in range(args.steps):
            actions = torch.zeros(args.num_envs, 8, device=env.device)
            observations, reward, dones, extras = env.step(actions)
            assert observations["policy"].shape == (args.num_envs, TASK_DIMS[args.task])
            assert torch.isfinite(observations["policy"]).all() and torch.isfinite(reward).all()
            assert reward.shape == (args.num_envs,) and reward.data_ptr() == base.reward_buf.data_ptr()
            scan = height_scan(base, SceneEntityCfg("height_scanner"), offset=0.5).clamp(-1.0, 1.0)
            contact = foot_contact_state(base, feet, CONTACT_FORCE_THRESHOLD)
            assert scan.shape == (args.num_envs, 63) and torch.isfinite(scan).all()
            assert ((contact == 0) | (contact == 1)).all()
            if TASK_DIMS[args.task] in (123, 127):
                assert torch.equal(observations["policy"][:, 60:123], scan)
            if TASK_DIMS[args.task] in (64, 127):
                assert torch.equal(observations["policy"][:, -4:], contact)
            seen_contact |= (contact == 1).any(dim=0)
            seen_air |= (contact == 0).any(dim=0)
            rewards.append(reward.mean().item())
        torch.cuda.synchronize() if str(env.device).startswith("cuda") else None
        assert seen_contact.all() and seen_air.all()
        result = {
            "task": args.task, "num_envs": args.num_envs, "seed": args.seed, "steps": args.steps,
            "policy_shape": list(observations["policy"].shape), "action_shape": list(actions.shape),
            "non_observation_configs_identical": True, "PPO_identical": True,
            "finite_observations_rewards": True, "initial_clearance": clearance.tolist(),
            "terrain_seed": cfg.scene.terrain.terrain_generator.seed,
            "terrain_mesh_sha256": geometry.hexdigest(), "env_origins_sha256": origin_hash,
            "robot_material_sha256": material_hash, "seen_contact": seen_contact.tolist(),
            "seen_air": seen_air.tolist(), "reward_mean": sum(rewards) / len(rewards), "PASS": True,
        }
        print("ABLATION_RESULT_JSON " + json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
