# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate one completed episode per environment for an RSL-RL checkpoint."""

"""Launch Isaac Sim Simulator first."""

import argparse
import hashlib
import json
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate one episode per environment with RSL-RL.")
parser.add_argument("--task", type=str, required=True, help="Name of the Isaac Lab task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path or URI to an RSL-RL checkpoint.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed. Uses the task default when omitted.")
parser.add_argument(
    "--foot_contacts", action="store_true", help="Append four foot-contact inputs (requires a matching checkpoint)."
)
parser.add_argument("--debug_contacts", action="store_true", help="Print env_0 contact diagnostics once.")
parser.add_argument("--preflight_steps", type=int, default=0, help="Run a short inference preflight instead of episode evaluation.")
parser.add_argument(
    "--result_json", type=str, default=None,
    help="Optional path for aggregate and per-environment first-episode results (refuses overwrite).",
)
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the evaluation.")
parser.add_argument("--video_length", type=int, default=960, help="Number of simulation steps to record.")
parser.add_argument(
    "--height_scan", action="store_true",
    help="Append the Ant height scan (original 63 rays unless --height_scan_pattern selects a variant).",
)
parser.add_argument("--height_scan_pattern", choices=("original", "dense", "forward_dense"), default=None)
parser.add_argument("--compact_terrain", action="store_true", help="Use the training 18-D compact descriptor; --height_scan additionally retains raw 63-D inputs.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RSL-RL configuration entry point."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.preflight_steps < 0:
    parser.error("--preflight_steps must be nonnegative")
if args_cli.compact_terrain and (args_cli.foot_contacts or args_cli.height_scan_pattern not in (None, "original")):
    parser.error("--compact_terrain requires Original scan and no --foot_contacts")

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
from pxr import UsdGeom

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from ant_observation_adapter import (
    configure_compact_terrain, configure_height_scan, scan_metadata, validate_checkpoint_inputs, validate_compact_scan,
)


def scene_signature(env_cfg, base):
    """Physical scene hashes for cross-policy comparisons; no simulation state changes."""
    terrain_mesh = UsdGeom.Mesh(base.scene.stage.GetPrimAtPath(env_cfg.scene.terrain.prim_path + "/terrain/mesh"))
    geometry_hash = hashlib.sha256()
    geometry_hash.update(np.asarray(terrain_mesh.GetPointsAttr().Get(), dtype=np.float32).tobytes())
    geometry_hash.update(np.asarray(terrain_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).tobytes())
    return {
        "terrain_mesh_sha256": geometry_hash.hexdigest(),
        "env_origins_sha256": hashlib.sha256(base.scene.env_origins.cpu().numpy().tobytes()).hexdigest(),
        "robot_material_sha256": hashlib.sha256(
            base.scene["robot"].root_physx_view.get_material_properties().cpu().numpy().tobytes()
        ).hexdigest(),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate the first completed episode in every parallel environment."""
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    if args_cli.result_json and os.path.exists(args_cli.result_json):
        raise FileExistsError(f"Refusing to overwrite evaluation result: {args_cli.result_json}")

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    configure_height_scan(env_cfg, args_cli.height_scan, args_cli.height_scan_pattern)
    configure_compact_terrain(env_cfg, args_cli.compact_terrain, args_cli.height_scan_pattern)
    if args_cli.compact_terrain:
        from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg

        assert env_cfg.rewards.to_dict() == AntEnvCfg().rewards.to_dict()
        print("REWARD_AUDIT_RESULT: MATCHES_BASIC_ANT")

    if args_cli.foot_contacts:
        from isaaclab_tasks.manager_based.classic.ant.ant_contact_observations import add_foot_contacts_for_evaluation

        add_foot_contacts_for_evaluation(env_cfg)
    if args_cli.debug_contacts and getattr(env_cfg.observations.policy, "foot_contacts", None) is None:
        raise ValueError("--debug_contacts requires --foot_contacts or a task with contact observations.")

    resume_path = retrieve_file_path(args_cli.checkpoint)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_one_episode"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording evaluation video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    inputs = validate_checkpoint_inputs(resume_path, env.unwrapped)
    scan = scan_metadata(env.unwrapped, args_cli.height_scan_pattern)
    print(f"[INFO] Checkpoint/scan inputs: {inputs}, {scan}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    compact_checks = validate_compact_scan(env.unwrapped, obs["policy"]) if args_cli.compact_terrain else {}
    if args_cli.preflight_steps:
        for _ in range(args_cli.preflight_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)
            assert tuple(obs["policy"].shape) == (env.num_envs, inputs["observation_dim"])
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rewards).all()
            if args_cli.compact_terrain:
                compact_checks = validate_compact_scan(env.unwrapped, obs["policy"])
        if str(env.device).startswith("cuda"):
            torch.cuda.synchronize()
        result = {"status": "preflight_pass", "task": args_cli.task, "checkpoint": resume_path,
                  "seed": int(env_cfg.seed), "num_envs": env.num_envs, "steps": args_cli.preflight_steps,
                  "finite_observation_reward": True, **inputs, **scan, **compact_checks, **scene_signature(env_cfg, env.unwrapped)}
        if args_cli.result_json:
            os.makedirs(os.path.dirname(os.path.abspath(args_cli.result_json)), exist_ok=True)
            with open(args_cli.result_json, "x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
        print("PREFLIGHT_RESULT " + json.dumps(result))
        env.close()
        return
    finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    returns = torch.zeros(env.num_envs, device=env.device)
    lengths = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    forward_displacement = torch.zeros(env.num_envs, device=env.device)
    time_out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fall = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    forward_velocity_sum = torch.zeros(env.num_envs, device=env.device)
    velocity_samples = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    termination_manager = env.unwrapped.termination_manager
    required_terms = {"time_out", "torso_height"}
    missing_terms = required_terms.difference(termination_manager.active_terms)
    if missing_terms:
        raise ValueError(f"Task does not expose the required Ant termination terms: {sorted(missing_terms)}")

    robot = env.unwrapped.scene["robot"]
    initial_x = robot.data.root_pos_w[:, 0].clone()
    latest_x = initial_x.clone()
    contact_debug_steps = 0
    contact_debug_logged = False

    while simulation_app.is_running() and not torch.all(finished):
        active = ~finished
        latest_x[active] = robot.data.root_pos_w[active, 0]
        forward_velocity_sum[active] += robot.data.root_lin_vel_w[active, 0]
        velocity_samples[active] += 1
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, _ = env.step(actions)

        # The manager-based environment has already auto-reset done environments here.  The reward and
        # TerminationManager term buffers still correspond to this step's pre-reset terminal transition.
        returns[active] += rewards[active]
        lengths[active] += 1
        newly_finished = dones.bool() & active
        if torch.any(newly_finished):
            time_out_this_step = termination_manager.get_term("time_out")
            fall_this_step = termination_manager.get_term("torso_height")
            forward_displacement[newly_finished] = latest_x[newly_finished] - initial_x[newly_finished]
            time_out[newly_finished] = time_out_this_step[newly_finished]
            fall[newly_finished] = fall_this_step[newly_finished]
            finished[newly_finished] = True

        if args_cli.debug_contacts and not contact_debug_logged:
            contact_debug_steps += 1
            if contact_debug_steps == 30 or torch.all(finished):
                from isaaclab_tasks.manager_based.classic.ant.ant_contact_observations import log_foot_contacts

                log_foot_contacts(env.unwrapped)
                contact_debug_logged = True

    if not torch.all(finished):
        raise RuntimeError("Evaluation stopped before every environment completed one episode.")

    time_out_count = time_out.sum().item()
    fall_count = fall.sum().item()
    print("\nEvaluation Results")
    print("------------------")
    print(f"task: {args_cli.task}")
    print(f"checkpoint: {resume_path}")
    print(f"seed: {env_cfg.seed}")
    print(f"num_envs: {env.num_envs}")
    print()
    print(f"episode_return_mean: {returns.mean().item():.3f}")
    print(f"episode_return_std: {returns.std(unbiased=False).item():.3f}")
    print(f"episode_return_min: {returns.min().item():.3f}")
    print(f"episode_return_max: {returns.max().item():.3f}")
    print()
    print(f"episode_length_mean: {lengths.float().mean().item():.3f}")
    print(f"episode_length_std: {lengths.float().std(unbiased=False).item():.3f}")
    print()
    print(f"forward_displacement_mean: {forward_displacement.mean().item():.3f}")
    print(f"forward_displacement_std: {forward_displacement.std(unbiased=False).item():.3f}")
    print()
    print(f"time_out_count: {time_out_count}")
    print(f"time_out_ratio: {time_out_count / env.num_envs:.3f}")
    print(f"fall_count: {fall_count}")
    print(f"fall_ratio: {fall_count / env.num_envs:.3f}")

    if args_cli.result_json:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.result_json)), exist_ok=True)
        mean_velocity = forward_velocity_sum / velocity_samples.clamp_min(1)
        progress_2m = forward_displacement >= 2.0
        progress_5m = forward_displacement >= 5.0
        result = {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "seed": int(env_cfg.seed),
            "num_envs": int(env.num_envs),
            "observation_dim": int(env.unwrapped.observation_manager.group_obs_dim["policy"][0]),
            "terrain_seed": getattr(env_cfg.scene.terrain.terrain_generator, "seed", None),
            **inputs, **scan, **compact_checks, **scene_signature(env_cfg, env.unwrapped),
            "aggregates": {
                "reward_mean": float(returns.mean().item()),
                "reward_std": float(returns.std(unbiased=False).item()),
                "episode_length_mean": float(lengths.float().mean().item()),
                "episode_length_std": float(lengths.float().std(unbiased=False).item()),
                "forward_displacement_mean": float(forward_displacement.mean().item()),
                "forward_displacement_std": float(forward_displacement.std(unbiased=False).item()),
                "forward_velocity_mean": float(mean_velocity.mean().item()),
                "timeout_count": int(time_out_count),
                "timeout_ratio": float(time_out_count / env.num_envs),
                "fall_count": int(fall_count),
                "fall_ratio": float(fall_count / env.num_envs),
                "progress_2m_count": int(progress_2m.sum().item()),
                "progress_2m_ratio": float(progress_2m.float().mean().item()),
                "progress_5m_count": int(progress_5m.sum().item()),
                "progress_5m_ratio": float(progress_5m.float().mean().item()),
            },
            "episodes": [
                {
                    "env_id": i,
                    "episode_return": float(returns[i].item()),
                    "episode_length": int(lengths[i].item()),
                    "forward_displacement_m": float(forward_displacement[i].item()),
                    "forward_velocity_mean_mps": float(mean_velocity[i].item()),
                    "time_out": bool(time_out[i].item()),
                    "fall": bool(fall[i].item()),
                    "progress_2m": bool(progress_2m[i].item()),
                    "progress_5m": bool(progress_5m[i].item()),
                }
                for i in range(env.num_envs)
            ],
        }
        # Exclusive creation protects prior results if separate jobs are launched accidentally.
        with open(args_cli.result_json, "x", encoding="utf-8") as result_file:
            json.dump(result, result_file, indent=2)
        print(f"[INFO] Per-episode evaluation data written to: {args_cli.result_json}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
