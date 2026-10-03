# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_name", type=str, default=None, help="Label for a unique video recording directory.")
parser.add_argument(
    "--follow_robot", action="store_true", help="Track env_0's robot root with a stable third-person camera."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--height_scan",
    action="store_true",
    help="Append the training height scan when playing the new 123-D Ant policy on an existing task.",
)
parser.add_argument(
    "--height_scan_pattern", choices=("original", "dense", "forward_dense"), default=None,
    help="Enable a specific Ant scan layout; --height_scan alone retains the original layout.",
)
parser.add_argument("--compact_terrain", action="store_true", help="Use the training compact descriptor; --height_scan also retains raw Original scan.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--foot_contacts", action="store_true", help="Append four foot-contact inputs (requires a matching checkpoint)."
)
parser.add_argument("--debug_contacts", action="store_true", help="Print env_0 contact diagnostics once, after 30 steps.")
parser.add_argument("--rollout_telemetry_dir", type=str, default=None,
                    help="Opt-in single-env first-episode telemetry; requires --video and an unused output directory.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.compact_terrain and (args_cli.foot_contacts or args_cli.height_scan_pattern not in (None, "original")):
    parser.error("--compact_terrain requires Original scan and no --foot_contacts")
if args_cli.rollout_telemetry_dir and (not args_cli.video or args_cli.num_envs != 1):
    parser.error("--rollout_telemetry_dir requires --video --num_envs 1")
if args_cli.video_name is not None and (
    not args_cli.video_name
    or args_cli.video_name in {".", ".."}
    or "/" in args_cli.video_name
    or "\\" in args_cli.video_name
):
    parser.error("--video_name must be a non-empty directory label, not a path.")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import tempfile
import time
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    from ant_observation_adapter import configure_compact_terrain, configure_height_scan, validate_checkpoint_inputs

    configure_height_scan(env_cfg, args_cli.height_scan, args_cli.height_scan_pattern)
    configure_compact_terrain(env_cfg, args_cli.compact_terrain, args_cli.height_scan_pattern)

    if args_cli.foot_contacts:
        from isaaclab_tasks.manager_based.classic.ant.ant_contact_observations import add_foot_contacts_for_evaluation

        add_foot_contacts_for_evaluation(env_cfg)
    if args_cli.debug_contacts and getattr(env_cfg.observations.policy, "foot_contacts", None) is None:
        raise ValueError("--debug_contacts requires --foot_contacts or a task with contact observations.")

    if args_cli.follow_robot:
        # Native tracking translates with the live root but does not inherit body
        # roll/pitch/yaw, keeping a stable oblique chase view for the Ant's +X motion.
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (-4.0, -3.0, 2.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.1)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_root = os.path.join(log_dir, "videos", "play")
        os.makedirs(video_root, exist_ok=True)
        video_label = args_cli.video_name or "play"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Atomic reservation also prevents collisions between simultaneous recordings.
        video_folder = tempfile.mkdtemp(prefix=f"{video_label}_{timestamp}_", dir=video_root)
        video_kwargs = {
            "video_folder": video_folder,
            "name_prefix": video_label,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording playback video to: {video_folder}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if task_name.startswith("Isaac-Ant-") and agent_cfg.class_name == "OnPolicyRunner":
        print(f"[INFO] Ant checkpoint inputs: {validate_checkpoint_inputs(resume_path, env.unwrapped)}")

    if args_cli.follow_robot and env.unwrapped.viewport_camera_controller is not None:
        # The wrapper resets the robot onto its assigned terrain origin. Center the
        # camera there immediately, rather than showing its authored spawn location.
        env.unwrapped.viewport_camera_controller.update_view_to_asset_root("robot")
        if args_cli.video:
            # Initialize the RGB render product before capturing the first frame.
            # These are render-only updates: no extra physics or policy steps.
            for _ in range(3):
                env.unwrapped.render()

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if not args_cli.rollout_telemetry_dir:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    telemetry = None
    if args_cli.rollout_telemetry_dir:
        from rollout_telemetry import RolloutTelemetry

        telemetry = RolloutTelemetry(env.unwrapped, args_cli.rollout_telemetry_dir, resume_path,
                                     video_folder, args_cli.height_scan_pattern)
    timestep = 0
    contact_debug_steps = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if telemetry is not None:
                telemetry.begin(obs)
            # env stepping
            obs, rewards, dones, _ = env.step(actions)
            if telemetry is not None:
                telemetry.end(rewards[0].item(), bool(dones[0]))
        if telemetry is not None and bool(dones[0]):
            break
        if args_cli.debug_contacts:
            contact_debug_steps += 1
            if contact_debug_steps == 30:
                from isaaclab_tasks.manager_based.classic.ant.ant_contact_observations import log_foot_contacts

                log_foot_contacts(env.unwrapped)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    if telemetry is not None:
        telemetry.finalize()


if __name__ == "__main__":
    # run the main function
    try:
        main()
    finally:
        simulation_app.close()
