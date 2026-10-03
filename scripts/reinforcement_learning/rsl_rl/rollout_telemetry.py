"""Opt-in, read-only first-episode Ant telemetry; never changes policy inputs or rewards."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.manager_based.classic.ant.ant_contact_observations import (
    CONTACT_FORCE_THRESHOLD, FOOT_BODY_NAMES, foot_contact_state,
)

from ant_observation_adapter import scan_metadata, validate_checkpoint_inputs
from rollout_metrics import window_metrics


class RolloutTelemetry:
    """Capture terminal state at the native pre-reset hook, not the reset pose.

    No recorder terms are added: the normal observation compute count is unchanged.
    The original pre-reset callback is forwarded, and restored after recording.
    """

    def __init__(self, env, directory, checkpoint, video_folder, pattern):
        if env.num_envs != 1:
            raise ValueError("Rollout telemetry requires exactly one environment")
        self.env, self.dt = env, env.step_dt
        self.path = Path(directory).resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        self.rows, self.scans, self.hits = [], [], []
        self.robot = env.scene["robot"]
        self.foot_ids, ordered = self.robot.find_bodies(FOOT_BODY_NAMES, preserve_order=True)
        assert ordered == FOOT_BODY_NAMES
        self.sensor_cfg = SceneEntityCfg("feet_contacts", body_names=FOOT_BODY_NAMES.copy(), preserve_order=True)
        self.sensor_cfg.resolve(env.scene)
        self.scan_info = scan_metadata(env, pattern)
        self.scan_slice = None
        offset = 0
        for name, dims in zip(env.observation_manager.active_terms["policy"],
                              env.observation_manager.group_obs_term_dim["policy"]):
            size = int(np.prod(dims))
            if name == "height_scan":
                self.scan_slice = slice(offset, offset + size)
            offset += size
        self.initial = self.snapshot()
        self.metadata = {
            "task": env.spec.id if getattr(env, "spec", None) else "Isaac-Ant-DevOOD-SteppingStones-v0",
            "seed": env.cfg.seed, "num_envs": 1, "step_dt": self.dt,
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "video_directory": str(Path(video_folder).resolve()),
            "foot_order": FOOT_BODY_NAMES, "joint_order": self.robot.joint_names,
            "contact_threshold_N": CONTACT_FORCE_THRESHOLD,
            "initial_state": self.initial, "env_origin": env.scene.env_origins[0].cpu().tolist(),
            "stagnation_definition": "fraction of post-step samples with abs(world vx) < 0.1 m/s",
            "backward_definition": "fraction of post-step samples with world vx < -0.1 m/s",
            "contact_switching_definition": "per-foot binary transitions / episode duration (Hz); aggregate is sum of foot rates",
            "scan_definition": "decision-time policy height_scan, unchanged offset 0.5 and clip [-1,1]; raw terrain hits saved separately",
            "discontinuity_definition": "raw hit-Z difference > 0.10 m between immediate grid neighbors; count distinct incident rays, finite hit pairs only",
            "timing": "row k: action/scan from decision k-1; state after control step k; terminal state captured before auto-reset",
            "video_terminal_caveat": "RecordVideo captures after env.step auto-reset; its last frame may show reset. No second episode is stepped.",
            **validate_checkpoint_inputs(checkpoint, env), **self.scan_info,
        }
        self.terminal = None
        self.armed = False
        self.recorder = env.recorder_manager
        self.original_pre_reset = self.recorder.record_pre_reset

        def pre_reset(env_ids, *args, **kwargs):
            if self.armed and 0 in env_ids:
                self.terminal = self.snapshot()
            return self.original_pre_reset(env_ids, *args, **kwargs)

        env.recorder_manager.record_pre_reset = pre_reset

    def snapshot(self):
        robot = self.robot.data
        terrain_z = float(self.env.scene["torso_ground_ray"].data.ray_hits_w[0, 0, 2])
        angles = euler_xyz_from_quat(robot.root_quat_w[:1])
        force = self.env.scene["feet_contacts"].data.net_forces_w[0, self.sensor_cfg.body_ids]
        root = robot.root_pos_w[0]
        result = {
            "terrain_z": terrain_z if np.isfinite(terrain_z) else None,
            "ground_hit_valid": bool(np.isfinite(terrain_z)),
            "clearance": float(root[2]) - terrain_z if np.isfinite(terrain_z) else None,
            "roll": float(angles[0][0]), "pitch": float(angles[1][0]), "yaw": float(angles[2][0]),
            "termination": bool(self.env.termination_manager.terminated[0]),
            "timeout": bool(self.env.termination_manager.time_outs[0]),
            "fall": bool(self.env.termination_manager.get_term("torso_height")[0]),
        }
        arrays = {
            "position": root, "linear_velocity": robot.root_lin_vel_w[0],
            "angular_velocity": robot.root_ang_vel_w[0], "action": self.env.action_manager.action[0],
            "joint_position": robot.joint_pos[0], "joint_velocity": robot.joint_vel[0],
            "foot_force_N": torch.linalg.vector_norm(force, dim=-1),
            "foot_contact": foot_contact_state(self.env, self.sensor_cfg, CONTACT_FORCE_THRESHOLD)[0],
            "foot_position": robot.body_pos_w[0, self.foot_ids].flatten(),
            "torso_to_foot_world": (robot.body_pos_w[0, self.foot_ids] - root).flatten(),
        }
        for name, tensor in arrays.items():
            for i, value in enumerate(tensor.cpu().tolist()):
                result[f"{name}_{i}"] = value
        return result

    def begin(self, obs):
        self.armed, self.terminal = True, None
        self.scan_stats = {}
        if self.scan_slice is None:
            return
        scanner = self.env.scene["height_scanner"]
        values = obs["policy"][0, self.scan_slice].detach().cpu().numpy().copy()
        hits = scanner.data.ray_hits_w[0].cpu().numpy().copy()
        xy = scanner.ray_starts[0, :, :2].cpu().numpy()
        nx, ny = len(np.unique(xy[:, 0])), len(np.unique(xy[:, 1]))
        assert nx * ny == len(values)
        z = hits[:, 2].reshape(ny, nx)
        incident = np.zeros_like(z, dtype=bool)
        diffs = []
        for axis in (0, 1):
            a = [slice(None), slice(None)]
            b = a.copy()
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            za, zb = z[tuple(a)], z[tuple(b)]
            valid = np.isfinite(za) & np.isfinite(zb)
            delta = np.abs(za[valid] - zb[valid])
            diffs.extend(delta.tolist())
            discontinuous = np.zeros_like(za, dtype=bool)
            discontinuous[valid] = delta > 0.10
            incident[tuple(a)] |= discontinuous
            incident[tuple(b)] |= discontinuous
        self.scan_stats = {"scan_min": float(values.min()), "scan_max": float(values.max()),
                           "scan_mean": float(values.mean()), "scan_std": float(values.std()),
                           "scan_adjacent_difference_max_m": max(diffs, default=0.0),
                           "scan_discontinuous_ray_count": int(incident.sum()),
                           "scan_missing_hit_count": int((~np.isfinite(hits[:, 2])).sum())}
        self.scans.append(values)
        self.hits.append(hits)
        self.scan_xy = xy.copy()

    def end(self, reward, done):
        state = self.terminal if done else self.snapshot()
        if state is None:
            raise RuntimeError("Missing pre-reset terminal snapshot")
        self.armed = False
        step = len(self.rows) + 1
        state.update(step=step, simulation_time=step * self.dt, step_reward=float(reward),
                     cumulative_reward=(self.rows[-1]["cumulative_reward"] if self.rows else 0.0) + float(reward),
                     action_l2=float(np.linalg.norm([state[f"action_{i}"] for i in range(8)])),
                     joint_velocity_l2=float(np.linalg.norm([state[f"joint_velocity_{i}"] for i in range(8)])),
                     contact_count=sum(state[f"foot_contact_{i}"] for i in range(4)), **self.scan_stats)
        self.rows.append(state)

    def finalize(self):
        self.recorder.record_pre_reset = self.original_pre_reset
        with (self.path / "telemetry.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        if self.scans:
            np.savez_compressed(self.path / "height_scan.npz", policy_scan=np.asarray(self.scans),
                                ray_hits_w=np.asarray(self.hits), ray_xy_body=self.scan_xy)
        values = lambda key: np.asarray([r[key] for r in self.rows], dtype=float)
        vx, roll, pitch = values("linear_velocity_0"), values("roll"), values("pitch")
        displacement = values("position_0") - self.initial["position_0"]
        contacts = np.asarray([[r[f"foot_contact_{i}"] for i in range(4)] for r in self.rows])
        duration = len(self.rows) * self.dt
        angles = lambda a: {"rms_rad": float(np.sqrt(np.mean(a*a))), "max_abs_rad": float(np.abs(a).max())}
        counts = contacts.sum(axis=1)
        summary = {**self.metadata, "episode_length": len(self.rows), "duration_s": duration,
                   "episode_reward_total": self.rows[-1]["cumulative_reward"],
                   "final_displacement_m": float(displacement[-1]), "max_displacement_m": float(displacement.max()),
                   "mean_vx": float(vx.mean()), "max_vx": float(vx.max()),
                   "fall": self.rows[-1]["fall"], "timeout": self.rows[-1]["timeout"],
                   "termination": "torso_height" if self.rows[-1]["fall"] else "time_out" if self.rows[-1]["timeout"] else "recording_limit",
                   "terminal_ground_hit_valid": self.rows[-1]["ground_hit_valid"],
                   "terminal_terrain_z": self.rows[-1]["terrain_z"],
                   "minimum_clearance_m": min(r["clearance"] for r in self.rows if r["clearance"] is not None),
                   "roll": angles(roll), "pitch": angles(pitch),
                   "mean_action_l2": float(values("action_l2").mean()), "max_action_l2": float(values("action_l2").max()),
                   "mean_joint_velocity_l2": float(values("joint_velocity_l2").mean()),
                   "max_joint_velocity_l2": float(values("joint_velocity_l2").max()),
                   "contact_duty_ratio": dict(zip(FOOT_BODY_NAMES, contacts.mean(axis=0).tolist())),
                   "mean_contact_count": float(counts.mean()),
                   "contact_count_ratio": {str(i): float(np.mean(counts == i)) for i in range(5)},
                   "contact_switching_hz": dict(zip(FOOT_BODY_NAMES, (np.abs(np.diff(contacts, axis=0)).sum(axis=0)/duration).tolist())),
                   "stagnation_ratio": float(np.mean(np.abs(vx) < 0.1)),
                   "backward_ratio": float(np.mean(vx < -0.1)), "stagnation_intervals": [], "terminal_windows": {}}
        mask = np.abs(vx) < 0.1
        edges = np.diff(np.r_[False, mask, False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if (end-start)*self.dt >= 1.0:
                summary["stagnation_intervals"].append(self.window(start, end))
        window = round(1.0/self.dt)
        summary["one_second_low_progress_ratio"] = float(np.mean(
            (displacement[window:] - displacement[:-window]) < 0.1)) if len(displacement)>window else None
        for seconds in (2.0, 1.0, 0.5):
            summary["terminal_windows"][str(seconds)] = self.window(max(0, len(self.rows)-round(seconds/self.dt)), len(self.rows))
        if self.scans:
            scan_delta = values("scan_adjacent_difference_max_m")
            summary["scan_discontinuity_vx_correlation"] = float(np.corrcoef(scan_delta, vx)[0, 1]) if scan_delta.std()>0 and vx.std()>0 else None
        summary["video_files"] = [str(p) for p in Path(summary["video_directory"]).glob("*.mp4")]
        with (self.path / "summary.json").open("x") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        print(f"[TELEMETRY] {self.path}: {summary['termination']}, displacement={displacement[-1]:.3f} m")

    def window(self, start, end):
        return window_metrics(self.rows, self.initial, self.dt, start, end, self.scan_slice is not None)
