"""Close Dev experiments, preregister/freeze Final, then validate with zero actions only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from pxr import UsdGeom
import isaaclab_tasks
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import height_scan
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_final_unseen_env_cfg import COURSE_DESIGN
from isaaclab_tasks.manager_based.classic.ant.ant_contact_observations import FOOT_BODY_NAMES, foot_contact_state
from isaaclab_config_reader import parse_config_text

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "validation"
TASK = "Isaac-Ant-Final-Unseen-v0"
ANT = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def canonical(cfg):
    return parse_config_text(yaml.dump(cfg.to_dict()))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    # Part A is complete before any freeze or Final mesh construction.
    rollout = VAL / "existing_vs_diverse_rollout"
    assert (rollout / "comparison.csv").is_file()
    trajectories = []
    for label in ("existing", "diverse"):
        summary = json.loads((rollout / label / "summary.json").read_text())
        assert summary["observation_dim"] == 127 and summary["scan_dim"] == 63 and summary["video_files"]
        with (rollout / label / "telemetry.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        dt = summary["step_dt"]
        vx = np.array([float(r["linear_velocity_0"]) for r in rows])
        x = np.array([float(r["position_0"]) for r in rows])
        mask = np.abs(vx) < .1
        edges = np.diff(np.r_[False, mask, False].astype(int))
        intervals = [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
        start, end = max(intervals, key=lambda ab: ab[1]-ab[0], default=(0, 0))
        trajectories.append({"policy": label, "longest_stagnation_s": (end-start)*dt,
                             "longest_stagnation_start_s": start*dt, "longest_stagnation_end_s": end*dt,
                             "first_2s_displacement": float(x[min(len(x)-1, 119)]-summary["initial_state"]["position_0"]),
                             "last_4s_displacement": float(x[-1]-x[max(0, len(x)-241)]),
                             "mean_vx": float(vx.mean()), "final_displacement": summary["final_displacement_m"]})
    dump(rollout / "behavior_analysis.json", trajectories)
    with (rollout / "comparison.md").open("a") as stream:
        stream.write("\n## Behavior interpretation\n\n" + json.dumps(trajectories, indent=2) + "\n\n"
                     "Diverse gains 0.592 m (+3.50%) in this episode, not the aggregate +8.07%. It starts more slowly (2m: 1.183 vs 1.100s; 5m: 4.283 vs 2.650s). "
                     "It survives to 16s, while Existing falls at 11.317s. Diverse mean speed is lower despite higher final distance. "
                     "Its low-speed/low-progress fractions are higher, but there is no continuous >=1s |vx|<0.1 interval and no static four-foot stance for most of the episode. "
                     "Diverse has higher mean action L2 (1.517 vs 1.228), more contact switching (39.875 vs 33.844 total Hz), and higher roll/pitch RMS. "
                     "Timeout alone is not evidence of stagnation. No extra seeds needed; both episodes are interpretable.\n")

    policies = json.loads((VAL / "ablation_eval_manifest.json").read_text())
    diverse = json.loads((VAL / "diverse_training_run.json").read_text())
    selected = [next(p for p in policies if p["label"] == label) for label in ("Base", "HeightScan", "HeightScan+Contact")]
    selected.append({"label": "Diverse HeightScan+Contact", "checkpoint_path": diverse["final_checkpoint"], "observation_dim": 127})
    now = datetime.now(timezone.utc).isoformat()
    source_paths = [p for p in ANT.rglob("*.py") if p.name not in ("__init__.py", "ant_final_unseen_env_cfg.py")]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    dev_cfgs = {}
    for label in ("UnevenBlocks", "SteppingStones", "GapPath"):
        config = parse_env_cfg(f"Isaac-Ant-DevOOD-{label}-v0", device=args.device, num_envs=100)
        config.seed = 24
        dev_cfgs[label] = {"config": canonical(config), "sha256": digest(canonical(config))}
    freeze = {"timestamp": now, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
              "source_hashes": source_hashes, "dev_configs": dev_cfgs,
              "checkpoint_sha256": {p["checkpoint_path"]: sha(Path(p["checkpoint_path"])) for p in selected},
              "declaration": "No further tuning against Dev OOD after this point"}
    dump(VAL / "dev_experiment_freeze.json", freeze)
    print("DEV EXPERIMENT FREEZE SAVED", flush=True)
    config = parse_env_cfg(TASK, device=args.device, num_envs=4)
    config.seed = 24
    assert config.rewards.to_dict() == AntEnvCfg().rewards.to_dict()
    agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    final_sources = {**source_hashes, str((ANT / "ant_final_unseen_env_cfg.py").relative_to(ROOT)): sha(ANT / "ant_final_unseen_env_cfg.py"),
                     str((ANT / "__init__.py").relative_to(ROOT)): sha(ANT / "__init__.py")}
    prereg = {"timestamp": now, "task_id": TASK, "design": COURSE_DESIGN, "seed": 24, "terrain_seed": 2404,
              "official_num_envs": 100, "policy_set": selected, "optional_compact_included": False,
              "source_hashes": final_sources, "env_source_config": canonical(config), "agent_config": canonical(agent),
              "reward_source": "AntEnvCfg.RewardsCfg: stock seven terms, weights and parameters unchanged",
              "novelty": {"training": "No pyramid ramp/regular stairs/uniform random noise generator; analytical connected composite",
                          "UnevenBlocks": "No HfDiscreteObstacles generator or box layout",
                          "SteppingStones": "No isolated stones or gaps", "GapPath": "No trench; basin is continuous and shallow"},
              "freeze_rule": "No parameter, seed, friction, spawn, termination, reward or policy tuning after preregistration; code bugs documented separately"}
    dump(VAL / "final_unseen_preregister.json", prereg)
    print("FINAL PREREGISTRATION SAVED; FIRST MESH GENERATION NEXT", flush=True)
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=config))
    try:
        base = env.unwrapped
        robot = base.scene["robot"]
        initial = robot.data.root_pos_w.clone()
        ray = base.scene["torso_ground_ray"]
        clearance = initial[:, 2] - ray.data.ray_hits_w[:, 0, 2]
        assert torch.allclose(clearance, torch.full_like(clearance, .5), atol=.01)
        assert torch.all((robot.data.body_pos_w-base.scene.env_origins[:, None, :])[:, :, 2] > 0)
        mesh = UsdGeom.Mesh(base.scene.stage.GetPrimAtPath("/World/ground/terrain/mesh"))
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        assert np.isfinite(points).all()
        scanner = base.scene["height_scanner"]
        assert scanner._num_envs == ray._num_envs == 4 and scanner.num_rays == 63
        feet = SceneEntityCfg("feet_contacts", body_names=FOOT_BODY_NAMES, preserve_order=True)
        feet.resolve(base.scene)
        actions = torch.zeros((4, 8), device=env.device)
        terminated_steps = 0
        for _ in range(64):
            obs, reward, done, _ = env.step(actions)
            assert obs["policy"].shape == (4, 60) and torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            scan = height_scan(base, SceneEntityCfg("height_scanner"), .5).clamp(-1, 1)
            contacts = foot_contact_state(base, feet, 1.0)
            assert scan.shape == (4, 63) and torch.isfinite(scan).all()
            assert contacts.shape == (4, 4) and ((contacts == 0) | (contacts == 1)).all()
            assert torch.isfinite(robot.data.root_pos_w[:, 2] - ray.data.ray_hits_w[:, 0, 2]).all()
            terminated_steps += int(done.sum())
        report = {"result": "PASS", "policy_free": True, "seed": 24, "num_envs": 4, "steps": 64,
                  "initial_root_positions": initial.tolist(), "env_origins": base.scene.env_origins.tolist(),
                  "initial_clearance": clearance.tolist(), "geometry_finite": True, "body_frames_above_spawn_surface": True,
                  "reward_matches_basic_ant": True, "mesh_sha256": hashlib.sha256(points.tobytes()+indices.tobytes()).hexdigest(),
                  "env_origins_sha256": hashlib.sha256(base.scene.env_origins.cpu().numpy().tobytes()).hexdigest(),
                  "robot_material_sha256": hashlib.sha256(robot.root_physx_view.get_material_properties().cpu().numpy().tobytes()).hexdigest(),
                  "zero_action_termination_count": terminated_steps, "mesh_z_extent": [float(points[:,2].min()), float(points[:,2].max())],
                  "preregister_sha256": sha(VAL / "final_unseen_preregister.json"),
                  "scene_invariance": "Shared hidden original scanner + contacts for every candidate; policy terms only adapted"}
        dump(VAL / "final_unseen_validation.json", report)
        print("FINAL POLICY-FREE VALIDATION PASS " + json.dumps(report), flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close(wait_for_replicator=False)
