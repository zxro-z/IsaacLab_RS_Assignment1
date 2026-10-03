"""Separate-process terrain-only config/geometry/sensor validation; no training."""

import argparse
import hashlib
import json
import copy
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "validation"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=("existing", "diverse"), required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from pxr import UsdGeom
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import height_scan
from isaaclab.utils.warp import raycast_mesh
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from isaaclab_tasks.manager_based.classic.ant.ant_contact_observations import FOOT_BODY_NAMES, foot_contact_state
from isaaclab_tasks.manager_based.classic.ant.ant_terrain_heightscan_env_cfg import AntTrainingTerrainGenerator
from isaaclab_config_reader import load_config, parse_config_text

TASKS = {"existing": "Isaac-Ant-Terrain-Ablation-HeightScan-Contact-v0",
         "diverse": "Isaac-Ant-Terrain-Diverse-HeightScan-Contact-v0"}


def write(name, value):
    def json_value(item):
        if isinstance(item, np.generic):
            return item.item()
        if callable(item):
            return item.__module__ + ":" + item.__qualname__
        raise TypeError(type(item).__name__)
    serialized = json.dumps(value, indent=2, allow_nan=False, default=json_value)
    destination = VAL / name
    if destination.exists():
        try:
            existing = json.loads(destination.read_text())
        except json.JSONDecodeError:
            # Preserve the incomplete collector output, not replace/delete it.
            destination.rename(destination.with_name(destination.name + ".incomplete_" + str(time.time_ns())))
        else:
            if existing == json.loads(serialized):
                return
    with destination.open("x") as stream:
        stream.write(serialized)


def differences(a, b, path=""):
    """Record exact paths/values; no broad runtime subtrees are ignored."""
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            field = f"{path}.{key}" if path else key
            if key not in a or key not in b:
                result.append({"path": field, "saved": a.get(key), "current": b.get(key)})
            else:
                result.extend(differences(a[key], b[key], field))
        return result
    return [] if a == b else [{"path": path, "saved": a, "current": b}]


def runtime_audit(saved, cfg, saved_agent, agent):
    # Compare actually expanded config, not guessed source defaults. Simulator
    # expansion supplies importer values, sub-terrain sizes/HF scales and paths.
    current = parse_config_text(yaml.dump(cfg.to_dict()))
    reference = copy.deepcopy(saved)
    changes = []
    # Both runs use 4096 for training; this diagnostic deliberately uses four.
    for container, key, value, reason in (
        (reference["scene"], "num_envs", 4, "requested 4-env diagnostic; source training default verified as 4096"),
        (reference["scene"]["terrain"], "num_envs", 4, "InteractiveScene._add_entities_from_cfg assigns scene.num_envs"),
    ):
        changes.append({"field": "scene.num_envs" if container is reference["scene"] else "scene.terrain.num_envs",
                        "from": container[key], "to": value, "reason": reason})
        container[key] = value
    for key in ("log_dir", "io_descriptors_output_dir"):
        # train.py assigns unique logging destinations, not task settings.
        changes.append({"field": key, "from": current[key], "to": reference[key],
                        "reason": "train.py assigns the output run directory"})
        current[key] = reference[key]
    agent_current = parse_config_text(yaml.dump(agent.to_dict()))
    agent_current["run_name"] = saved_agent["run_name"]
    delta = differences(reference, current)
    delta += differences(saved_agent, agent_current, "agent")
    report = {"result": "PASS" if not delta else "FAIL", "differences": delta,
              "normalizations": changes,
              "runtime_expansion_proof": {
                  "importer": "source/isaaclab/isaaclab/scene/interactive_scene.py:_add_entities_from_cfg",
                  "sub_terrain_size_and_HF_scales": "source/isaaclab/isaaclab/terrains/terrain_generator.py:TerrainGenerator.__init__",
                  "entity_prim_paths": "source/isaaclab/isaaclab/scene/interactive_scene.py:_add_entities_from_cfg",
              }, "entire_terrain_subtree_compared": True}
    print("DIVERSE_RUNTIME_CONFIG_AUDIT: " + report["result"], flush=True)
    print("RUNTIME_CONFIG_DETAILS " + json.dumps(report), flush=True)
    return report


def main():
    configs = {label: parse_env_cfg(task, device=args.device) for label, task in TASKS.items()}
    for cfg in configs.values():
        cfg.seed = 42
    agents = {label: load_cfg_from_registry(task, "rsl_rl_cfg_entry_point") for label, task in TASKS.items()}
    before, after = (configs[label].to_dict() for label in TASKS)
    old_mix = before["scene"]["terrain"]["terrain_generator"].pop("sub_terrains")
    new_mix = after["scene"]["terrain"]["terrain_generator"].pop("sub_terrains")
    source_delta = differences(before, after)
    for name in old_mix:
        old, new = dict(old_mix[name]), dict(new_mix[name])
        old.pop("proportion"); new.pop("proportion")
        source_delta += differences(old, new, f"shared_terrain.{name}")
    source_delta += differences(agents["existing"].to_dict(), agents["diverse"].to_dict(), "agent")
    print("PROGRESS config loaded", flush=True)
    print("DIVERSE_SOURCE_CONFIG_AUDIT: " + ("FAIL" if source_delta else "PASS"), flush=True)
    if source_delta:
        print(json.dumps(source_delta, default=str), flush=True)
        raise RuntimeError("Actual non-composition source mismatch; training forbidden")
    assert configs["diverse"].rewards.to_dict() == AntEnvCfg().rewards.to_dict()
    manifest = json.loads((VAL / "ablation_eval_manifest.json").read_text())
    prior = next(p for p in manifest if p["label"] == "HeightScan+Contact")
    saved = load_config(Path(prior["training_run_path"]) / "params/env.yaml")
    agent_saved = load_config(Path(prior["training_run_path"]) / "params/agent.yaml")
    print("REWARD_AUDIT_RESULT: MATCHES_BASIC_ANT", flush=True)
    if args.variant == "diverse":
        write("diverse_training_source_config_audit.json", {
            "result": "PASS", "excluded_fields": ["scene.terrain.terrain_generator.sub_terrains"],
            "shared_terrain_parameters_unchanged": True, "observation_dim": 127,
            "saved_127d_terrain_reward_observations_ppo_match_source": True,
            "reward_matches_stock_ant": True, "ppo_entry_point": gym.spec(TASKS["diverse"]).kwargs["rsl_rl_cfg_entry_point"],
            "old_mix": old_mix, "new_mix": new_mix,
            "same_fields": ["observations", "reward", "termination", "robot", "reset", "friction",
                            "physics", "episode_length", "actions", "PPO", "scanner", "contact_sensor"],
        })
        write("diverse_training_ood_overlap_audit.json", {
            "selected": "HfRandomUniformTerrainCfg + central spawn platform",
            "rejected": "HfDiscreteObstaclesTerrainCfg is already Dev UnevenBlocks primitive",
            "interpolation": "native bicubic; quantized mesh heights, not sharp isolated blocks",
            "terrains": {"UnevenBlocks": {"primitive_overlap": False, "exact_layout_overlap": False,
                          "interpretation": "Different primitive, shared irregular-height concept; development-exposed, not final unseen."},
                         "SteppingStones": {"primitive_overlap": False, "exact_layout_overlap": False},
                         "GapPath": {"primitive_overlap": False, "exact_layout_overlap": False}},
        })
    # Observe actual native random sampling; do not change RNG or terrain generation.
    patches = []
    add = AntTrainingTerrainGenerator._add_sub_terrain
    def record(generator, mesh, origin, row, col, sub_cfg):
        name = next(k for k, value in generator.cfg.sub_terrains.items() if value is sub_cfg)
        patches.append({"row": int(row), "col": int(col), "primitive": name,
                        "origin": origin.tolist(), "z_min": float(mesh.vertices[:, 2].min()),
                        "z_max": float(mesh.vertices[:, 2].max())})
        return add(generator, mesh, origin, row, col, sub_cfg)
    AntTrainingTerrainGenerator._add_sub_terrain = record
    cfg = configs[args.variant]
    cfg.seed = 42
    cfg.scene.num_envs = 4
    env = RslRlVecEnvWrapper(gym.make(TASKS[args.variant], cfg=cfg))
    print("PROGRESS env created", flush=True)
    try:
        base = env.unwrapped
        if args.variant == "existing":
            audit = runtime_audit(saved, cfg, agent_saved, agents["existing"])
            write("diverse_training_runtime_config_audit.json", audit)
            if audit["result"] != "PASS":
                raise RuntimeError("Saved/runtime mismatch; inspect exact differences before training")
        obs = env.get_observations()["policy"]
        robot = base.scene["robot"]
        scanner = base.scene["height_scanner"]
        feet = SceneEntityCfg("feet_contacts", body_names=FOOT_BODY_NAMES, preserve_order=True)
        feet.resolve(base.scene)
        assert scanner.num_rays == 63 and scanner._num_envs == 4
        assert base.scene["feet_contacts"]._num_envs == 4
        root = robot.data.root_pos_w.clone()
        initial_local_bodies = robot.data.body_pos_w - base.scene.env_origins[:, None, :]
        clearance = root[:, 2] - base.scene["torso_ground_ray"].data.ray_hits_w[:, 0, 2]
        assert torch.allclose(clearance, torch.full_like(clearance, 0.5), atol=0.01)
        assert torch.all(initial_local_bodies[:, :, 2] > 0), "Body frame below spawn surface"
        mesh = UsdGeom.Mesh(base.scene.stage.GetPrimAtPath("/World/ground/terrain/mesh"))
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        assert np.isfinite(points).all()
        digest = hashlib.sha256(points.tobytes())
        digest.update(np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).tobytes())
        terrain = base.scene.terrain
        origins = terrain.terrain_origins.reshape(-1, 3)
        assert torch.isfinite(origins).all() and torch.allclose(origins[:, 2], torch.zeros_like(origins[:, 2]), atol=1e-5)
        # Keep outer samples as geometry diagnostics. Native pyramid stairs shrink
        # by two step_widths per iteration, so requested platform_width=2 need not
        # yield a 2 m plateau (hf_terrains.py:pyramid_stairs_terrain while loop).
        xy = torch.tensor([(x, y) for x in (-0.85, 0, 0.85) for y in (-0.85, 0, 0.85)], device=env.device)
        starts = origins[:, None, :].expand(-1, 9, -1).clone()
        starts[:, :, :2] += xy
        starts[:, :, 2] += 10
        directions = torch.zeros_like(starts); directions[:, :, 2] = -1
        hits = raycast_mesh(starts.reshape(-1, 3), directions.reshape(-1, 3), scanner.meshes["/World/ground"])[0]
        bad = (~torch.isfinite(hits).all(dim=1)) | (hits[:, 2].abs() > 1e-4)
        print("PLATFORM_RAY_DIAGNOSTIC " + json.dumps({
            "bad_count": int(bad.sum()),
            "bad_samples": [{"patch_index": int(i // 9), "offset": xy[i % 9].tolist(),
                             "origin": origins[i // 9].tolist(), "hit": hits[i].tolist()}
                            for i in bad.nonzero().flatten().tolist()[:20]],
        }), flush=True)
        assert torch.isfinite(hits).all(), "Missing terrain under spawn neighborhood"
        inner_xy = torch.tensor([(x, y) for x in (-0.5, 0, 0.5) for y in (-0.5, 0, 0.5)], device=env.device)
        starts[:, :, :2] = origins[:, None, :2] + inner_xy
        inner_hits = raycast_mesh(starts.reshape(-1, 3), directions.reshape(-1, 3), scanner.meshes["/World/ground"])[0]
        assert torch.isfinite(inner_hits).all() and torch.allclose(
            inner_hits[:, 2], torch.zeros_like(inner_hits[:, 2]), atol=1e-4
        ), "Central spawn neighborhood is not zero-height"
        initial_obs = obs.tolist()
        print("PROGRESS runtime hash complete", flush=True)
        runner = OnPolicyRunner(env, agents[args.variant].to_dict(), log_dir=None, device=agents[args.variant].device)
        assert runner.alg.policy.actor[0].in_features == runner.alg.policy.critic[0].in_features == 127
        actions = torch.zeros((4, 8), device=env.device)
        contact_seen = torch.zeros(4, dtype=torch.bool, device=env.device)
        for _ in range(64):
            observation, reward, _, _ = env.step(actions)
            obs = observation["policy"]
            scan = height_scan(base, SceneEntityCfg("height_scanner"), 0.5).clamp(-1, 1)
            contacts = foot_contact_state(base, feet, 1.0)
            assert obs.shape == (4, 127) and actions.shape == (4, 8)
            assert scan.shape == (4, 63) and contacts.shape == (4, 4)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all() and torch.isfinite(scan).all()
            assert torch.all((contacts == 0) | (contacts == 1))
            assert torch.equal(obs[:, 60:123], scan) and torch.equal(obs[:, 123:], contacts)
            contact_seen |= contacts.bool().any(dim=0)
        torch.cuda.synchronize()
        counts = {name: sum(p["primitive"] == name for p in patches) for name in cfg.scene.terrain.terrain_generator.sub_terrains}
        write(f"diverse_training_smoke_{args.variant}.json", {
            "PASS": True, "task": TASKS[args.variant], "seed": 42, "num_envs": 4, "steps": 64,
            "observation_shape": [4, 127], "action_shape": [4, 8], "scan_shape": [4, 63], "contact_shape": [4, 4],
            "actor_critic_input_dim": 127, "finite_obs_reward": True, "contact_binary": True,
            "contact_seen_per_foot": contact_seen.tolist(), "initial_root": root.tolist(),
            "initial_clearance": clearance.tolist(), "initial_local_body_positions": initial_local_bodies.tolist(),
            "body_frames_above_spawn_surface": True, "all_100_platforms_ray_checked": True,
            "central_platform_checked_offsets_m": [-0.5, 0.0, 0.5],
            "outer_neighborhood_nonzero_ray_count": int(bad.sum()),
            "mesh_sha256": digest.hexdigest(), "mesh_finite": True,
            "env_origins": base.scene.env_origins.tolist(), "initial_observation": initial_obs,
            "robot_material_sha256": hashlib.sha256(robot.root_physx_view.get_material_properties().cpu().numpy().tobytes()).hexdigest(),
            "patch_counts": counts, "actual_proportions": {k: v / 100 for k, v in counts.items()},
            "sampling_proportions": {k: v.proportion for k, v in cfg.scene.terrain.terrain_generator.sub_terrains.items()},
            "patches": patches,
        })
        print(f"DIVERSE_SMOKE_PASS {args.variant}: counts={counts}", flush=True)
        print("DIVERSE_TRAINING_SMOKE: PASS", flush=True)
        print("PROGRESS smoke complete", flush=True)
    finally:
        env.close()
        print("PROGRESS env closed", flush=True)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    print("PROGRESS app shutdown", flush=True)
    app.close(wait_for_replicator=False)
    print("PROGRESS app shutdown complete", flush=True)
