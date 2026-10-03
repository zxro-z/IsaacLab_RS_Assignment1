"""Resume the validated terrain-only experiment; preserve earlier runs/results.

Training and postprocessing have separate statuses. Raw Isaac Lab YAML is read
with the existing inert node reader, never object-deserialized or rewritten.
"""

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import run_ablation_training as collector
import evaluate_dev_ood as evaluation

ROOT, VAL = collector.ROOT, collector.VALIDATION
STATE = VAL / "diverse_training_run.json"
TASK = "Isaac-Ant-Terrain-Diverse-HeightScan-Contact-v0"
NAME = "ablation_heightscan_contact_diverse_s42"


def save(path, data):
    collector.save_json(path, data)


def json_read(path):
    return json.loads(path.read_text())


def gates():
    source = json_read(VAL / "diverse_training_source_config_audit.json")
    runtime = json_read(VAL / "diverse_training_runtime_config_audit.json")
    existing = json_read(VAL / "diverse_training_smoke_existing.json")
    diverse = json_read(VAL / "diverse_training_smoke_diverse.json")
    overlap = json_read(VAL / "diverse_training_ood_overlap_audit.json")
    assert source["result"] == runtime["result"] == "PASS"
    assert source["reward_matches_stock_ant"]
    assert runtime["entire_terrain_subtree_compared"] and not runtime["differences"]
    assert all(not item["primitive_overlap"] and not item["exact_layout_overlap"]
               for item in overlap["terrains"].values())
    for smoke in (existing, diverse):
        assert smoke["PASS"] and smoke["actor_critic_input_dim"] == 127
        assert smoke["observation_shape"] == [4, 127] and smoke["contact_binary"]
        assert smoke["scan_shape"] == [4, 63] and smoke["contact_shape"] == [4, 4]
    assert existing["mesh_sha256"] != diverse["mesh_sha256"]
    assert existing["robot_material_sha256"] == diverse["robot_material_sha256"]
    audit = {"source_config_audit": source, "runtime_config_audit": runtime,
             "ood_leakage_audit": "PASS", "reward_audit": "MATCHES_BASIC_ANT",
             "actor_critic_input_dim": 127, "result": "PASS"}
    if not (VAL / "diverse_training_config_audit.json").exists():
        save(VAL / "diverse_training_config_audit.json", audit)
    if not (VAL / "diverse_training_smoke.json").exists():
        save(VAL / "diverse_training_smoke.json", {"result": "PASS", "existing": existing, "diverse": diverse})


def collect(run):
    collector.training_status(run)
    directory = Path(run["run_directory"])
    try:
        reference = next(p for p in json_read(VAL / "ablation_eval_manifest.json")
                         if p["label"] == "HeightScan+Contact")
        old_dir = Path(reference["training_run_path"])
        old_env, new_env = [collector.load_yaml(p / "params/env.yaml") for p in (old_dir, directory)]
        old_agent, new_agent = [collector.load_yaml(p / "params/agent.yaml") for p in (old_dir, directory)]
        # Deliberately retain observations, sensors and every importer field.
        for config in (old_env, new_env):
            config.pop("log_dir")
            config.pop("io_descriptors_output_dir")
            config["scene"]["terrain"]["terrain_generator"].pop("sub_terrains")
        diffs = collector.differences(old_env, new_env)
        diffs += collector.differences(collector.normalized_agent(old_agent), collector.normalized_agent(new_agent), "agent")
        assert (new_agent["seed"], new_agent["num_steps_per_env"], new_agent["max_iterations"], new_agent["resume"]) == (42, 32, 1000, False)
        assert new_env["seed"] == 42 and new_env["scene"]["num_envs"] == 4096
        run["config_differences"] = diffs
        run["config_audit_status"] = "pass" if not diffs else "failed"
    except Exception as exc:
        collector.phase_error(run, "config_audit", exc)
    try:
        checkpoint = directory / "model_999.pt"
        run["checkpoint_integrity"] = collector.checkpoint_audit(checkpoint, 127)
        run["checkpoint_status"] = "pass" if run["checkpoint_integrity"]["PASS"] else "failed"
        run["final_checkpoint"] = str(checkpoint)
        (VAL / "diverse_training_final_checkpoint.sha256").write_text(
            run["checkpoint_integrity"]["sha256"] + "  " + str(checkpoint.relative_to(ROOT)) + "\n")
    except Exception as exc:
        collector.phase_error(run, "checkpoint", exc)
    try:
        tags, mapping, values, nonfinite = collector.scalar_data(directory)
        run["scalar_tags"], run["scalar_mapping"] = tags, mapping
        run["nonfinite_scalar_tags"] = nonfinite
        run["event_files"] = [str(p) for p in directory.glob("events.out.tfevents.*")]
        run["tensorboard_status"] = "pass" if not nonfinite and all(values.values()) else "failed"
        rows = [{"run": NAME, "iteration": step, **{k: data.get(step) for k, data in values.items()}}
                for step in sorted({step for data in values.values() for step in data})]
        evaluation._write_csv(VAL / "diverse_training_scalars.csv", ["run", "iteration", *collector.TAGS], rows)
        rewards = values["mean_reward"]
        best = max(rewards, key=rewards.get) if rewards else None
        run["training_statistics"] = {**{f"final_{k}": data[max(data)] if data else None for k, data in values.items()},
                                      "maximum_mean_reward": rewards.get(best), "maximum_reward_iteration": best}
    except Exception as exc:
        collector.phase_error(run, "tensorboard", exc)
    run["completed"] = run["training_status"] == "completed" and all(
        run.get(phase + "_status") == "pass" for phase in ("config_audit", "checkpoint", "tensorboard"))
    save(STATE, run)
    if not run["completed"]:
        raise RuntimeError("Training or postprocessing gate failed; see separate phase statuses")


def simulate(command, log, result):
    if result.exists():
        return json_read(result)  # Recovery does not repeat a completed evaluation.
    with log.open("x") as stream:
        child = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
    if child.returncode:
        raise RuntimeError(f"Simulation exit={child.returncode}: {log}")
    return json_read(result)  # A missing result is failure even if Sim exits zero.


def evaluate(run):
    destination = VAL / "diverse_training_dev_ood_runs"
    destination.mkdir(exist_ok=True)
    entries = []
    # Gate every full evaluation on its fresh-process one-env preflight.
    for label, task in evaluation.TASKS:
        result = destination / f"preflight_{label}.json"
        cmd = ["./isaaclab.sh", "-p", "scripts/assignment/run.py", "scripts/reinforcement_learning/rsl_rl/play_one_episode.py",
               "--task", task, "--checkpoint", run["final_checkpoint"], "--seed", "24", "--num_envs", "1",
               "--headless", "--height_scan", "--foot_contacts", "--preflight_steps", "20", "--result_json", str(result)]
        data = simulate(cmd, destination / f"preflight_{label}.log", result)
        assert data["status"] == "preflight_pass" and data["observation_dim"] == 127
        print("PREFLIGHT PASS " + label, flush=True)
    rows, raw, comparisons = [], [], []
    for label, task in evaluation.TASKS:
        result = destination / f"{label}.json"
        cmd = ["./isaaclab.sh", "-p", "scripts/assignment/run.py", "scripts/reinforcement_learning/rsl_rl/play_one_episode.py",
               "--task", task, "--checkpoint", run["final_checkpoint"], "--seed", "24", "--num_envs", "100",
               "--headless", "--height_scan", "--foot_contacts", "--result_json", str(result)]
        print("EVALUATE " + label, flush=True)
        data = simulate(cmd, destination / f"{label}.log", result)
        old = json_read(VAL / "dev_ood_runs_20261003_025922" / f"{label}__HeightScan_Contact.json")
        assert data["observation_dim"] == 127 and len(data["episodes"]) == 100
        assert old["seed"] == data["seed"] == 24 and old["num_envs"] == data["num_envs"] == 100
        for key in ("terrain_mesh_sha256", "env_origins_sha256", "robot_material_sha256"):
            assert old[key] == data[key], f"Physical Dev OOD mismatch: {label}/{key}"
        aggregate = data["aggregates"]
        common = {"environment": label, "policy": "HeightScan+Contact-Diverse", "observation_dim": 127,
                  "checkpoint": run["final_checkpoint"], "seed": 24, "num_envs": 100}
        rows.append({**common, **{key: aggregate[key] for key in evaluation.RESULT_FIELDS if key in aggregate},
                     **{key: data[key] for key in ("terrain_seed", "terrain_mesh_sha256", "env_origins_sha256", "robot_material_sha256")}})
        raw.extend({**common, **episode} for episode in data["episodes"])
        prior = copy.deepcopy(old["aggregates"])
        if "progress_5m_ratio" not in prior:
            prior["progress_5m_ratio"] = sum(p["forward_displacement_m"] >= 5 for p in old["episodes"]) / 100
        comparisons.append({"environment": label, "existing": prior, "diverse": aggregate,
                            "effects": {"reward_percent": evaluation._relative_change(aggregate["reward_mean"], prior["reward_mean"]),
                                        "displacement_percent": evaluation._relative_change(aggregate["forward_displacement_mean"], prior["forward_displacement_mean"]),
                                        **{key + "_delta_pp": 100 * (aggregate[key] - prior[key]) for key in
                                           ("fall_ratio", "timeout_ratio", "progress_2m_ratio", "progress_5m_ratio")}}})
        print("RESULT " + label + " " + json.dumps(aggregate), flush=True)
    assert len(raw) == 300
    evaluation._write_csv(VAL / "diverse_training_dev_ood_results.csv", evaluation.RESULT_FIELDS, rows)
    evaluation._write_csv(VAL / "diverse_training_dev_ood_raw.csv", evaluation.RAW_FIELDS, raw)
    save(VAL / "diverse_training_dev_ood_comparison.json", comparisons)
    run["evaluation_status"] = "completed"
    run["comparisons"] = comparisons
    save(STATE, run)
    return comparisons


def summary(run):
    lines = ["# Terrain diversity: identical 127-D HeightScan + Contact", "",
             "Training distribution only changes. Reward is stock Ant; original scan/contact, termination and PPO unchanged.",
             "Training statistics are not Dev OOD scores. One training seed; no statistical causal/convergence claim.",
             "Final unseen is not created. Dev OOD is development-exposed and reusable for model selection.", "",
             "## Training", "", "```json", json.dumps({k: run.get(k) for k in
                 ("training_status", "wall_clock_duration", "checkpoint_integrity", "training_statistics", "config_audit_status")}, indent=2), "```", "",
             "## Dev OOD: existing vs diverse", "",
             "Seed 24, 100 envs, first completed episode only; original environment-returned Ant reward.",
             "All three meshes, env origins and robot material hashes matched the preserved existing-policy evaluations."]
    for comparison in run.get("comparisons", []):
        lines.extend(["", "### " + comparison["environment"], "",
                      "| Environment | Training | Reward mean ± std | Distance (m) | Fall % | Timeout % | ≥2m % | ≥5m % |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"])
        for variant in ("existing", "diverse"):
            m = comparison[variant]
            lines.append(f"| {comparison['environment']} | {variant} | {m['reward_mean']:.3f} ± {m['reward_std']:.3f} | "
                         f"{m['forward_displacement_mean']:.3f} | {100*m['fall_ratio']:.1f} | {100*m['timeout_ratio']:.1f} | "
                         f"{100*m['progress_2m_ratio']:.1f} | {100*m['progress_5m_ratio']:.1f} |")
        lines.extend(["", "Pairwise effects: `" + json.dumps(comparison["effects"]) + "`", ""])
    lines.extend(["GapPath remains a stress case unless traversal actually increases; reward without progress is not traversal improvement.",
                  "The older ≥5m ratio is derived from preserved first-episode displacement records using the same ≥5m criterion.", ""])
    smoke = json_read(VAL / "diverse_training_smoke_diverse.json")
    rough = [patch for patch in smoke["patches"] if patch["primitive"] == "random_uniform"]
    lines.extend(["## Audit and failure recovery", "",
                  "DIVERSE_SOURCE_CONFIG_AUDIT: PASS; DIVERSE_RUNTIME_CONFIG_AUDIT: PASS; REWARD_AUDIT_RESULT: MATCHES_BASIC_ANT.",
                  "The saved/runtime audit compares the entire terrain subtree. It runs InteractiveScene/TerrainGenerator's actual native expansion,",
                  "which fills importer num_envs/env_spacing, formats entity paths and assigns sub-terrain size/HF scales.",
                  "Explicit comparison normalizations are only the requested 4-env diagnostic size, unique train output paths and agent run_name.",
                  "Both saved and source dictionaries use the same inert YAML-node representation; raw YAML is never rewritten or object-deserialized.",
                  "Proof: interactive_scene.py:_add_entities_from_cfg and terrain_generator.py:TerrainGenerator.__init__.",
                  "Earlier failures were actual config assertions followed by native shutdown hang/segfault, not a completed training run.",
                  "The stale failed validation worker was stopped; its logs remain. Current env.close and app shutdown return promptly.",
                  "Native fast shutdown can mask exception exit codes, so PASS JSON, explicit phase markers and checkpoint/iteration checks are required.",
                  "The outer ±0.85 m spawn check hit pre-existing stair steps. Native 0.4 m stairs shrink their plateau discretely,",
                  "so platform_width=2.0 does not guarantee an exact 2 m plateau. Geometry was not changed; center ±0.5 m checked zero for all 100 patches.",
                  "Both 4-env tests completed 64 zero-action steps: obs (4,127), action (4,8), scan (4,63), contact (4,4), finite and binary.", "",
                  "## Training terrain and leakage", "",
                  "Original: flat 30%, slopes 60%, low upward stairs 10%. Diverse: flat 20%, slopes 50%, low stairs 10%, random uniform 20%.",
                  "Easy/medium/harder up and down proportions are 12.5%, 8.333333%, 4.166667% each; existing slope/stair parameters are unchanged.",
                  "Random uniform: sampled heights -0.05..0.05 m in 0.025 m increments, coarse spacing 0.4 m, central platform 2 m, native mesh spacing 0.1 m.",
                  "Native bicubic interpolation is retained; this is continuous rough height variation, not literal separated blocks/footholds.",
                  f"Measured random-uniform patch Z extent: {min(p['z_min'] for p in rough):.3f}..{max(p['z_max'] for p in rough):.3f} m. Sampling range is not a strict post-interpolation surface bound.",
                  "10×10 layout, seed 42, curriculum disabled; sampled proportions are probabilities, not exact patch quotas.",
                  "Actual patch counts: `" + json.dumps(smoke["patch_counts"]) + "`.", "",
                  "| Dev OOD | Native primitive | Exact training primitive/layout overlap | Qualification |",
                  "|---|---|---|---|",
                  "| UnevenBlocks | HfDiscreteObstaclesTerrainCfg | No / No | Shared irregular-height concept; not a final unseen claim |",
                  "| SteppingStones | HfSteppingStonesTerrainCfg | No / No | No stone-width/gap layout in training |",
                  "| GapPath | MeshGapTerrainCfg | No / No | No explicit trench/gap in training |", "",
                  "## Interpretation", "",
                  "SteppingStones: reward +3.56%, distance +8.07%, fall -5 pp, timeout +5 pp, ≥2m +1 pp, ≥5m +1 pp.",
                  "There is a modest progress/stability improvement for this training/evaluation seed pair, not evidence of a large or statistically significant generalization gain.",
                  "Timeout did not decrease. Aggregate results do not establish that stable stagnation was removed; trajectory analysis would be needed.",
                  "UnevenBlocks: distance -3.07%, fall +6 pp. No broad improvement across Dev OOD was observed.",
                  "GapPath: distance decreased and ≥2m/≥5m remain zero; traversal generalization was not improved.",
                  "The observation and PPO did not change, but one training seed cannot establish that distribution diversity is generally more important than perception complexity.", "",
                  "## Integrity and next step", "",
                  "Protected checkpoints, previous validation/results and Ant training/Dev OOD/reward/PPO source checksums: "
                  + run.get("protected_files_audit", {}).get("result", "PENDING") + ".",
                  "py_compile and git diff --check are checked separately after generation.",
                  "Recommended next step: inspect the existing-vs-diverse SteppingStones trajectory without changing geometry, then use paired additional training seeds if authorized.",
                  "No Final unseen environment, new sensor, reward shaping or PPO tuning was added.", ""])
    (VAL / "diverse_training_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover", action="store_true", help="Only collect/evaluate this batch; never launch training")
    parser.add_argument("--summary_only", action="store_true", help="Regenerate this batch's report only; no training/evaluation")
    args = parser.parse_args()
    if args.summary_only:
        summary(json_read(STATE))
        return
    gates()
    if args.recover:
        run = json_read(STATE)
    else:
        assert not STATE.exists(), "Existing batch: inspect status and use --recover; no duplicate training"
        assert not list((ROOT / "logs/rsl_rl/ant").glob("*_" + NAME)), "Existing diverse run; do not retrain"
        protected = [p for p in VAL.rglob("*") if p.is_file() and "diverse_training" not in p.name]
        protected += list((ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant").rglob("*.py"))
        protected += list((ROOT / "logs/rsl_rl/ant").glob("*/model_999.pt"))
        run = {"task_id": TASK, "run_name": NAME, "observation_dim": 127, "action_dim": 8, "seed": 42,
               "num_envs": 4096, "max_iterations": 1000, "num_steps_per_env": 32, "sample_budget": 131072000,
               "protected_files": {str(p.relative_to(ROOT)): collector.sha256(p) for p in protected},
               "git_commit": collector.command_output(["git", "rev-parse", "HEAD"]), "training_status": "running"}
        command = ["./isaaclab.sh", "-p", "scripts/assignment/run.py", "scripts/reinforcement_learning/rsl_rl/train.py",
                   "--task", TASK, "--headless", "--seed", "42", "--num_envs", "4096", "--max_iterations", "1000",
                   "--run_name", NAME, "agent.resume=false"]
        run["command"] = command
        log = VAL / (NAME + "_training.log")
        run["console_log"], run["training_start_time"] = str(log), collector.now()
        save(STATE, run)
        start = time.monotonic()
        print("START " + NAME, flush=True)
        with log.open("x") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
            run["pid"] = process.pid
            save(STATE, run)
            while process.poll() is None:
                directories = list((ROOT / "logs/rsl_rl/ant").glob("*_" + NAME))
                if len(directories) == 1:
                    run["run_directory"] = str(directories[0])
                    save(STATE, run)
                time.sleep(5)
            run["exit_code"] = process.returncode
        run["training_end_time"], run["wall_clock_duration"] = collector.now(), round(time.monotonic() - start, 3)
        collector.training_status(run)
        save(STATE, run)
    collect(run)
    print("TRAINING AND COLLECTION PASS", flush=True)
    evaluate(run)
    changed = [name for name, digest in run["protected_files"].items() if collector.sha256(ROOT / name) != digest]
    run["protected_files_audit"] = {"result": "FAIL" if changed else "PASS", "changed": changed}
    save(STATE, run)
    summary(run)
    assert not changed, changed
    print("DIVERSE TRAINING + DEV OOD PASS", flush=True)


if __name__ == "__main__":
    main()
