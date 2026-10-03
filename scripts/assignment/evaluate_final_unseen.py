"""Official four-policy evaluation on the preregistered frozen Final course.

Never trains, changes physical config or overwrites existing results. Gate on
policy-free validation and all separate-process input/shape preflights.
"""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import run_ablation_training as collector
import evaluate_dev_ood as evaluation

ROOT, VAL = collector.ROOT, collector.VALIDATION
TASK = "Isaac-Ant-Final-Unseen-v0"
MANIFEST = VAL / "final_unseen_eval_manifest.json"


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    validation = read(VAL / "final_unseen_validation.json")
    prereg = read(VAL / "final_unseen_preregister.json")
    dev = read(VAL / "dev_experiment_freeze.json")
    assert validation["result"] == "PASS" and validation["policy_free"]
    assert collector.sha256(VAL / "final_unseen_preregister.json") == validation["preregister_sha256"]
    for name, digest in prereg["source_hashes"].items():
        assert collector.sha256(ROOT / name) == digest, f"Frozen source changed: {name}"
    entries = []
    for entry in prereg["policy_set"]:
        dim = entry["observation_dim"]
        checkpoint = Path(entry["checkpoint_path"])
        integrity = collector.checkpoint_audit(checkpoint, dim)
        assert integrity["PASS"] and integrity["sha256"] == dev["checkpoint_sha256"][str(checkpoint)]
        label = "Original HeightScan" if entry["label"] == "HeightScan" else entry["label"]
        flags = ([] if dim == 60 else ["--height_scan"]) + (["--foot_contacts"] if dim == 127 else [])
        entries.append({"label": label, "checkpoint_path": str(checkpoint), "observation_dim": dim,
                        "flags": flags, "checkpoint_integrity": integrity})
    # First policy inference cannot precede this durable freeze manifest.
    old_files = [p for p in VAL.rglob("*") if p.is_file() and "final_unseen" not in p.name]
    protected = {str(p.relative_to(ROOT)): collector.sha256(p) for p in old_files}
    manifest = {"timestamp": datetime.now(timezone.utc).isoformat(), "task": TASK, "seed": 24, "num_envs": 100,
                "first_episode_only": True, "policy_set": entries, "source_hashes": prereg["source_hashes"],
                "preregister_sha256": validation["preregister_sha256"],
                "dev_freeze_sha256": collector.sha256(VAL / "dev_experiment_freeze.json"),
                "protected_files": protected, "freeze_rule": "No tuning or physical environment edits after first policy inference"}
    save(MANIFEST, manifest)
    runs = VAL / "final_unseen_runs"
    runs.mkdir(exist_ok=False)
    preflights = []

    def simulate(entry, count, preflight=False):
        stem = ("preflight_" if preflight else "") + entry["label"].replace(" ", "_").replace("+", "_")
        result, log = runs / (stem + ".json"), runs / (stem + ".log")
        command = ["./isaaclab.sh", "-p", "scripts/assignment/run.py", "scripts/reinforcement_learning/rsl_rl/play_one_episode.py",
                   "--task", TASK, "--checkpoint", entry["checkpoint_path"], "--seed", "24", "--num_envs", str(count),
                   "--headless", "--result_json", str(result), *entry["flags"]]
        if preflight:
            command += ["--preflight_steps", "20"]
        # Check the frozen implementation before every process, not just at end.
        for name, digest in prereg["source_hashes"].items():
            assert collector.sha256(ROOT / name) == digest
        print(("PREFLIGHT " if preflight else "OFFICIAL EVALUATION ") + entry["label"], flush=True)
        with log.open("x") as stream:
            child = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
        assert child.returncode == 0, f"Failed process; preserve {log}"
        data = read(result)
        assert data["observation_dim"] == entry["observation_dim"]
        if preflight:
            assert data["status"] == "preflight_pass"
        else:
            assert len(data["episodes"]) == count
        return data

    for entry in entries:
        data = simulate(entry, 1, True)
        for key in ("terrain_mesh_sha256", "env_origins_sha256", "robot_material_sha256"):
            if preflights:
                assert data[key] == preflights[0][key], "Policy-specific physical scene detected"
        preflights.append(data)
    save(runs / "preflight_summary.json", preflights)
    rows, raw, by_label = [], [], {}
    for entry in entries:
        data = simulate(entry, 100)
        if by_label:
            reference = next(iter(by_label.values()))
            for key in ("terrain_mesh_sha256", "env_origins_sha256", "robot_material_sha256"):
                assert data[key] == reference[key], "Official physical scene differs between policies"
        by_label[entry["label"]] = data
        metrics = data["aggregates"]
        common = {"environment": "FinalUnseen", "policy": entry["label"], "observation_dim": entry["observation_dim"],
                  "checkpoint": entry["checkpoint_path"], "seed": 24, "num_envs": 100}
        rows.append({**common, **{k: metrics[k] for k in evaluation.RESULT_FIELDS if k in metrics},
                     **{k: data[k] for k in ("terrain_seed", "terrain_mesh_sha256", "env_origins_sha256", "robot_material_sha256")}})
        raw.extend({**common, **episode} for episode in data["episodes"])
        print("RESULT " + entry["label"] + " " + json.dumps(metrics), flush=True)
    assert len(raw) == 400
    evaluation._write_csv(VAL / "final_unseen_results.csv", evaluation.RESULT_FIELDS, rows)
    evaluation._write_csv(VAL / "final_unseen_raw.csv", evaluation.RAW_FIELDS, raw)
    pairs = []
    for first, second in (("Base", "Original HeightScan"), ("Original HeightScan", "HeightScan+Contact"),
                          ("HeightScan+Contact", "Diverse HeightScan+Contact")):
        a, b = by_label[first]["aggregates"], by_label[second]["aggregates"]
        pairs.append({"from": first, "to": second,
                      "reward_percent": evaluation._relative_change(b["reward_mean"], a["reward_mean"]),
                      "displacement_percent": evaluation._relative_change(b["forward_displacement_mean"], a["forward_displacement_mean"]),
                      **{k + "_delta_pp": 100*(b[k]-a[k]) for k in ("fall_ratio", "timeout_ratio", "progress_2m_ratio", "progress_5m_ratio")}})
    save(VAL / "final_unseen_pairwise.json", pairs)
    changed = [name for name, digest in protected.items() if collector.sha256(ROOT / name) != digest]
    changed += [name for name, digest in prereg["source_hashes"].items() if collector.sha256(ROOT / name) != digest]
    changed += [name for name, digest in dev["checkpoint_sha256"].items() if collector.sha256(Path(name)) != digest]
    integrity = {"result": "PASS" if not changed else "FAIL", "changed": changed,
                 "protected_previous_files": len(protected), "raw_rows": len(raw), "policy_count": len(entries),
                 "policy_observation_only_differences": True, "source_unchanged_after_preregistration": not changed,
                 "preregister_sha256": collector.sha256(VAL / "final_unseen_preregister.json"),
                 "shared_official_physical_hashes": {k: next(iter(by_label.values()))[k] for k in
                                                     ("terrain_mesh_sha256", "env_origins_sha256", "robot_material_sha256")}}
    save(VAL / "final_unseen_integrity.json", integrity)
    assert not changed, changed
    lines = ["# Frozen Final Unseen official evaluation", "",
             "Preregistered before mesh generation; validated with zero actions before any candidate inference. No post-result terrain or policy changes.",
             "Seed 24, 100 envs per candidate, exactly the first completed episode; original Ant reward from env.step, population std.",
             "One deterministic repeated composite course and one evaluation seed: conclusions are limited to this test, not universal policy rankings.", "",
             "| Policy | Obs | Training difference | Return mean ± std | Distance mean ± std m | Fall | Timeout | ≥2m | ≥5m |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        training = "20% uniform roughness added" if row["policy"].startswith("Diverse") else "Flat + slopes + low stairs"
        lines.append(f"| {row['policy']} | {row['observation_dim']} | {training} | {row['reward_mean']:.3f} ± {row['reward_std']:.3f} | "
                     f"{row['forward_displacement_mean']:.3f} ± {row['forward_displacement_std']:.3f} | "
                     f"{100*row['fall_ratio']:.0f}% | {100*row['timeout_ratio']:.0f}% | {100*row['progress_2m_ratio']:.0f}% | {100*row['progress_5m_ratio']:.0f}% |")
    lines += ["", "## Prespecified pairwise comparisons", "", "```json", json.dumps(pairs, indent=2), "```", "",
              "No composite score/ranking. Pairwise deltas are descriptive; one training seed does not establish causal or statistically significant sensor/diversity effects.",
              "Timeout is not equivalent to traversal success; use distance and ≥2m/≥5m alongside survival. Stagnation needs trajectory evidence.", "",
              "## Freeze and integrity", "", json.dumps(integrity, indent=2), "",
              "The -1 m minimum of the combined mesh includes the native outer terrain border, not a deep hole in the course. The course basin is 0.07 m deep; terraces are at most 0.075 m high.",
              "100 patches repeat the same 40×8 m course; crossing a tile end enters another course or native padding. No new success termination was added.",
              "The Final test is now exposed: no further tuning/retraining against it and no geometry/difficulty changes, regardless of results.", ""]
    (VAL / "final_unseen_summary.md").write_text("\n".join(lines))
    print("FINAL UNSEEN OFFICIAL EVALUATION PASS", flush=True)


if __name__ == "__main__":
    main()
