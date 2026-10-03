"""Sequentially evaluate four manifest-selected Ant policies on fixed Development OOD scenes.

Example (inside IsaacLab_RS):
    TERM=xterm ./isaaclab.sh -p scripts/assignment/run.py \
        scripts/assignment/evaluate_dev_ood.py

The script runs each simulation in a fresh process, sequentially. It never trains,
modifies a checkpoint, or writes results over an existing result set.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "validation"
MANIFEST = VALIDATION / "ablation_eval_manifest.json"
TASKS = (
    ("UnevenBlocks", "Isaac-Ant-DevOOD-UnevenBlocks-v0"),
    ("SteppingStones", "Isaac-Ant-DevOOD-SteppingStones-v0"),
    ("GapPath", "Isaac-Ant-DevOOD-GapPath-v0"),
)
SENSOR_FLAGS = {
    "Base": (),
    "HeightScan": ("--height_scan",),
    "Contact": ("--foot_contacts",),
    "HeightScan+Contact": ("--height_scan", "--foot_contacts"),
}
OBS_DIMS = {"Base": 60, "HeightScan": 123, "Contact": 64, "HeightScan+Contact": 127}
POLICY_ORDER = tuple(SENSOR_FLAGS)
RESULT_FIELDS = (
    "environment", "policy", "observation_dim", "checkpoint", "seed", "num_envs",
    "reward_mean", "reward_std", "episode_length_mean", "episode_length_std",
    "forward_displacement_mean", "forward_displacement_std", "forward_velocity_mean",
    "timeout_count", "timeout_ratio", "fall_count", "fall_ratio",
    "progress_2m_count", "progress_2m_ratio", "progress_5m_count", "progress_5m_ratio", "terrain_seed",
    "terrain_mesh_sha256", "env_origins_sha256", "robot_material_sha256",
)
RAW_FIELDS = (
    "environment", "policy", "observation_dim", "checkpoint", "seed", "num_envs",
    "env_id", "episode_return", "episode_length", "forward_displacement_m",
    "forward_velocity_mean_mps", "time_out", "fall", "progress_2m", "progress_5m",
)
PAIRS = (
    ("Base", "HeightScan"),
    ("Base", "Contact"),
    ("Base", "HeightScan+Contact"),
    ("HeightScan", "HeightScan+Contact"),
    ("Contact", "HeightScan+Contact"),
)
SPATIAL_PAIRS = (("Original", "Dense"), ("Original", "ForwardDense"),
                 ("Dense", "ForwardDense"), ("Base", "Dense"), ("Base", "ForwardDense"))
COMPACT_PAIRS = (("Compact-only", "Original+Compact"),)


def sensor_flags(entry):
    """Read explicit spatial metadata, retaining legacy contact manifest behavior."""
    if "height_scan_enabled" not in entry:
        return list(SENSOR_FLAGS[entry["label"]])
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("ant_observation_adapter", ROOT / "scripts/reinforcement_learning/rsl_rl/ant_observation_adapter.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    flags = []
    if entry["height_scan_enabled"]:
        pattern = module.canonical_pattern(entry.get("height_scan_pattern") or "original")
        flags += ["--height_scan", "--height_scan_pattern", pattern]
    if entry.get("foot_contacts_enabled", entry.get("foot_contacts", False)):
        flags += ["--foot_contacts"]
    compact = entry.get("compact_enabled", False)
    if compact:
        if entry.get("height_scan_pattern") not in (None, "original", "original_9x7") or "--foot_contacts" in flags:
            raise ValueError("Compact manifest requires Original internal scanner and no Contact")
        flags += ["--compact_terrain"]
    expected = 60 + (module.SCAN_DIMS[pattern] if entry["height_scan_enabled"] else 0) + (18 if compact else 0) + (4 if "--foot_contacts" in flags else 0)
    if entry["observation_dim"] != expected:
        raise ValueError(f"Manifest observation/pattern mismatch: {entry}")
    return flags


def _mean_std(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def _relative_change(new, old):
    return None if old == 0 else 100.0 * (new - old) / abs(old)


def _write_csv(path: Path, fields, rows):
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            old_rows = list(reader)
            expected = [{key: str(row[key]) for key in fields} for row in rows]
            if reader.fieldnames != list(fields) or old_rows != expected:
                raise FileExistsError(f"Existing CSV differs; refusing overwrite: {path}")
        return  # Collection recovery preserves byte-identical existing CSVs.
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--num_envs", type=int, default=100)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--preflight_steps", type=int, default=0,
                        help="Gate all full evaluations on separate one-env policy preflights.")
    parser.add_argument("--collect_only", action="store_true",
                        help="Recover summaries from existing per-run JSON; never launch simulations.")
    parser.add_argument("--runs_dir", type=Path, default=None, help="Unique folder for per-run JSON and logs.")
    parser.add_argument("--results_csv", type=Path, default=VALIDATION / "dev_ood_evaluation_results.csv")
    parser.add_argument("--raw_csv", type=Path, default=VALIDATION / "dev_ood_evaluation_raw.csv")
    parser.add_argument("--summary_md", type=Path, default=VALIDATION / "dev_ood_evaluation_summary.md")
    args = parser.parse_args()
    for name in ("manifest", "results_csv", "raw_csv", "summary_md"):
        setattr(args, name, getattr(args, name).resolve())
    if args.runs_dir is not None:
        args.runs_dir = args.runs_dir.resolve()
    if args.preflight_steps < 0:
        parser.error("--preflight_steps must be nonnegative")

    for path in (args.results_csv, args.raw_csv, args.summary_md):
        if path.exists() and (not args.collect_only or path == args.summary_md):
            raise FileExistsError(f"Refusing to overwrite existing evaluation output: {path}")
    policies = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_label = {entry["label"]: entry for entry in policies}
    if len(by_label) != len(policies) or len(policies) not in (2, 4):
        raise ValueError("Expected two compact or four legacy uniquely labeled policies")
    policy_order = tuple(by_label)
    pairs = (COMPACT_PAIRS if set(by_label) == {"Compact-only", "Original+Compact"} else
             SPATIAL_PAIRS if set(by_label) == {p for pair in SPATIAL_PAIRS for p in pair} else PAIRS)
    if any(label not in by_label for pair in pairs for label in pair):
        raise ValueError(f"Unsupported comparison labels: {policy_order}")
    obs_dims = {label: entry["observation_dim"] for label, entry in by_label.items()}
    flags = {label: sensor_flags(entry) for label, entry in by_label.items()}
    for label, entry in by_label.items():
        if entry.get("status", "completed") != "completed":
            raise ValueError(f"Manifest status/dimension mismatch for {label}: {entry}")
        if not Path(entry["checkpoint_path"]).is_file():
            raise FileNotFoundError(entry["checkpoint_path"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = args.runs_dir or (VALIDATION / f"dev_ood_runs_{stamp}")
    if args.collect_only:
        if args.runs_dir is None or not runs_dir.is_dir():
            parser.error("--collect_only requires an existing --runs_dir")
    else:
        runs_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.setdefault("TERM", "xterm")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    eval_script = ROOT / "scripts/reinforcement_learning/rsl_rl/play_one_episode.py"
    aggregate_rows, raw_rows, results_by_env = [], [], {}

    if args.preflight_steps and not args.collect_only:
        preflights = []
        for environment, task_id in TASKS:
            for policy in policy_order:
                entry = by_label[policy]
                stem = f"preflight__{environment}__{policy.replace(' ', '_')}"
                result_path, log_path = runs_dir / f"{stem}.json", runs_dir / f"{stem}.log"
                command = ["./isaaclab.sh", "-p", "scripts/assignment/run.py", str(eval_script.relative_to(ROOT)),
                           "--task", task_id, "--checkpoint", entry["checkpoint_path"], "--num_envs", "1",
                           "--seed", str(args.seed), "--headless", "--preflight_steps", str(args.preflight_steps),
                           "--result_json", str(result_path), *flags[policy]]
                print(f"[PREFLIGHT] {environment} × {policy}", flush=True)
                with log_path.open("x", encoding="utf-8") as log:
                    completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f"Preflight failed; full evaluations not started: {log_path}")
                data = json.loads(result_path.read_text())
                if data["status"] != "preflight_pass" or data["observation_dim"] != obs_dims[policy]:
                    raise RuntimeError(f"Invalid preflight result: {result_path}")
                reference = next((p for p in preflights if p["environment"] == environment), None)
                if reference is not None:
                    for key in ("terrain_mesh_sha256", "env_origins_sha256", "robot_material_sha256"):
                        if data[key] != reference[key]:
                            raise RuntimeError(f"Preflight physical scene mismatch: {environment}/{key}")
                preflights.append({"environment": environment, "policy": policy, **data})
        (runs_dir / "preflight_summary.json").write_text(json.dumps(preflights, indent=2))
        print(f"[PASS] All {len(preflights)} preflights; starting full evaluations", flush=True)

    for environment, task_id in TASKS:
        results_by_env[environment] = {}
        for policy in policy_order:
            entry = by_label[policy]
            tag = f"{environment}__{policy.replace('+', '_')}.json"
            result_path, log_path = runs_dir / tag, runs_dir / f"{tag[:-5]}.log"
            command = [
                "./isaaclab.sh", "-p", "scripts/assignment/run.py", str(eval_script.relative_to(ROOT)),
                "--task", task_id, "--checkpoint", entry["checkpoint_path"],
                "--num_envs", str(args.num_envs), "--seed", str(args.seed), "--headless",
                "--result_json", str(result_path), *flags[policy],
            ]
            if not args.collect_only:
                print(f"[RUN] {environment} × {policy}: {task_id}", flush=True)
                with log_path.open("x", encoding="utf-8") as log:
                    completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode != 0:
                    print(f"[FAIL] exit={completed.returncode}; retained log: {log_path}", file=sys.stderr)
                    raise SystemExit(completed.returncode)
            data = json.loads(result_path.read_text(encoding="utf-8"))
            if data["observation_dim"] != obs_dims[policy] or len(data["episodes"]) != args.num_envs:
                raise RuntimeError(f"Unexpected observation/episode dimensions in {result_path}")
            results_by_env[environment][policy] = data
            for hash_name in ("terrain_mesh_sha256", "env_origins_sha256", "robot_material_sha256"):
                if hash_name in results_by_env[environment][policy_order[0]] and (
                    data[hash_name] != results_by_env[environment][policy_order[0]][hash_name]
                ):
                    raise RuntimeError(
                        f"Physical evaluation scene mismatch for {environment}: {hash_name} differs by policy"
                    )
            metrics = data["aggregates"]
            aggregate_rows.append({
                "environment": environment, "policy": policy, "observation_dim": obs_dims[policy],
                "checkpoint": entry["checkpoint_path"], "seed": args.seed, "num_envs": args.num_envs,
                "reward_mean": metrics["reward_mean"], "reward_std": metrics["reward_std"],
                "episode_length_mean": metrics["episode_length_mean"],
                "episode_length_std": metrics["episode_length_std"],
                "forward_displacement_mean": metrics["forward_displacement_mean"],
                "forward_displacement_std": metrics["forward_displacement_std"],
                "forward_velocity_mean": metrics["forward_velocity_mean"],
                "timeout_count": metrics["timeout_count"], "timeout_ratio": metrics["timeout_ratio"],
                "fall_count": metrics["fall_count"], "fall_ratio": metrics["fall_ratio"],
                "progress_2m_count": metrics["progress_2m_count"],
                "progress_2m_ratio": metrics["progress_2m_ratio"], "terrain_seed": data["terrain_seed"],
                "progress_5m_count": metrics["progress_5m_count"], "progress_5m_ratio": metrics["progress_5m_ratio"],
                "terrain_mesh_sha256": data["terrain_mesh_sha256"],
                "env_origins_sha256": data["env_origins_sha256"],
                "robot_material_sha256": data["robot_material_sha256"],
            })
            for episode in data["episodes"]:
                raw_rows.append({
                    "environment": environment, "policy": policy, "observation_dim": obs_dims[policy],
                    "checkpoint": entry["checkpoint_path"], "seed": args.seed, "num_envs": args.num_envs,
                    **episode,
                })
            print(f"[DONE] {environment} × {policy}: reward={metrics['reward_mean']:.3f}, "
                  f"distance={metrics['forward_displacement_mean']:.3f} m", flush=True)

    _write_csv(args.results_csv, RESULT_FIELDS, aggregate_rows)
    _write_csv(args.raw_csv, RAW_FIELDS, raw_rows)
    lines = [
        "# Development OOD evaluation summary", "",
        "Training terrain is used for policy optimization. These fixed Dev OOD terrains are for repeated "
        "development comparison/model selection. Final unseen remains unbuilt and reserved until candidate "
        "policies are selected.", "",
        f"Protocol: seed={args.seed}, num_envs={args.num_envs}, exactly the first completed episode per env; "
        "reward is the original Ant environment reward. Progress success is forward displacement ≥ 2.0 m "
        "during the first episode.", "",
        "All selected policies in each environment use the same terrain task/seed, fixed 1.0 static/dynamic "
        "friction, stock Ant rewards, and the same terrain-relative 0.31 m torso clearance termination. "
        "The per-policy runtime adapters alter observation inputs only.", "",
    ]
    for environment, _ in TASKS:
        lines.extend([f"## {environment}", "", "| Policy | Reward mean ± std | Distance mean ± std (m) | "
                      "Length mean ± std (steps) | Fall % | Timeout % | Progress ≥2m % |", "|---|---:|---:|---:|---:|---:|---:|"])
        for policy in policy_order:
            row = next(r for r in aggregate_rows if r["environment"] == environment and r["policy"] == policy)
            lines.append(
                f"| {policy} ({row['observation_dim']}D) | {_mean_std(row['reward_mean'], row['reward_std'])} | "
                f"{_mean_std(row['forward_displacement_mean'], row['forward_displacement_std'])} | "
                f"{_mean_std(row['episode_length_mean'], row['episode_length_std'])} | "
                f"{100*row['fall_ratio']:.1f}% | {100*row['timeout_ratio']:.1f}% | "
                f"{100*row['progress_2m_ratio']:.1f}% |"
            )
        lines.extend(["", "### Pairwise ablation differences", "", "Changes are new minus old for fall/timeout "
                      "ratios (percentage points); reward and displacement are relative percent change. No overall "
                      "policy score or ranking is computed.", "", "| Pair | Reward change | Displacement change | Fall ratio Δ (pp) | "
                      "Timeout ratio Δ (pp) | Progress ≥2m Δ (pp) |", "|---|---:|---:|---:|---:|---:|"])
        env_data = results_by_env[environment]
        for old, new in pairs:
            old_m, new_m = env_data[old]["aggregates"], env_data[new]["aggregates"]
            reward_change = _relative_change(new_m["reward_mean"], old_m["reward_mean"])
            distance_change = _relative_change(new_m["forward_displacement_mean"], old_m["forward_displacement_mean"])
            lines.append(
                f"| {old} → {new} | {('n/a' if reward_change is None else f'{reward_change:+.1f}%')} | "
                f"{('n/a' if distance_change is None else f'{distance_change:+.1f}%')} | "
                f"{100*(new_m['fall_ratio']-old_m['fall_ratio']):+.1f} | "
                f"{100*(new_m['timeout_ratio']-old_m['timeout_ratio']):+.1f} | "
                f"{100*(new_m['progress_2m_ratio']-old_m['progress_2m_ratio']):+.1f} |"
            )
        lines.append("")
    lines.extend([
        "## Files", "",
        f"- Aggregate metrics: `{args.results_csv.relative_to(ROOT)}`", f"- Episode-level raw data: `{args.raw_csv.relative_to(ROOT)}`",
        f"- Simulator logs and source JSON: `{runs_dir.relative_to(ROOT)}/`", "",
        "## Representative video commands", "",
        "SteppingStones is selected to inspect separated footholds. Each command uses one environment, seed 24, "
        "a distinct recording label, and only the observation adapter required by that checkpoint.", "",
    ])
    lines.extend(["## Observation cost", "", "| Policy | Scan dimensions / rays | Total observation dimensions |", "|---|---:|---:|"])
    for policy in policy_order:
        entry = by_label[policy]
        scan_dim = obs_dims[policy] - 60 - (4 if "--foot_contacts" in flags[policy] else 0)
        lines.append(f"| {policy} | {scan_dim} terrain input dims; "
                     f"{63 if entry.get('compact_enabled') else scan_dim} scanner rays | {obs_dims[policy]} |")
    lines.append("")
    lines.extend(["| Environment | Policy | Scan rays | Total obs | Reward mean | Distance mean (m) | Progress ≥2m % |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for row in aggregate_rows:
        policy = row["policy"]
        rays = (63 if by_label[policy].get("compact_enabled") else
                obs_dims[policy] - 60 - (4 if "--foot_contacts" in flags[policy] else 0))
        lines.append(f"| {row['environment']} | {policy} | {rays} | {obs_dims[policy]} | "
                     f"{row['reward_mean']:.3f} | {row['forward_displacement_mean']:.3f} | {100*row['progress_2m_ratio']:.1f} |")
    lines.extend(["", "These measurements do not establish that denser rays caused improved gap detection. "
                  "They compare one seed per representation on unchanged development scenes; no composite/final score is computed.", ""])
    for policy in policy_order:
        entry = by_label[policy]
        command = (
            "./isaaclab.sh -p scripts/assignment/run.py scripts/reinforcement_learning/rsl_rl/play.py "
            f"--task Isaac-Ant-DevOOD-SteppingStones-v0 --checkpoint {entry['checkpoint_path']} "
            "--num_envs 1 --seed 24 --video --video_length 960 --follow_robot "
            f"--video_name stepping_{policy.lower().replace(' heightscan', '').replace('+', '_').replace(' ', '_')}"
        )
        command += " " + " ".join(flags[policy])
        lines.extend([f"### {policy}", "", "```bash", command, "```", ""])
    lines.extend(["No final unseen/composite environment was created or evaluated.", ""])
    with args.summary_md.open("x", encoding="utf-8") as summary:
        summary.write("\n".join(lines))
    print(f"[PASS] Wrote {args.results_csv}, {args.raw_csv}, {args.summary_md}")


if __name__ == "__main__":
    main()
