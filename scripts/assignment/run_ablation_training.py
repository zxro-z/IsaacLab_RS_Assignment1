"""Run the four fixed Ant ablations sequentially and collect training artifacts.

No task, reward, PPO, checkpoint or existing validation file is changed. A failed
run stops the batch without reducing the sample budget. --collect refreshes only
this batch's reports, without launching training. Statistics are NOT OOD scores.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import time
import traceback
import fcntl

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from isaaclab_config_reader import load_config

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "validation"
RUNS_FILE = VALIDATION / "ablation_training_runs.json"
VARIANTS = (
    ("Base", "Base", "ablation_base_obs60_s42", 60),
    ("HeightScan", "HeightScan", "ablation_heightscan_obs123_s42", 123),
    ("Contact", "Contact", "ablation_contact_obs64_s42", 64),
    ("HeightScan+Contact", "HeightScan-Contact", "ablation_heightscan_contact_obs127_s42", 127),
)
TAGS = {
    "mean_reward": ("Train/mean_reward",),
    "episode_length": ("Train/mean_episode_length",),
    "value_loss": ("Loss/value_function",),
    "surrogate_loss": ("Loss/surrogate",),
    "noise_std": ("Policy/mean_noise_std",),
    "learning_rate": ("Loss/learning_rate",),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    # Atomic replacement applies ONLY to newly created batch reports.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def command_output(command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def load_yaml(path):
    # Syntax-only parser: Python-specific tags stay inert tag/value data.
    return load_config(path)


def phase_error(run, phase, exc):
    run[phase + "_status"] = "failed"
    run.setdefault("postprocessing_errors", []).append(
        {"phase": phase, "error": str(exc), "traceback": traceback.format_exc(), "time": now()}
    )


def training_status(run):
    """Training completion comes from the child exit/log, never a YAML parser."""
    if run.get("exit_code") is None:
        run.setdefault("training_status", "not_started")
        return
    console = Path(run["console_log"]).read_text(errors="replace")
    clean = re.sub(r"\x1b\[[0-9;]*m", "", console)
    run["logged_iterations"] = [int(i) for i in re.findall(r"Learning iteration\s+(\d+)/1000", clean)]
    run["oom_detected"] = bool(re.search(r"out of memory|CUDA_ERROR_OUT_OF_MEMORY|bad_alloc", clean, re.I))
    run["nan_detected"] = bool(re.search(r"(?<![A-Za-z])nan(?![A-Za-z])", clean, re.I))
    run["training_status"] = "completed" if (
        run["exit_code"] == 0 and set(run["logged_iterations"]) == set(range(1000))
        and not run["oom_detected"] and not run["nan_detected"]
    ) else "failed"


def differences(first, second, path=""):
    if isinstance(first, dict) and isinstance(second, dict):
        result = []
        for key in sorted(first.keys() | second.keys()):
            name = f"{path}.{key}" if path else key
            if key not in first or key not in second:
                result.append(name + ": missing key")
            else:
                result.extend(differences(first[key], second[key], name))
        return result
    return [] if first == second else [f"{path}: {first!r} != {second!r}"]


def tensor_finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(tensor_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(tensor_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def checkpoint_audit(path, dimension):
    # These are locally generated, trusted RSL-RL checkpoints. Never load an
    # arbitrary downloaded checkpoint with weights_only=False.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    result = {
        "iteration": checkpoint.get("iter"),
        "actor_input_dim": state["actor.0.weight"].shape[1],
        "critic_input_dim": state["critic.0.weight"].shape[1],
        "action_dim": state["actor.6.weight"].shape[0],
        "optimizer_state_present": bool(checkpoint.get("optimizer_state_dict", {}).get("state")),
        "all_tensors_finite": tensor_finite(checkpoint),
        "sha256": sha256(path),
    }
    result["PASS"] = (
        result["iteration"] == 999 and result["actor_input_dim"] == dimension
        and result["critic_input_dim"] == dimension and result["action_dim"] == 8
        and result["optimizer_state_present"] and result["all_tensors_finite"]
    )
    return result


def scalar_data(directory):
    accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
    tags = accumulator.Tags()["scalars"]
    mapping = {key: next((tag for tag in candidates if tag in tags), None) for key, candidates in TAGS.items()}
    data = {
        key: {event.step: event.value for event in accumulator.Scalars(tag)} if tag else {}
        for key, tag in mapping.items()
    }
    # Check all scalar tags (not just the six selected metrics) for nonfinite values.
    nonfinite = [tag for tag in tags if any(not math.isfinite(e.value) for e in accumulator.Scalars(tag))]
    return tags, mapping, data, nonfinite


def normalized_env(config):
    config = dict(config)
    # Native train.py injects unique output directories. These two fields are
    # logging destinations, NOT experimental or physical parameters.
    for key in ("observations", "log_dir", "io_descriptors_output_dir"):
        config.pop(key, None)
    return config


def normalized_agent(config):
    config = dict(config)
    config.pop("run_name", None)  # Required unique label; all other fields compared.
    return config


def collect(batch):
    rows, sampled, checksum_lines = [], [], []
    configs = []
    for run in batch["runs"]:
        training_status(run)
        for phase in ("checkpoint", "tensorboard", "config_audit", "summary"):
            run.setdefault(phase + "_status", "pending")
        directory = Path(run["run_directory"]) if run.get("run_directory") else None
        if directory is None or not directory.is_dir():
            continue
        env_path, agent_path = directory / "params/env.yaml", directory / "params/agent.yaml"
        run["artifact_presence"] = {"env_yaml": env_path.exists(), "agent_yaml": agent_path.exists(),
                                    "model_999": (directory / "model_999.pt").exists()}
        if env_path.exists() and agent_path.exists():
            try:
                env_cfg, agent_cfg = load_yaml(env_path), load_yaml(agent_path)
                assert env_cfg["seed"] == agent_cfg["seed"] == 42
                assert env_cfg["scene"]["num_envs"] == 4096
                assert env_cfg["scene"]["terrain"]["terrain_generator"]["seed"] == 42
                assert agent_cfg["max_iterations"] == 1000 and agent_cfg["resume"] is False
                assert agent_cfg["run_name"] == run["run_name"]
                run["params_sha256"] = {"env.yaml": sha256(env_path), "agent.yaml": sha256(agent_path)}
                configs.append((run["label"], normalized_env(env_cfg), normalized_agent(agent_cfg)))
                run["config_audit_status"] = "parsed_pending_comparison"
            except Exception as exc:
                phase_error(run, "config_audit", exc)
        events = sorted(directory.glob("events.out.tfevents.*"))
        run["event_files"] = [str(path) for path in events]
        if events:
            try:
                tags, mapping, data, nonfinite = scalar_data(directory)
                run["tensorboard_status"] = "pass" if not nonfinite else "nonfinite_scalars"
            except Exception as exc:
                phase_error(run, "tensorboard", exc)
                tags, mapping, data, nonfinite = [], {}, {key: {} for key in TAGS}, []
            run["available_scalar_tags"], run["scalar_tag_mapping"] = tags, mapping
            run["nonfinite_scalar_tags"] = nonfinite
            steps = sorted({step for values in data.values() for step in values})
            for step in steps:
                rows.append({"run": run["run_name"], "iteration": step,
                             **{key: values.get(step) for key, values in data.items()}})
            for target in list(range(0, 1000, 100)) + [999]:
                row = {"run": run["run_name"], "iteration": target}
                for key, values in data.items():
                    nearest = min(values, key=lambda step: abs(step-target)) if values else None
                    row[key] = values.get(nearest)
                    row[key + "_source_step"] = nearest
                sampled.append(row)
            reward, length = data["mean_reward"], data["episode_length"]
            finite_reward = {step: val for step, val in reward.items() if math.isfinite(val)}
            maximum_step = max(finite_reward, key=finite_reward.get) if finite_reward else None
            run["training_statistics"] = {
                "final_mean_reward": reward.get(max(reward)) if reward else None,
                "final_reward_iteration": max(reward) if reward else None,
                "maximum_mean_reward": finite_reward.get(maximum_step),
                "maximum_reward_iteration": maximum_step,
                "final_episode_length": length.get(max(length)) if length else None,
                "final_episode_length_iteration": max(length) if length else None,
            }
        final = directory / "model_999.pt"
        if final.exists():
            run["final_checkpoint"] = str(final)
            try:
                run["checkpoint_integrity"] = checkpoint_audit(final, run["observation_dim"])
                run["final_checkpoint_iteration"] = run["checkpoint_integrity"]["iteration"]
                run["checkpoint_status"] = "pass" if run["checkpoint_integrity"]["PASS"] else "integrity_failed"
            except Exception as exc:
                phase_error(run, "checkpoint", exc)
        elif run.get("exit_code") is not None:
            run["checkpoint_status"] = "missing_final_checkpoint"
        if run.get("exit_code") is not None:
            run["completed"] = run["training_status"] == "completed" and run["checkpoint_status"] == "pass"
            run["status"] = "completed" if run["completed"] else (
                "training_completed_artifacts_unverified" if run["training_status"] == "completed" else "training_failed")
            if run["completed"]:
                checksum_lines.append(f'{run["checkpoint_integrity"]["sha256"]}  {final.relative_to(ROOT)}')
    audit = {"result": "PENDING", "logging_only_exclusions": {
        "env": ["log_dir", "io_descriptors_output_dir"], "agent": ["run_name"]}, "differences": []}
    if configs:
        label0, env0, agent0 = configs[0]
        for label, env_cfg, agent_cfg in configs[1:]:
            audit["differences"].extend([f"{label0} vs {label}: {diff}" for diff in
                                        differences(env0, env_cfg, "env") + differences(agent0, agent_cfg, "agent")])
        audit["saved_configs_available"] = len(configs)
        audit["result"] = "FAIL" if audit["differences"] else ("PASS" if len(configs) == 4 else "PENDING")
    batch["config_audit"] = audit
    save_json(VALIDATION / "ablation_training_normalized_configs.json",
              {label: {"env": env_cfg, "agent": agent_cfg} for label, env_cfg, agent_cfg in configs})
    for run in batch["runs"]:
        if run["config_audit_status"] == "parsed_pending_comparison":
            run["config_audit_status"] = audit["result"].lower()
    save_json(RUNS_FILE, batch)
    for name, values, fields in (
        ("ablation_training_scalars.csv", sampled, ["run", "iteration"] +
         [field for key in TAGS for field in (key, key + "_source_step")]),
        ("ablation_training_scalars_raw.csv", rows, ["run", "iteration", *TAGS]),
    ):
        with (VALIDATION / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in values:
                writer.writerow({key: value if not isinstance(value, float) or math.isfinite(value) else "NaN"
                                 for key, value in row.items()})
    (VALIDATION / "ablation_final_checkpoints.sha256").write_text("\n".join(checksum_lines) + ("\n" if checksum_lines else ""))
    manifest = [{"label": run["label"], "task_id": run["task_id"], "observation_dim": run["observation_dim"],
                 "checkpoint_path": run.get("final_checkpoint") if run.get("completed") else None,
                 "seed": 42, "training_run_path": run.get("run_directory"), "status": run["status"]}
                for run in batch["runs"]]
    save_json(VALIDATION / "ablation_eval_manifest.json", manifest)
    summary = ["# Ant sensor ablation training", "", "Training statistics only; NOT dev/final unseen evaluation scores.",
               "No performance ranking or policy selection is made here.", "", "## Experiment Conditions", "",
               "| Variant | Obs dim | Seed | Envs | Iterations | Terrain | Reward | PPO |",
               "|---|---:|---:|---:|---:|---|---|---|"]
    for run in batch["runs"]:
        summary.append(f'| {run["label"]} | {run["observation_dim"]} | 42 | 4096 | 1000 | shared mixed, seed 42 | stock Ant | original Ant PPO |')
    summary.extend(["", "## Training Completion", "", "| Variant | Completed | Final checkpoint | Duration (s) | Exit code |", "|---|---|---|---:|---:|"])
    for run in batch["runs"]:
        summary.append(f'| {run["label"]} | {run.get("completed", False)} ({run["status"]}) | {run.get("final_checkpoint")} | {run.get("wall_clock_duration")} | {run.get("exit_code")} |')
    summary.extend(["", "## Training Scalars", "", "| Variant | Final train reward | Max train reward | Max iteration | Final episode length |", "|---|---:|---:|---:|---:|"])
    for run in batch["runs"]:
        stats = run.get("training_statistics", {})
        summary.append(f'| {run["label"]} | {stats.get("final_mean_reward")} | {stats.get("maximum_mean_reward")} | {stats.get("maximum_reward_iteration")} | {stats.get("final_episode_length")} |')
    summary.extend(["", "## Integrity", "", f'CONFIG_AUDIT_RESULT: {audit["result"]}',
                    "Only observation and unique output/name fields are excluded from the saved-config comparison.",
                    "Checkpoint hashes: `ablation_final_checkpoints.sha256`.", ""])
    for run in batch["runs"]:
        summary.append(f'- {run["label"]}: SHA256={run.get("checkpoint_integrity", {}).get("sha256")}; NaN={run.get("nan_detected")}; OOM={run.get("oom_detected")}; status={run["status"]}')
    if audit["differences"]:
        summary.extend(["", "```", *audit["differences"], "```"])
    summary.extend(["", "## Next Step", "", "Design Dev OOD environments and evaluate all four policies under identical conditions.",
                    "No Dev OOD environment or evaluation is implemented by this script.", ""])
    (VALIDATION / "ablation_training_summary.md").write_text("\n".join(summary))
    for run in batch["runs"]:
        run["summary_status"] = "pass"
    save_json(RUNS_FILE, batch)


def safe_collect(batch):
    """Post-processing can fail without changing any child's training status."""
    try:
        collect(batch)
    except Exception as exc:
        for run in batch["runs"]:
            phase_error(run, "summary", exc)
        batch.setdefault("collector_errors", []).append(traceback.format_exc())
        save_json(RUNS_FILE, batch)
        print("COLLECTOR_WARNING " + str(exc), flush=True)


def snapshot():
    files = [path for path in (ROOT / "source").rglob("*") if path.is_file() and path.suffix in (".py", ".yaml", ".toml")]
    files += [path for path in (ROOT / "scripts").rglob("*.py") if path.is_file()]
    files += [path for path in VALIDATION.iterdir() if path.is_file()]
    files += [ROOT / "isaaclab.sh"]
    # Protect every file in the four migrated run folders using the existing manifest.
    for line in (VALIDATION / "ant_core_runs_checksums.sha256").read_text().splitlines():
        files.append(ROOT / line.split(maxsplit=1)[1])
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(files))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true", help="Launch all four fresh, fixed-budget runs.")
    group.add_argument("--collect", action="store_true", help="Refresh reports only, never train.")
    group.add_argument("--continue-batch", action="store_true", help="Verify existing completed runs and train only pending variants.")
    args = parser.parse_args()
    if args.collect:
        safe_collect(json.loads(RUNS_FILE.read_text()))
        return
    if RUNS_FILE.exists() and not args.continue_batch:
        raise SystemExit("Batch metadata already exists; refusing to overwrite/restart. Use --collect.")
    validation = json.loads((VALIDATION / "ant_terrain_ablation_summary.json").read_text())
    assert validation["non_observation_to_dict_equality"] == validation["registration"] == "PASS"
    for label, variant, name, dim in VARIANTS:
        assert validation["policy_shapes"][f"Isaac-Ant-Terrain-Ablation-{variant}-v0"] == [4, dim]
        if not args.continue_batch:
            assert not list((ROOT / "logs/rsl_rl/ant").glob(f"*_{name}")), name
    environment = os.environ.copy()
    environment.update(TERM="xterm", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    environment.pop("CUDA_LAUNCH_BLOCKING", None)  # Native training defaults, identical for all four.
    sim_site = Path(environment["ISAAC_SIM_SITE_PACKAGES"])
    runtime = {
        "gpu": torch.cuda.get_device_name(0), "python_version": platform.python_version(),
        "isaac_sim_version": (sim_site / "isaacsim/VERSION").read_text().strip(),
        "isaac_lab_version": (ROOT / "VERSION").read_text().strip(),
        "rsl_rl_version": metadata.version("rsl-rl-lib"), "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "git_commit": command_output(["git", "rev-parse", "HEAD"]),
    }
    if args.continue_batch:
        batch = json.loads(RUNS_FILE.read_text())
        protected = json.loads((VALIDATION / "ablation_training_protected_files.json").read_text())
        batch["authorized_collector_changes"] = ["scripts/assignment/run_ablation_training.py"]
        for run in batch["runs"]:
            assert (run["seed"], run["num_envs"], run["max_iterations"], run["num_steps_per_env"]) == (42, 4096, 1000, 32)
        safe_collect(batch)
        batch["continued_at"] = now()
    else:
        protected = snapshot()
        save_json(VALIDATION / "ablation_training_protected_files.json", protected)
        batch = {"created_at": now(), "runtime": runtime, "git_branch": command_output(["git", "branch", "--show-current"]),
                 "git_status_before": command_output(["git", "status", "--short"]), "runs": []}
        for label, variant, name, dim in VARIANTS:
            batch["runs"].append({"label": label, "task_id": f"Isaac-Ant-Terrain-Ablation-{variant}-v0",
                                  "run_name": name, "seed": 42, "num_envs": 4096, "max_iterations": 1000,
                                  "num_steps_per_env": 32, "sample_budget": 131072000,
                                  "observation_dim": dim, "action_dim": 8, "status": "pending", **runtime})
    batch["collector_source_sha256"] = {name: sha256(ROOT / "scripts/assignment" / name)
                                        for name in ("run_ablation_training.py", "isaaclab_config_reader.py")}
    save_json(RUNS_FILE, batch)
    for run in batch["runs"]:
        if run.get("completed"):
            print("SKIP_COMPLETED " + run["run_name"], flush=True)
            continue
        if run.get("exit_code") is not None or run["status"] != "pending":
            raise SystemExit("Existing partial/unverified run preserved; explicit retry name required: " + run["run_name"])
        if list((ROOT / "logs/rsl_rl/ant").glob(f'*_{run["run_name"]}')):
            raise SystemExit("Untracked existing run directory; refusing duplicate training: " + run["run_name"])
        run.update(runtime)
        run["training_start_time"], start = now(), time.monotonic()
        run["status"] = "running"
        run["training_status"] = "running"
        console = VALIDATION / f'{run["run_name"]}_training.log'
        run["console_log"] = str(console)
        before = set((ROOT / "logs/rsl_rl/ant").glob(f'*_{run["run_name"]}'))
        command = ["./isaaclab.sh", "-p", "scripts/assignment/run.py", "scripts/reinforcement_learning/rsl_rl/train.py",
                   "--task", run["task_id"], "--headless", "--seed", "42", "--num_envs", "4096",
                   "--max_iterations", "1000", "--run_name", run["run_name"], "agent.resume=false"]
        run["command"] = command
        print("START " + run["task_id"] + " " + now(), flush=True)
        save_json(RUNS_FILE, batch)
        with console.open("x") as output:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT)
            run["pid"] = process.pid
            save_json(RUNS_FILE, batch)
            while process.poll() is None:
                directories = set((ROOT / "logs/rsl_rl/ant").glob(f'*_{run["run_name"]}')) - before
                if len(directories) == 1:
                    run["run_directory"] = str(directories.pop())
                    save_json(RUNS_FILE, batch)
                time.sleep(5)
            run["exit_code"] = process.returncode
        run["training_end_time"] = now()
        run["wall_clock_duration"] = round(time.monotonic() - start, 3)
        directories = set((ROOT / "logs/rsl_rl/ant").glob(f'*_{run["run_name"]}')) - before
        if len(directories) == 1:
            run["run_directory"] = str(directories.pop())
        run["status"] = "finished_unverified"
        save_json(RUNS_FILE, batch)
        training_status(run)
        safe_collect(batch)
        print("END " + run["run_name"] + " status=" + run["status"] + " exit=" + str(run["exit_code"]), flush=True)
        if not run.get("completed"):
            batch["stopped_reason"] = "Run failed completion/integrity checks; remaining runs not started. No config changes."
            for pending in batch["runs"]:
                if pending["status"] == "pending":
                    pending["status"] = "not_started_after_failure"
            break
    changed = [name for name, value in protected.items()
               if name not in batch.get("authorized_collector_changes", [])
               and (not (ROOT / name).is_file() or sha256(ROOT / name) != value)]
    batch["protected_files_audit"] = {"result": "PASS" if not changed else "FAIL", "changed_files": changed,
                                      "file_count": len(protected)}
    batch["git_status_after"] = command_output(["git", "status", "--short"])
    batch["ended_at"] = now()
    safe_collect(batch)
    result = batch.get("config_audit", {}).get("result", "PENDING")
    print("BATCH_FINISHED config_audit=" + result + " protected=" + batch["protected_files_audit"]["result"], flush=True)
    if not all(run.get("completed") for run in batch["runs"]) or changed or result != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    with (VALIDATION / "ablation_training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another collector/batch owns the lock; refusing concurrent execution.")
        main()
