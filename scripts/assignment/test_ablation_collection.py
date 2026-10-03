"""Read-only regression tests for inert config parsing and independent phases."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import run_ablation_training as collector
from isaaclab_config_reader import parse_config_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing regression report")
    expected = {}
    for line in (collector.VALIDATION / "ant_core_runs_checksums.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name] = digest
    assert all(collector.sha256(collector.ROOT / name) == digest for name, digest in expected.items())
    runs = (
        ("baseline", "2026-09-29_17-15-49_ant_baseline", 60),
        ("friction_random", "2026-09-29_19-32-15_friction_random", 60),
        ("terrain_heightscan", "2026-10-01_05-01-44_ant_terrain_heightscan", 123),
        ("contact_127d", "2026-10-02_00-46-02_contact_127d", 127),
    )
    result = {"reader": "syntax-only, no Python object construction", "runs": []}
    for label, name, dim in runs:
        directory = collector.ROOT / "logs/rsl_rl/ant" / name
        env = collector.load_yaml(directory / "params/env.yaml")
        agent = collector.load_yaml(directory / "params/agent.yaml")
        integrity = collector.checkpoint_audit(directory / "model_999.pt", dim)
        assert integrity["PASS"]
        tags, mapping, values, nonfinite = collector.scalar_data(directory)
        assert not nonfinite and all(values[key] for key in collector.TAGS)
        result["runs"].append({"label": label, "seed": env["seed"], "num_envs": env["scene"]["num_envs"],
                               "run_name": agent["run_name"], "experiment_name": agent["experiment_name"],
                               "observation_dim_from_checkpoint": integrity["actor_input_dim"],
                               "terrain_type": env["scene"]["terrain"]["terrain_type"],
                               "scalar_tag_mapping": mapping,
                               "final_scalars": {key: data[max(data)] for key, data in values.items()}, "PASS": True})
    # Even a callable-tagged input is inert data, never a shell/Python invocation.
    tagged = parse_config_text("value: !!python/object/apply:os.system ['not executed']")
    assert tagged["value"] == {"__yaml_tag__": "tag:yaml.org,2002:python/object/apply:os.system", "value": ["not executed"]}
    slice_value = parse_config_text("index: !!python/object/apply:builtins.slice [null, null, null]")
    assert slice_value["index"]["value"] == [None, None, None]
    for invalid in ("x: 1\nx: 2", "x: &recursive [*recursive]"):
        try:
            parse_config_text(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate/cyclic config was accepted")
    actual = json.loads(collector.RUNS_FILE.read_text())
    with tempfile.TemporaryDirectory(prefix="ant_collector_regression_") as temporary:
        collector.VALIDATION = Path(temporary)
        collector.RUNS_FILE = collector.VALIDATION / "runs.json"
        original_reader = collector.load_yaml
        def fail_reader(path):
            raise ValueError("intentional config parser fault for phase-isolation test")
        collector.load_yaml = fail_reader
        isolated = deepcopy(actual)
        collector.collect(isolated)
        base = isolated["runs"][0]
        assert base["training_status"] == "completed" and base["completed"]
        assert base["config_audit_status"] == "failed"
        assert base["checkpoint_status"] == base["tensorboard_status"] == base["summary_status"] == "pass"
        collector.load_yaml = original_reader
        original_scalars = collector.scalar_data
        def fail_scalars(path):
            raise ValueError("intentional TensorBoard reader fault")
        collector.scalar_data = fail_scalars
        isolated = deepcopy(actual)
        collector.collect(isolated)
        assert isolated["runs"][0]["training_status"] == "completed"
        assert isolated["runs"][0]["completed"] and isolated["runs"][0]["tensorboard_status"] == "failed"
        collector.scalar_data = original_scalars
    assert all(collector.sha256(collector.ROOT / name) == digest for name, digest in expected.items())
    result.update(python_tags_inert=True, parser_failure_isolated=True, tensorboard_failure_isolated=True,
                  migrated_checksums="16/16 unchanged", PASS=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
