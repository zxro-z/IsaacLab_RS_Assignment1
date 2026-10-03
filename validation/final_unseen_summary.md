# Frozen Final Unseen official evaluation

Preregistered before mesh generation; validated with zero actions before any candidate inference. No post-result terrain or policy changes.
Seed 24, 100 envs per candidate, exactly the first completed episode; original Ant reward from env.step, population std.
One deterministic repeated composite course and one evaluation seed: conclusions are limited to this test, not universal policy rankings.

| Policy | Obs | Training difference | Return mean ± std | Distance mean ± std m | Fall | Timeout | ≥2m | ≥5m |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Base | 60 | Flat + slopes + low stairs | 60.504 ± 15.942 | 55.772 ± 14.751 | 19% | 81% | 97% | 96% |
| Original HeightScan | 123 | Flat + slopes + low stairs | 67.512 ± 15.075 | 62.437 ± 13.955 | 23% | 77% | 98% | 97% |
| HeightScan+Contact | 127 | Flat + slopes + low stairs | 69.610 ± 16.748 | 61.914 ± 14.977 | 16% | 84% | 96% | 96% |
| Diverse HeightScan+Contact | 127 | 20% uniform roughness added | 65.277 ± 21.282 | 59.295 ± 19.369 | 33% | 67% | 97% | 97% |

## Prespecified pairwise comparisons

```json
[
  {
    "from": "Base",
    "to": "Original HeightScan",
    "reward_percent": 11.581262369684657,
    "displacement_percent": 11.951561422186412,
    "fall_ratio_delta_pp": 4.000000000000001,
    "timeout_ratio_delta_pp": -4.0000000000000036,
    "progress_2m_ratio_delta_pp": 0.9999990463256836,
    "progress_5m_ratio_delta_pp": 0.9999990463256836
  },
  {
    "from": "Original HeightScan",
    "to": "HeightScan+Contact",
    "reward_percent": 3.1084081922324547,
    "displacement_percent": -0.8383135778255242,
    "fall_ratio_delta_pp": -7.000000000000001,
    "timeout_ratio_delta_pp": 6.999999999999995,
    "progress_2m_ratio_delta_pp": -1.9999980926513672,
    "progress_5m_ratio_delta_pp": -0.9999990463256836
  },
  {
    "from": "HeightScan+Contact",
    "to": "Diverse HeightScan+Contact",
    "reward_percent": -6.2243360289191205,
    "displacement_percent": -4.229141410394835,
    "fall_ratio_delta_pp": 17.0,
    "timeout_ratio_delta_pp": -16.999999999999993,
    "progress_2m_ratio_delta_pp": 0.9999990463256836,
    "progress_5m_ratio_delta_pp": 0.9999990463256836
  }
]
```

No composite score/ranking. Pairwise deltas are descriptive; one training seed does not establish causal or statistically significant sensor/diversity effects.
Timeout is not equivalent to traversal success; use distance and ≥2m/≥5m alongside survival. Stagnation needs trajectory evidence.

## Freeze and integrity

{
  "result": "PASS",
  "changed": [],
  "protected_previous_files": 396,
  "raw_rows": 400,
  "policy_count": 4,
  "policy_observation_only_differences": true,
  "source_unchanged_after_preregistration": true,
  "preregister_sha256": "5a06d1a934fcec84a3e842358f925c234bf5574e2f1cbf04a65e02182fe95ab2",
  "shared_official_physical_hashes": {
    "terrain_mesh_sha256": "c0e1eb34d72dbc86d0a64774a27878aeb9acd9de2174cb0853f99703234cc282",
    "env_origins_sha256": "cd6e637900ffe35bf76ff3c805ff8085e42d3afdc63d5c036d8f57c22ac9d957",
    "robot_material_sha256": "37aa67b2f006497c1e88625a1461557e8b5d5db2c425bbbed6076de236f01fe5"
  }
}

The -1 m minimum of the combined mesh includes the native outer terrain border, not a deep hole in the course. The course basin is 0.07 m deep; terraces are at most 0.075 m high.
100 patches repeat the same 40×8 m course; crossing a tile end enters another course or native padding. No new success termination was added.
The Final test is now exposed: no further tuning/retraining against it and no geometry/difficulty changes, regardless of results.

## Prespecified effect / trade-off interpretation

- Base → Original: return +11.58%, displacement +11.95%, fall +4 pp. Higher progress did not coincide with fewer falls.
- Original → HeightScan+Contact: return +3.11%, displacement −0.84%, fall −7 pp, timeout +7 pp. Contact fusion coincided with longer survival rather than extra distance in this test.
- Existing 127-D → Diverse 127-D: return −6.22%, displacement −4.23%, fall +17 pp, timeout −17 pp. The SteppingStones Dev improvement did not carry over to this frozen Final.
- ≥5m is 96–97% for every candidate and is close to a ceiling on this mild course. It should not be treated as a strong discriminator.
- Distances exceed the 40 m patch length for many episodes. This is a repeated/padded composite terrain exposure, not a single bounded course completion test; no completion termination was added or retrofitted.
- These are descriptive comparisons for one training seed and one fixed Final realization. No overall score, ranking, statistical significance or universal sensor/diversity claim is made.

## Representative trajectory and freeze provenance

Before Dev closure, Existing/Diverse seed24 SteppingStones episodes used identical initial state and 127-D inputs, 1280×720 video at 60 fps with the same root-follow camera.

| Metric | Existing | Diverse |
|---|---:|---:|
| Final displacement (m) | 16.892 | 17.483 |
| Duration (s) / termination | 11.317 / fall | 16.000 / timeout |
| Time to 2m (s) | 1.100 | 1.183 |
| Time to 5m (s) | 2.650 | 4.283 |
| Mean vx (m/s) | 1.482 | 1.085 |
| Low-speed step fraction | 3.09% | 6.56% |
| 1s signed-low-progress window fraction | 0% | 4.33% |
| Longest continuous low-speed interval (s) | 0.05 | 0.15 |
| Mean action L2 | 1.228 | 1.517 |
| Mean contact feet | 1.041 | 1.121 |
| Contact transitions summed across feet (Hz) | 33.844 | 39.875 |

Diverse's representative distance gain is +3.50%, not the aggregate +8.07%. It starts more slowly, survives longer and keeps progressing late; the timeout is not explained by a sustained static stance. Additional seeds were not necessary and were not run. This trajectory does not replace the 100-env Dev results.

Video paths, complete metrics, contacts, pre-reset terminal state and separate 63-D scan NPZ are in `existing_vs_diverse_rollout/manifest.json`, `comparison.md`, `behavior_analysis.json`, and the two policy subdirectories. Sampled late video frames confirm the robot stays centered; original videos were not altered.

`dev_experiment_freeze.json` records composed Dev config hashes, relevant source hashes, selected checkpoint SHA256, Git commit/dirty status and the no-further-tuning declaration. `final_unseen_preregister.json` was saved before first mesh generation, and includes all analytic geometry parameters, seed/friction, inherited reward/termination/config and source hashes. Four policy-free envs completed 64 zero-action steps with approximately 0.5 m initial clearance before any candidate inference.

Final task: `Isaac-Ant-Final-Unseen-v0`; native HF mesher, 40×8 m patch, 10×10 layout, seed 2404. Spawn local position (2,4,0), flat X 0–4 m; asymmetric ridge X 4–10 m; offset terrace heights 0.040/0.075/0.025 m at unequal lengths; smooth 0.070 m basin X 18–25 m; deterministic smooth mixed-frequency landing X 25–40 m. Static/dynamic friction both 1.0, restitution zero. Original stock Ant reward and terrain-relative clearance <0.31 m termination are unchanged.

Next step is presentation/submission packaging only. No new Final-driven policy or environment variants are authorized by this result.
