# Terrain diversity: identical 127-D HeightScan + Contact

Training distribution only changes. Reward is stock Ant; original scan/contact, termination and PPO unchanged.
Training statistics are not Dev OOD scores. One training seed; no statistical causal/convergence claim.
Final unseen is not created. Dev OOD is development-exposed and reusable for model selection.

## Training

```json
{
  "training_status": "completed",
  "wall_clock_duration": 545.093,
  "checkpoint_integrity": {
    "iteration": 999,
    "actor_input_dim": 127,
    "critic_input_dim": 127,
    "action_dim": 8,
    "optimizer_state_present": true,
    "all_tensors_finite": true,
    "sha256": "2f36d5fc0280a9000b1719552ab1c68cb64861a22b9088440f19acd5217dfe48",
    "PASS": true
  },
  "training_statistics": {
    "final_mean_reward": 45.85015106201172,
    "final_episode_length": 582.9600219726562,
    "final_value_loss": 0.3744092583656311,
    "final_surrogate_loss": -0.004204336553812027,
    "final_noise_std": 0.13370171189308167,
    "final_learning_rate": 0.0002601229352876544,
    "maximum_mean_reward": 51.0830192565918,
    "maximum_reward_iteration": 905
  },
  "config_audit_status": "pass"
}
```

## Dev OOD: existing vs diverse

Seed 24, 100 envs, first completed episode only; original environment-returned Ant reward.
All three meshes, env origins and robot material hashes matched the preserved existing-policy evaluations.

### UnevenBlocks

| Environment | Training | Reward mean ± std | Distance (m) | Fall % | Timeout % | ≥2m % | ≥5m % |
|---|---|---:|---:|---:|---:|---:|---:|
| UnevenBlocks | existing | 35.770 ± 23.242 | 31.777 | 87.0 | 13.0 | 97.0 | 83.0 |
| UnevenBlocks | diverse | 33.916 ± 22.621 | 30.802 | 93.0 | 7.0 | 96.0 | 82.0 |

Pairwise effects: `{"reward_percent": -5.180799496971034, "displacement_percent": -3.068663879787866, "fall_ratio_delta_pp": 6.000000000000005, "timeout_ratio_delta_pp": -6.0, "progress_2m_ratio_delta_pp": -0.9999990463256836, "progress_5m_ratio_delta_pp": -1.0000007152557333}`


### SteppingStones

| Environment | Training | Reward mean ± std | Distance (m) | Fall % | Timeout % | ≥2m % | ≥5m % |
|---|---|---:|---:|---:|---:|---:|---:|
| SteppingStones | existing | 17.577 ± 9.844 | 12.099 | 43.0 | 57.0 | 95.0 | 71.0 |
| SteppingStones | diverse | 18.202 ± 10.184 | 13.075 | 38.0 | 62.0 | 96.0 | 72.0 |

Pairwise effects: `{"reward_percent": 3.559749496456886, "displacement_percent": 8.067682983926982, "fall_ratio_delta_pp": -4.999999999999999, "timeout_ratio_delta_pp": 5.000000000000004, "progress_2m_ratio_delta_pp": 0.9999990463256836, "progress_5m_ratio_delta_pp": 0.9999969005584752}`


### GapPath

| Environment | Training | Reward mean ± std | Distance (m) | Fall % | Timeout % | ≥2m % | ≥5m % |
|---|---|---:|---:|---:|---:|---:|---:|
| GapPath | existing | 1.540 ± 1.276 | 1.011 | 96.0 | 4.0 | 0.0 | 0.0 |
| GapPath | diverse | 1.390 ± 1.490 | 0.943 | 94.0 | 6.0 | 0.0 | 0.0 |

Pairwise effects: `{"reward_percent": -9.736228668593272, "displacement_percent": -6.682944738015551, "fall_ratio_delta_pp": -2.0000000000000018, "timeout_ratio_delta_pp": 1.9999999999999998, "progress_2m_ratio_delta_pp": 0.0, "progress_5m_ratio_delta_pp": 0.0}`

GapPath remains a stress case unless traversal actually increases; reward without progress is not traversal improvement.
The older ≥5m ratio is derived from preserved first-episode displacement records using the same ≥5m criterion.

## Audit and failure recovery

DIVERSE_SOURCE_CONFIG_AUDIT: PASS; DIVERSE_RUNTIME_CONFIG_AUDIT: PASS; REWARD_AUDIT_RESULT: MATCHES_BASIC_ANT.
The saved/runtime audit compares the entire terrain subtree. It runs InteractiveScene/TerrainGenerator's actual native expansion,
which fills importer num_envs/env_spacing, formats entity paths and assigns sub-terrain size/HF scales.
Explicit comparison normalizations are only the requested 4-env diagnostic size, unique train output paths and agent run_name.
Both saved and source dictionaries use the same inert YAML-node representation; raw YAML is never rewritten or object-deserialized.
Proof: interactive_scene.py:_add_entities_from_cfg and terrain_generator.py:TerrainGenerator.__init__.
Earlier failures were actual config assertions followed by native shutdown hang/segfault, not a completed training run.
The stale failed validation worker was stopped; its logs remain. Current env.close and app shutdown return promptly.
Native fast shutdown can mask exception exit codes, so PASS JSON, explicit phase markers and checkpoint/iteration checks are required.
The outer ±0.85 m spawn check hit pre-existing stair steps. Native 0.4 m stairs shrink their plateau discretely,
so platform_width=2.0 does not guarantee an exact 2 m plateau. Geometry was not changed; center ±0.5 m checked zero for all 100 patches.
Both 4-env tests completed 64 zero-action steps: obs (4,127), action (4,8), scan (4,63), contact (4,4), finite and binary.

## Training terrain and leakage

Original: flat 30%, slopes 60%, low upward stairs 10%. Diverse: flat 20%, slopes 50%, low stairs 10%, random uniform 20%.
Easy/medium/harder up and down proportions are 12.5%, 8.333333%, 4.166667% each; existing slope/stair parameters are unchanged.
Random uniform: sampled heights -0.05..0.05 m in 0.025 m increments, coarse spacing 0.4 m, central platform 2 m, native mesh spacing 0.1 m.
Native bicubic interpolation is retained; this is continuous rough height variation, not literal separated blocks/footholds.
Measured random-uniform patch Z extent: -0.130..0.095 m. Sampling range is not a strict post-interpolation surface bound.
10×10 layout, seed 42, curriculum disabled; sampled proportions are probabilities, not exact patch quotas.
Actual patch counts: `{"flat": 25, "up_easy": 11, "down_easy": 14, "up_medium": 7, "down_medium": 9, "up_harder": 2, "down_harder": 3, "low_up_stairs": 15, "random_uniform": 14}`.

| Dev OOD | Native primitive | Exact training primitive/layout overlap | Qualification |
|---|---|---|---|
| UnevenBlocks | HfDiscreteObstaclesTerrainCfg | No / No | Shared irregular-height concept; not a final unseen claim |
| SteppingStones | HfSteppingStonesTerrainCfg | No / No | No stone-width/gap layout in training |
| GapPath | MeshGapTerrainCfg | No / No | No explicit trench/gap in training |

## Interpretation

SteppingStones: reward +3.56%, distance +8.07%, fall -5 pp, timeout +5 pp, ≥2m +1 pp, ≥5m +1 pp.
There is a modest progress/stability improvement for this training/evaluation seed pair, not evidence of a large or statistically significant generalization gain.
Timeout did not decrease. Aggregate results do not establish that stable stagnation was removed; trajectory analysis would be needed.
UnevenBlocks: distance -3.07%, fall +6 pp. No broad improvement across Dev OOD was observed.
GapPath: distance decreased and ≥2m/≥5m remain zero; traversal generalization was not improved.
The observation and PPO did not change, but one training seed cannot establish that distribution diversity is generally more important than perception complexity.

## Integrity and next step

Protected checkpoints, previous validation/results and Ant training/Dev OOD/reward/PPO source checksums: PASS.
py_compile and git diff --check are checked separately after generation.
Recommended next step: inspect the existing-vs-diverse SteppingStones trajectory without changing geometry, then use paired additional training seeds if authorized.
No Final unseen environment, new sensor, reward shaping or PPO tuning was added.
