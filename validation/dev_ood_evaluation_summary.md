# Development OOD evaluation summary

Training terrain is used for policy optimization. These fixed Dev OOD terrains are for repeated development comparison/model selection. Final unseen remains unbuilt and reserved until candidate policies are selected.

Protocol: seed=24, num_envs=100, exactly the first completed episode per env; reward is the original Ant environment reward. Progress success is forward displacement ≥ 2.0 m during the first episode.

All four policies in each environment use the same terrain task/seed, fixed 1.0 static/dynamic friction, stock Ant rewards, and the same terrain-relative 0.31 m torso clearance termination. The per-policy runtime adapters alter observation inputs only.

## UnevenBlocks

| Policy | Reward mean ± std | Distance mean ± std (m) | Length mean ± std (steps) | Fall % | Timeout % | Progress ≥2m % |
|---|---:|---:|---:|---:|---:|---:|
| Base (60D) | 31.214 ± 21.297 | 28.662 ± 19.711 | 502.500 ± 320.628 | 86.0% | 14.0% | 92.0% |
| HeightScan (123D) | 32.510 ± 21.552 | 29.833 ± 19.953 | 477.370 ± 290.532 | 93.0% | 7.0% | 98.0% |
| Contact (64D) | 30.694 ± 22.006 | 27.449 ± 19.879 | 487.540 ± 321.792 | 84.0% | 16.0% | 90.0% |
| HeightScan+Contact (127D) | 35.770 ± 23.242 | 31.777 ± 20.756 | 506.960 ± 304.469 | 87.0% | 13.0% | 97.0% |

### Pairwise ablation differences

Changes are new minus old for fall/timeout ratios (percentage points); reward and displacement are relative percent change. No overall policy score or ranking is computed.

| Pair | Reward change | Displacement change | Fall ratio Δ (pp) | Timeout ratio Δ (pp) |
|---|---:|---:|---:|---:|
| Base → HeightScan | +4.2% | +4.1% | +7.0 | -7.0 |
| Base → Contact | -1.7% | -4.2% | -2.0 | +2.0 |
| Base → HeightScan+Contact | +14.6% | +10.9% | +1.0 | -1.0 |
| HeightScan → HeightScan+Contact | +10.0% | +6.5% | -6.0 | +6.0 |
| Contact → HeightScan+Contact | +16.5% | +15.8% | +3.0 | -3.0 |

## SteppingStones

| Policy | Reward mean ± std | Distance mean ± std (m) | Length mean ± std (steps) | Fall % | Timeout % | Progress ≥2m % |
|---|---:|---:|---:|---:|---:|---:|
| Base (60D) | 19.093 ± 11.387 | 15.822 ± 9.969 | 652.080 ± 350.234 | 52.0% | 48.0% | 92.0% |
| HeightScan (123D) | 16.605 ± 8.183 | 10.969 ± 6.724 | 699.860 ± 327.431 | 49.0% | 51.0% | 96.0% |
| Contact (64D) | 17.556 ± 11.730 | 14.151 ± 10.148 | 558.190 ± 353.776 | 67.0% | 33.0% | 91.0% |
| HeightScan+Contact (127D) | 17.577 ± 9.844 | 12.099 ± 7.761 | 706.680 ± 328.288 | 43.0% | 57.0% | 95.0% |

### Pairwise ablation differences

Changes are new minus old for fall/timeout ratios (percentage points); reward and displacement are relative percent change. No overall policy score or ranking is computed.

| Pair | Reward change | Displacement change | Fall ratio Δ (pp) | Timeout ratio Δ (pp) |
|---|---:|---:|---:|---:|
| Base → HeightScan | -13.0% | -30.7% | -3.0 | +3.0 |
| Base → Contact | -8.1% | -10.6% | +15.0 | -15.0 |
| Base → HeightScan+Contact | -7.9% | -23.5% | -9.0 | +9.0 |
| HeightScan → HeightScan+Contact | +5.9% | +10.3% | -6.0 | +6.0 |
| Contact → HeightScan+Contact | +0.1% | -14.5% | -24.0 | +24.0 |

## GapPath

| Policy | Reward mean ± std | Distance mean ± std (m) | Length mean ± std (steps) | Fall % | Timeout % | Progress ≥2m % |
|---|---:|---:|---:|---:|---:|---:|
| Base (60D) | 1.492 ± 0.391 | 1.074 ± 0.375 | 64.310 ± 27.521 | 100.0% | 0.0% | 0.0% |
| HeightScan (123D) | 3.515 ± 3.784 | 0.873 ± 0.573 | 351.140 ± 392.618 | 71.0% | 29.0% | 0.0% |
| Contact (64D) | 1.490 ± 0.404 | 1.062 ± 0.359 | 65.080 ± 30.460 | 100.0% | 0.0% | 0.0% |
| HeightScan+Contact (127D) | 1.540 ± 1.276 | 1.011 ± 0.447 | 118.270 ± 174.891 | 96.0% | 4.0% | 0.0% |

### Pairwise ablation differences

Changes are new minus old for fall/timeout ratios (percentage points); reward and displacement are relative percent change. No overall policy score or ranking is computed.

| Pair | Reward change | Displacement change | Fall ratio Δ (pp) | Timeout ratio Δ (pp) |
|---|---:|---:|---:|---:|
| Base → HeightScan | +135.6% | -18.7% | -29.0 | +29.0 |
| Base → Contact | -0.1% | -1.1% | +0.0 | +0.0 |
| Base → HeightScan+Contact | +3.2% | -5.9% | -4.0 | +4.0 |
| HeightScan → HeightScan+Contact | -56.2% | +15.7% | +25.0 | -25.0 |
| Contact → HeightScan+Contact | +3.4% | -4.8% | -4.0 | +4.0 |

## Files

- Aggregate metrics: `validation/dev_ood_evaluation_results.csv`
- Episode-level raw data: `validation/dev_ood_evaluation_raw.csv`
- Simulator logs and source JSON: `validation/dev_ood_runs_20261003_025922/`

## Representative video commands

SteppingStones is selected to inspect separated footholds. Commands use env_0, seed 24, 960 steps, and follow-camera.

### Base

```bash
./isaaclab.sh -p scripts/assignment/run.py scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Ant-DevOOD-SteppingStones-v0 --checkpoint /home/zxro/IsaacLab_RS/logs/rsl_rl/ant/2026-10-02_03-39-50_ablation_base_obs60_s42/model_999.pt --num_envs 1 --seed 24 --video --video_length 960 --follow_robot --video_name dev_ood_steppingstones_base
```

### HeightScan

```bash
./isaaclab.sh -p scripts/assignment/run.py scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Ant-DevOOD-SteppingStones-v0 --checkpoint /home/zxro/IsaacLab_RS/logs/rsl_rl/ant/2026-10-03_00-52-49_ablation_heightscan_obs123_s42/model_999.pt --num_envs 1 --seed 24 --video --video_length 960 --follow_robot --video_name dev_ood_steppingstones_heightscan --height_scan
```

### Contact

```bash
./isaaclab.sh -p scripts/assignment/run.py scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Ant-DevOOD-SteppingStones-v0 --checkpoint /home/zxro/IsaacLab_RS/logs/rsl_rl/ant/2026-10-03_01-01-49_ablation_contact_obs64_s42/model_999.pt --num_envs 1 --seed 24 --video --video_length 960 --follow_robot --video_name dev_ood_steppingstones_contact --foot_contacts
```

### HeightScan+Contact

```bash
./isaaclab.sh -p scripts/assignment/run.py scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Ant-DevOOD-SteppingStones-v0 --checkpoint /home/zxro/IsaacLab_RS/logs/rsl_rl/ant/2026-10-03_01-10-01_ablation_heightscan_contact_obs127_s42/model_999.pt --num_envs 1 --seed 24 --video --video_length 960 --follow_robot --video_name dev_ood_steppingstones_heightscan_contact --height_scan --foot_contacts
```

No final unseen/composite environment was created or evaluated.
