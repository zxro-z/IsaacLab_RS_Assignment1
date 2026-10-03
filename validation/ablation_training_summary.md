# Ant sensor ablation training

Training statistics only; NOT dev/final unseen evaluation scores.
No performance ranking or policy selection is made here.

## Experiment Conditions

| Variant | Obs dim | Seed | Envs | Iterations | Terrain | Reward | PPO |
|---|---:|---:|---:|---:|---|---|---|
| Base | 60 | 42 | 4096 | 1000 | shared mixed, seed 42 | stock Ant | original Ant PPO |
| HeightScan | 123 | 42 | 4096 | 1000 | shared mixed, seed 42 | stock Ant | original Ant PPO |
| Contact | 64 | 42 | 4096 | 1000 | shared mixed, seed 42 | stock Ant | original Ant PPO |
| HeightScan+Contact | 127 | 42 | 4096 | 1000 | shared mixed, seed 42 | stock Ant | original Ant PPO |

## Training Completion

| Variant | Completed | Final checkpoint | Duration (s) | Exit code |
|---|---|---|---:|---:|
| Base | True (completed) | /home/zxro/IsaacLab_RS/logs/rsl_rl/ant/2026-10-02_03-39-50_ablation_base_obs60_s42/model_999.pt | 440.05 | 0 |
| HeightScan | True (completed) | /home/zxro/IsaacLab_RS/logs/rsl_rl/ant/2026-10-03_00-52-49_ablation_heightscan_obs123_s42/model_999.pt | 540.093 | 0 |
| Contact | True (completed) | /home/zxro/IsaacLab_RS/logs/rsl_rl/ant/2026-10-03_01-01-49_ablation_contact_obs64_s42/model_999.pt | 490.148 | 0 |
| HeightScan+Contact | True (completed) | /home/zxro/IsaacLab_RS/logs/rsl_rl/ant/2026-10-03_01-10-01_ablation_heightscan_contact_obs127_s42/model_999.pt | 550.154 | 0 |

## Training Scalars

| Variant | Final train reward | Max train reward | Max iteration | Final episode length |
|---|---:|---:|---:|---:|
| Base | 36.259361267089844 | 44.189422607421875 | 765 | 566.8300170898438 |
| HeightScan | 47.37748336791992 | 49.22859573364258 | 816 | 608.760009765625 |
| Contact | 39.715797424316406 | 43.66286849975586 | 814 | 564.5 |
| HeightScan+Contact | 49.58768844604492 | 52.08970260620117 | 855 | 649.1799926757812 |

## Integrity

CONFIG_AUDIT_RESULT: PASS
Only observation and unique output/name fields are excluded from the saved-config comparison.
Checkpoint hashes: `ablation_final_checkpoints.sha256`.

- Base: SHA256=a7304b58e1d06ba656b910d33edfdbcd9e08c28c8e541f80c095073242ff2570; NaN=False; OOM=False; status=completed
- HeightScan: SHA256=716a6bbb860078beb41766b31f3d35ff0775f875c4411f5c5c7055be95379aca; NaN=False; OOM=False; status=completed
- Contact: SHA256=d6b93941aa3732519f323e4c05e049621ad0ea66da86948eda8c66111722cd6c; NaN=False; OOM=False; status=completed
- HeightScan+Contact: SHA256=8fa361827664fce1a42a2f4d42152853c81a6a8cf63deb3174aca5b9d80d937a; NaN=False; OOM=False; status=completed

## Next Step

Design Dev OOD environments and evaluate all four policies under identical conditions.
No Dev OOD environment or evaluation is implemented by this script.
