# Frozen Final: confidence intervals and paired bootstrap

## A. Pairing validation

Pairing passed. All four runs use seed 24, terrain seed 2404, 100 envs, and matching ordered terrain-mesh/env-origin/material hashes. The stored ordered env origins and initial root XY arrays match exactly at every env_id. Original and measurement JSONs retain env_id 0–99 ordering, and measurement episodes reproduce all original outcomes. Root reset is deterministic relative to each origin; joint positions/velocities use common uniform reset samplers. The shared sensor scene and reset occur before policy-specific networks are constructed; observation corruption is disabled. Full initial joint/root vectors were not stored: matching those components is supported by the reviewed code/RNG initialization path, not a direct full-state measurement. Detailed source/manifest evidence is in the JSON.

## B. Statistical method

50,000 nonparametric bootstrap resamples, RNG seed 2404, NumPy PCG64. Each draw resamples 100 env_id blocks jointly across all policies/metrics. Percentile 95% intervals use the 2.5th/97.5th percentiles. A multinomial multiplicity vector implements uniform episode sampling with replacement. Conditional timing uses its own joint-completer subset. Continuous per-policy bootstrap CIs and normal reference CIs are both saved; reference SE uses sample variance (ddof=1), while displayed population std remains ddof=0. Binary per-policy CIs use Wilson. Paired binary differences use the same paired bootstrap and are reported in percentage points.

No p-values or multiplicity correction are used. Intervals are marginal, and the many reported metrics do not provide simultaneous confidence guarantees. The resampling units are episode/env indices within this single realization, not independent terrain or training seeds.

## C. Per-policy confidence intervals

| Policy | Return mean [bootstrap CI] | Distance mean m [bootstrap CI] | Length mean steps [bootstrap CI] |
|---|---:|---:|---:|
| Base | 60.504 [57.179, 63.399] | 55.772 [52.692, 58.453] | 889.940 [846.519, 927.330] |
| Original HeightScan | 67.512 [64.293, 70.236] | 62.437 [59.464, 64.960] | 890.280 [851.960, 922.300] |
| HeightScan+Contact | 69.610 [66.056, 72.607] | 61.914 [58.742, 64.604] | 901.560 [859.689, 936.210] |
| Diverse HeightScan+Contact | 65.277 [60.945, 69.266] | 59.295 [55.351, 62.929] | 829.410 [778.930, 875.260] |

| Policy | Fall | Original timeout | 40 m reach | Corridor-valid 40 m |
|---|---:|---:|---:|---:|
| Base | 19/100 (19%; CI 12.5–27.8) | 81/100 (81%; CI 72.2–87.5) | 92/100 (92%; CI 85.0–95.9) | 89/100 (89%; CI 81.4–93.7) |
| Original HeightScan | 23/100 (23%; CI 15.8–32.2) | 77/100 (77%; CI 67.8–84.2) | 94/100 (94%; CI 87.5–97.2) | 87/100 (87%; CI 79.0–92.2) |
| HeightScan+Contact | 16/100 (16%; CI 10.1–24.4) | 84/100 (84%; CI 75.6–89.9) | 94/100 (94%; CI 87.5–97.2) | 93/100 (93%; CI 86.3–96.6) |
| Diverse HeightScan+Contact | 33/100 (33%; CI 24.6–42.7) | 67/100 (67%; CI 57.3–75.4) | 85/100 (85%; CI 76.7–90.7) | 80/100 (80%; CI 71.1–86.7) |

All continuous mean/std/SE/normal reference intervals, velocity, and optional 18/25 m reach are in CSV/JSON. Existing means/std agree within rtol=1e-6, atol=1e-6; small reduction differences reflect original float32 versus new float64 arithmetic. Timeout retains the original evaluator labels, including the Base simultaneous fall/timeout episode classified as fall only.

## D. Prespecified paired continuous differences

| Comparison | Metric | Ref | New | Δ | 95% paired bootstrap CI |
|---|---|---:|---:|---:|---:|
| Base → Original HeightScan | Return | 60.504 | 67.512 | +7.007 | [+3.986, +10.266] |
| Base → Original HeightScan | Final displacement (m) | 55.772 | 62.437 | +6.666 | [+3.874, +9.673] |
| Base → Original HeightScan | Episode length (steps) | 889.940 | 890.280 | +0.340 | [-35.630, +39.250] |
| Original HeightScan → HeightScan+Contact | Return | 67.512 | 69.610 | +2.099 | [-0.439, +4.643] |
| Original HeightScan → HeightScan+Contact | Final displacement (m) | 62.437 | 61.914 | -0.523 | [-2.820, +1.795] |
| Original HeightScan → HeightScan+Contact | Episode length (steps) | 890.280 | 901.560 | +11.280 | [-19.480, +40.400] |
| HeightScan+Contact → Diverse HeightScan+Contact | Return | 69.610 | 65.277 | -4.333 | [-8.530, -0.199] |
| HeightScan+Contact → Diverse HeightScan+Contact | Final displacement (m) | 61.914 | 59.295 | -2.618 | [-6.406, +1.098] |
| HeightScan+Contact → Diverse HeightScan+Contact | Episode length (steps) | 901.560 | 829.410 | -72.150 | [-120.641, -25.180] |

All unconditional comparisons have 100 pairs. Δ=new−reference. Bootstrap distribution medians and mean centers are recorded alongside the intervals in CSV/JSON.

## E. Prespecified paired binary differences

| Comparison | Metric | Ref | New | Δ pp | 95% paired CI pp | Discordant pairs 1→0 / 0→1 |
|---|---|---:|---:|---:|---:|---:|
| Base → Original HeightScan | Fall | 19% | 23% | +4.0 | [-4.000, +12.000] | 6 / 10 |
| Base → Original HeightScan | 40 m reach | 92% | 94% | +2.0 | [-3.000, +8.000] | 3 / 5 |
| Base → Original HeightScan | Corridor-valid 40 m completion | 89% | 87% | -2.0 | [-10.000, +6.000] | 10 / 8 |
| Original HeightScan → HeightScan+Contact | Fall | 23% | 16% | -7.0 | [-14.000, +0.000] | 10 / 3 |
| Original HeightScan → HeightScan+Contact | 40 m reach | 94% | 94% | +0.0 | [-5.000, +5.000] | 3 / 3 |
| Original HeightScan → HeightScan+Contact | Corridor-valid 40 m completion | 87% | 93% | +6.0 | [-1.000, +13.000] | 3 / 9 |
| HeightScan+Contact → Diverse HeightScan+Contact | Fall | 16% | 33% | +17.0 | [+8.000, +26.000] | 3 / 20 |
| HeightScan+Contact → Diverse HeightScan+Contact | 40 m reach | 94% | 85% | -9.0 | [-17.000, -2.000] | 12 / 3 |
| HeightScan+Contact → Diverse HeightScan+Contact | Corridor-valid 40 m completion | 93% | 80% | -13.0 | [-21.000, -5.000] | 16 / 3 |

For fall, 1→0 means fall→no fall and a negative Δ is fewer falls. For reach/corridor outcomes, 1→0 means success→failure. Full 2×2 tables including both-zero/both-one counts, timeout and 18/25 m reach are saved in the binary CSV. No McNemar p-values are added.

## F. Conditional 40 m timing

**conditional on both policies completing the course**

| Comparison | Joint pairs | Ref mean s | New mean s | Δ s | 95% paired CI s |
|---|---:|---:|---:|---:|---:|
| Base → Original HeightScan | 89 | 10.198 | 9.238 | -0.959 | [-1.151, -0.764] |
| Original HeightScan → HeightScan+Contact | 91 | 9.161 | 9.434 | +0.273 | [+0.101, +0.442] |
| HeightScan+Contact → Diverse HeightScan+Contact | 82 | 9.424 | 9.005 | -0.418 | [-0.621, -0.202] |

The subset is defined using existing terrain-local x=40 reach, not corridor validity. Pairs where only one policy reaches 40 m are excluded. This selection changes the comparison population, so shorter timing does not imply higher overall completion or robustness. Missing t_40m values belong only to non-reachers and are not imputed.

## G. Interpretation within this preregistered Final realization

### Base → Original HeightScan

Return: observed Δ +7.007 stock reward, CI [+3.986, +10.266]; the observed direction was positive, and the marginal paired interval excluded zero.
Final displacement (m): observed Δ +6.666 m, CI [+3.874, +9.673]; the observed direction was positive, and the marginal paired interval excluded zero.
Episode length (steps): observed Δ +0.340 steps, CI [-35.630, +39.250]; the observed direction was positive, and the paired interval included zero.
Fall: observed Δ +4.000 pp, CI [-4.000, +12.000]; the observed direction was positive, and the paired interval included zero.
40 m reach: observed Δ +2.000 pp, CI [-3.000, +8.000]; the observed direction was positive, and the paired interval included zero.
Corridor-valid 40 m completion: observed Δ -2.000 pp, CI [-10.000, +6.000]; the observed direction was negative, and the paired interval included zero.
Descriptive Return relative change: +11.58% (preserved original published value). This percentage is not the absolute-difference CI.
Descriptive Final displacement (m) relative change: +11.95% (preserved original published value). This percentage is not the absolute-difference CI.
The shorter first-pass timing and positive whole-episode displacement difference are related progress measurements but measure different portions of the trajectory. First-course endpoint reach changes by only +2 pp, while whole-episode distance includes movement beyond the initial course. The pattern is consistent with terrain-awareness-related progress differences in this realization; it does not identify a causal sensor effect.

### Original HeightScan → HeightScan+Contact

Return: observed Δ +2.099 stock reward, CI [-0.439, +4.643]; the observed direction was positive, and the paired interval included zero.
Final displacement (m): observed Δ -0.523 m, CI [-2.820, +1.795]; the observed direction was negative, and the paired interval included zero.
Episode length (steps): observed Δ +11.280 steps, CI [-19.480, +40.400]; the observed direction was positive, and the paired interval included zero.
Fall: observed Δ -7.000 pp, CI [-14.000, +0.000]; the observed direction was negative, and the paired interval included zero.
40 m reach: observed Δ +0.000 pp, CI [-5.000, +5.000]; the observed direction was zero, and the paired interval included zero.
Corridor-valid 40 m completion: observed Δ +6.000 pp, CI [-1.000, +13.000]; the observed direction was positive, and the paired interval included zero.
Descriptive Return relative change: +3.11% (preserved original published value). This percentage is not the absolute-difference CI.
Descriptive Final displacement (m) relative change: -0.84% (preserved original published value). This percentage is not the absolute-difference CI.
Existing section analysis localized the total fall reduction (23→16) to post-course falls (17→10); pre-course falls remained 6→6, and X endpoint reach remained 94%→94%. The paired fall interval describes uncertainty around that observed reduction, rather than establishing it. Near-zero displacement and unchanged X completion are consistent with the prior post-course-stability interpretation, while corridor-valid completion increases descriptively.

### HeightScan+Contact → Diverse HeightScan+Contact

Return: observed Δ -4.333 stock reward, CI [-8.530, -0.199]; the observed direction was negative, and the marginal paired interval excluded zero.
Final displacement (m): observed Δ -2.618 m, CI [-6.406, +1.098]; the observed direction was negative, and the paired interval included zero.
Episode length (steps): observed Δ -72.150 steps, CI [-120.641, -25.180]; the observed direction was negative, and the marginal paired interval excluded zero.
Fall: observed Δ +17.000 pp, CI [+8.000, +26.000]; the observed direction was positive, and the marginal paired interval excluded zero.
40 m reach: observed Δ -9.000 pp, CI [-17.000, -2.000]; the observed direction was negative, and the marginal paired interval excluded zero.
Corridor-valid 40 m completion: observed Δ -13.000 pp, CI [-21.000, -5.000]; the observed direction was negative, and the marginal paired interval excluded zero.
Descriptive Return relative change: -6.22% (preserved original published value). This percentage is not the absolute-difference CI.
Descriptive Final displacement (m) relative change: -4.23% (preserved original published value). This percentage is not the absolute-difference CI.
Diverse has fewer X/corridor completions and more falls in the observed episodes, despite shorter timing among joint completers. The timing subset excludes one-policy failures; it cannot establish an overall speed/robustness advantage. Earlier section analysis localized six Diverse pre-course falls in Landing versus zero for Contact. Broader training distribution did not translate into uniformly higher observed Final outcomes in this fixed realization.

The result does not establish a general effect across unseen terrains. There is one fixed terrain realization, one evaluation seed and one training seed, with no independent retraining replication. Resampling 100 episode indices quantifies conditional episode variability and does not make the terrain/test realization itself representative.

## H. Integrity and validation

No simulation, training, policy selection or tuning. All 520 pre-existing protected artifacts/checkpoints/sources retained their hashes. Original 400 rows and all first-pass/preregistration/parity artifacts were preserved. Original population means/std and fall/timeout counts were verified. Missingness, sample sizes, interval bounds, paired weighting, bootstrap centers and estimate containment were checked.

New script: `scripts/assignment/analyze_final_unseen_statistics.py`. Outputs: `final_unseen_policy_ci.csv`, `final_unseen_pairwise_bootstrap.csv`, `final_unseen_pairwise_binary.csv`, `final_unseen_statistical_analysis.json`, this report, and the paired-CI figure. Full protection snapshot, pairing source fingerprints, independent validation and commands are in `final_unseen_statistics_runs/`.

![Paired confidence intervals](final_unseen_statistical_analysis.png)
