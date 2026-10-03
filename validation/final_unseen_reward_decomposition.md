# Frozen Final stock reward decomposition

## A. Actual stock reward definition

Final inherits AntEnvCfg.RewardsCfg without replacing any reward term. Ant imports reward helpers from classic/humanoid/mdp and common isaaclab/envs/mdp. The seven runtime terms, exact function identifiers, weights and parameters were read from RewardManager and checked against the stock config.

| Term | Function | Weight | Definition |
|---|---|---:|---|
| progress | `isaaclab_tasks.manager_based.classic.humanoid.mdp.rewards.progress_reward` | 1 | Potential difference towards world target (1000,0,0). __call__ zeros target-vector z before distance; reset uses initial 3-D distance. Potential=-distance/control_dt, then manager multiplies by dt. No separate linear-velocity reward. |
| alive | `isaaclab.envs.mdp.rewards.is_alive` | 0.5 | (~termination_manager.terminated).float(): 1 for non-fall steps including pure timeout; 0 on non-timeout termination, including simultaneous fall+timeout. |
| upright | `isaaclab_tasks.manager_based.classic.humanoid.mdp.rewards.upright_posture_bonus` | 0.1 | 1 if base-up world-Z projection > 0.93, else 0. |
| move_to_target | `isaaclab_tasks.manager_based.classic.humanoid.mdp.rewards.move_to_target_bonus` | 0.5 | Heading projection towards target: 1 if projection > 0.8, else projection/0.8 (may be negative). |
| action_l2 | `isaaclab.envs.mdp.rewards.action_l2` | -0.005 | sum(action_manager.action squared). |
| energy | `isaaclab_tasks.manager_based.classic.humanoid.mdp.rewards.power_consumption` | -0.05 | sum(abs(action * joint_velocity * normalized_gear_ratio)); all gear ratios 15, normalized by maximum => 1. This is the stock proxy, not a new physical energy measurement. |
| joint_pos_limits | `isaaclab_tasks.manager_based.classic.humanoid.mdp.rewards.joint_pos_limits_penalty_ratio` | -0.1 | Normalize joints to [-1,1] using soft limits; sum (abs(normalized_position)>0.99) * (abs(position)-0.99)/(1-0.99) * normalized_gear_ratio. |

Every control step uses raw_value × weight × control_dt (0.0166666666666667 s). Terminations are computed first, then rewards before reset, so terminal-step contributions are included. Alive alone gates on terminated; timeout alone retains alive. There is no explicit terminal penalty or standalone velocity term. Progress is a target-potential difference and is not exactly the legacy forward-displacement metric.

## B. Data sufficiency and instrumentation

Prior Final/first-pass logs had total reward only. Four frozen 100-env first-episode measurement runs logged existing RewardManager._step_reward buffers. This buffer is already weighted but has dt divided out; dt was restored for analysis. No term was recomputed, RewardManager.compute was not replaced, and no evaluator/environment source/config was edited. Existing reset extras average over reset envs and normalize by max duration, so they cannot recover per-env component trajectories.

Every old episode return/length/legacy displacement/fall/timeout was reproduced. All saved step X positions, total rewards and done/terminated/truncated/fall signals also matched the first-pass trajectory, preserving the first-reach boundary split. The native env.step total remains the authoritative Final reward.

## C. Reward identity validation

Across 351119 first-episode control transitions, max |sum components − env reward| = 2.80560925603e-08; mean absolute step error = 3.85584635359e-09.
Max accumulated component-vs-float64 env trace error = 2.82873224933e-06. Max component-vs-native float32 episode return error = 0.000110682182139; mean absolute error = 2.40521127248e-05.
Step/float64-trace checks use rtol=1e-6, atol=1e-6. Separate component accumulation changes summation order: 8/400 native comparisons exceed nominal rtol=1e-6, while all pass rtol=2e-6, atol=1e-6 and the explicit float32 sequential-sum gamma_N error bound. This small reduction difference does not change any env reward or original episode return. The read-only buffer’s dt divide/multiply round trip is included in the measured step error.

Pairwise reconciliation:

| Comparison | Native total Δ | Sum component Δ | Residual |
|---|---:|---:|---:|
| Base → Original HeightScan | +7.007175798 | +7.007174727 | -1.07130266e-06 |
| Original HeightScan → HeightScan+Contact | +2.098533256 | +2.098530837 | -2.41969955e-06 |
| HeightScan+Contact → Diverse HeightScan+Contact | -4.332762902 | -4.332751981 | +1.09207355e-05 |

## D. Policy-level accumulated weighted contributions

| Reward component | Base | Original | Contact | Diverse |
|---|---:|---:|---:|---:|
| progress | 55.823319 | 62.495533 | 61.970421 | 59.360711 |
| alive | 7.414584 | 7.417084 | 7.511667 | 6.909000 |
| upright | 1.387350 | 1.381767 | 1.464467 | 1.294800 |
| move_to_target | 7.240414 | 7.250634 | 7.396902 | 6.597338 |
| action_l2 | -0.139703 | -0.181474 | -0.105305 | -0.124915 |
| energy | -7.404162 | -6.320442 | -5.448845 | -6.143183 |
| joint_pos_limits | -3.817451 | -4.531576 | -3.179251 | -2.616447 |
| total_reward_from_env | 60.504350 | 67.511526 | 69.610059 | 65.277296 |

Mean/std/median/paired-block bootstrap mean CIs and supplementary signed contribution fractions are in CSV/JSON. Fractions are not a share of positive performance because negative terms offset positive terms.

Per-step contribution, mean of per-episode rates:

| Component | Base | Original | Contact | Diverse |
|---|---:|---:|---:|---:|
| progress | 0.0603771 | 0.0686967 | 0.0662014 | 0.0689896 |
| alive | 0.0083268 | 0.0083280 | 0.0083265 | 0.0083241 |
| upright | 0.0014981 | 0.0015109 | 0.0015638 | 0.0014914 |
| move_to_target | 0.0079938 | 0.0079999 | 0.0080230 | 0.0077794 |
| action_l2 | -0.0001582 | -0.0002063 | -0.0001162 | -0.0001533 |
| energy | -0.0081151 | -0.0070059 | -0.0058732 | -0.0072309 |
| joint_pos_limits | -0.0043451 | -0.0051747 | -0.0036544 | -0.0033265 |
| total_reward_from_env | 0.0655774 | 0.0741486 | 0.0744709 | 0.0758739 |

Per-second rates equal these per-step rates divided by control_dt; their means/std/median/CIs are also saved. Rates use equal episode weighting; pooled exposure rates would answer a different question.

## E. Prespecified paired component deltas

50,000 bootstrap resamples, RNG seed 2404, percentile 95% CI; same env_id blocks across policies/components. All main pairs have n=100. Component CIs are exploratory/marginal; no p-values or multiplicity correction. Δ=new−reference.

| Comparison | Component | Δ contribution | 95% paired CI |
|---|---|---:|---:|
| Base → Original HeightScan | progress | +6.672214 | [+3.8770, +9.6829] |
| Base → Original HeightScan | alive | +0.002500 | [-0.2978, +0.3273] |
| Base → Original HeightScan | upright | -0.005583 | [-0.0713, +0.0656] |
| Base → Original HeightScan | move_to_target | +0.010220 | [-0.3211, +0.3731] |
| Base → Original HeightScan | action_l2 | -0.041771 | [-0.0486, -0.0352] |
| Base → Original HeightScan | energy | +1.083720 | [+0.7423, +1.3958] |
| Base → Original HeightScan | joint_pos_limits | -0.714125 | [-0.8788, -0.5570] |
| Base → Original HeightScan | total_reward_from_env | +7.007176 | [+3.9862, +10.2656] |
| Original HeightScan → HeightScan+Contact | progress | -0.525112 | [-2.8207, +1.7964] |
| Original HeightScan → HeightScan+Contact | alive | +0.094583 | [-0.1623, +0.3377] |
| Original HeightScan → HeightScan+Contact | upright | +0.082700 | [+0.0283, +0.1354] |
| Original HeightScan → HeightScan+Contact | move_to_target | +0.146268 | [-0.1180, +0.4026] |
| Original HeightScan → HeightScan+Contact | action_l2 | +0.076170 | [+0.0708, +0.0813] |
| Original HeightScan → HeightScan+Contact | energy | +0.871597 | [+0.6586, +1.0835] |
| Original HeightScan → HeightScan+Contact | joint_pos_limits | +1.352325 | [+1.2228, +1.4775] |
| Original HeightScan → HeightScan+Contact | total_reward_from_env | +2.098533 | [-0.4391, +4.6435] |
| HeightScan+Contact → Diverse HeightScan+Contact | progress | -2.609710 | [-6.4002, +1.1100] |
| HeightScan+Contact → Diverse HeightScan+Contact | alive | -0.602667 | [-1.0075, -0.2107] |
| HeightScan+Contact → Diverse HeightScan+Contact | upright | -0.169667 | [-0.2574, -0.0834] |
| HeightScan+Contact → Diverse HeightScan+Contact | move_to_target | -0.799564 | [-1.2143, -0.3871] |
| HeightScan+Contact → Diverse HeightScan+Contact | action_l2 | -0.019610 | [-0.0261, -0.0127] |
| HeightScan+Contact → Diverse HeightScan+Contact | energy | -0.694338 | [-1.0474, -0.3203] |
| HeightScan+Contact → Diverse HeightScan+Contact | joint_pos_limits | +0.562804 | [+0.4236, +0.7044] |
| HeightScan+Contact → Diverse HeightScan+Contact | total_reward_from_env | -4.332763 | [-8.5296, -0.1986] |

## F. Pre/post first 40 m endpoint

Pre40 includes the first crossing control transition. Post40 starts on its following transition. Noncompleters contribute their whole episode to pre40 and zero to post40. All 100 envs remain in each mean/pair; pre40 group durations differ by policy, so segment differences are descriptive rather than matched-exposure effects.

| Comparison | Component | Pre40 Δ | Pre40 CI | Post40 Δ | Post40 CI |
|---|---|---:|---:|---:|---:|
| Base → Original HeightScan | progress | +1.039847 | [-0.0705, +2.3559] | +5.632367 | [+3.7006, +7.5672] |
| Base → Original HeightScan | alive | -0.360333 | [-0.5043, -0.2050] | +0.362833 | [+0.1401, +0.5878] |
| Base → Original HeightScan | upright | -0.069667 | [-0.1020, -0.0338] | +0.064083 | [+0.0210, +0.1077] |
| Base → Original HeightScan | move_to_target | -0.332687 | [-0.4930, -0.1539] | +0.342907 | [+0.1170, +0.5715] |
| Base → Original HeightScan | action_l2 | -0.018415 | [-0.0216, -0.0153] | -0.023356 | [-0.0283, -0.0183] |
| Base → Original HeightScan | energy | +1.016958 | [+0.8344, +1.1837] | +0.066762 | [-0.1418, +0.2706] |
| Base → Original HeightScan | joint_pos_limits | -0.227825 | [-0.3073, -0.1520] | -0.486300 | [-0.6164, -0.3549] |
| Base → Original HeightScan | total_reward_from_env | +1.047878 | [-0.1568, +2.4758] | +5.959296 | [+3.8740, +8.0534] |
| Original HeightScan → HeightScan+Contact | progress | -0.592949 | [-1.7331, +0.3436] | +0.067837 | [-1.5007, +1.7497] |
| Original HeightScan → HeightScan+Contact | alive | +0.031833 | [-0.1369, +0.1743] | +0.062750 | [-0.1031, +0.2417] |
| Original HeightScan → HeightScan+Contact | upright | +0.044000 | [+0.0118, +0.0719] | +0.038700 | [+0.0045, +0.0750] |
| Original HeightScan → HeightScan+Contact | move_to_target | +0.050205 | [-0.1111, +0.1892] | +0.096063 | [-0.0768, +0.2825] |
| Original HeightScan → HeightScan+Contact | action_l2 | +0.046968 | [+0.0442, +0.0499] | +0.029202 | [+0.0247, +0.0333] |
| Original HeightScan → HeightScan+Contact | energy | +0.536230 | [+0.4190, +0.6636] | +0.335366 | [+0.1794, +0.4788] |
| Original HeightScan → HeightScan+Contact | joint_pos_limits | +0.824136 | [+0.7465, +0.9119] | +0.528189 | [+0.4176, +0.6301] |
| Original HeightScan → HeightScan+Contact | total_reward_from_env | +0.940423 | [-0.3531, +1.9960] | +1.158107 | [-0.5553, +2.9869] |
| HeightScan+Contact → Diverse HeightScan+Contact | progress | -0.963747 | [-2.5696, +0.6207] | -1.645964 | [-4.1758, +0.7914] |
| HeightScan+Contact → Diverse HeightScan+Contact | alive | -0.293000 | [-0.4873, -0.1013] | -0.309667 | [-0.5802, -0.0482] |
| HeightScan+Contact → Diverse HeightScan+Contact | upright | -0.088633 | [-0.1305, -0.0470] | -0.081033 | [-0.1360, -0.0274] |
| HeightScan+Contact → Diverse HeightScan+Contact | move_to_target | -0.413516 | [-0.6038, -0.2219] | -0.386048 | [-0.6582, -0.1211] |
| HeightScan+Contact → Diverse HeightScan+Contact | action_l2 | -0.013548 | [-0.0167, -0.0102] | -0.006062 | [-0.0106, -0.0013] |
| HeightScan+Contact → Diverse HeightScan+Contact | energy | -0.450880 | [-0.6183, -0.2753] | -0.243458 | [-0.4750, -0.0014] |
| HeightScan+Contact → Diverse HeightScan+Contact | joint_pos_limits | +0.309862 | [+0.2300, +0.3861] | +0.252942 | [+0.1503, +0.3591] |
| HeightScan+Contact → Diverse HeightScan+Contact | total_reward_from_env | -1.913461 | [-3.7078, -0.1341] | -2.419290 | [-5.2067, +0.2714] |

## G. Duration versus per-step rates

For each paired episode and component S=L×q, the exact symmetric algebra is ΔS=ΔL×(q_new+q_ref)/2 + Δq×(L_new+L_ref)/2. The table calls these duration-associated and rate-associated contributions. They are algebraic associations, not causal mediation or a matched-duration counterfactual.

| Comparison | Component | Duration-associated Δ | Rate-associated Δ | Per-step Δ |
|---|---|---:|---:|---:|
| Base → Original HeightScan | progress | -0.312488 | +6.984702 | +0.0083196 |
| Base → Original HeightScan | alive | +0.002478 | +0.000022 | +0.0000012 |
| Base → Original HeightScan | upright | -0.009025 | +0.003441 | +0.0000128 |
| Base → Original HeightScan | move_to_target | -0.035154 | +0.045374 | +0.0000060 |
| Base → Original HeightScan | action_l2 | +0.000509 | -0.042281 | -0.0000481 |
| Base → Original HeightScan | energy | +0.041781 | +1.041939 | +0.0011091 |
| Base → Original HeightScan | joint_pos_limits | -0.013761 | -0.700363 | -0.0008296 |
| Original HeightScan → HeightScan+Contact | progress | +1.245133 | -1.770245 | -0.0024953 |
| Original HeightScan → HeightScan+Contact | alive | +0.094328 | +0.000255 | -0.0000015 |
| Original HeightScan → HeightScan+Contact | upright | +0.027472 | +0.055228 | +0.0000529 |
| Original HeightScan → HeightScan+Contact | move_to_target | +0.127822 | +0.018447 | +0.0000232 |
| Original HeightScan → HeightScan+Contact | action_l2 | -0.002513 | +0.078683 | +0.0000901 |
| Original HeightScan → HeightScan+Contact | energy | -0.108091 | +0.979687 | +0.0011327 |
| Original HeightScan → HeightScan+Contact | joint_pos_limits | -0.033119 | +1.385444 | +0.0015203 |
| HeightScan+Contact → Diverse HeightScan+Contact | progress | -5.016461 | +2.406750 | +0.0027882 |
| HeightScan+Contact → Diverse HeightScan+Contact | alive | -0.600415 | -0.002252 | -0.0000025 |
| HeightScan+Contact → Diverse HeightScan+Contact | upright | -0.109859 | -0.059808 | -0.0000724 |
| HeightScan+Contact → Diverse HeightScan+Contact | move_to_target | -0.599561 | -0.200003 | -0.0002436 |
| HeightScan+Contact → Diverse HeightScan+Contact | action_l2 | +0.010284 | -0.029894 | -0.0000371 |
| HeightScan+Contact → Diverse HeightScan+Contact | energy | +0.483822 | -1.178160 | -0.0013576 |
| HeightScan+Contact → Diverse HeightScan+Contact | joint_pos_limits | +0.253020 | +0.309784 | +0.0003279 |

## H. Main interpretation

### Base → Original HeightScan

progress: accumulated Δ +6.672214, CI [+3.8770, +9.6829]; pre40 Δ +1.039847; post40 Δ +5.632367; per-step Δ +0.0083196.
alive: accumulated Δ +0.002500, CI [-0.2978, +0.3273]; pre40 Δ -0.360333; post40 Δ +0.362833; per-step Δ +0.0000012.
upright: accumulated Δ -0.005583, CI [-0.0713, +0.0656]; pre40 Δ -0.069667; post40 Δ +0.064083; per-step Δ +0.0000128.
move_to_target: accumulated Δ +0.010220, CI [-0.3211, +0.3731]; pre40 Δ -0.332687; post40 Δ +0.342907; per-step Δ +0.0000060.
action_l2: accumulated Δ -0.041771, CI [-0.0486, -0.0352]; pre40 Δ -0.018415; post40 Δ -0.023356; per-step Δ -0.0000481.
energy: accumulated Δ +1.083720, CI [+0.7423, +1.3958]; pre40 Δ +1.016958; post40 Δ +0.066762; per-step Δ +0.0011091.
joint_pos_limits: accumulated Δ -0.714125, CI [-0.8788, -0.5570]; pre40 Δ -0.227825; post40 Δ -0.486300; per-step Δ -0.0008296.
Native total Δ +7.007176: pre40 +1.047878, post40 +5.959296. Most of the episode-total increase occurs after the initial course endpoint; earlier passage leaves more episode exposure for post-course movement.
Most of the observed return difference is associated with higher target progress. Lower energy penalty adds to it, while larger joint-limit/action penalties offset part of it. Alive accumulation is almost unchanged, consistent with nearly equal mean episode length. The progress rate increase is consistent with the previous shorter joint-completer t40 and greater whole-episode displacement, but those are distinct metrics and include different survivor/exposure effects.

### Original HeightScan → HeightScan+Contact

progress: accumulated Δ -0.525112, CI [-2.8207, +1.7964]; pre40 Δ -0.592949; post40 Δ +0.067837; per-step Δ -0.0024953.
alive: accumulated Δ +0.094583, CI [-0.1623, +0.3377]; pre40 Δ +0.031833; post40 Δ +0.062750; per-step Δ -0.0000015.
upright: accumulated Δ +0.082700, CI [+0.0283, +0.1354]; pre40 Δ +0.044000; post40 Δ +0.038700; per-step Δ +0.0000529.
move_to_target: accumulated Δ +0.146268, CI [-0.1180, +0.4026]; pre40 Δ +0.050205; post40 Δ +0.096063; per-step Δ +0.0000232.
action_l2: accumulated Δ +0.076170, CI [+0.0708, +0.0813]; pre40 Δ +0.046968; post40 Δ +0.029202; per-step Δ +0.0000901.
energy: accumulated Δ +0.871597, CI [+0.6586, +1.0835]; pre40 Δ +0.536230; post40 Δ +0.335366; per-step Δ +0.0011327.
joint_pos_limits: accumulated Δ +1.352325, CI [+1.2228, +1.4775]; pre40 Δ +0.824136; post40 Δ +0.528189; per-step Δ +0.0015203.
Native total Δ +2.098533: pre40 +0.940423, post40 +1.158107. Joint-limit/energy/action penalty reductions occur in both segments, so the reward gains are not confined to the post-course region where the observed fall reduction was localized.
The return increase trend is chiefly associated with smaller joint-limit and energy penalties, with smaller action penalties too. Progress contribution falls slightly; the small alive gain alone does not explain the total increase. Original total-return CI includes zero. Earlier first-pass analysis found equal 94% endpoint reach and fewer post-course falls; pre/post and duration/rate tables describe how control-related contribution differences coexist with that pattern, without asserting a causal stability effect.

### HeightScan+Contact → Diverse HeightScan+Contact

progress: accumulated Δ -2.609710, CI [-6.4002, +1.1100]; pre40 Δ -0.963747; post40 Δ -1.645964; per-step Δ +0.0027882.
alive: accumulated Δ -0.602667, CI [-1.0075, -0.2107]; pre40 Δ -0.293000; post40 Δ -0.309667; per-step Δ -0.0000025.
upright: accumulated Δ -0.169667, CI [-0.2574, -0.0834]; pre40 Δ -0.088633; post40 Δ -0.081033; per-step Δ -0.0000724.
move_to_target: accumulated Δ -0.799564, CI [-1.2143, -0.3871]; pre40 Δ -0.413516; post40 Δ -0.386048; per-step Δ -0.0002436.
action_l2: accumulated Δ -0.019610, CI [-0.0261, -0.0127]; pre40 Δ -0.013548; post40 Δ -0.006062; per-step Δ -0.0000371.
energy: accumulated Δ -0.694338, CI [-1.0474, -0.3203]; pre40 Δ -0.450880; post40 Δ -0.243458; per-step Δ -0.0013576.
joint_pos_limits: accumulated Δ +0.562804, CI [+0.4236, +0.7044]; pre40 Δ +0.309862; post40 Δ +0.252942; per-step Δ +0.0003279.
Native total Δ -4.332763: pre40 -1.913461, post40 -2.419290. The symmetric duration-associated sum is -5.579169, while the rate-associated sum is +1.246417. The shorter exposure contribution dominates this algebraic decomposition; it is not a causal attribution.
Lower accumulated progress and smaller alive/upright/move_to_target accumulations accompany the shorter episodes. Energy penalty is more negative even though episodes are shorter; joint-limit penalty is less negative and partially offsets other losses. Per-step progress is higher among equal-weighted episode rates, so lower accumulated progress does not imply slower progress per control step. This is consistent with shorter timing among joint completers but lower completion and more falls. Selection into completed/fall/timeout groups is descriptive only.

Within this Final realization, the decomposition indicates associations between observed total-return differences and fixed reward components. The result does not establish causal effects of sensors or general effects across unseen terrains/training seeds. Component intervals are exploratory and no component ranking is treated as a discovery.

## I. Conditioning and artifacts

Supplementary component summaries for 40 m completers/noncompleters, falls and original timeout labels are in `final_unseen_reward_components_conditioned.csv` and JSON. These condition on outcomes and select different populations. Negative penalty fractions and positive fractions above 100% are supplementary bookkeeping, not performance scores.

Step telemetry is stored per policy as gzip CSV in `final_unseen_reward_runs/<policy>/trajectory.csv.gz`; it contains weighted reward contributions, env total, step/time, positions and done/terminated/truncated. Episode, policy summary and pairwise CSVs accompany this report. Native total Δ CIs exactly reproduce the previous statistical bootstrap.

## J. Integrity and validation

All 533 protected files/checkpoints retain their hashes. Original Final/first-pass/statistical/parity/preregistration artifacts are preserved. Frozen checkpoints, terrain seed 2404, evaluation seed 24, 100 envs, first-episode semantics and 60/123/127/127 dimensions were retained. No tuning, training, selection or reward/environment modification.

Independent identity/bootstrap verification, py_compile, whitespace checks and exact commands are recorded under `final_unseen_reward_runs/`. The separate `final_unseen_reward_decomposition_addendum.json` documents measurement-only instrumentation.

![Accumulated contributions](final_unseen_reward_components_policy.png)

![Paired component deltas](final_unseen_reward_components_pairwise.png)
