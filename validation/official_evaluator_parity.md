# Official evaluator parity verification

Task `Isaac-Ant-Final-Unseen-v0`, seed 24, 4 envs per policy. Verified against the downloaded assignment source, rather than assuming the locally named script is identical.

Official URL: https://github.com/cailab-hy/IsaacLab_RS/blob/main/scripts/reinforcement_learning/rsl_rl/play_one_episode.py
Official snapshot SHA256: `70dd6d6674ca211f335b57fea5d8dca417fe670753adedf2d002a638b2b089d0`

## A. Official semantics

The provided evaluator initializes per-env rewards as float64, steps as int64, and finished as bool. After policy inference and env.step, its operations are:

```python
active = ~finished
episode_rewards[active] += rewards[active]
episode_steps[active] += 1
finished |= dones.bool()
```

The terminal transition contributes reward and one step before the finished mask changes. Later auto-reset episodes are ignored for that env, even though the batched policy and simulation continue stepping it. Each env has its own accumulator and completion mask. Rewards come directly from RslRlVecEnvWrapper.step; the wrapper forwards the underlying env reward without replacement, and merges terminated | truncated into long dones.

CLI seed 24 is assigned to agent_cfg.seed by cli_args.update_rsl_rl_cfg, then to env_cfg.seed before gym.make. The local script performs the equivalent assignment before creation. No late reseeding was injected.

Multi-env return mean/std use float64 accumulated returns; step mean/std first convert steps to float64. Both std calls use unbiased=False (population, ddof=0). With one env the official script prints the single return and step count.

Official loop also stops at env.max_episode_length or simulator shutdown. It warns and reports partial episodes if any remain unfinished. The local script instead requires all first episodes to complete and raises on premature simulator shutdown. This guard difference is outside the completed-episode accounting: every smoke env completed by the 960-step official cap. The original raw 400 rows also each record completion via timeout or fall. Seed -1 also differs in general (official CLI randomizes it); this verification uses the fixed seed 24 only.

## B. Current evaluator comparison

`evaluate_final_unseen.py` contains orchestration, not a separate simulation accounting loop. It calls the LOCAL `scripts/reinforcement_learning/rsl_rl/play_one_episode.py` with the frozen checkpoint and flags, then copies its episode and aggregate data into CSV. This local script differs from the downloaded official source and supplies the required policy observation adapters.

| Check | Verdict | Execution-order evidence |
|---|---|---|
| First episode only | PASS | Separate finished mask for every env; local loop ends only after all finish. |
| Terminal reward included | PASS | Local returns[active] += rewards[active] before newly_finished updates finished. |
| Terminal step counted | PASS | Local lengths[active] += 1 before updating finished. |
| Post-reset episode excluded | PASS | active computed before env.step from previous finished; finished cannot change during env.step. Next iteration masks completed envs. |
| Done handling | PASS | Both use the same wrapper terminated OR truncated, then bool; local newly_finished = dones.bool() & active is equivalent to official finished OR dones.bool(). |
| Reward source | PASS | Same direct env.step reward vector; no recomputation or reward shaping. |
| Seed handling | PASS (seed 24) | Agent seed -> env_cfg.seed before gym.make in both. |
| Per-env accounting | PASS | Independent indexed reward/step tensors, with equal finished masks and step counts asserted on every transition. |
| Std definition | PASS | Both population std, unbiased=False. No pandas ddof=1 path. |

Numerical precision differs: official float64 return accumulation and float64 step reductions versus local float32 return accumulation and float32 step reductions. This is not an episode/reward-source discrepancy. The actual return differences below all pass the requested tolerance without relaxing it. Raw CSV population std was independently reconstructed and agrees with the published std within 1e-6 relative/absolute tolerance; existing summaries were preserved.

## C. Small parity method and results

The new wrapper parses the official snapshot AST, verifies and extracts its exact four accounting statements, and renames only shadow accumulator variables. It instruments an in-memory AST of the existing local script, leaving both original source files unchanged. The current script performs the real env creation, checkpoint load, policy-specific observation construction and inference. Immediately after each actual env.step, the official shadow accumulator consumes the very same reward/done vector. The existing local accounting statements then execute unchanged. This implements the requested observation-adapter fallback; it is not an independent unadapted official-script rollout.

Every transition asserts identical finished masks and step counts, unchanged official accumulators for already-finished envs, and unchanged current returns for already-finished envs. Runtime observation shape is checked on every step against the checkpoint actor dimension. Terminal reward addition is explicitly checked in the reference, and current totals and terminal-step counts are compared. The JSON records first_done_step, terminal reward, and observed post-reset transitions. No checkpoint export is invoked.

Tolerance: `np.isclose(current, official, rtol=1e-6, atol=1e-6)`. Because both accumulators share one trajectory, their differences are accumulation precision, not GPU trajectory nondeterminism. No tolerance adjustment was needed. Full precision values are in JSON; the table prints nine decimal places.

### Policy: Base

Actor input: 60; runtime observation: 60. Parity: **PASS**.

| Env | Official return | Current return | Official steps | Current steps | Absolute return difference | Post-reset transitions observed / counted |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 67.033315057 | 67.033370972 | 960 | 960 | 5.59147447e-05 | 0 / 0 |
| 1 | 71.142503156 | 71.142532349 | 960 | 960 | 2.91930046e-05 | 0 / 0 |
| 2 | 63.843871874 | 63.843883514 | 960 | 960 | 1.16406009e-05 | 0 / 0 |
| 3 | 60.107147593 | 60.107128143 | 879 | 879 | 1.94498571e-05 | 81 / 0 |

### Policy: Original HeightScan

Actor input: 123; runtime observation: 123. Parity: **PASS**.

| Env | Official return | Current return | Official steps | Current steps | Absolute return difference | Post-reset transitions observed / counted |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 70.693214853 | 70.693252563 | 960 | 960 | 3.77102988e-05 | 0 / 0 |
| 1 | 80.717491074 | 80.717521667 | 960 | 960 | 3.05939466e-05 | 0 / 0 |
| 2 | 71.561177254 | 71.561149597 | 960 | 960 | 2.76570208e-05 | 0 / 0 |
| 3 | 62.816197805 | 62.816215515 | 765 | 765 | 1.77097972e-05 | 195 / 0 |

### Policy: HeightScan+Contact

Actor input: 127; runtime observation: 127. Parity: **PASS**.

| Env | Official return | Current return | Official steps | Current steps | Absolute return difference | Post-reset transitions observed / counted |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 73.456270375 | 73.456245422 | 936 | 936 | 2.49524601e-05 | 24 / 0 |
| 1 | 77.239062662 | 77.239059448 | 960 | 960 | 3.21376137e-06 | 0 / 0 |
| 2 | 72.532648164 | 72.532638550 | 960 | 960 | 9.61404294e-06 | 0 / 0 |
| 3 | 65.082197953 | 65.082214355 | 861 | 861 | 1.6402686e-05 | 99 / 0 |

### Policy: Diverse HeightScan+Contact

Actor input: 127; runtime observation: 127. Parity: **PASS**.

| Env | Official return | Current return | Official steps | Current steps | Absolute return difference | Post-reset transitions observed / counted |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 76.994469559 | 76.994506836 | 960 | 960 | 3.72768845e-05 | 0 / 0 |
| 1 | 74.487283746 | 74.487266541 | 960 | 960 | 1.72054861e-05 | 0 / 0 |
| 2 | 75.130773126 | 75.130813599 | 960 | 960 | 4.04729508e-05 | 0 / 0 |
| 3 | 62.962849358 | 62.962856293 | 789 | 789 | 6.93509355e-06 | 171 / 0 |

Maximum absolute return difference: 5.59147447348e-05. All 16 first episodes completed; terminal rewards and steps included; no post-reset contribution.

## D. Decision

Case A: all four policies PASS. Preserve the existing 400 raw rows and the existing aggregate/summary files. No Final 100-env reevaluation was launched. Population std already matches, so summary recalculation is unnecessary. Checkpoints, training, terrain, preregistration, rewards, terminations, sensors and observations were not modified.

The existing Final evaluation follows the first-episode accounting
semantics of the provided play_one_episode.py.
No Final re-evaluation is required.

The next analysis steps may proceed: (1) 40 m first-pass / terrain-section analysis; (2) confidence intervals or paired bootstrap using the raw 400 episodes; (3) stock reward component decomposition. Those analyses were not performed in this parity task.

## E. Artifacts and integrity

- New wrapper: `scripts/assignment/check_official_evaluator_parity.py`.
- New four-policy smoke runner: `scripts/assignment/run_official_evaluator_parity.py`.
- Main results: `validation/official_evaluator_parity.json`.
- Per-policy JSON/logs, official source snapshot, source/artifact hashes and full simulation commands: `validation/official_evaluator_parity_runs/`.
- Separate protocol addendum: `validation/final_unseen_evaluation_protocol_addendum.json`.
- This report: `validation/official_evaluator_parity.md`.

All 447 pre-existing validation artifacts, preregistered sources and selected checkpoints covered by the pre-test snapshot retained their hashes. The addendum records before/after relevant source hashes and checkpoint SHA256. Existing preregistration and Dev artifacts were not overwritten.

The first sandbox attempt could not access CUDA (No CUDA GPUs are available). Its logs/results were preserved under `official_evaluator_parity_runs/sandbox_attempt/`. The successful GPU run used explicitly approved execution outside the sandbox, with the same Python/Isaac Sim installation as the original Final evaluation.

## F. Validation

`python -m py_compile` and `git diff --check` results are recorded in `official_evaluator_parity_runs/validation.log`. Full smoke commands are in `commands.log`; validation/inspection command provenance is in `session_commands.log`. All policy subprocesses exited zero on the approved GPU run.
