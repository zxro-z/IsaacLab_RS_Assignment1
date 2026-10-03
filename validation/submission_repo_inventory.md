# Submission Repository Inventory

Inventory captured before any move/removal. All sizes are logical bytes. `.git` history and read-only session metadata (`.agents`, `.aws`, `.codex`) are excluded from the file inventory and are not moved.

Files: 3,640; tracked project files: 1,816; logical size: 919,882,638 bytes.

Every file is classified in [submission_repo_inventory.csv](submission_repo_inventory.csv); original SHA256 and Git status are in [submission_cleanup_plan.json](submission_cleanup_plan.json). No uncertain file is moved. Existing tracked framework and task registrations remain intact to preserve runtime dependencies.

## KEEP

2,011 files; 115.12 MiB.

| Path | Reason |
|---|---|
| `.dockerignore` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `.flake8` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `.gitattributes` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `.github` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `.gitignore` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `.pre-commit-config.yaml` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `.vscode` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `CITATION.cff` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `CONTRIBUTING.md` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `CONTRIBUTORS.md` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `LICENSE` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `LICENSE-mimic` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `README.md` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `SECURITY.md` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `VERSION` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `apps` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `docker` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `docs` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `environment.yml` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `isaaclab.bat` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `isaaclab.sh` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `logs/rsl_rl/ant/2026-10-02_03-39-50_ablation_base_obs60_s42` | One of four Final representative weights, exact saved configs, or original training curves |
| `logs/rsl_rl/ant/2026-10-02_03-39-50_ablation_base_obs60_s42/model_999.pt` | One of four Final representative weights, exact saved configs, or original training curves |
| `logs/rsl_rl/ant/2026-10-02_03-39-50_ablation_base_obs60_s42/params` | One of four Final representative weights, exact saved configs, or original training curves |
| `logs/rsl_rl/ant/2026-10-03_00-52-49_ablation_heightscan_obs123_s42` | One of four Final representative weights, exact saved configs, or original training curves |
| `logs/rsl_rl/ant/2026-10-03_00-52-49_ablation_heightscan_obs123_s42/model_999.pt` | One of four Final representative weights, exact saved configs, or original training curves |
| `logs/rsl_rl/ant/2026-10-03_00-52-49_ablation_heightscan_obs123_s42/params` | One of four Final representative weights, exact saved configs, or original training curves |
| `logs/rsl_rl/ant/2026-10-03_01-10-01_ablation_heightscan_contact_obs127_s42` | One of four Final representative weights, exact saved configs, or original training curves |
| `logs/rsl_rl/ant/2026-10-03_01-10-01_ablation_heightscan_contact_obs127_s42/model_999.pt` | One of four Final representative weights, exact saved configs, or original training curves |
| `logs/rsl_rl/ant/2026-10-03_01-10-01_ablation_heightscan_contact_obs127_s42/params` | One of four Final representative weights, exact saved configs, or original training curves |
| `logs/rsl_rl/ant/2026-10-03_18-57-46_ablation_heightscan_contact_diverse_s42` | One of four Final representative weights, exact saved configs, or original training curves |
| `logs/rsl_rl/ant/2026-10-03_18-57-46_ablation_heightscan_contact_diverse_s42/model_999.pt` | One of four Final representative weights, exact saved configs, or original training curves |
| `logs/rsl_rl/ant/2026-10-03_18-57-46_ablation_heightscan_contact_diverse_s42/params` | One of four Final representative weights, exact saved configs, or original training curves |
| `pyproject.toml` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `pytest.ini` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `scripts` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `source` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `tools` | Framework/task dependency, original project scaffolding/license, or retained experiment source; do not break registered tasks |
| `validation/ablation_eval_manifest.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/ablation_final_checkpoints.sha256` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/ablation_training_normalized_configs.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/ablation_training_protected_files.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/ablation_training_runs.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/ablation_training_scalars.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/ablation_training_scalars_raw.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/ablation_training_summary.md` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/ant_core_runs_checksums.sha256` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/ant_core_runs_migration_manifest.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/ant_terrain_ablation_summary.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/dev_experiment_freeze.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/dev_ood_evaluation_raw.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/dev_ood_evaluation_results.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/dev_ood_evaluation_summary.md` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/dev_ood_preflight.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/dev_ood_runs_20261003_025922` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_config_audit.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_dev_ood_comparison.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_dev_ood_raw.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_dev_ood_results.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_dev_ood_runs` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_final_checkpoint.sha256` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_ood_overlap_audit.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_run.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_runtime_config_audit.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_scalars.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_smoke.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_smoke_diverse.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_smoke_existing.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_source_config_audit.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/diverse_training_summary.md` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_eval_manifest.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_evaluation_protocol_addendum.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_first_pass_analysis_addendum.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_first_pass_raw.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_first_pass_runs` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_first_pass_trajectory_manifest.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_integrity.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_pairwise.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_pairwise_binary.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_pairwise_bootstrap.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_policy_ci.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_preregister.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_raw.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_results.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_reward_components_conditioned.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_reward_components_episode.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_reward_components_pairwise.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_reward_components_pairwise.png` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_reward_components_policy.png` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_reward_components_summary.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_reward_decomposition.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_reward_decomposition.md` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_reward_decomposition_addendum.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_reward_label_consistency_fix.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_reward_measurement_manifest.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_reward_runs` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_runs` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_section_analysis.csv` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_section_analysis.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_section_analysis.md` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_section_analysis.png` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_statistical_analysis.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_statistical_analysis.md` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_statistical_analysis.png` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_statistics_runs` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_summary.md` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/final_unseen_validation.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/official_evaluator_parity.json` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/official_evaluator_parity.md` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
| `validation/official_evaluator_parity_runs` | Final/Dev evidence, analysis inputs, parity, preregistration, or four-policy training integrity |
## ARCHIVE

1,130 files; 758.36 MiB.

| Path | Reason |
|---|---|
| `README.assignment.md` | Hydra development outputs or superseded assignment README |
| `logs/rsl_rl/ant/2026-09-29_17-15-49_ant_baseline` | Historical run, intermediate checkpoint, or development video; final four weights retained at original paths |
| `logs/rsl_rl/ant/2026-09-29_19-32-15_friction_random` | Historical run, intermediate checkpoint, or development video; final four weights retained at original paths |
| `logs/rsl_rl/ant/2026-10-01_05-01-44_ant_terrain_heightscan` | Historical run, intermediate checkpoint, or development video; final four weights retained at original paths |
| `logs/rsl_rl/ant/2026-10-02_00-46-02_contact_127d` | Historical run, intermediate checkpoint, or development video; final four weights retained at original paths |
| `logs/rsl_rl/ant/2026-10-02_02-42-25_baseline_rs_sanity` | Historical run, intermediate checkpoint, or development video; final four weights retained at original paths |
| `logs/rsl_rl/ant/2026-10-02_03-39-50_ablation_base_obs60_s42` | Historical run, intermediate checkpoint, or development video; final four weights retained at original paths |
| `logs/rsl_rl/ant/2026-10-03_00-52-49_ablation_heightscan_obs123_s42` | Historical run, intermediate checkpoint, or development video; final four weights retained at original paths |
| `logs/rsl_rl/ant/2026-10-03_01-01-49_ablation_contact_obs64_s42` | Historical run, intermediate checkpoint, or development video; final four weights retained at original paths |
| `logs/rsl_rl/ant/2026-10-03_01-10-01_ablation_heightscan_contact_obs127_s42` | Historical run, intermediate checkpoint, or development video; final four weights retained at original paths |
| `logs/rsl_rl/ant/2026-10-03_13-38-43_ablation_heightscan_dense_s42` | Historical run, intermediate checkpoint, or development video; final four weights retained at original paths |
| `logs/rsl_rl/ant/2026-10-03_13-47-49_ablation_heightscan_forwarddense_s42` | Historical run, intermediate checkpoint, or development video; final four weights retained at original paths |
| `logs/rsl_rl/ant/2026-10-03_17-17-51_ablation_compact_obs78_s42` | Historical run, intermediate checkpoint, or development video; final four weights retained at original paths |
| `logs/rsl_rl/ant/2026-10-03_17-36-32_ablation_heightscan_compact_obs141_s42` | Historical run, intermediate checkpoint, or development video; final four weights retained at original paths |
| `logs/rsl_rl/ant/2026-10-03_18-57-46_ablation_heightscan_contact_diverse_s42` | Historical run, intermediate checkpoint, or development video; final four weights retained at original paths |
| `outputs` | Hydra development outputs or superseded assignment README |
| `scripts` | Standalone compact/spatial/video/transfer tooling outside submission story; core helpers retained |
| `validation/ablation_base_obs60_s42_training.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/ablation_collector_original_traceback.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/ablation_contact_obs64_s42_training.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/ablation_heightscan_contact_diverse_s42_training.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/ablation_heightscan_contact_obs127_s42_training.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/ablation_heightscan_dense_s42_training.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/ablation_heightscan_forwarddense_s42_training.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/ablation_heightscan_obs123_s42_training.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/ablation_reader_fix_summary.json` | Superseded diagnostic evidence; completed training provenance is retained |
| `validation/ablation_reader_regression.json` | Superseded diagnostic evidence; completed training provenance is retained |
| `validation/ablation_training_before_reader_fix.json` | Superseded diagnostic evidence; completed training provenance is retained |
| `validation/ablation_training_continuation.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/ant_terrain_ablation_base.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/ant_terrain_ablation_contact.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/ant_terrain_ablation_heightscan.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/ant_terrain_ablation_heightscan_contact.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/baseline_clean_smoke.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/baseline_clean_train.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/baseline_rs_train_sanity.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/compact_checkpoint_audit.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/compact_dev_ood_batch.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/compact_dev_ood_batch_retry1.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/compact_dev_ood_evaluation_raw.csv` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/compact_dev_ood_evaluation_results.csv` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/compact_dev_ood_evaluation_summary.md` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/compact_dev_ood_runs` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission; Verbose development console, incomplete output, or superseded source backup |
| `validation/compact_dev_ood_runs_retry1` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission; Verbose development console, incomplete output, or superseded source backup |
| `validation/compact_eval_manifest.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/compact_evaluation_integrity.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/compact_evaluation_protected_files.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/compact_terrain_descriptor_combined.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/compact_terrain_descriptor_compact.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/compact_terrain_descriptor_config_audit.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/compact_terrain_descriptor_original.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/compact_terrain_descriptor_schema.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/compact_terrain_descriptor_smoke.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/compact_terrain_descriptor_smoke_combined.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/compact_terrain_descriptor_smoke_compact.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/compact_terrain_descriptor_smoke_original.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/compact_terrain_descriptor_summary.md` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/compact_terrain_descriptor_unit_tests.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/dev_ood_runs_20261003_025922` | Verbose development console, incomplete output, or superseded source backup |
| `validation/diverse_training_dev_ood_runs` | Verbose development console, incomplete output, or superseded source backup |
| `validation/diverse_training_diverse_smoke_retry1.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/diverse_training_existing_smoke.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/diverse_training_existing_smoke_retry1.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/diverse_training_existing_smoke_retry2.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/diverse_training_existing_smoke_retry3.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/diverse_training_existing_smoke_retry4.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/diverse_training_existing_smoke_retry5.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/diverse_training_existing_smoke_retry6.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/diverse_training_orchestration.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/diverse_training_smoke_existing.json.incomplete_1791020835039940535` | Verbose development console, incomplete output, or superseded source backup |
| `validation/existing_vs_diverse_recording.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/existing_vs_diverse_rollout` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission; Verbose development console, incomplete output, or superseded source backup |
| `validation/final_unseen_evaluation.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/final_unseen_first_pass_runs` | Verbose development console, incomplete output, or superseded source backup |
| `validation/final_unseen_reward_runs` | Verbose development console, incomplete output, or superseded source backup |
| `validation/final_unseen_runs` | Verbose development console, incomplete output, or superseded source backup |
| `validation/final_unseen_statistics_runs` | Verbose development console, incomplete output, or superseded source backup |
| `validation/final_unseen_validation.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/heightscan_pattern_dense.png` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_pattern_forward_dense.png` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_pattern_original.png` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_patterns.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_smoke_dense.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_smoke_dense.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/heightscan_smoke_forward_dense.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_smoke_forward_dense.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/heightscan_smoke_original.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_smoke_original.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/heightscan_spatial_dev_ood_raw.csv` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_spatial_dev_ood_results.csv` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_spatial_dev_ood_runs` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission; Verbose development console, incomplete output, or superseded source backup |
| `validation/heightscan_spatial_dev_ood_summary.md` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_spatial_eval_manifest.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_spatial_evaluation_batch.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/heightscan_spatial_final_checkpoints.sha256` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_spatial_training_batch.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/heightscan_spatial_training_runs.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_spatial_training_scalars.csv` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_spatial_training_summary.md` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_spatial_validation.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_variant_summary.md` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/heightscan_variants_config_audit.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/list_envs.log` | Verbose development console, incomplete output, or superseded source backup |
| `validation/official_evaluator_parity_runs` | Verbose development console, incomplete output, or superseded source backup |
| `validation/run_ablation_training_before_reader_fix.py` | Verbose development console, incomplete output, or superseded source backup |
| `validation/stepping_rollout_analysis` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission; Verbose development console, incomplete output, or superseded source backup |
| `validation/stepping_rollout_analysis_failed_finalize_20261003` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission; Verbose development console, incomplete output, or superseded source backup |
| `validation/stepping_rollout_analysis_failed_init_20261003` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission; Verbose development console, incomplete output, or superseded source backup |
| `validation/teammate_terrain_transfer` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission; Verbose development console, incomplete output, or superseded source backup |
| `validation/teammate_terrain_transfer_manifest.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/terrain_representation_dev_ood_results.csv` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/terrain_representation_dev_ood_summary.md` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/terrain_representation_eval_manifest.json` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
| `validation/terrain_representation_pairwise.csv` | Auxiliary representation/spatial/debug/teammate-transfer study outside the four-policy Final submission |
## REMOVE-GENERATED

499 files; 3.78 MiB.

| Path | Reason |
|---|---|
| `scripts/assignment/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `scripts/reinforcement_learning/rsl_rl/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/actuators/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/app/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/assets/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/assets/articulation/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/assets/deformable_object/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/assets/rigid_object/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/assets/rigid_object_collection/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/assets/surface_gripper/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/controllers/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/devices/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/devices/gamepad/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/devices/keyboard/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/devices/openxr/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/devices/openxr/retargeters/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/devices/openxr/retargeters/humanoid/fourier/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/devices/openxr/retargeters/humanoid/unitree/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/devices/openxr/retargeters/humanoid/unitree/inspire/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/devices/openxr/retargeters/humanoid/unitree/trihand/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/devices/openxr/retargeters/manipulator/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/devices/spacemouse/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/envs/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/envs/mdp/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/envs/mdp/actions/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/envs/mdp/commands/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/envs/mdp/recorders/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/envs/ui/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/envs/utils/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/managers/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/markers/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/markers/config/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/scene/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sensors/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sensors/camera/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sensors/contact_sensor/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sensors/frame_transformer/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sensors/imu/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sensors/ray_caster/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sensors/ray_caster/patterns/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sim/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sim/converters/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sim/schemas/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sim/spawners/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sim/spawners/from_files/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sim/spawners/lights/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sim/spawners/materials/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sim/spawners/meshes/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sim/spawners/sensors/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sim/spawners/shapes/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/sim/spawners/wrappers/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/terrains/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/terrains/height_field/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/terrains/trimesh/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/ui/widgets/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/utils/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/utils/buffers/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/utils/datasets/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/utils/interpolation/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/utils/io/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/utils/modifiers/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/utils/noise/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab/isaaclab/utils/warp/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_assets/isaaclab_assets/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_assets/isaaclab_assets/robots/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_assets/isaaclab_assets/sensors/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_mimic/isaaclab_mimic/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_rl/isaaclab_rl/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_rl/isaaclab_rl/rsl_rl/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/allegro_hand/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/allegro_hand/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/ant/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/ant/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/automate/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/automate/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/cart_double_pendulum/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/cart_double_pendulum/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/cartpole_showcase/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/cartpole_showcase/cartpole/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/cartpole_showcase/cartpole/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/cartpole_showcase/cartpole_camera/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/cartpole_showcase/cartpole_camera/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/factory/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/factory/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/forge/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/forge/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/humanoid/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/humanoid/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/locomotion/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand_over/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand_over/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/humanoid/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/humanoid/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/humanoid/mdp/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/tracking/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/tracking/config/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/tracking/config/digit/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/tracking/config/digit/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/a1/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/a1/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_b/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_b/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/cassie/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/cassie/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/digit/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/digit/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/config/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/config/franka/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/config/franka/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/reach/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/reach/config/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/reach/config/ur_10e/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/reach/config/ur_10e/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/config/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/config/kuka_allegro/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/config/kuka_allegro/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/ur_10/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/ur_10/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/galbot/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/ur10_gripper/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/mdp/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/config/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/config/anymal_c/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/config/anymal_c/agents/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `source/isaaclab_tasks/isaaclab_tasks/utils/__pycache__` | Regenerable bytecode/cache or stale experiment lock; no research measurements |
| `validation` | Regenerable bytecode/cache or stale experiment lock; no research measurements |

## Uncertain

| Path | Why review is needed |
|---|---|
| Registered compact/spatial/teammate task source configs | Retained as KEEP: deleting them would leave existing task registrations dangling; no frozen source edits. |
| `.git/` and read-only session metadata | Local administration; not submission artifacts. Leave intact, ignore session directories. |

## Safety and preservation

- Four representative `model_999.pt` files, saved env/agent configs, original TensorBoard events, and Diverse H3 evidence are KEEP. Intermediate checkpoint and development videos are ARCHIVE.
- All `final_unseen_*` reports, CSVs, manifests, and trajectory inputs needed by offline analysis are KEEP. Console logs may be archived; scientific evidence and protocol hashes stay unchanged.
- Dev evidence supporting experiment provenance remains; compact/spatial/teammate-transfer outputs are archived outside the repository.
- REMOVE-GENERATED contains only cache/bytecode/stale locks. For reversibility, these will also be relocated to the external archive, rather than deleted.
- Archive uses original relative paths and verifies every SHA256. No history rewrite or Git staging.

## Large files

| Path | MiB |
|---|---:|
| `validation/final_unseen_reward_runs/original_heightscan/trajectory.csv.gz` | 7.51 |
| `validation/final_unseen_reward_runs/heightscan_contact/trajectory.csv.gz` | 7.48 |
| `validation/final_unseen_reward_runs/base/trajectory.csv.gz` | 7.44 |
| `validation/final_unseen_reward_runs/diverse_heightscan_contact/trajectory.csv.gz` | 6.97 |
| `logs/rsl_rl/ant/2026-10-03_13-38-43_ablation_heightscan_dense_s42/model_100.pt` | 3.96 |
| `logs/rsl_rl/ant/2026-10-03_13-38-43_ablation_heightscan_dense_s42/model_150.pt` | 3.96 |
| `logs/rsl_rl/ant/2026-10-03_13-38-43_ablation_heightscan_dense_s42/model_200.pt` | 3.96 |
| `logs/rsl_rl/ant/2026-10-03_13-38-43_ablation_heightscan_dense_s42/model_250.pt` | 3.96 |
| `logs/rsl_rl/ant/2026-10-03_13-38-43_ablation_heightscan_dense_s42/model_300.pt` | 3.96 |
| `logs/rsl_rl/ant/2026-10-03_13-38-43_ablation_heightscan_dense_s42/model_350.pt` | 3.96 |
| `logs/rsl_rl/ant/2026-10-03_13-38-43_ablation_heightscan_dense_s42/model_400.pt` | 3.96 |
| `logs/rsl_rl/ant/2026-10-03_13-38-43_ablation_heightscan_dense_s42/model_450.pt` | 3.96 |
