# Assignment 1 — Robust Ant Locomotion on Unseen Terrain

## 1. Project Goal

과제의 `Isaac-Ant-v0`를 출발점으로, 처음 보는 지형에서도 안정적으로 걷는 Ant를 개발한다. 실제 네 정책은 stock Ant를 확장한 terrain ablation task에서 학습했고, 별도의 FinalUnseen task에서 평가했다. IsaacLab 환경·PPO 학습 코드·학습된 가중치·평가 및 분석 자료를 함께 제공한다. **Current submission candidate: HeightScan+Contact** — 팀 최종 확인 전의 provisional candidate이다. PPT는 별도 제출 예정이며 이 문서가 발표 구성의 기반이다.

**Quick summary — within this Final realization:**

- **HeightScan:** Base보다 높은 progress와, 두 정책 모두 완주한 pair에서 짧은 traversal time이 관측됐다.
- **Contact:** HeightScan-only와 유사한 이동량에 stability/control-efficiency pattern이 나타났다. Fall 감소 방향의 paired CI는 0에 닿으므로 확실한 감소로 단정하지 않는다.
- **Diverse:** 성공한 joint completer에서는 더 빠른 timing이 관측됐지만 전체 completion robustness는 낮았다. 더 넓은 training distribution이 이 Final realization의 robustness를 개선하지 못했다.

팀 공유 시 먼저 확인할 항목:

- 실행: [공식 평가 명령](#9-official-evaluation-command), [SUBMISSION_COMMAND.txt](SUBMISSION_COMMAND.txt)
- 모델: [현재 제출 후보](#16-current-submission-candidate), [네 정책의 checkpoint/config mapping](validation/submission_model_manifest.json)
- 근거: [Final 결과](#10-main-results) → [first-pass](#12-40-m-first-pass-analysis) → [paired CI](#13-paired-statistical-analysis) → [reward decomposition](#14-reward-decomposition)

## 2. Research Question

> 처음 보는 지형에서 Ant가 안정적으로 걷기 위해 어떤 정보가 필요한가?

## 3. Hypotheses

- **H1 — HeightScan:** 국소 지형 형상을 미리 관측하면 unseen locomotion mobility가 향상된다.
- **H2 — Contact:** 지형 정보에 실제 foot contact를 더하면 stability/control efficiency가 향상된다.
- **H3 — Training Diversity:** 더 넓은 roughness 분포에서 학습하면 unseen robustness가 향상된다.

## 4. Method Overview

| Policy | Obs dim | Difference |
|---|---:|---|
| Base | 60 | Stock Ant state observation |
| Original HeightScan | 123 | Base + 63-ray terrain scan |
| HeightScan+Contact | 127 | Original + 4 binary foot contacts |
| Diverse HeightScan+Contact | 127 | Contact + broader training terrain distribution |

Base/Original/Contact는 같은 학습 지형·asset·action·reward·PPO 설정을 사용한다. Diverse는 Contact의 observation을 유지하고 학습 지형 분포만 바꾼 H3 ablation이다. Diverse의 실패 패턴도 핵심 연구 결과로 보존한다.

## 5. Sensor / Observation Design

**HeightScan:** torso에 부착된 yaw-aligned RayCaster로 전방 X=0–1.6 m, Y=−0.6–0.6 m를 0.2 m 간격으로 관측한다(9×7=63 rays). Ground mesh까지의 높이를 torso-relative observation으로 변환하고 offset 0.5, clipping [−1,1]을 적용한다.

**Contact:** front-left/front-right/left-back/right-back foot 순서로 net contact force norm >1 N을 0/1로 전달한다. Stock 60-D의 foot incoming wrench와 별도로 추가된 4-D binary feedback이다. Base의 60-D에는 torso height, linear/angular velocity, orientation/target projections, joint state, foot wrenches, previous action이 포함된다.

이는 hardware 변경 없이 sensor/config/software 수준에서 수행한 실험이다. Checkpoint actor 입력은 실제로 60/123/127/127-D이며, runtime adapter는 각 모델의 학습 observation ordering을 유지한다.

Source: [HeightScan](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_terrain_heightscan_env_cfg.py), [Contact](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_contact_observations.py), [runtime adapter](scripts/reinforcement_learning/rsl_rl/ant_observation_adapter.py).

## 6. Training Setup

저장된 각 run의 `params/env.yaml`, `params/agent.yaml` 기준이다.

| Setting | Value |
|---|---|
| Algorithm | RSL-RL PPO |
| Training seed / terrain seed | 42 / 42 |
| Parallel envs / iterations | 4096 / 1000 |
| Rollout steps per env | 32 |
| Total transitions per policy | 4096 × 32 × 1000 = 131,072,000 |
| Actor / critic hidden layers | [400, 200, 100], ELU |
| Learning rate / schedule | 0.0005 / adaptive |
| γ / λ | 0.99 / 0.95 |
| Epochs / mini-batches / clip | 5 / 4 / 0.2 |
| Entropy coefficient | 0.0 |
| Actor / critic observation normalization | Disabled |
| Physics dt / decimation / control dt | 1/120 s / 2 / 1/60 s |
| Episode duration / actions | 16 s / 8-D joint effort, scale 7.5 |

Base/Original/Contact의 training mix는 flat 30%, 양방향 slope 60%(easy/medium/harder), low stairs 10%이다. 8×8 m tiles, 10×10 layout, curriculum disabled. Robot friction startup randomization과 joint reset randomization을 공유한다.

Diverse는 flat 20%, slopes 50%, stairs 10%, random-uniform roughness 20%로 바꾼다. 추가 roughness는 height range [−0.05,0.05] m, noise step 0.025 m, downsampled scale 0.4 m이며 중앙 2 m spawn platform을 유지한다.

Source: [ablation training config](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_terrain_ablation_env_cfg.py), [Diverse config](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_terrain_diverse_env_cfg.py), [PPO config](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/agents/rsl_rl_ppo_cfg.py).

## 7. Self-designed Unseen Evaluation Environment

**`Isaac-Ant-Final-Unseen-v0`**: 사전 등록한 40×8 m analytic course. Training terrain의 복제본이 아니라 smooth ridge, connected terraces, basin, mixed-frequency landing을 연결한 별도 평가 지형이다.

| Section | Terrain-local X |
|---|---|
| Flat spawn | 0–4 m |
| Asymmetric smooth ridge | 4–10 m |
| Connected terraces | 10–18 m |
| Basin | 18–25 m |
| Mixed-frequency landing | 25–40 m |

Spawn은 terrain-local (2,4)이므로 **endpoint x=40 도달은 약 38 m forward displacement**에 해당한다. Boundary 분석은 시작 tile의 local coordinate를 유지하며 modulo/repeated-tile 좌표를 사용하지 않는다.

[Final config](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_final_unseen_env_cfg.py) · [Preregistration](validation/final_unseen_preregister.json)

## 8. Evaluation Protocol

- Environment seed **24**, terrain seed **2404**, policy당 **100 env**, 총 **400 first episodes**.
- Deterministic inference, 최대 16 s. Terrain-relative torso clearance <0.31 m(또는 유효 ground hit 없음)을 fall로 종료하며 timeout은 별도로 기록한다.
- `env.step()`이 반환하는 stock Ant reward를 누적한다. Seven terms: `progress` (1.0), `alive` (0.5), `upright` (0.1), `move_to_target` (0.5), `action_l2` (−0.005), `energy` (−0.05), `joint_pos_limits` (−0.1). Weighted per-step contributions에는 control dt가 곱해진다.
- 제공된 `play_one_episode.py`와 **네 정책 모두 episode accounting parity PASS**: terminal-step reward/step 포함, 종료 후 auto-reset episode 제외. Std는 population std (`ddof=0`).
- 기존 Final 결과를 유지한 채 first-pass/reward telemetry만 계측했으며 400 episode consistency를 검증했다.

[Parity report](validation/official_evaluator_parity.md) · [Parity JSON](validation/official_evaluator_parity.json) · [Protocol addendum](validation/final_unseen_evaluation_protocol_addendum.json)

## 9. Official Evaluation Command

Repository root에서 Isaac Sim / Isaac Lab 환경을 활성화한 뒤 실행한다. `run.py`는 다른 editable checkout 대신 이 repository를 import하도록 경로만 지정하며 evaluation semantics를 바꾸지 않는다.

```bash
./isaaclab.sh -p scripts/assignment/run.py \
    scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
    --task Isaac-Ant-Final-Unseen-v0 \
    --seed 24 \
    --num_envs 100 \
    --checkpoint logs/rsl_rl/ant/2026-10-03_01-10-01_ablation_heightscan_contact_obs127_s42/model_999.pt \
    --height_scan --foot_contacts \
    --headless
```

같은 명령이 [SUBMISSION_COMMAND.txt](SUBMISSION_COMMAND.txt)에 있다. `--height_scan --foot_contacts`는 이 checkpoint의 학습 observation을 구성하기 위한 필수 flags다. 평가 task는 `Isaac-Ant-Final-Unseen-v0`이고 training task와 구분한다. 기본 명령은 stdout 평가이며 결과 파일을 지정하지 않는다.

## 10. Main Results

Frozen Final; policy당 N=100. 표의 ±는 population std이다. Distance는 초기 root 위치를 기준으로 한 forward X displacement이며 누적 경로 길이가 아니다. 결과와 CI의 원본은 아래 retained CSV/JSON/보고서이며, README는 그 값을 발표용 자릿수로 표시한다.

| Policy | Return mean ± std | Distance mean ± std | Fall | Timeout |
|---|---:|---:|---:|---:|
| Base | 60.504 ± 15.942 | 55.772 ± 14.751 m | 19% | 81% |
| Original HeightScan | 67.512 ± 15.075 | 62.437 ± 13.955 m | 23% | 77% |
| HeightScan+Contact | 69.610 ± 16.748 | 61.914 ± 14.977 m | 16% | 84% |
| Diverse HeightScan+Contact | 65.277 ± 21.282 | 59.295 ± 19.369 m | 33% | 67% |

[Original raw 400 episodes](validation/final_unseen_raw.csv) · [Results CSV](validation/final_unseen_results.csv) · [Final summary](validation/final_unseen_summary.md)

## 11. Why Mean Distance Alone Was Not Enough

평균 55–62 m는 첫 episode의 forward displacement이며 initial 40 m course 이후의 이동도 포함한다. 반복 tile이나 외곽 padding 이동을 bounded-course completion으로 해석할 수 없다. 따라서 시작 tile의 endpoint 최초 통과를 별도로 분석했다. 기존 evaluator의 final displacement는 종료 transition 직전(step N−1)의 pose를 사용한다. 이 관례를 그대로 보존했으며, first-pass trajectory는 auto-reset 전의 terminal-step N pose까지 포함한다. Reward와 episode step count는 두 경우 모두 terminal step을 포함한다.

## 12. 40 m First-pass Analysis

| Policy | X-direction endpoint reach | Corridor-valid completion |
|---|---:|---:|
| Base | 92% | 89% |
| Original HeightScan | 94% | 87% |
| HeightScan+Contact | 94% | 93% |
| Diverse | 85% | 80% |

Endpoint reach는 최초 terrain-local x≥40을 뜻한다. Corridor-valid는 endpoint에 도달하면서 episode 시작부터 최초 crossing step까지(해당 step 포함) 어느 sampled root position에서도 local y∉[0,8]인 lateral exit가 없었던 episode이다. Contact는 Original과 X reach가 같으나 corridor-valid completion은 관측상 6 pp 높다. 이는 서로 다른 metric이며 X reach를 corridor completion으로 대체하지 않는다.

![First-pass rates and section outcomes](docs/figures/final_unseen_section_analysis.png)

[Section report](validation/final_unseen_section_analysis.md) · [Episode first-pass CSV](validation/final_unseen_first_pass_raw.csv)

## 13. Paired Statistical Analysis

동일 env_id의 terrain/origin/initial condition pairing을 검증했다. 50,000 paired bootstrap resamples, RNG seed 2404, percentile 95% CI; rate 단독 CI는 Wilson이다. Δ는 new − reference이며 fall의 음수 Δ는 감소를 뜻한다.

| Comparison | Metric | Δ [95% paired CI] |
|---|---|---:|
| Base → Original | Return | +7.007 [3.986, 10.266] |
| Base → Original | Displacement | +6.666 m [3.874, 9.673] |
| Base → Original | 40 m reach | +2 pp [−3, 8] |
| Original → Contact | Return | +2.099 [−0.439, 4.643] |
| Original → Contact | Displacement | −0.523 m [−2.820, 1.795] |
| Original → Contact | Fall | −7 pp [−14, 0] |
| Original → Contact | 40 m reach | 0 pp [−5, 5] |
| Original → Contact | Corridor completion | +6 pp [−1, 13] |
| Contact → Diverse | Return | −4.333 [−8.530, −0.199] |
| Contact → Diverse | Episode length | −72.15 steps [−120.64, −25.18] |
| Contact → Diverse | Fall | +17 pp [8, 26] |
| Contact → Diverse | 40 m reach | −9 pp [−17, −2] |
| Contact → Diverse | Corridor completion | −13 pp [−21, −5] |

Original→Contact의 일부 CI는 0을 포함하거나 0에 닿는다. 관측된 방향을 일반적 causal effect로 주장하지 않는다. Component/metric별 intervals는 multiple comparisons 보정 없는 descriptive intervals이다.

[Statistical report](validation/final_unseen_statistical_analysis.md) · [Paired CSV](validation/final_unseen_pairwise_bootstrap.csv) · [Paired CI figure](docs/figures/final_unseen_statistical_analysis.png)

## 14. Reward Decomposition

Stock RewardManager의 실제 canonical key를 유지했다. Weighted component 합과 `env.step()` total의 identity를 step/episode 수준에서 floating-point tolerance 이내로 검증했다. `energy`는 stock action/joint-velocity proxy이며 실제 에너지(J)를 측정한 값은 아니다.

- **Base → Original:** total +7.007 중 `progress` +6.672. Terrain awareness와 locomotion progress 향상이 함께 나타난다.
- **Original → Contact:** total +2.099, `progress` −0.525. `energy`/`action_l2`/`joint_pos_limits` penalties가 덜 음수가 되는 패턴으로, 이동량 증가보다 control/stability efficiency 해석에 부합한다.
- **Contact → Diverse:** total −4.333. Per-step progress는 높지만 episode가 짧고 fall이 많아 accumulated progress가 감소한다. “Fast when active / among completers, but lower robustness”라는 제한된 해석이다.

![Weighted episode reward contributions](docs/figures/final_unseen_reward_components_policy.png)

[Reward report](validation/final_unseen_reward_decomposition.md) · [Episode components](validation/final_unseen_reward_components_episode.csv) · [Component delta figure](docs/figures/final_unseen_reward_components_pairwise.png)

## 15. Findings by Hypothesis

| Hypothesis | Finding within this Final realization |
|---|---|
| H1 — HeightScan | Progress/return/displacement 향상에 의해 지지됨; first-course reach 차이는 작고 CI에 0 포함 |
| H2 — Contact | Stability/control-efficiency pattern 관측; 일부 paired CI에 0 포함 |
| H3 — Training Diversity | 이 realization에서 지지되지 않음; 더 빠른 conditional motion과 낮은 overall robustness가 공존 |

## 16. Current Submission Candidate

**HeightScan+Contact — provisional, pending team confirmation.** 네 정책 중 관측된 Final mean return이 높고 fall rate가 낮으며, X endpoint reach 94%, corridor-valid completion 93%를 보였다. Original의 이동성과 Contact의 안정성/control-efficiency를 함께 고려한 후보이며 reward ranking만으로 선택한 결론이 아니다.

Checkpoint: [model_999.pt](logs/rsl_rl/ant/2026-10-03_01-10-01_ablation_heightscan_contact_obs127_s42/model_999.pt). Training config: [env.yaml](logs/rsl_rl/ant/2026-10-03_01-10-01_ablation_heightscan_contact_obs127_s42/params/env.yaml), [agent.yaml](logs/rsl_rl/ant/2026-10-03_01-10-01_ablation_heightscan_contact_obs127_s42/params/agent.yaml). Actor input 127-D, saved iteration 999.

## 17. Limitations

한 training seed와 한 preregistered Final terrain realization의 결과다. Bootstrap CI는 이 realization 내부 episode variability를 기술하며 여러 unseen terrain에서의 일반적 causal effect를 입증하지 않는다. Mean displacement는 post-course motion을 포함한다. Traversal timing은 completer subset에 조건부이고, paired t40 비교는 두 정책 모두 endpoint에 도달한 joint completer만 사용한다. Survivor selection이 있으므로 완주자에서 빠르다는 결과가 더 높은 overall robustness를 뜻하지 않는다. Frozen fall/timeout 집계와 simultaneous termination/timeout flag의 차이는 [section report](validation/final_unseen_section_analysis.md)에 설명되어 있다. Final은 이미 결과가 공개된 evidence set이며 새로운 blind model-selection holdout은 아니다.

## 18. Reproduction

**Recorded runtime:** Python 3.11.14, Isaac Lab 2.3.0, Isaac Sim `5.0.0-rc.45+release.23960.184afb15.gl`, RSL-RL 3.0.1, PyTorch 2.7.0+cu128, CUDA 12.8. [Training runtime manifest](validation/ablation_training_runs.json)의 실제 기록이다. Simulator/assets와 NVIDIA GPU는 별도 준비하며 repository가 Isaac Sim을 포함하지 않는다. 이 실험 runtime과 upstream 설치 요구사항이 다를 수 있으므로 원본 기록을 우선 확인한다.

설정 참고: [upstream Isaac Lab README](docs/README.isaaclab.md), [environment.yml](environment.yml), [RL dependency definition](source/isaaclab_rl/setup.py). 준비된 Isaac Sim Python 환경에서 이 checkout의 packages를 설치할 수 있다:

```bash
./isaaclab.sh -i rsl_rl
```

**Evaluation:** Section 9의 명령을 사용한다. 나머지 ablation 재현 시 같은 task/seed/num_envs에 아래 checkpoint와 flags를 사용한다.

| Policy | Checkpoint (repo-relative) | Adapter flags |
|---|---|---|
| Base | [2026-10-02_03-39-50_ablation_base_obs60_s42/model_999.pt](logs/rsl_rl/ant/2026-10-02_03-39-50_ablation_base_obs60_s42/model_999.pt) | none |
| Original | [2026-10-03_00-52-49_ablation_heightscan_obs123_s42/model_999.pt](logs/rsl_rl/ant/2026-10-03_00-52-49_ablation_heightscan_obs123_s42/model_999.pt) | `--height_scan` |
| Contact | [2026-10-03_01-10-01_ablation_heightscan_contact_obs127_s42/model_999.pt](logs/rsl_rl/ant/2026-10-03_01-10-01_ablation_heightscan_contact_obs127_s42/model_999.pt) | `--height_scan --foot_contacts` |
| Diverse | [2026-10-03_18-57-46_ablation_heightscan_contact_diverse_s42/model_999.pt](logs/rsl_rl/ant/2026-10-03_18-57-46_ablation_heightscan_contact_diverse_s42/model_999.pt) | `--height_scan --foot_contacts` |

**Training reference:** Contact의 실제 task는 `Isaac-Ant-Terrain-Ablation-HeightScan-Contact-v0`이며 seed 42, num_envs 4096, max_iterations 1000이다. [Training script](scripts/reinforcement_learning/rsl_rl/train.py), [saved PPO config](logs/rsl_rl/ant/2026-10-03_01-10-01_ablation_heightscan_contact_obs127_s42/params/agent.yaml), [ablation orchestrator](scripts/assignment/run_ablation_training.py), [Diverse orchestrator](scripts/assignment/run_diverse_training.py)에 실행 경로와 설정을 보존했다. 제출 가중치 평가에는 재학습이 필요하지 않다.

**Offline analysis:** [section script](scripts/assignment/analyze_final_unseen_sections.py), [statistics script](scripts/assignment/analyze_final_unseen_statistics.py), [reward script](scripts/assignment/analyze_final_unseen_reward_components.py) 및 그 CSV/gzip telemetry 입력을 보존했다. 기존 outputs는 보호 대상이므로 이미 존재하는 report에 대한 overwrite 방지 검사에 유의한다. Historical manifests의 absolute path는 원본 provenance로 보존되어 있으며, 다른 머신에서 분석을 재실행하려면 경로 매핑을 별도 확인해야 한다. 평가 명령은 repo-relative이다.

**Video:** 현재 확인된 Contact video는 Dev SteppingStones rollout이며 Final representative video로 표시하지 않는다. Final Contact video는 확인되지 않아 README에 영상 결과를 넣지 않았다. 개발 영상은 외부 archive에 보존했다. 추후 동일 evaluation 명령에 `--video --video_length 960`을 추가해 별도로 생성할 수 있으나 이번 정리에서는 실행하지 않았다.

## 19. Repository Structure

```text
IsaacLab_RS/
├── README.md / SUBMISSION_COMMAND.txt
├── source/                       # Isaac Lab core, Ant tasks, observations, sensors, PPO
├── scripts/reinforcement_learning/rsl_rl/
│   ├── train.py / play_one_episode.py
│   └── ant_observation_adapter.py
├── scripts/assignment/           # reproducibility, measurement, offline analysis
├── logs/rsl_rl/ant/              # four representative weights + saved configs/curves
├── validation/                   # frozen Final/Dev evidence, manifests, analysis inputs
├── docs/figures/                 # selected unchanged presentation figures
└── isaaclab.sh / environment.yml / LICENSE
```

Framework dependencies와 기존 task registrations를 깨뜨리지 않기 위해 공유 source는 유지했다. 중간 checkpoint·폐기 실험·개발 영상·cache는 repository 외부 archive로 옮겼다. [Inventory](validation/submission_repo_inventory.md)와 [file-level classification](validation/submission_repo_inventory.csv)에 정리 근거를 기록했다. Git history는 수정하지 않았다. 아직 commit/push하지 않은 working tree이며 제출 전 네 weight/config가 포함되는지 확인해야 한다.

## 20. Key Artifacts

| Evidence | Artifact |
|---|---|
| Frozen Final | [Summary](validation/final_unseen_summary.md), [raw 400](validation/final_unseen_raw.csv) |
| First-pass / sections | [Report](validation/final_unseen_section_analysis.md), [JSON](validation/final_unseen_section_analysis.json) |
| Paired bootstrap | [Report](validation/final_unseen_statistical_analysis.md), [JSON](validation/final_unseen_statistical_analysis.json) |
| Stock reward decomposition | [Report](validation/final_unseen_reward_decomposition.md), [JSON](validation/final_unseen_reward_decomposition.json) |
| Official accounting | [Parity](validation/official_evaluator_parity.json), [addendum](validation/final_unseen_evaluation_protocol_addendum.json) |
| Preregistration / checkpoint mapping | [Preregistration](validation/final_unseen_preregister.json), [evaluation manifest](validation/final_unseen_eval_manifest.json), [portable model mapping](validation/submission_model_manifest.json) |
| Development / H3 provenance | [Dev report](validation/dev_ood_evaluation_summary.md), [Diverse report](validation/diverse_training_summary.md) |
| Submission integrity | [Cleanup validation](validation/submission_cleanup_validation.json) |

원본 Isaac Lab attribution 및 BSD-3-Clause license를 유지한다: [LICENSE](LICENSE), [CONTRIBUTORS](CONTRIBUTORS.md).
