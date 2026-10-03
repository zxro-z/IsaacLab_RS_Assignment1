"""Offline CIs and paired bootstrap from preserved Final and first-pass episodes.

Resample env_id blocks jointly across policies/metrics; no simulator or policy
imports. Verify pairing, original aggregates and protected hashes first. All
output files use exclusive creation. Population std is retained; reference SE
uses the sample variance. Percentile intervals are marginal, within-realization.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / 'validation'
RUNS = VAL / 'final_unseen_statistics_runs'
SEED = 2404
ITERATIONS = 50000
Z95 = 1.959963984540054
POLICIES = ('Base', 'Original HeightScan', 'HeightScan+Contact', 'Diverse HeightScan+Contact')
PAIRS = ((0, 1), (1, 2), (2, 3))
CONTINUOUS = ('episode_return', 'final_displacement_m', 'episode_length_steps', 'mean_forward_velocity_mps')
BINARY = ('fall', 'timeout', 'reached_40m', 'corridor_valid_40m', 'reached_25m', 'reached_18m')
METRICS = CONTINUOUS + BINARY
UNITS = dict(episode_return='stock reward', final_displacement_m='m', episode_length_steps='steps',
             mean_forward_velocity_mps='m/s', **{m: 'proportion' for m in BINARY}, t_40m_joint_completers='s')
LABELS = dict(episode_return='Return', final_displacement_m='Final displacement (m)',
             episode_length_steps='Episode length (steps)', mean_forward_velocity_mps='Mean forward velocity (m/s)',
             fall='Fall', timeout='Timeout (original label)', reached_40m='40 m reach',
             corridor_valid_40m='Corridor-valid 40 m completion', reached_25m='25 m reach', reached_18m='18 m reach')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boolean(value):
    assert value in ('True', 'False'), f'Invalid/missing boolean: {value!r}'
    return value == 'True'


def index_csv(path):
    rows = list(csv.DictReader(Path(path).open()))
    result = {(row['policy'], int(row['env_id'])): row for row in rows}
    assert len(rows) == len(result) == 400
    assert set(result) == {(p, i) for p in POLICIES for i in range(100)}
    return result


def wilson(count, n):
    assert 0 <= count <= n and n > 0
    q = count / n
    den = 1 + Z95**2 / n
    mid = (q + Z95**2 / (2*n)) / den
    half = Z95 * math.sqrt(q*(1-q)/n + Z95**2/(4*n*n)) / den
    return [max(0., mid-half), min(1., mid+half)]


def bootstrap_means(values, rng, iterations=ITERATIONS):
    """Multinomial episode multiplicities equal uniform sampling with replacement."""
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 2 and len(values) > 0 and np.isfinite(values).all()
    n = len(values)
    weights = rng.multinomial(n, np.full(n, 1/n), size=iterations)
    assert np.all(weights.sum(axis=1) == n)
    return weights @ values / n, weights


def boot_summary(estimate, samples):
    samples = np.asarray(samples)
    low, high = np.quantile(samples, [.025, .975], method='linear')
    center = float(samples.mean())
    median = float(np.median(samples))
    mc_se = float(samples.std(ddof=1) / math.sqrt(len(samples)))
    assert low-1e-12 <= estimate <= high+1e-12
    assert abs(center-estimate) <= 6*mc_se + 1e-10
    return dict(estimate=float(estimate), bootstrap_mean=center, bootstrap_median=median,
                ci95_low=float(low), ci95_high=float(high), monte_carlo_se_of_center=mc_se)


def validate_pairing(raw, first):
    manifest = read(VAL / 'final_unseen_eval_manifest.json')
    trajectory = read(VAL / 'final_unseen_first_pass_trajectory_manifest.json')
    assert manifest['task'] == trajectory['task'] == 'Isaac-Ant-Final-Unseen-v0'
    assert manifest['seed'] == trajectory['seed'] == 24
    assert manifest['num_envs'] == trajectory['num_envs'] == 100
    assert manifest['first_episode_only'] and trajectory['first_episode_only']
    assert read(VAL / 'official_evaluator_parity.json')['overall_parity']
    assert trajectory['consistency_pass'] and all(r['pass'] for r in trajectory['comparisons'])
    assert trajectory['original_raw_sha256'] == digest(VAL / 'final_unseen_raw.csv')
    reference = None
    evidence = []
    physical_keys = ('terrain_mesh_sha256', 'env_origins_sha256', 'robot_material_sha256')
    shared_hashes = None
    for policy, entry, measured in zip(POLICIES, manifest['policy_set'], trajectory['policies']):
        assert entry['label'] == measured['policy'] == policy
        original = read(VAL / 'final_unseen_runs' / (policy.replace(' ', '_').replace('+', '_') + '.json'))
        metadata = read(ROOT / measured['directory'] / 'metadata.json')
        assert original['seed'] == metadata['seed'] == 24
        assert original['terrain_seed'] == metadata['terrain_seed'] == 2404
        assert original['num_envs'] == metadata['num_envs'] == 100
        assert [r['env_id'] for r in original['episodes']] == list(range(100))
        assert [r['env_id'] for r in metadata['episodes']] == list(range(100))
        physical = {key: original[key] for key in physical_keys}
        if shared_hashes is None:
            shared_hashes = physical
            reference = metadata
        assert physical == shared_hashes
        for key in ('env_origins_xy', 'initial_root_xy', 'initial_terrain_local_xy'):
            assert np.array_equal(metadata[key], reference[key]), (policy, key)
        assert digest(entry['checkpoint_path']) == entry['checkpoint_integrity']['sha256'] == metadata['checkpoint_sha256']
        for i in range(100):
            r, f = raw[policy, i], first[policy, i]
            assert int(r['seed']) == 24 and int(r['num_envs']) == 100
            assert r['checkpoint'] == entry['checkpoint_path']
            assert float(r['episode_return']) == float(f['episode_return'])
            assert int(r['episode_length']) == int(f['episode_steps'])
            assert float(r['forward_displacement_m']) == float(f['final_displacement'])
            assert boolean(r['fall']) == boolean(f['fall'])
            assert boolean(r['time_out']) == boolean(f['timeout'])
        evidence.append(dict(policy=policy, initial_xy_exact_match=True, ordered_env_origins_exact_match=True,
                             env_id_order=list(range(100)), checkpoint_sha256=metadata['checkpoint_sha256']))
    # Verify the reviewed creation/reset order directly in source without importing Isaac Sim.
    local = (ROOT / 'scripts/reinforcement_learning/rsl_rl/play_one_episode.py').read_text()
    assert local.index('env_cfg.seed = agent_cfg.seed') < local.index('env = gym.make(')
    assert local.index('env = RslRlVecEnvWrapper(') < local.index('runner = OnPolicyRunner(') < local.index('runner.load(')
    wrapper = (ROOT / 'source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py').read_text()
    assert 'self.env.reset()' in wrapper and 'torch.randperm' not in wrapper
    ant = (ROOT / 'source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py').read_text()
    assert 'params={"pose_range": {}, "velocity_range": {}}' in ant
    assert '"position_range": (-0.2, 0.2)' in ant and '"velocity_range": (-0.1, 0.1)' in ant
    assert 'self.enable_corruption = False' in ant
    sections = read(VAL / 'final_unseen_section_analysis.json')
    for policy in POLICIES:
        n = sum(boolean(first[policy, i]['reach_40m']) and not boolean(first[policy, i]['lateral_exit_before_first_pass']) for i in range(100))
        assert n == sections['policies'][policy]['course_end_reached_without_lateral_exit']['count']
    return dict(valid=True, unit='env_id 0..99; resample the same env index across policies',
                seed=24, terrain_seed=2404, shared_physical_hashes=shared_hashes, policy_evidence=evidence,
                initial_state_evidence=dict(empirical='Ordered env origins and initial root XY exactly equal for all 100 indices.',
                    code='Common root reset with zero pose/velocity ranges; common joint offset sampling U(-0.2,0.2) and velocity U(-0.1,0.1); shared seed, sensor scene and ordered reset occur before policy-network construction.',
                    observation='Final scene contains height/contact sensors for all policies. Adapters retain the same sensor definitions and root/joint reset events; observation corruption disabled.',
                    limitation='Full initial joint/root state vectors were not logged. Their equivalence is inferred from reviewed initialization/RNG order and frozen common config, not claimed as a direct full-state measurement.'),
                ordering='Wrapper forwards ordered vectors; original per-env JSONs explicitly enumerate env_id 0..99; no permutation is introduced.',
                fallback='Unpaired analysis would be required if these checks failed; checks passed.')


def check_aggregates(raw):
    results = {r['policy']: r for r in csv.DictReader((VAL / 'final_unseen_results.csv').open())}
    summary = (VAL / 'final_unseen_summary.md').read_text()
    checks = []
    for policy in POLICIES:
        r = [raw[policy, i] for i in range(100)]
        a = results[policy]
        for source, mean_key, std_key in (
            ('episode_return', 'reward_mean', 'reward_std'),
            ('forward_displacement_m', 'forward_displacement_mean', 'forward_displacement_std'),
            ('episode_length', 'episode_length_mean', 'episode_length_std'),
            ('forward_velocity_mean_mps', 'forward_velocity_mean', None)):
            values = np.asarray([float(x[source]) for x in r])
            for key, value in [(mean_key, float(values.mean()))] + ([(std_key, float(values.std(ddof=0)))] if std_key else []):
                old = float(a[key])
                assert np.isclose(value, old, rtol=1e-6, atol=1e-6), (policy, key)
                checks.append(dict(policy=policy, metric=key, raw_float64=value, published_float32=old, difference=value-old))
        for key, field in [('fall', 'fall'), ('timeout', 'time_out')]:
            count = sum(boolean(x[field]) for x in r)
            assert count == int(a[key+'_count'])
            assert np.isclose(count/100, float(a[key+'_ratio']), atol=1e-8, rtol=1e-6)
        # Verify the displayed Markdown table's three-decimal population summaries too.
        assert f"{float(a['reward_mean']):.3f} ± {float(a['reward_std']):.3f}" in summary
        assert f"{float(a['forward_displacement_mean']):.3f} ± {float(a['forward_displacement_std']):.3f}" in summary
    return checks


def prepare(raw, first):
    data = np.empty((100, len(POLICIES), len(METRICS)), dtype=float)
    times = np.full((100, len(POLICIES)), np.nan)
    for j, policy in enumerate(POLICIES):
        for i in range(100):
            r, f = raw[policy, i], first[policy, i]
            values = [float(r[k]) for k in ('episode_return', 'forward_displacement_m', 'episode_length', 'forward_velocity_mean_mps')]
            reach = boolean(f['reach_40m'])
            values += [boolean(r['fall']), boolean(r['time_out']), reach,
                       reach and not boolean(f['lateral_exit_before_first_pass']), boolean(f['reach_25m']), boolean(f['reach_18m'])]
            data[i, j] = values
            if reach:
                times[i, j] = float(f['t_40m'])
                assert np.isfinite(times[i, j])
            else:
                assert f['t_40m'] == ''
    assert np.isfinite(data).all()
    return data, times


def main():
    outputs = ('final_unseen_policy_ci.csv', 'final_unseen_pairwise_bootstrap.csv', 'final_unseen_pairwise_binary.csv',
               'final_unseen_statistical_analysis.json', 'final_unseen_statistical_analysis.md')
    assert not RUNS.exists() and not any((VAL / p).exists() for p in outputs), 'Refuse overwrite'
    protected = {str(p.relative_to(ROOT)): digest(p) for p in VAL.rglob('*') if p.is_file()}
    prereg = read(VAL / 'final_unseen_preregister.json')
    for name, expected in prereg['source_hashes'].items():
        assert digest(ROOT / name) == expected
        protected[name] = expected
    for p in (ROOT / 'scripts/assignment').glob('*.py'):
        if p.resolve() != Path(__file__).resolve(): protected[str(p.relative_to(ROOT))] = digest(p)
    for e in read(VAL / 'final_unseen_eval_manifest.json')['policy_set']:
        path = Path(e['checkpoint_path'])
        assert digest(path) == e['checkpoint_integrity']['sha256']
        protected[str(path.relative_to(ROOT))] = digest(path)
    RUNS.mkdir()
    save(RUNS / 'protected_before.json', protected)
    raw, first = index_csv(VAL / 'final_unseen_raw.csv'), index_csv(VAL / 'final_unseen_first_pass_raw.csv')
    pairing = validate_pairing(raw, first)
    reviewed_sources = ['scripts/assignment/evaluate_final_unseen.py', 'scripts/assignment/evaluate_final_unseen_first_pass.py',
        'scripts/reinforcement_learning/rsl_rl/play_one_episode.py', 'scripts/reinforcement_learning/rsl_rl/ant_observation_adapter.py',
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py',
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_final_unseen_env_cfg.py',
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_terrain_heightscan_env_cfg.py',
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_contact_observations.py',
        'source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_dev_ood_env_cfg.py',
        'source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py', 'source/isaaclab/isaaclab/envs/manager_based_env.py',
        'source/isaaclab/isaaclab/envs/mdp/events.py', 'source/isaaclab/isaaclab/terrains/terrain_importer.py',
        'source/isaaclab/isaaclab/managers/observation_manager.py',
        '/home/zxro/anaconda3/envs/env_isaaclab_231/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner.py']
    pairing['reviewed_source_sha256'] = {n: digest(ROOT/n) for n in reviewed_sources}
    save(RUNS/'pairing_evidence.json', pairing)
    consistency = check_aggregates(raw)
    data, times = prepare(raw, first)
    # One common block resampling distribution preserves pairing across all policies and metrics.
    rng = np.random.default_rng(SEED)
    means, weights = bootstrap_means(data.reshape(100, -1), rng)
    boots = means.reshape(ITERATIONS, len(POLICIES), len(METRICS))
    estimates = data.mean(axis=0)
    policy_rows, pair_rows, binary_rows, policy_json, pair_json, timing_json = [], [], [], {}, {}, []
    for j, policy in enumerate(POLICIES):
        policy_json[policy] = {}
        for k, metric in enumerate(METRICS):
            values = data[:, j, k]
            estimate = float(estimates[j, k]); std = float(values.std(ddof=0))
            se = float(values.std(ddof=1) / math.sqrt(100))
            base = dict(policy=policy, metric=metric, n=100, estimate=estimate, std=std, se=se)
            if metric in CONTINUOUS:
                b = boot_summary(estimate, boots[:, j, k])
                normal = [estimate-Z95*se, estimate+Z95*se]
                policy_rows.append(dict(**base, ci_method='percentile_episode_bootstrap', ci95_low=b['ci95_low'], ci95_high=b['ci95_high']))
                policy_rows.append(dict(**base, ci_method='normal_mean_plus_minus_1.96_sample_SE', ci95_low=normal[0], ci95_high=normal[1]))
                policy_json[policy][metric] = dict(**base, bootstrap=b, reference_normal_ci95=normal)
            else:
                count = int(values.sum()); ci = wilson(count, 100)
                assert 0 <= ci[0] <= estimate <= ci[1] <= 1
                policy_rows.append(dict(**base, ci_method='Wilson_95', ci95_low=ci[0], ci95_high=ci[1]))
                policy_json[policy][metric] = dict(**base, count=count, wilson_ci95=ci)
    original_relative = {(x['from'], x['to']): x for x in read(VAL / 'final_unseen_pairwise.json')}
    for ref, new in PAIRS:
        name = f'{POLICIES[ref]} → {POLICIES[new]}'
        pair_json[name] = {}
        for k, metric in enumerate(METRICS):
            delta = data[:, new, k]-data[:, ref, k]
            observed = float(delta.mean())
            distribution = boots[:, new, k]-boots[:, ref, k]
            # Verify paired resampling directly, rather than independently resampling two means.
            assert np.allclose(distribution, weights @ delta / 100, rtol=1e-12, atol=1e-12)
            b = boot_summary(observed, distribution)
            reference_mean, new_mean = float(estimates[ref, k]), float(estimates[new, k])
            relative = 100*(new_mean-reference_mean)/reference_mean if metric in CONTINUOUS and reference_mean != 0 else None
            published_relative = original_relative[POLICIES[ref], POLICIES[new]].get({'episode_return':'reward_percent', 'final_displacement_m':'displacement_percent'}.get(metric, ''))
            if published_relative is not None:
                assert math.isclose(relative, published_relative, rel_tol=1e-6, abs_tol=1e-4)
            row = dict(comparison=name, metric=metric, n_pairs=100, reference_mean=reference_mean, new_mean=new_mean,
                       observed_delta=observed, bootstrap_median_delta=b['bootstrap_median'], ci95_low=b['ci95_low'],
                       ci95_high=b['ci95_high'], relative_change_percent=relative,
                       published_relative_change_percent=published_relative, unit=UNITS[metric],
                       condition='all paired first episodes')
            pair_rows.append(row)
            pair_json[name][metric] = dict(**row, bootstrap=b, zero_in_interval=b['ci95_low'] <= 0 <= b['ci95_high'])
            if metric in BINARY:
                a, c = data[:, ref, k].astype(bool), data[:, new, k].astype(bool)
                counts = dict(both_0=int((~a & ~c).sum()), reference_1_new_0=int((a & ~c).sum()),
                              reference_0_new_1=int((~a & c).sum()), both_1=int((a & c).sum()))
                assert sum(counts.values()) == 100
                assert math.isclose((counts['reference_0_new_1']-counts['reference_1_new_0'])/100, observed, abs_tol=1e-12)
                assert -1-1e-12 <= b['ci95_low'] <= b['ci95_high'] <= 1+1e-12
                binary = dict(comparison=name, metric=metric, n_pairs=100, **counts,
                              reference_rate=reference_mean, new_rate=new_mean, delta_percentage_points=100*observed,
                              ci95_low=100*b['ci95_low'], ci95_high=100*b['ci95_high'], ci_unit='percentage_points',
                              discordant_pairs=counts['reference_1_new_0']+counts['reference_0_new_1'])
                binary_rows.append(binary); pair_json[name][metric]['discordance'] = binary
        mask = np.isfinite(times[:, ref]) & np.isfinite(times[:, new])
        n = int(mask.sum())
        assert n > 0
        tdelta = times[mask, new]-times[mask, ref]
        tboots, _ = bootstrap_means(tdelta[:, None], rng)
        b = boot_summary(float(tdelta.mean()), tboots[:, 0])
        timing = dict(comparison=name, metric='t_40m_joint_completers', n_pairs=n,
                      reference_mean=float(times[mask, ref].mean()), new_mean=float(times[mask, new].mean()),
                      observed_delta=float(tdelta.mean()), bootstrap_median_delta=b['bootstrap_median'],
                      ci95_low=b['ci95_low'], ci95_high=b['ci95_high'], relative_change_percent=None,
                      published_relative_change_percent=None, unit='s',
                      condition='conditional on both policies completing the course')
        pair_rows.append(timing)
        timing_json.append(dict(**timing, bootstrap=b, joint_env_ids=np.flatnonzero(mask).tolist(),
                                excluded_pairs=100-n, zero_in_interval=b['ci95_low'] <= 0 <= b['ci95_high']))
    changed = [n for n, h in protected.items() if digest(ROOT/n) != h]
    assert not changed
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(), analysis='offline post-hoc Final descriptive statistics',
        settings=dict(bootstrap_seed=SEED, iterations=ITERATIONS, rng='numpy.default_rng PCG64',
            resampling='Uniform paired env_id blocks with replacement, implemented as Multinomial(n, 1/n) multiplicities; shared across all policy/metric estimates.',
            timing_resampling='Same RNG stream continues in the three prespecified pair order, within each fixed joint-completer subset.',
            ci_method='Percentile 2.5%/97.5%, numpy quantile method=linear', binary_ci_method='Wilson 95%',
            reference_ci='Mean ± z(0.975) * sample_std(ddof=1)/sqrt(n)', reported_std='population ddof=0',
            paired_delta_direction='new policy minus reference policy; negative fall delta means fewer falls',
            tests='No p-values or McNemar tests; no multiplicity correction. All intervals are marginal.',
            numpy_version=np.__version__),
        pairing=pairing, definitions=dict(
            episode_return='Original stock reward first-episode sum from raw CSV',
            final_displacement_m='Original raw forward_displacement_m, measured at pre-terminal step N-1',
            episode_length_steps='Original terminal-inclusive first-episode length',
            mean_forward_velocity_mps='Original pre-step forward_velocity_mean_mps',
            fall='Original raw fall label', timeout='Original raw time_out label: preserve 81/77/84/67 counts; simultaneous Base fall/timeout remains legacy-labelled fall only',
            reached_40m='Existing reach_40m at terrain-local x=40, approximately 38 m displacement from local spawn x=2',
            corridor_valid_40m='Existing first-pass definition: reach_40m AND NOT lateral_exit_before_first_pass; no new criterion',
            reached_18m='Existing reach_18m', reached_25m='Existing reach_25m',
            t_40m_joint_completers='Existing t_40m on env indices with both reach_40m true; conditional on both policies completing the course; not corridor-conditioned',
            relative_change='Descriptive 100*(new mean-reference mean)/reference mean for continuous metrics; published original percentages retained separately'),
        policy_ci=policy_json, pairwise=pair_json, conditional_timing=timing_json,
        consistency=dict(passed=True, tolerance=dict(rtol=1e-6, atol=1e-6), aggregates=consistency,
            continuous_difference_reason='Float64 raw reductions versus original float32 Torch reductions; population std definition unchanged.'),
        validation=dict(each_policy_n=100, each_unconditional_pair_n=100, raw_rows=400, first_pass_rows=400,
            missing_continuous=0, missing_binary=0, missing_timing='Only unreachable boundaries; handled by explicitly reported joint subset',
            paired_weights_direct_crosscheck=True, bootstrap_center_checked=True, estimate_inside_ci_checked=True,
            binary_interval_bounds_checked=True),
        integrity=dict(protected_count=len(protected), changed=changed, simulation_performed=False,
            source_sha256=digest(Path(__file__)), original_raw_sha256=digest(VAL/'final_unseen_raw.csv'),
            first_pass_raw_sha256=digest(VAL/'final_unseen_first_pass_raw.csv')),
        limitation='One fixed preregistered Final realization, one evaluation seed and one training seed; env bootstrap estimates within-realization episode variability. It does not establish causal sensor/training effects or generalization across unseen terrains/training seeds. Marginal intervals across many metrics are not simultaneous confidence guarantees.')
    write_csv(VAL / outputs[0], policy_rows)
    write_csv(VAL / outputs[1], pair_rows)
    write_csv(VAL / outputs[2], binary_rows)
    save(VAL / outputs[3], result)
    report(result)
    plot(result)
    save(RUNS/'validation.json', dict(passed=True, **result['validation'], protected_count=len(protected), changed=changed))
    print('STATISTICAL ANALYSIS PASS: pairing valid; 50,000 resamples; no simulation; prior artifacts preserved', flush=True)


def number(x):
    return f'{x:+.3f}' if x is not None else 'NA'


def interval(r, scale=1):
    return f"[{scale*r['ci95_low']:+.3f}, {scale*r['ci95_high']:+.3f}]"


def report(a):
    lines = ['# Frozen Final: confidence intervals and paired bootstrap', '',
        '## A. Pairing validation', '',
        'Pairing passed. All four runs use seed 24, terrain seed 2404, 100 envs, and matching ordered terrain-mesh/env-origin/material hashes. The stored ordered env origins and initial root XY arrays match exactly at every env_id. Original and measurement JSONs retain env_id 0–99 ordering, and measurement episodes reproduce all original outcomes. Root reset is deterministic relative to each origin; joint positions/velocities use common uniform reset samplers. The shared sensor scene and reset occur before policy-specific networks are constructed; observation corruption is disabled. Full initial joint/root vectors were not stored: matching those components is supported by the reviewed code/RNG initialization path, not a direct full-state measurement. Detailed source/manifest evidence is in the JSON.',
        '', '## B. Statistical method', '',
        f"50,000 nonparametric bootstrap resamples, RNG seed {SEED}, NumPy PCG64. Each draw resamples 100 env_id blocks jointly across all policies/metrics. Percentile 95% intervals use the 2.5th/97.5th percentiles. A multinomial multiplicity vector implements uniform episode sampling with replacement. Conditional timing uses its own joint-completer subset. Continuous per-policy bootstrap CIs and normal reference CIs are both saved; reference SE uses sample variance (ddof=1), while displayed population std remains ddof=0. Binary per-policy CIs use Wilson. Paired binary differences use the same paired bootstrap and are reported in percentage points.",
        '', 'No p-values or multiplicity correction are used. Intervals are marginal, and the many reported metrics do not provide simultaneous confidence guarantees. The resampling units are episode/env indices within this single realization, not independent terrain or training seeds.',
        '', '## C. Per-policy confidence intervals', '',
        '| Policy | Return mean [bootstrap CI] | Distance mean m [bootstrap CI] | Length mean steps [bootstrap CI] |',
        '|---|---:|---:|---:|']
    for policy,d in a['policy_ci'].items():
        cells=[]
        for m in CONTINUOUS[:3]:
            b=d[m]['bootstrap'];cells.append(f"{b['estimate']:.3f} [{b['ci95_low']:.3f}, {b['ci95_high']:.3f}]")
        lines.append('| '+policy+' | '+' | '.join(cells)+' |')
    lines += ['', '| Policy | Fall | Original timeout | 40 m reach | Corridor-valid 40 m |', '|---|---:|---:|---:|---:|']
    for policy,d in a['policy_ci'].items():
        cells=[]
        for m in BINARY[:4]:
            r=d[m];lo,hi=r['wilson_ci95'];cells.append(f"{r['count']}/100 ({100*r['estimate']:.0f}%; CI {100*lo:.1f}–{100*hi:.1f})")
        lines.append('| '+policy+' | '+' | '.join(cells)+' |')
    lines += ['', 'All continuous mean/std/SE/normal reference intervals, velocity, and optional 18/25 m reach are in CSV/JSON. Existing means/std agree within rtol=1e-6, atol=1e-6; small reduction differences reflect original float32 versus new float64 arithmetic. Timeout retains the original evaluator labels, including the Base simultaneous fall/timeout episode classified as fall only.',
        '', '## D. Prespecified paired continuous differences', '',
        '| Comparison | Metric | Ref | New | Δ | 95% paired bootstrap CI |',
        '|---|---|---:|---:|---:|---:|']
    for comparison,d in a['pairwise'].items():
        for metric in CONTINUOUS[:3]:
            r=d[metric]
            lines.append(f"| {comparison} | {LABELS[metric]} | {r['reference_mean']:.3f} | {r['new_mean']:.3f} | {number(r['observed_delta'])} | {interval(r)} |")
    lines += ['', 'All unconditional comparisons have 100 pairs. Δ=new−reference. Bootstrap distribution medians and mean centers are recorded alongside the intervals in CSV/JSON.',
        '', '## E. Prespecified paired binary differences', '',
        '| Comparison | Metric | Ref | New | Δ pp | 95% paired CI pp | Discordant pairs 1→0 / 0→1 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for comparison,d in a['pairwise'].items():
        for metric in ('fall','reached_40m','corridor_valid_40m'):
            r=d[metric];b=r['discordance']
            lines.append(f"| {comparison} | {LABELS[metric]} | {100*r['reference_mean']:.0f}% | {100*r['new_mean']:.0f}% | {100*r['observed_delta']:+.1f} | {interval(r,100)} | {b['reference_1_new_0']} / {b['reference_0_new_1']} |")
    lines += ['', 'For fall, 1→0 means fall→no fall and a negative Δ is fewer falls. For reach/corridor outcomes, 1→0 means success→failure. Full 2×2 tables including both-zero/both-one counts, timeout and 18/25 m reach are saved in the binary CSV. No McNemar p-values are added.',
        '', '## F. Conditional 40 m timing', '',
        '**conditional on both policies completing the course**', '',
        '| Comparison | Joint pairs | Ref mean s | New mean s | Δ s | 95% paired CI s |',
        '|---|---:|---:|---:|---:|---:|']
    for r in a['conditional_timing']:
        lines.append(f"| {r['comparison']} | {r['n_pairs']} | {r['reference_mean']:.3f} | {r['new_mean']:.3f} | {number(r['observed_delta'])} | {interval(r)} |")
    lines += ['', 'The subset is defined using existing terrain-local x=40 reach, not corridor validity. Pairs where only one policy reaches 40 m are excluded. This selection changes the comparison population, so shorter timing does not imply higher overall completion or robustness. Missing t_40m values belong only to non-reachers and are not imputed.',
        '', '## G. Interpretation within this preregistered Final realization', '']
    for pair_index,(comparison,d) in enumerate(a['pairwise'].items()):
        lines += [f'### {comparison}', '']
        for metric in ('episode_return','final_displacement_m','episode_length_steps','fall','reached_40m','corridor_valid_40m'):
            r=d[metric];scale=100 if metric in BINARY else 1
            direction='negative' if r['observed_delta']<0 else 'positive' if r['observed_delta']>0 else 'zero'
            statement='the paired interval included zero' if r['zero_in_interval'] else 'the marginal paired interval excluded zero'
            lines.append(f"{LABELS[metric]}: observed Δ {scale*r['observed_delta']:+.3f} {'pp' if metric in BINARY else UNITS[metric]}, CI {interval(r,scale)}; the observed direction was {direction}, and {statement}.")
        for metric in ('episode_return','final_displacement_m'):
            r=d[metric]
            lines.append(f"Descriptive {LABELS[metric]} relative change: {r['published_relative_change_percent']:+.2f}% (preserved original published value). This percentage is not the absolute-difference CI.")
        if pair_index==0:
            lines += ['The shorter first-pass timing and positive whole-episode displacement difference are related progress measurements but measure different portions of the trajectory. First-course endpoint reach changes by only +2 pp, while whole-episode distance includes movement beyond the initial course. The pattern is consistent with terrain-awareness-related progress differences in this realization; it does not identify a causal sensor effect.']
        elif pair_index==1:
            lines += ['Existing section analysis localized the total fall reduction (23→16) to post-course falls (17→10); pre-course falls remained 6→6, and X endpoint reach remained 94%→94%. The paired fall interval describes uncertainty around that observed reduction, rather than establishing it. Near-zero displacement and unchanged X completion are consistent with the prior post-course-stability interpretation, while corridor-valid completion increases descriptively.']
        else:
            lines += ['Diverse has fewer X/corridor completions and more falls in the observed episodes, despite shorter timing among joint completers. The timing subset excludes one-policy failures; it cannot establish an overall speed/robustness advantage. Earlier section analysis localized six Diverse pre-course falls in Landing versus zero for Contact. Broader training distribution did not translate into uniformly higher observed Final outcomes in this fixed realization.']
        lines += ['']
    lines += ['The result does not establish a general effect across unseen terrains. There is one fixed terrain realization, one evaluation seed and one training seed, with no independent retraining replication. Resampling 100 episode indices quantifies conditional episode variability and does not make the terrain/test realization itself representative.',
        '', '## H. Integrity and validation', '',
        f"No simulation, training, policy selection or tuning. All {a['integrity']['protected_count']} pre-existing protected artifacts/checkpoints/sources retained their hashes. Original 400 rows and all first-pass/preregistration/parity artifacts were preserved. Original population means/std and fall/timeout counts were verified. Missingness, sample sizes, interval bounds, paired weighting, bootstrap centers and estimate containment were checked.",
        '', 'New script: `scripts/assignment/analyze_final_unseen_statistics.py`. Outputs: `final_unseen_policy_ci.csv`, `final_unseen_pairwise_bootstrap.csv`, `final_unseen_pairwise_binary.csv`, `final_unseen_statistical_analysis.json`, this report, and the paired-CI figure. Full protection snapshot, pairing source fingerprints, independent validation and commands are in `final_unseen_statistics_runs/`.',
        '', '![Paired confidence intervals](final_unseen_statistical_analysis.png)', '']
    with (VAL/'final_unseen_statistical_analysis.md').open('x') as stream:
        stream.write('\n'.join(lines))


def plot(a):
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/final-unseen-statistics-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=VAL/'final_unseen_statistical_analysis.png'
    assert not target.exists()
    metrics=('episode_return','final_displacement_m','fall','reached_40m','corridor_valid_40m')
    labels=('Base → Original','Original → Contact','Contact → Diverse')
    fig,axes=plt.subplots(1,len(metrics),figsize=(15,4.5),sharey=True)
    for ax,metric in zip(axes,metrics):
        scale=100 if metric in BINARY else 1
        for y,(comparison,d) in enumerate(a['pairwise'].items()):
            r=d[metric];point=r['observed_delta']*scale
            lo,hi=r['ci95_low']*scale,r['ci95_high']*scale
            ax.errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',capsize=4)
        ax.axvline(0,color='gray',linestyle='--',linewidth=1)
        ax.set_title(LABELS[metric].replace('Corridor-valid 40 m completion','Corridor-valid 40 m'),fontsize=10)
        ax.set_xlabel('Δ pp' if metric in BINARY else 'Δ '+UNITS[metric]);ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(3),labels);axes[0].invert_yaxis()
    fig.suptitle('Fixed Final realization: paired env-index differences with marginal 95% bootstrap CI',fontsize=12)
    fig.text(.5,.015,'New − reference; 100 pairs; 50,000 resamples. Negative fall Δ means fewer falls. No inference across terrain/training seeds.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.91));fig.savefig(target,dpi=180);plt.close(fig)


if __name__ == '__main__':
    main()
