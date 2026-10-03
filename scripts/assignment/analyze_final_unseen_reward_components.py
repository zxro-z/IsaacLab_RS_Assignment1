"""Offline decomposition of unchanged stock env.step rewards from buffer telemetry.

Keep native evaluator returns authoritative. Component sums use float64 for
analysis, so identity checks quantify native float32 accumulation roundoff.
All rates/segments/conditioning are explicit; bootstrap is paired by env_id.
"""
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from analyze_final_unseen_statistics import bootstrap_means, boot_summary, POLICIES, PAIRS

ROOT=Path(__file__).resolve().parents[2];VAL=ROOT/'validation';RUNS=VAL/'final_unseen_reward_runs'
TERMS=('progress','alive','upright','move_to_target','action_l2','energy','joint_pos_limits')
MEASURES=('episode','pre40','post40','per_step','per_second')
BOOTSTRAP_SEED=2404;BOOTSTRAP_ITERATIONS=50000
EPISODE_RTOL=2e-6;EPISODE_ATOL=1e-6
DESCRIPTIONS=dict(
 progress='Potential difference towards world target (1000,0,0). __call__ zeros target-vector z before distance; reset uses initial 3-D distance. Potential=-distance/control_dt, then manager multiplies by dt. No separate linear-velocity reward.',
 alive='(~termination_manager.terminated).float(): 1 for non-fall steps including pure timeout; 0 on non-timeout termination, including simultaneous fall+timeout.',
 upright='1 if base-up world-Z projection > 0.93, else 0.',
 move_to_target='Heading projection towards target: 1 if projection > 0.8, else projection/0.8 (may be negative).',
 action_l2='sum(action_manager.action squared).',
 energy='sum(abs(action * joint_velocity * normalized_gear_ratio)); all gear ratios 15, normalized by maximum => 1. This is the stock proxy, not a new physical energy measurement.',
 joint_pos_limits='Normalize joints to [-1,1] using soft limits; sum (abs(normalized_position)>0.99) * (abs(position)-0.99)/(1-0.99) * normalized_gear_ratio.')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,value):
    with Path(p).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
def write_csv(p,rows):
    with Path(p).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def stats(values,samples):
    values=np.asarray(values,float);b=boot_summary(float(values.mean()),samples)
    return dict(n=len(values),mean=float(values.mean()),std=float(values.std(ddof=0)),median=float(np.median(values)),
        ci95_low=b['ci95_low'],ci95_high=b['ci95_high'],bootstrap_median=b['bootstrap_median'])


def load():
    manifest=read(VAL/'final_unseen_reward_measurement_manifest.json')
    assert manifest['consistency_pass'] and len(manifest['comparisons'])==400
    assert all(c['pass'] for c in manifest['comparisons'])
    assert read(VAL/'final_unseen_statistical_analysis.json')['pairing']['valid']
    original={(r['policy'],int(r['env_id'])):r for r in csv.DictReader((VAL/'final_unseen_raw.csv').open())}
    first={(r['policy'],int(r['env_id'])):r for r in csv.DictReader((VAL/'final_unseen_first_pass_raw.csv').open())}
    # Last channel always env.step total reward, never replace it by the component sum.
    values=np.zeros((100,4,len(MEASURES),len(TERMS)+1));lengths=np.zeros((100,4),int)
    episode_rows=[];checks=[];definitions=None;dt=None;subsets={}
    all_step_errors=[];ordered_errors=[];strict_failures=[]
    for j,entry in enumerate(manifest['policies']):
        policy=entry['policy'];assert policy==POLICIES[j]
        directory=ROOT/entry['directory'];meta=read(directory/'metadata.json')
        assert digest(directory/'trajectory.csv.gz')==entry['trajectory_sha256']
        assert meta['stock_reward_config_match'] and not meta['reward_compute_or_weights_modified']
        if definitions is None:definitions=meta['reward_terms'];dt=meta['control_dt']
        assert meta['reward_terms']==definitions and meta['control_dt']==dt
        data={i:[] for i in range(100)}
        with gzip.open(directory/'trajectory.csv.gz','rt') as stream:
            for r in csv.DictReader(stream):
                if int(r['step'])>0:data[int(r['env_id'])].append(r)
        subsets[policy]={k:[] for k in ['40m_completers','40m_noncompleters','fall','timeout_original_label']}
        for ep in meta['episodes']:
            i=ep['env_id'];rows=data[i];old=original[policy,i];f=first[policy,i];n=len(rows)
            assert n==ep['episode_steps']==int(old['episode_length'])
            assert [int(r['step']) for r in rows]==list(range(1,n+1))
            assert rows[-1]['done']=='True' and all(r['done']=='False' for r in rows[:-1])
            contribution=np.array([[float(r['reward_'+k]) for k in TERMS] for r in rows])
            total=np.array([float(r['total_reward']) for r in rows])
            assert np.isfinite(contribution).all() and np.isfinite(total).all()
            step_error=contribution.sum(1)-total
            ordered=np.zeros(n,np.float32)
            for k in range(len(TERMS)):ordered+=contribution[:,k].astype(np.float32)
            assert np.allclose(contribution.sum(1),total,rtol=1e-6,atol=1e-6)
            native=ep['episode_return'];assert native==float(old['episode_return'])
            native_from_trace=float(np.cumsum(total.astype(np.float32),dtype=np.float32)[-1]);assert native_from_trace==native
            summed=contribution.sum(0);trace_total=float(total.sum());component_total=float(summed.sum())
            # Strict trace identity isolates buffer dt roundtrip + per-step native reward reduction.
            assert np.isclose(component_total,trace_total,rtol=1e-6,atol=1e-6)
            strict=bool(np.isclose(component_total,native,rtol=1e-6,atol=1e-6))
            if not strict:strict_failures.append(dict(policy=policy,env_id=i,error=component_total-native))
            assert np.isclose(component_total,native,rtol=EPISODE_RTOL,atol=EPISODE_ATOL)
            # Float32 sequential sum error bound: gamma_N * sum(abs(env rewards)), plus
            # observed component-vs-native per-step reduction discrepancy. No reward is redefined.
            u=float(np.finfo(np.float32).eps)/2;gamma=n*u/(1-n*u)
            bound=gamma*float(np.abs(total).sum())+float(np.abs(step_error).sum())+1e-12
            assert abs(component_total-native)<=bound
            # Alive is independently determined by the actual termination signal, not legacy timeout label.
            terminated=np.array([r['terminated']=='True' for r in rows])
            alive_expected=(~terminated).astype(float)*definitions[1]['weight']*dt
            assert np.allclose(contribution[:,1],alive_expected,atol=1e-9,rtol=1e-6)
            reach=f['reach_40m']=='True';hit=int(f['first_reach_step_40m']) if reach else n
            observed_hits=[int(r['step']) for r in rows if float(r['terrain_local_x'])>=40]
            assert bool(observed_hits)==reach and (not reach or observed_hits[0]==hit)
            pre=contribution[:hit].sum(0);post=contribution[hit:].sum(0)
            pre_total=float(total[:hit].sum());post_total=float(total[hit:].sum())
            assert np.allclose(pre+post,summed,atol=1e-10)
            values[i,j,0]=np.r_[summed,native]
            values[i,j,1]=np.r_[pre,pre_total];values[i,j,2]=np.r_[post,post_total]
            values[i,j,3]=values[i,j,0]/n;values[i,j,4]=values[i,j,0]/(n*dt);lengths[i,j]=n
            row=dict(policy=policy,env_id=i,episode_return=native,episode_steps=n,episode_duration_s=n*dt,
                final_displacement=ep['legacy_final_displacement'],fall=ep['fall'],timeout=ep['timeout'],
                reach_40m=reach,first_reach_step_40m=hit if reach else None,pre40_steps=hit,post40_steps=n-hit,
                total_from_step_sum_float64=trace_total,component_sum=component_total,
                component_minus_native_return=component_total-native,pre40_total_reward=pre_total,post40_total_reward=post_total)
            for k,term in enumerate(TERMS):
                for scope,s in [('sum',summed),('pre40',pre),('post40',post)]:row[term+'_'+scope]=float(s[k])
                row[term+'_per_step']=float(summed[k]/n);row[term+'_per_second']=float(summed[k]/(n*dt))
            episode_rows.append(row)
            checks.append(dict(policy=policy,env_id=i,episode_steps=n,max_absolute_step_error=float(np.max(abs(step_error))),
                mean_absolute_step_error=float(np.mean(abs(step_error))),component_minus_native_return=component_total-native,
                component_minus_float64_trace_total=component_total-trace_total,
                float64_trace_minus_native_return=trace_total-native,
                component_sum_native_float32_accumulation=float(np.cumsum(ordered,dtype=np.float32)[-1]),
                float32_roundoff_bound=bound,episode_identity_pass=True))
            all_step_errors.extend(np.abs(step_error).tolist());ordered_errors.extend(np.abs(ordered.astype(float)-total).tolist())
            subsets[policy]['40m_completers' if reach else '40m_noncompleters'].append(i)
            if ep['fall']:subsets[policy]['fall'].append(i)
            if ep['timeout']:subsets[policy]['timeout_original_label'].append(i)
    assert len(episode_rows)==400 and np.isfinite(values).all()
    identity=dict(step_count=len(all_step_errors),step_max_absolute_error=max(all_step_errors),
        step_mean_absolute_error=float(np.mean(all_step_errors)),ordered_float32_step_max_error=max(ordered_errors),
        episode_max_absolute_native_error=max(abs(x['component_minus_native_return']) for x in checks),
        episode_mean_absolute_native_error=float(np.mean([abs(x['component_minus_native_return']) for x in checks])),
        episode_max_absolute_float64_trace_error=max(abs(x['component_minus_float64_trace_total']) for x in checks),
        step_tolerance=dict(rtol=1e-6,atol=1e-6),episode_tolerance=dict(rtol=EPISODE_RTOL,atol=EPISODE_ATOL),
        nominal_episode_1e6_tolerance_failures=len(strict_failures),nominal_failures=strict_failures,
        justification='Nominal failures reflect native float32 sequential-accumulation roundoff; actual total trajectory exactly reproduces existing evaluation. Episode tolerance 2e-6 relative + 1e-6 absolute retains all observed errors and every episode also passes the explicit gamma_N roundoff bound.',
        all_step_checks_pass=True,all_episode_checks_pass=True,episodes=checks)
    return manifest,definitions,dt,values,lengths,episode_rows,identity,subsets


def main():
    outputs=['final_unseen_reward_components_episode.csv','final_unseen_reward_components_summary.csv',
        'final_unseen_reward_components_pairwise.csv','final_unseen_reward_components_conditioned.csv',
        'final_unseen_reward_decomposition.json','final_unseen_reward_decomposition.md',
        'final_unseen_reward_components_policy.png','final_unseen_reward_components_pairwise.png',
        'final_unseen_reward_decomposition_addendum.json']
    assert not any((VAL/x).exists() for x in outputs),'Refuse overwrite'
    manifest,definitions,dt,values,lengths,episode_rows,identity,subsets=load()
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    means,weights=bootstrap_means(values.reshape(100,-1),rng,iterations=BOOTSTRAP_ITERATIONS)
    boots=means.reshape(BOOTSTRAP_ITERATIONS,4,len(MEASURES),len(TERMS)+1)
    summary_rows=[];policy_json={};pair_rows=[];pair_json={};reconciliation=[]
    channels=TERMS+('total_reward_from_env',)
    for j,policy in enumerate(POLICIES):
        policy_json[policy]={}
        for s,measure in enumerate(MEASURES):
            policy_json[policy][measure]={}
            denom=float(values[:,j,s,-1].mean())
            for k,term in enumerate(channels):
                info=stats(values[:,j,s,k],boots[:,j,s,k])
                row=dict(policy=policy,component=term,measure=measure,**info,
                    contribution_fraction_of_total_mean=info['mean']/denom if denom else None)
                summary_rows.append(row);policy_json[policy][measure][term]=row
    old_stats=read(VAL/'final_unseen_statistical_analysis.json')
    for ref,new in PAIRS:
        name=f'{POLICIES[ref]} → {POLICIES[new]}';pair_json[name]={}
        for s,measure in enumerate(MEASURES):
            pair_json[name][measure]={}
            for k,term in enumerate(channels):
                delta=values[:,new,s,k]-values[:,ref,s,k]
                distribution=boots[:,new,s,k]-boots[:,ref,s,k]
                assert np.allclose(distribution,weights@delta/100,atol=1e-12)
                b=boot_summary(float(delta.mean()),distribution)
                row=dict(comparison=name,component=term,measure=measure,n_pairs=100,
                    reference_mean=float(values[:,ref,s,k].mean()),new_mean=float(values[:,new,s,k].mean()),
                    observed_delta=float(delta.mean()),median_paired_delta=float(np.median(delta)),
                    bootstrap_median_delta=b['bootstrap_median'],ci95_low=b['ci95_low'],ci95_high=b['ci95_high'])
                pair_rows.append(row);pair_json[name][measure][term]=row
        total_delta=pair_json[name]['episode']['total_reward_from_env']['observed_delta']
        component_delta=sum(pair_json[name]['episode'][t]['observed_delta'] for t in TERMS)
        residual=component_delta-total_delta
        assert np.isclose(total_delta,old_stats['pairwise'][name]['episode_return']['observed_delta'],atol=1e-12)
        mean_bound=float(np.mean([x['float32_roundoff_bound'] for x in identity['episodes'] if x['policy'] in [POLICIES[ref],POLICIES[new]]]))*2
        assert abs(residual)<=mean_bound
        assert np.isclose(component_delta,total_delta,rtol=3e-6,atol=1e-6)
        reconciliation.append(dict(comparison=name,total_return_delta=total_delta,sum_component_deltas=component_delta,
            difference=residual,tolerance=dict(rtol=3e-6,atol=1e-6),float32_roundoff_bound=mean_bound,pass_=True))
        # Exact symmetric product decomposition S=L*q. Descriptive association, not causal mediation.
        q_ref,q_new=values[:,ref,3,:-1],values[:,new,3,:-1]
        Lref,Lnew=lengths[:,ref,None],lengths[:,new,None]
        duration=(Lnew-Lref)*(q_new+q_ref)/2
        rate=(q_new-q_ref)*(Lnew+Lref)/2
        assert np.allclose(duration+rate,values[:,new,0,:-1]-values[:,ref,0,:-1],atol=1e-12)
        for measure,data in [('duration_associated',duration),('rate_associated',rate)]:
            pair_json[name][measure]={}
            sampled=weights@data/100
            for k,term in enumerate(TERMS):
                b=boot_summary(float(data[:,k].mean()),sampled[:,k])
                row=dict(comparison=name,component=term,measure=measure,n_pairs=100,
                    reference_mean=None,new_mean=None,observed_delta=float(data[:,k].mean()),
                    median_paired_delta=float(np.median(data[:,k])),bootstrap_median_delta=b['bootstrap_median'],
                    ci95_low=b['ci95_low'],ci95_high=b['ci95_high'])
                pair_rows.append(row);pair_json[name][measure][term]=row
    conditioned=[]
    for j,policy in enumerate(POLICIES):
        for subset,ids in subsets[policy].items():
            count=len(ids);assert count>0
            subset_data=values[ids,j,0,:];b,_=bootstrap_means(subset_data,rng,iterations=BOOTSTRAP_ITERATIONS)
            for k,term in enumerate(channels):conditioned.append(dict(policy=policy,subset=subset,component=term,**stats(subset_data[:,k],b[:,k])))
    # Original total delta CI exactly matches prior bootstrap settings/shared env block weights.
    for name,d in pair_json.items():
        old=old_stats['pairwise'][name]['episode_return'];new=d['episode']['total_reward_from_env']
        assert np.allclose([old['ci95_low'],old['ci95_high']],[new['ci95_low'],new['ci95_high']],atol=1e-12)
    before=read(RUNS/'protected_before.json');changed=[n for n,h in before.items() if digest(ROOT/n)!=h];assert not changed
    result=dict(timestamp=datetime.now(timezone.utc).isoformat(),task=manifest['task'],seed=24,terrain_seed=2404,
        num_envs=100,first_episode_only=True,control_dt=dt,measurement_only=True,reward_terms=[dict(**d,description=DESCRIPTIONS[d['name']]) for d in definitions],
        reward_definition=dict(inheritance='AntFinalUnseenEnvCfg -> AntDevOODBaseEnvCfg -> AntOODTerrainEnvCfg -> AntEnvCfg.RewardsCfg',
            total='Authoritative unchanged env.step returned reward; no replacement/redefinition',
            manager='compute evaluates raw term once, multiplies weight * control_dt, sequentially adds float32 values',
            buffer='_step_reward stores weighted per-second value: value/control_dt, not raw/unweighted. Logging restores control_dt without recomputing terms.',
            extras='reset logging averages over resetting envs and divides episode component sums by max episode length seconds; not sufficient per-env episode data',
            terminal='All seven terms evaluated after termination computation and before auto-reset; alive zero on terminated step, pure timeout alive remains one',
            no='No explicit terminal penalty, standalone linear-velocity reward or new reward component'),
        bootstrap=dict(iterations=BOOTSTRAP_ITERATIONS,seed=BOOTSTRAP_SEED,method='paired env_id block percentile 95%, shared multinomial resamples across policies/components/measures',
            std='population ddof=0',multiple_comparisons='Marginal exploratory component CIs; no p-values, no multiplicity correction'),
        definitions=dict(pre40='Steps 1..first_reach_step_40m inclusive; entire episode if unreached',post40='Strictly after first reach through terminal step; zero for noncompleters',
            per_step='Per-episode component_sum / own episode_steps, then equal-env average; not a pooled exposure-weighted rate',
            per_second='component_sum / own episode_steps / control_dt',fractions='Ratio of mean component to mean env total; supplementary signed ratio, not a share of positive utility',
            duration_rate_identity='delta(S)=delta(L)*(q_new+q_ref)/2 + delta(q)*(L_new+L_ref)/2. This is symmetric algebra, not a causal decomposition.',
            conditioned='Exploratory 40m-completer/noncompleter/fall/original-timeout groups; different selected populations'),
        identity=identity,original_consistency=manifest,policy_summaries=policy_json,pairwise=pair_json,
        total_delta_reconciliation=reconciliation,conditioned_summaries=conditioned,
        integrity=dict(protected_count=len(before),changed=changed,original_raw_sha256=digest(VAL/'final_unseen_raw.csv'),
            first_pass_raw_sha256=digest(VAL/'final_unseen_first_pass_raw.csv'),source_sha256=digest(Path(__file__))),
        limitation='One frozen Final realization and one training seed; exploratory components and conditioned timing do not establish causal effects or general unseen-terrain effects.')
    write_csv(VAL/outputs[0],episode_rows);write_csv(VAL/outputs[1],summary_rows);write_csv(VAL/outputs[2],pair_rows)
    write_csv(VAL/outputs[3],conditioned);save(VAL/outputs[4],result);report(result);plot(result)
    sources=['scripts/assignment/evaluate_final_unseen_reward_components.py','scripts/assignment/run_final_unseen_reward_measurement.py','scripts/assignment/analyze_final_unseen_reward_components.py']
    save(VAL/outputs[-1],dict(timestamp=result['timestamp'],purpose='Measurement-only stock reward buffer instrumentation and post-hoc decomposition',
        unchanged=['reward definition and weights','env.step total reward','training','checkpoints','Final terrain/config/seed','termination','observation','sensor',
            'original Final results','first-pass results','statistical results','preregistration','official parity'],
        existing_final_results_reproduced=True,all_existing_step_trajectories_reproduced=True,identity_checks_pass=True,
        protected_count=len(before),protected_changed=[],source_hashes={n:digest(ROOT/n) for n in sources},
        declaration='No tuning, retraining, policy selection, or policy/environment/reward modification was performed based on Final results.'))
    print('REWARD DECOMPOSITION PASS: 400 episodes, identities and original total delta CI verified',flush=True)


def ci(r,scale=1):return f"[{r['ci95_low']*scale:+.4f}, {r['ci95_high']*scale:+.4f}]"


def report(a):
    lines=['# Frozen Final stock reward decomposition','', '## A. Actual stock reward definition','',
        'Final inherits AntEnvCfg.RewardsCfg without replacing any reward term. Ant imports reward helpers from classic/humanoid/mdp and common isaaclab/envs/mdp. The seven runtime terms, exact function identifiers, weights and parameters were read from RewardManager and checked against the stock config.','',
        '| Term | Function | Weight | Definition |','|---|---|---:|---|']
    for d in a['reward_terms']:lines.append(f"| {d['name']} | `{d['function']}` | {d['weight']:g} | {d['description']} |")
    lines += ['', f"Every control step uses raw_value × weight × control_dt ({a['control_dt']:.15g} s). Terminations are computed first, then rewards before reset, so terminal-step contributions are included. Alive alone gates on terminated; timeout alone retains alive. There is no explicit terminal penalty or standalone velocity term. Progress is a target-potential difference and is not exactly the legacy forward-displacement metric.",
        '', '## B. Data sufficiency and instrumentation','',
        'Prior Final/first-pass logs had total reward only. Four frozen 100-env first-episode measurement runs logged existing RewardManager._step_reward buffers. This buffer is already weighted but has dt divided out; dt was restored for analysis. No term was recomputed, RewardManager.compute was not replaced, and no evaluator/environment source/config was edited. Existing reset extras average over reset envs and normalize by max duration, so they cannot recover per-env component trajectories.',
        '', 'Every old episode return/length/legacy displacement/fall/timeout was reproduced. All saved step X positions, total rewards and done/terminated/truncated/fall signals also matched the first-pass trajectory, preserving the first-reach boundary split. The native env.step total remains the authoritative Final reward.',
        '', '## C. Reward identity validation','']
    i=a['identity']
    lines += [f"Across {i['step_count']} first-episode control transitions, max |sum components − env reward| = {i['step_max_absolute_error']:.12g}; mean absolute step error = {i['step_mean_absolute_error']:.12g}.",
        f"Max accumulated component-vs-float64 env trace error = {i['episode_max_absolute_float64_trace_error']:.12g}. Max component-vs-native float32 episode return error = {i['episode_max_absolute_native_error']:.12g}; mean absolute error = {i['episode_mean_absolute_native_error']:.12g}.",
        f"Step/float64-trace checks use rtol=1e-6, atol=1e-6. Separate component accumulation changes summation order: {i['nominal_episode_1e6_tolerance_failures']}/400 native comparisons exceed nominal rtol=1e-6, while all pass rtol=2e-6, atol=1e-6 and the explicit float32 sequential-sum gamma_N error bound. This small reduction difference does not change any env reward or original episode return. The read-only buffer’s dt divide/multiply round trip is included in the measured step error.",
        '', 'Pairwise reconciliation:', '', '| Comparison | Native total Δ | Sum component Δ | Residual |','|---|---:|---:|---:|']
    for r in a['total_delta_reconciliation']:lines.append(f"| {r['comparison']} | {r['total_return_delta']:+.9f} | {r['sum_component_deltas']:+.9f} | {r['difference']:+.9g} |")
    lines += ['', '## D. Policy-level accumulated weighted contributions','',
        '| Reward component | Base | Original | Contact | Diverse |','|---|---:|---:|---:|---:|']
    for term in TERMS+('total_reward_from_env',):
        lines.append('| '+term+' | '+' | '.join(f"{a['policy_summaries'][p]['episode'][term]['mean']:.6f}" for p in POLICIES)+' |')
    lines += ['', 'Mean/std/median/paired-block bootstrap mean CIs and supplementary signed contribution fractions are in CSV/JSON. Fractions are not a share of positive performance because negative terms offset positive terms.',
        '', 'Per-step contribution, mean of per-episode rates:', '',
        '| Component | Base | Original | Contact | Diverse |','|---|---:|---:|---:|---:|']
    for term in TERMS+('total_reward_from_env',):
        lines.append('| '+term+' | '+' | '.join(f"{a['policy_summaries'][p]['per_step'][term]['mean']:.7f}" for p in POLICIES)+' |')
    lines += ['', 'Per-second rates equal these per-step rates divided by control_dt; their means/std/median/CIs are also saved. Rates use equal episode weighting; pooled exposure rates would answer a different question.',
        '', '## E. Prespecified paired component deltas','',
        '50,000 bootstrap resamples, RNG seed 2404, percentile 95% CI; same env_id blocks across policies/components. All main pairs have n=100. Component CIs are exploratory/marginal; no p-values or multiplicity correction. Δ=new−reference.',
        '', '| Comparison | Component | Δ contribution | 95% paired CI |','|---|---|---:|---:|']
    for comparison,d in a['pairwise'].items():
        for term in TERMS+('total_reward_from_env',):
            r=d['episode'][term];lines.append(f"| {comparison} | {term} | {r['observed_delta']:+.6f} | {ci(r)} |")
    lines += ['', '## F. Pre/post first 40 m endpoint','',
        'Pre40 includes the first crossing control transition. Post40 starts on its following transition. Noncompleters contribute their whole episode to pre40 and zero to post40. All 100 envs remain in each mean/pair; pre40 group durations differ by policy, so segment differences are descriptive rather than matched-exposure effects.',
        '', '| Comparison | Component | Pre40 Δ | Pre40 CI | Post40 Δ | Post40 CI |','|---|---|---:|---:|---:|---:|']
    for comparison,d in a['pairwise'].items():
        for term in TERMS+('total_reward_from_env',):
            pre,post=d['pre40'][term],d['post40'][term]
            lines.append(f"| {comparison} | {term} | {pre['observed_delta']:+.6f} | {ci(pre)} | {post['observed_delta']:+.6f} | {ci(post)} |")
    lines += ['', '## G. Duration versus per-step rates','',
        'For each paired episode and component S=L×q, the exact symmetric algebra is ΔS=ΔL×(q_new+q_ref)/2 + Δq×(L_new+L_ref)/2. The table calls these duration-associated and rate-associated contributions. They are algebraic associations, not causal mediation or a matched-duration counterfactual.',
        '', '| Comparison | Component | Duration-associated Δ | Rate-associated Δ | Per-step Δ |','|---|---|---:|---:|---:|']
    for comparison,d in a['pairwise'].items():
        for term in TERMS:
            lines.append(f"| {comparison} | {term} | {d['duration_associated'][term]['observed_delta']:+.6f} | {d['rate_associated'][term]['observed_delta']:+.6f} | {d['per_step'][term]['observed_delta']:+.7f} |")
    lines += ['', '## H. Main interpretation','']
    for index,(comparison,d) in enumerate(a['pairwise'].items()):
        lines += [f'### {comparison}','']
        for term in TERMS:
            r=d['episode'][term]
            lines.append(f"{term}: accumulated Δ {r['observed_delta']:+.6f}, CI {ci(r)}; pre40 Δ {d['pre40'][term]['observed_delta']:+.6f}; post40 Δ {d['post40'][term]['observed_delta']:+.6f}; per-step Δ {d['per_step'][term]['observed_delta']:+.7f}.")
        if index==0:
            lines += [f"Native total Δ {d['episode']['total_reward_from_env']['observed_delta']:+.6f}: pre40 {d['pre40']['total_reward_from_env']['observed_delta']:+.6f}, post40 {d['post40']['total_reward_from_env']['observed_delta']:+.6f}. Most of the episode-total increase occurs after the initial course endpoint; earlier passage leaves more episode exposure for post-course movement.", 'Most of the observed return difference is associated with higher target progress. Lower energy penalty adds to it, while larger joint-limit/action penalties offset part of it. Alive accumulation is almost unchanged, consistent with nearly equal mean episode length. The progress rate increase is consistent with the previous shorter joint-completer t40 and greater whole-episode displacement, but those are distinct metrics and include different survivor/exposure effects.']
        elif index==1:
            lines += [f"Native total Δ {d['episode']['total_reward_from_env']['observed_delta']:+.6f}: pre40 {d['pre40']['total_reward_from_env']['observed_delta']:+.6f}, post40 {d['post40']['total_reward_from_env']['observed_delta']:+.6f}. Joint-limit/energy/action penalty reductions occur in both segments, so the reward gains are not confined to the post-course region where the observed fall reduction was localized.", 'The return increase trend is chiefly associated with smaller joint-limit and energy penalties, with smaller action penalties too. Progress contribution falls slightly; the small alive gain alone does not explain the total increase. Original total-return CI includes zero. Earlier first-pass analysis found equal 94% endpoint reach and fewer post-course falls; pre/post and duration/rate tables describe how control-related contribution differences coexist with that pattern, without asserting a causal stability effect.']
        else:
            lines += [f"Native total Δ {d['episode']['total_reward_from_env']['observed_delta']:+.6f}: pre40 {d['pre40']['total_reward_from_env']['observed_delta']:+.6f}, post40 {d['post40']['total_reward_from_env']['observed_delta']:+.6f}. The symmetric duration-associated sum is {sum(r['observed_delta'] for r in d['duration_associated'].values()):+.6f}, while the rate-associated sum is {sum(r['observed_delta'] for r in d['rate_associated'].values()):+.6f}. The shorter exposure contribution dominates this algebraic decomposition; it is not a causal attribution.", 'Lower accumulated progress and smaller alive/upright/move_to_target accumulations accompany the shorter episodes. Energy penalty is more negative even though episodes are shorter; joint-limit penalty is less negative and partially offsets other losses. Per-step progress is higher among equal-weighted episode rates, so lower accumulated progress does not imply slower progress per control step. This is consistent with shorter timing among joint completers but lower completion and more falls. Selection into completed/fall/timeout groups is descriptive only.']
        lines += ['']
    lines += ['Within this Final realization, the decomposition indicates associations between observed total-return differences and fixed reward components. The result does not establish causal effects of sensors or general effects across unseen terrains/training seeds. Component intervals are exploratory and no component ranking is treated as a discovery.',
        '', '## I. Conditioning and artifacts','',
        'Supplementary component summaries for 40 m completers/noncompleters, falls and original timeout labels are in `final_unseen_reward_components_conditioned.csv` and JSON. These condition on outcomes and select different populations. Negative penalty fractions and positive fractions above 100% are supplementary bookkeeping, not performance scores.',
        '', 'Step telemetry is stored per policy as gzip CSV in `final_unseen_reward_runs/<policy>/trajectory.csv.gz`; it contains weighted reward contributions, env total, step/time, positions and done/terminated/truncated. Episode, policy summary and pairwise CSVs accompany this report. Native total Δ CIs exactly reproduce the previous statistical bootstrap.',
        '', '## J. Integrity and validation','',
        f"All {a['integrity']['protected_count']} protected files/checkpoints retain their hashes. Original Final/first-pass/statistical/parity/preregistration artifacts are preserved. Frozen checkpoints, terrain seed 2404, evaluation seed 24, 100 envs, first-episode semantics and 60/123/127/127 dimensions were retained. No tuning, training, selection or reward/environment modification.",
        '', 'Independent identity/bootstrap verification, py_compile, whitespace checks and exact commands are recorded under `final_unseen_reward_runs/`. The separate `final_unseen_reward_decomposition_addendum.json` documents measurement-only instrumentation.',
        '', '![Accumulated contributions](final_unseen_reward_components_policy.png)','',
        '![Paired component deltas](final_unseen_reward_components_pairwise.png)','']
    with (VAL/'final_unseen_reward_decomposition.md').open('x') as stream:stream.write('\n'.join(lines))


def plot(a):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/final-reward-decomposition-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=plt.cm.tab10.colors
    fig,ax=plt.subplots(figsize=(10,6));positive=np.zeros(4);negative=np.zeros(4)
    for k,term in enumerate(TERMS):
        heights=np.array([a['policy_summaries'][p]['episode'][term]['mean'] for p in POLICIES])
        bottom=np.where(heights>=0,positive,negative)
        ax.bar(np.arange(4),heights,bottom=bottom,label=term,color=colors[k])
        positive+=np.maximum(heights,0);negative+=np.minimum(heights,0)
    totals=[a['policy_summaries'][p]['episode']['total_reward_from_env']['mean'] for p in POLICIES]
    ax.plot(range(4),totals,'kD',label='Native env total',markersize=6)
    ax.axhline(0,color='gray',linewidth=.8);ax.set_xticks(range(4),['Base','Original','Contact','Diverse'])
    ax.set(ylabel='Mean accumulated weighted contribution',title='Unchanged stock Ant reward — fixed Final realization')
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(VAL/'final_unseen_reward_components_policy.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,6),sharey=True)
    for ax,(comparison,d) in zip(axes,a['pairwise'].items()):
        for k,term in enumerate(TERMS+('total_reward_from_env',)):
            r=d['episode'][term];point=r['observed_delta']
            ax.errorbar(point,k,xerr=[[point-r['ci95_low']],[r['ci95_high']-point]],fmt='o',capsize=3,color=colors[k%len(colors)])
        ax.axvline(0,color='gray',linestyle='--',linewidth=1);ax.set_xlabel('New − reference contribution')
        ax.set_title(comparison.replace('Original HeightScan','Original').replace('Diverse HeightScan+Contact','Diverse').replace('HeightScan+Contact','Contact'),fontsize=10)
        ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(8),list(TERMS)+['Native total']);axes[0].invert_yaxis()
    fig.suptitle('Paired reward components: exploratory marginal 95% bootstrap CI',fontsize=13)
    fig.text(.5,.015,'100 paired env indices; 50,000 resamples; RNG seed 2404. Fixed terrain/training realization; no causal or multiplicity-adjusted claims.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(VAL/'final_unseen_reward_components_pairwise.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
