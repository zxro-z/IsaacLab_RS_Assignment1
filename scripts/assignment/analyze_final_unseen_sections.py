"""Post-hoc first-pass analysis, gated by 400-row frozen-result consistency.

No simulator or policy loading. Boundaries are derived from the logged runtime
COURSE_DESIGN and checked against the frozen source fingerprint. Outputs refuse
overwrite. Bootstrap CIs describe episode variability within this fixed scene.
"""
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
VAL=ROOT/'validation'
BOOTSTRAP_REPLICATES=10000
BOOTSTRAP_SEED=240403


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):
    with p.open('x') as s: json.dump(data,s,indent=2,allow_nan=False)
def write_csv(p,rows):
    with p.open('x',newline='') as s:
        writer=csv.DictWriter(s,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def wilson(k,n):
    if n==0: return [None,None]
    z=1.959963984540054; rate=k/n; den=1+z*z/n
    mid=(rate+z*z/(2*n))/den
    half=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
    return [max(0.,mid-half),min(1.,mid+half)]


def stats(values,rng):
    a=np.asarray(values,dtype=float); n=len(a)
    if n==0:
        return dict(n=0,mean=None,median=None,std=None,mean_bootstrap_ci95=[None,None],median_bootstrap_ci95=[None,None])
    samples=a[rng.integers(0,n,size=(BOOTSTRAP_REPLICATES,n))]
    return dict(n=n,mean=float(a.mean()),median=float(np.median(a)),std=float(a.std(ddof=0)),
        mean_bootstrap_ci95=np.quantile(samples.mean(axis=1),[.025,.975]).tolist(),
        median_bootstrap_ci95=np.quantile(np.median(samples,axis=1),[.025,.975]).tolist())


def rate(k,n): return dict(count=k,denominator=n,rate=k/n if n else None,wilson_ci95=wilson(k,n))
def fmt_rate(k,n):
    ci=wilson(k,n)
    return f'{k}/{n} ({100*k/n:.0f}%; {100*ci[0]:.1f}–{100*ci[1]:.1f})' if n else 'NA (0 entrants)'
def fmt_stat(s):
    if not s['n']:return 'NA (n=0)'
    lo,hi=s['mean_bootstrap_ci95']
    return f"{s['mean']:.3f} ± {s['std']:.3f}; median {s['median']:.3f}; mean CI [{lo:.3f}, {hi:.3f}]; n={s['n']}"


def main():
    output_names=['final_unseen_first_pass_raw.csv','final_unseen_section_analysis.csv',
        'final_unseen_section_analysis.json','final_unseen_section_analysis.md','final_unseen_section_analysis.png',
        'final_unseen_first_pass_analysis_addendum.json']
    assert not any((VAL/n).exists() for n in output_names),'Refuse overwrite'
    manifest=json.loads((VAL/'final_unseen_first_pass_trajectory_manifest.json').read_text())
    assert manifest['consistency_pass'] and len(manifest['comparisons'])==400
    assert all(c['pass'] for c in manifest['comparisons'])
    original=list(csv.DictReader((VAL/'final_unseen_raw.csv').open()))
    assert len(original)==400 and digest(VAL/'final_unseen_raw.csv')==manifest['original_raw_sha256']
    original_index={(r['policy'],int(r['env_id'])):r for r in original}
    before=json.loads((VAL/'final_unseen_first_pass_runs/protected_before.json').read_text())
    assert all(digest(ROOT/n)==h for n,h in before.items())
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    raw=[]; summaries=[]; policy_summaries={}; design=None; dt=None
    for entry in manifest['policies']:
        directory=ROOT/entry['directory']
        assert digest(directory/'trajectory.csv.gz')==entry['trajectory_sha256']
        meta=json.loads((directory/'metadata.json').read_text())
        if design is None:
            design=meta['course_design'];dt=meta['control_dt']
            sections=[('Flat',design['spawn_flat_x'][0],design['spawn_flat_x'][1]),
                ('Ridge',*design['ridge']['x']),('Terraces',design['terraces'][0]['x'][0],design['terraces'][-1]['x'][1]),
                ('Basin',*design['basin']['x']),('Landing',*design['landing']['x'])]
            assert all(sections[i][2]==sections[i+1][1] for i in range(len(sections)-1))
            assert sections[-1][2]==design['size_m'][0]
            boundaries=[b for name,a,b in sections]
        assert meta['course_design']==design and meta['control_dt']==dt
        assert math.isclose(dt,meta['physics_dt']*meta['decimation'],rel_tol=0,abs_tol=1e-15)
        trajectories={i:[] for i in range(100)}
        with gzip.open(directory/'trajectory.csv.gz','rt') as stream:
            for r in csv.DictReader(stream): trajectories[int(r['env_id'])].append(r)
        episodes=[]
        for episode in meta['episodes']:
            i=episode['env_id']; rows=trajectories[i]; steps=episode['episode_steps']; final=rows[-1]
            assert [int(r['step']) for r in rows]==list(range(steps+1))
            assert final['done']=='True' and all(r['done']=='False' for r in rows[:-1])
            assert (final['fall']=='True')==episode['fall']
            # The frozen TerminationManager get_term table records only the last true term.
            # With time_out then torso_height, simultaneous signals are legacy-classified as fall.
            assert episode['timeout']==((final['truncated']=='True') and not episode['fall'])
            initial=meta['initial_root_xy'][i]; origin=meta['env_origins_xy'][i]
            assert math.isclose(float(rows[0]['terrain_local_x']),initial[0]-origin[0]+design['spawn_local_xyz'][0],abs_tol=1e-9)
            assert math.isclose(float(rows[0]['terrain_local_x']),design['spawn_local_xyz'][0],abs_tol=1e-6)
            x=np.array([float(r['terrain_local_x']) for r in rows]); y=np.array([float(r['terrain_local_y']) for r in rows])
            # The existing evaluator's legacy displacement is from step N-1, not reset or terminal pose.
            legacy=float(rows[-2]['forward_displacement'])
            assert math.isclose(legacy,episode['legacy_final_displacement'],rel_tol=1e-6,abs_tol=1e-6)
            old=original_index[(entry['policy'],i)]
            item=dict(policy=entry['policy'],env_id=i,episode_return=episode['episode_return'],episode_steps=steps,
                final_displacement=episode['legacy_final_displacement'],
                terminal_displacement=float(final['forward_displacement']),terminal_local_x=float(x[-1]),
                termination='fall+timeout' if episode['fall'] and final['truncated']=='True' else 'fall' if episode['fall'] else 'timeout',
                terminated=final['terminated']=='True',truncated=final['truncated']=='True',
                fall=episode['fall'],timeout=episode['timeout'],initial_local_x=float(x[0]),initial_local_y=float(y[0]))
            hits=[]
            for boundary in boundaries:
                indices=np.flatnonzero(x>=boundary); first=int(indices[0]) if len(indices) else None
                tag=f'{boundary:g}m'
                item['reach_'+tag]=first is not None
                item['first_reach_step_'+tag]=first
                item['t_'+tag]=first*dt if first is not None else None
                hits.append(first)
            assert all(hits[j] is None or (hits[j-1] is not None and hits[j]>=hits[j-1]) for j in range(1,len(hits)))
            previous=0
            for (name,a,b),first in zip(sections,hits):
                item['traversal_time_'+name.lower()]=None if first is None or previous is None else (first-previous)*dt
                previous=first
            stop=hits[-1] if hits[-1] is not None else steps
            item['lateral_exit_before_first_pass']=bool(np.any((y[:stop+1]<0)|(y[:stop+1]>design['size_m'][1])))
            item['max_local_x']=float(x.max())
            item['post_course_net_distance_m']=float(x[-1]-boundaries[-1]) if hits[-1] is not None else None
            item['post_course_distance_since_crossing_m']=float(x[-1]-x[hits[-1]]) if hits[-1] is not None else None
            item['max_post_course_extent_m']=float(x.max()-boundaries[-1]) if hits[-1] is not None else None
            item['fall_first_uncompleted_section']=next((name for name,a,b in sections if not item[f'reach_{b:g}m']), 'After course') if item['fall'] else None
            raw.append(item);episodes.append(item)
        policy=entry['policy']; n=len(episodes); assert n==100
        details=[]
        for index,(name,a,b) in enumerate(sections):
            tag=f'{b:g}m'; reached=[r for r in episodes if r['reach_'+tag]]
            entrants=episodes if index==0 else [r for r in episodes if r[f'reach_{boundaries[index-1]:g}m']]
            times=stats([r['t_'+tag] for r in reached],rng)
            traverse=stats([r['traversal_time_'+name.lower()] for r in reached],rng)
            fall_before=sum(r['fall'] and not r['reach_'+tag] for r in episodes)
            timeout_before=sum(r['truncated'] and not r['reach_'+tag] for r in episodes)
            detail=dict(section=name,start_local_x=a,end_local_x=b,completion=rate(len(reached),n),
                conditional_completion=rate(len(reached),len(entrants)),first_reach_time_s=times,traversal_time_s=traverse,
                fall_before_boundary=rate(fall_before,n),timeout_before_boundary=rate(timeout_before,n))
            details.append(detail)
            row=dict(policy=policy,section=name,start_local_x=a,end_local_x=b,
                reached=len(reached),episodes=n,reach_rate=len(reached)/n,
                reach_wilson_ci95_low=wilson(len(reached),n)[0],reach_wilson_ci95_high=wilson(len(reached),n)[1],
                entrants=len(entrants),conditional_completion=len(reached)/len(entrants) if entrants else None,
                conditional_wilson_ci95_low=wilson(len(reached),len(entrants))[0],conditional_wilson_ci95_high=wilson(len(reached),len(entrants))[1],
                fall_before_boundary=fall_before,fall_before_rate=fall_before/n,timeout_before_boundary=timeout_before,
                timeout_before_rate=timeout_before/n)
            for prefix,s in [('first_reach',times),('traversal',traverse)]:
                for field in ['n','mean','median','std']:row[prefix+'_'+field]=s[field]
                for field in ['mean','median']:
                    row[prefix+'_'+field+'_ci95_low']=s[field+'_bootstrap_ci95'][0]
                    row[prefix+'_'+field+'_ci95_high']=s[field+'_bootstrap_ci95'][1]
            summaries.append(row)
        reach40=[r for r in episodes if r[f'reach_{boundaries[-1]:g}m']]
        not40=[r for r in episodes if not r[f'reach_{boundaries[-1]:g}m']]
        counts={name:sum(r['fall_first_uncompleted_section']==name for r in episodes) for name in [s[0] for s in sections]+['After course']}
        policy_summaries[policy]=dict(actor_input_dim=entry['actor_input_dim'],sections=details,
            fall_localization_first_uncompleted_section=counts,fall_total=sum(r['fall'] for r in episodes),
            timeout_total=sum(r['truncated'] for r in episodes),legacy_timeout_total=sum(r['timeout'] for r in episodes),
            simultaneous_fall_timeout_count=sum(r['fall'] and r['truncated'] for r in episodes),
            timeout_before_course_end=sum(r['truncated'] for r in not40),timeout_after_course_end=sum(r['truncated'] for r in reach40),
            course_completed=len(reach40),course_end_reached_without_lateral_exit=rate(sum(not r['lateral_exit_before_first_pass'] for r in reach40),n),
            lateral_exit_before_first_pass=sum(r['lateral_exit_before_first_pass'] for r in episodes),
            post_course_net_distance_m=stats([r['post_course_net_distance_m'] for r in reach40],rng),
            post_course_distance_since_crossing_m=stats([r['post_course_distance_since_crossing_m'] for r in reach40],rng),
            unreached_course_end_final_displacement_m=stats([r['final_displacement'] for r in not40],rng),
            legacy_mean_displacement_m=float(np.mean([r['final_displacement'] for r in episodes])))
    assert len(raw)==400
    analysis=dict(timestamp=datetime.now(timezone.utc).isoformat(),task=manifest['task'],seed=24,terrain_seed=design['terrain_seed'],
        num_envs_per_policy=100,measurement_only=True,post_hoc=True,consistency_pass=True,
        consistency_max_absolute_differences={f:max(abs(c[f]) for c in manifest['comparisons']) for f in ['return_difference','length_difference','legacy_displacement_difference']},
        boundaries_terrain_local_x=boundaries,sections=[dict(name=n,start=a,end=b) for n,a,b in sections],
        spawn_local_xyz=design['spawn_local_xyz'],coordinate_definition='Fixed initial tile: x_local = root_world_x - env_origin_x + spawn_local_x. No modulo wrapping.',
        first_pass_definition='First logged post-control-step root x_local >= boundary, including terminal pose before auto-reset. Initial step 0 included.',
        time_definition='first_reach_step * control_dt; no interpolation; temporal sampling resolution is control_dt',
        control_dt=dt,physics_dt=manifest['policies'][0]['physics_dt'],decimation=manifest['policies'][0]['decimation'],
        statistics=dict(rate_ci='Wilson 95%',std='population ddof=0',bootstrap='IID episodes, percentile 95%, conditional on reaching boundary/completing section',
            bootstrap_replicates=BOOTSTRAP_REPLICATES,bootstrap_seed=BOOTSTRAP_SEED,
            limitation='CIs describe episode variation within one fixed Final realization and training seed; no multiplicity correction or causal/generalization test.'),
        metric_definitions=dict(section_completion='Unconditional reach endpoint / 100',conditional_completion='Reach endpoint / reach previous endpoint; Flat denominator 100',
            fall_before='Fall occurred in first episode AND boundary never reached, divided by 100',
            failure_localization='First section endpoint never reached by a fallen episode; after-course category if all reached',
            timeout_split='Actual truncated signal before/after first reach of local x=course end, irrespective of terminal x; overlaps fall on simultaneous termination',
            legacy_timeout='Frozen evaluator get_term(time_out): false when a later torso_height term also fires; preserved for original consistency',
            post_course='Signed terminal x_local minus course-end x among first-pass completers; also log movement since sampled first crossing',
            legacy_final_displacement='Preserved evaluator displacement at pre-terminal step N-1; true terminal displacement kept separately',
            lateral='Report exits from y_local [0,width] before first course-end crossing or episode end; endpoint reach alone is an x-progress measure'),
        policies=policy_summaries,integrity=dict(protected_count=len(before),changed=[]),
        trajectory_manifest_sha256=digest(VAL/'final_unseen_first_pass_trajectory_manifest.json'))
    write_csv(VAL/'final_unseen_first_pass_raw.csv',raw)
    write_csv(VAL/'final_unseen_section_analysis.csv',summaries)
    save(VAL/'final_unseen_section_analysis.json',analysis)
    report(analysis)
    plot(analysis)
    assert all(digest(ROOT/n)==h for n,h in before.items())
    save(VAL/'final_unseen_first_pass_analysis_addendum.json',dict(timestamp=analysis['timestamp'],
        purpose='Post-hoc measurement of frozen Final first-pass progress; no tuning or new Final result replacement.',
        unchanged=['training','checkpoints and SHA256','terrain geometry/config/seed','reward','termination','observation','sensor',
            'original Final raw/results/summary','preregistration','official parity and protocol addendum'],
        protected_count=len(before),protected_changed=[],consistency_pass=True,
        source_hashes={n:digest(ROOT/n) for n in ['scripts/assignment/evaluate_final_unseen_first_pass.py',
            'scripts/assignment/run_final_unseen_first_pass_measurement.py','scripts/assignment/analyze_final_unseen_sections.py']},
        original_final_hashes={n:digest(VAL/n) for n in ['final_unseen_raw.csv','final_unseen_results.csv','final_unseen_summary.md',
            'final_unseen_preregister.json','official_evaluator_parity.json','final_unseen_evaluation_protocol_addendum.json']},
        declaration='No tuning, retraining, policy selection, or terrain modification was performed based on Final results.'))
    print('SECTION ANALYSIS PASS: 400 episodes, original results preserved',flush=True)


def report(a):
    p=a['policies']; boundaries=a['boundaries_terrain_local_x']
    lines=['# Frozen Final: first-pass / terrain-section analysis','',
        '## A. Analysis definition','',
        'Post-hoc measurement only. Existing Final artifacts contained final metrics but no position trajectories; four frozen 100-env first-episode runs were instrumented. Seed 24, terrain seed 2404, same checkpoints and policy observation adapters. All 400 episodes matched original return, length, legacy final displacement and fall/timeout within rtol=1e-6, atol=1e-6 (steps/flags exact). Original summaries remain unchanged.',
        '', 'Boundaries were read from the runtime COURSE_DESIGN and frozen source: Flat 0–4; ridge 4–10; terraces 10–18 (subranges 10–12.2, 12.2–15.6, 15.6–18); basin 18–25; landing 25–40 m.',
        '', 'The terrain generator centers the full 400×80 m layout at world (0,0); tile (row,col) starts at (40*row-200, 8*col-40). Its env origin is tile-start + (2,4,0). Root reset adds robot default (0,0,0.5) to that origin. Runtime initial XY confirms local (2,4) for every episode. We use x_local = root_world_x - env_origin_x + 2, always relative to the INITIAL tile and never modulo 40. Consequently local boundaries 4/10/18/25/40 correspond to initial forward displacement 2/8/16/23/38 m. Flat traversal starts at local x=2, so it covers the remaining spawn flat, not the unvisited x=0–2.',
        '', f"Physics dt={a['physics_dt']:.15g} s, decimation={a['decimation']}, runtime control dt={a['control_dt']:.15g} s. First-pass is the first sampled x_local >= boundary, inclusive of pre-reset terminal pose. Time=step*control_dt; no interpolation. Sample timing resolution is one control step. Backtracking does not undo reach.",
        '', 'The legacy evaluator records displacement from step N-1 on a terminal transition. This was preserved and checked separately. First-pass uses true step-N terminal position, captured through the existing pre-reset callback without extra observation/sensor computations or recorder terms.',
        '', 'One Base episode (env 94, step 960) has both terminated and truncated true. The frozen TerminationManager.compute clears the per-term table when each later true term fires, so get_term(time_out) is false after torso_height fires. The existing evaluator therefore labels it fall only. Original flags are preserved in the raw timeout/fall columns; actual terminal flags are separate terminated/truncated columns. Timeout localization uses actual truncated, yielding Base 82 timeouts (one overlapping fall), versus the original exclusive-label count 81. This is documented classification behavior, not a rerun inconsistency.',
        '', 'Wilson 95% intervals are shown for rates. Time std uses ddof=0; percentile bootstrap 95% mean and median intervals use 10,000 resamples, RNG seed 240403. Times are conditional on reaching/completing, so survivor selection affects comparisons. These intervals describe episode variation within one fixed scene, not training-seed/terrain-seed generalization.',
        '', '## B. Boundary reach table','', '| Policy | '+ ' | '.join(f'≥{b:g} m' for b in boundaries)+' |', '|---|'+ '|'.join(['---:']*len(boundaries))+'|']
    for policy,d in p.items():lines.append('| '+policy+' | '+' | '.join(fmt_rate(s['completion']['count'],100) for s in d['sections'])+' |')
    lines+=['','Cells: n/100 (percentage; Wilson 95% interval). These are local-X milestone reach rates; lateral exits are reported below.','',
        '## C. Conditional section completion','',
        '| Policy | Ridge given Flat | Terraces given Ridge | Basin given Terraces | Landing given Basin |',
        '|---|---:|---:|---:|---:|']
    for policy,d in p.items():lines.append('| '+policy+' | '+' | '.join(fmt_rate(s['conditional_completion']['count'],s['conditional_completion']['denominator']) for s in d['sections'][1:])+' |')
    lines+=['','## D. First-reach time','', 'Seconds; mean ± population std; median; bootstrap 95% CI for mean. Full median bootstrap CIs are in CSV/JSON.','',
        '| Policy | Boundary local x | Time |','|---|---:|---|']
    for policy,d in p.items():
        for s in d['sections']:lines.append(f"| {policy} | {s['end_local_x']:g} m | {fmt_stat(s['first_reach_time_s'])} |")
    lines+=['','## E. Section traversal time','',
        'Seconds, conditional on completion. Flat starts at t=0 at local x=2. Other times subtract the two first-pass milestone times. Incomplete traversals are excluded from time statistics and retained in rate denominators.','',
        '| Policy | Section | Time |','|---|---|---|']
    for policy,d in p.items():
        for s in d['sections']:lines.append(f"| {policy} | {s['section']} | {fmt_stat(s['traversal_time_s'])} |")
    lines+=['','## F. Failure localization','',
        'Cumulative fall-before-boundary: first-episode fall and no prior reach of that boundary. Denominator is all 100 episodes. A fall after completing the first course is separated from a first-pass failure.','',
        '| Policy | Before 4 | Before 10 | Before 18 | Before 25 | Before 40 |','|---|---:|---:|---:|---:|---:|']
    for policy,d in p.items():lines.append('| '+policy+' | '+' | '.join(fmt_rate(s['fall_before_boundary']['count'],100) for s in d['sections'])+' |')
    lines+=['','Noncumulative localization by first uncompleted section endpoint (counts):','',
        '| Policy | Flat | Ridge | Terraces | Basin | Landing | After course | Timeout before/after 40 | Lateral exits before first pass |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for policy,d in p.items():
        counts=d['fall_localization_first_uncompleted_section']
        lines.append('| '+policy+' | '+' | '.join(str(counts[name]) for name in ['Flat','Ridge','Terraces','Basin','Landing','After course'])+f" | {d['timeout_before_course_end']}/{d['timeout_after_course_end']} | {d['lateral_exit_before_first_pass']} |")
    lines+=['','## G. Interpretation','']
    names=list(p)
    for first,second,title in [(names[0],names[1],'HeightScan'),(names[1],names[2],'Contact fusion'),(names[2],names[3],'Diverse training')]:
        one,two=p[first],p[second]
        lines.append(f'**{title}: {first} → {second}.** Within this preregistered Final realization:')
        lines.append('')
        differences=[]
        for s,t in zip(one['sections'],two['sections']):
            delta=t['completion']['count']-s['completion']['count']
            td=t['traversal_time_s']['mean']-s['traversal_time_s']['mean'] if t['traversal_time_s']['n'] and s['traversal_time_s']['n'] else None
            differences.append(f"{s['section']}: endpoint reach {s['completion']['count']}→{t['completion']['count']}/100; completed-section mean traversal {s['traversal_time_s']['mean']:.3f}→{t['traversal_time_s']['mean']:.3f} s ({td:+.3f} s)")
        lines.append('; '.join(differences)+'.')
        lines.append(f"Fall count before first course end: {one['sections'][-1]['fall_before_boundary']['count']}→{two['sections'][-1]['fall_before_boundary']['count']}; fall after course completion: {one['fall_localization_first_uncompleted_section']['After course']}→{two['fall_localization_first_uncompleted_section']['After course']}; total original falls: {one['fall_total']}→{two['fall_total']}.")
        lines.append('')
    lines += ['The section analysis suggests that HeightScan’s shorter traversal times appear across several sections, with the larger observed reductions in terraces and landing. Its first-course end reach is 94% versus Base 92%, while more of its falls occur after completing that first course.',
        '', 'Contact fusion’s total fall reduction (23→16) occurs after the first course (17→10): first-course falls remain 6→6, and first-course end reach remains 94%→94%. Landing conditional completion changes from 94/96 to 94/94, but earlier misses offset this. The overall fall reduction should therefore not be claimed as a reduction concentrated within the first 40 m course.',
        '', 'Diverse training has 15 first-course falls versus 6 for HeightScan+Contact. Six Diverse episodes fail after reaching the basin endpoint but before reaching the landing endpoint, versus zero for HeightScan+Contact; this is the largest local concentration of the additional first-course failures. Earlier sections also contribute differences. Faster times among Diverse survivors do not offset its lower completion count or imply a causal trade-off.', '']
    lines+=['These comparisons are descriptive; no policy ranking or aggregate score is computed. Time estimates condition on different survivor sets. The section analysis suggests where progress and failures occur, but it does not establish a general causal effect across unseen terrains. CIs are marginal within-realization intervals; no claim of significant between-policy section differences is made.',
        '', '### First course versus post-course movement','',
        '| Policy | Reach local 40 | Original mean displacement m | Mean terminal distance beyond local 40 among reachers m | Mean movement since sampled crossing m | Unreached: mean legacy final displacement m |',
        '|---|---:|---:|---:|---:|---:|']
    for policy,d in p.items():
        def f(v):return 'NA' if v is None else f'{v:.3f}'
        lines.append(f"| {policy} | {d['course_completed']}/100 | {d['legacy_mean_displacement_m']:.3f} | {f(d['post_course_net_distance_m']['mean'])} | {f(d['post_course_distance_since_crossing_m']['mean'])} | {f(d['unreached_course_end_final_displacement_m']['mean'])} |")
    lines += ['', 'A lateral-exit sensitivity check separates X progress from staying within the initial 8 m-wide tile. First-course-end reach without any sampled y_local exit from [0,8] before that crossing is: '+ '; '.join(f"{name} {d['course_end_reached_without_lateral_exit']['count']}/100" for name,d in p.items())+'. The requested main metric remains X first-pass reach; these sensitivity counts must not be silently substituted for it. Lateral excursions can change terrain exposure even when the same X section is reached.', '']
    lines+=['', 'The 40 m analytic tile starts 2 m behind the robot. First-course end therefore requires about 38 m forward displacement, not 40 m displacement. Original 55–62 m mean displacement is a whole-first-episode metric that includes movement beyond the initial tile. It must not be read as first-course completion or section performance. The first-pass tables stop at the initial local x=40 boundary; subsequent x progress is separated above. Adjacent repeated tiles and outer padding cannot be apportioned from world-X alone, and no claim of completing a bounded 2D corridor is inferred from an X threshold.',
        '', '## Integrity and provenance','',
        f"All {a['integrity']['protected_count']} protected file/checkpoint hashes remained unchanged. No training, tuning, selection, geometry, seed, reward, termination, observation or sensor changes. Existing 400-row Final CSV, summary, preregistration, parity results and prior addendum were preserved.",
        '', 'Trajectory/consistency manifest: `final_unseen_first_pass_trajectory_manifest.json`; episode data: `final_unseen_first_pass_raw.csv`; summary CSV/JSON: `final_unseen_section_analysis.*`; separate addendum: `final_unseen_first_pass_analysis_addendum.json`. Full runtime commands, original hash snapshot, per-policy terminal-safe telemetry and validation logs are under `final_unseen_first_pass_runs/`.',
        '', '![First-pass completion with Wilson intervals](final_unseen_section_analysis.png)','']
    with (VAL/'final_unseen_section_analysis.md').open('x') as stream:stream.write('\n'.join(lines))


def plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    bounds=np.array(a['boundaries_terrain_local_x'])
    for offset,(name,p) in zip(np.linspace(-.25,.25,4),a['policies'].items()):
        y=np.array([s['completion']['rate'] for s in p['sections']])*100
        ci=np.array([s['completion']['wilson_ci95'] for s in p['sections']])*100
        ax.errorbar(bounds+offset,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),marker='o',capsize=3,label=name)
    ax.set(xlabel='Boundary in initial tile: terrain-local x (m)',ylabel='First-pass reach (%)',
        title='Frozen Final realization — seed 24, 100 first episodes per policy',xticks=bounds,ylim=(0,102))
    ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=9)
    fig.text(.5,.01,'95% Wilson intervals; spawn at local x=2 m; no inference across terrain/training seeds.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,1));fig.savefig(VAL/'final_unseen_section_analysis.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
