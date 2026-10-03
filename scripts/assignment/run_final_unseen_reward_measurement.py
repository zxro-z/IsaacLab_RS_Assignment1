"""Fixed four-policy reward instrumentation; gate all analysis on original reproduction."""
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[2];VAL=ROOT/'validation';RUNS=VAL/'final_unseen_reward_runs'
PYTHON='/home/zxro/anaconda3/envs/env_isaaclab_231/bin/python'
SIM_SITE='/home/zxro/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    with Path(p).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def main():
    assert json.loads((VAL/'official_evaluator_parity.json').read_text())['overall_parity']
    manifest=json.loads((VAL/'final_unseen_eval_manifest.json').read_text())
    prereg=json.loads((VAL/'final_unseen_preregister.json').read_text())
    protected={str(p.relative_to(ROOT)):digest(p) for p in VAL.rglob('*') if p.is_file()}
    for p in (ROOT/'scripts/assignment').glob('*.py'):
        if p.name not in ['evaluate_final_unseen_reward_components.py','run_final_unseen_reward_measurement.py']:
            protected[str(p.relative_to(ROOT))]=digest(p)
    protected.update({n:digest(ROOT/n) for n in prereg['source_hashes']})
    for n,h in prereg['source_hashes'].items():assert protected[n]==h,n
    for e in manifest['policy_set']:
        assert digest(e['checkpoint_path'])==e['checkpoint_integrity']['sha256']
        protected[str(Path(e['checkpoint_path']).relative_to(ROOT))]=digest(e['checkpoint_path'])
    RUNS.mkdir(exist_ok=False);save(RUNS/'protected_before.json',protected)
    original={(r['policy'],int(r['env_id'])):r for r in csv.DictReader((VAL/'final_unseen_raw.csv').open())}
    first={(r['policy'],int(r['env_id'])):r for r in csv.DictReader((VAL/'final_unseen_first_pass_raw.csv').open())}
    env=os.environ.copy();env.update(ISAAC_SIM_SITE_PACKAGES=SIM_SITE,PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1',CONDA_PREFIX=str(Path(PYTHON).parents[1]))
    policies=[];comparisons=[]
    with (RUNS/'commands.log').open('x') as commands:
        for key,e in zip(['base','original_heightscan','heightscan_contact','diverse_heightscan_contact'],manifest['policy_set']):
            directory=RUNS/key
            command=[PYTHON,'scripts/assignment/run.py','scripts/assignment/evaluate_final_unseen_reward_components.py',
                '--reward_dir',str(directory),'--task',manifest['task'],'--seed','24','--num_envs','100',
                '--checkpoint',e['checkpoint_path'],'--headless',*e['flags']]
            commands.write('ISAAC_SIM_SITE_PACKAGES='+shlex.quote(SIM_SITE)+' PYTHONDONTWRITEBYTECODE=1 '+shlex.join(command)+'\n');commands.flush()
            print('REWARD MEASUREMENT START '+e['label'],flush=True)
            with (RUNS/(key+'.log')).open('x') as log:
                child=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            assert child.returncode==0,f'Preserve failure log: {key}'
            data=json.loads((directory/'metadata.json').read_text())
            assert data['actor_input_dim']==data['observation_dim']==e['observation_dim']
            checks=[]
            for ep in data['episodes']:
                old=original[e['label'],ep['env_id']]
                check=dict(policy=e['label'],env_id=ep['env_id'],return_difference=ep['episode_return']-float(old['episode_return']),
                    steps_difference=ep['episode_steps']-int(old['episode_length']),
                    displacement_difference=ep['legacy_final_displacement']-float(old['forward_displacement_m']),
                    fall_equal=ep['fall']==(old['fall']=='True'),timeout_equal=ep['timeout']==(old['time_out']=='True'))
                check['pass']=math.isclose(ep['episode_return'],float(old['episode_return']),rel_tol=1e-6,abs_tol=1e-6) and \
                    check['steps_difference']==0 and math.isclose(ep['legacy_final_displacement'],float(old['forward_displacement_m']),rel_tol=1e-6,abs_tol=1e-6) and check['fall_equal'] and check['timeout_equal']
                checks.append(check)
            save(directory/'consistency.json',checks)
            assert all(x['pass'] for x in checks),f'Original-result inconsistency; stop {key}'
            # Confirm entire logged original trajectory, not just final metrics, before reusing first-pass times.
            max_position_error=max_reward_error=0.0;matched_steps=0
            old_path=VAL/'final_unseen_first_pass_runs'/key/'trajectory.csv.gz'
            with gzip.open(directory/'trajectory.csv.gz','rt') as a,gzip.open(old_path,'rt') as b:
                new_rows,old_rows=csv.DictReader(a),csv.DictReader(b)
                for new,old in zip(new_rows,old_rows,strict=True):
                    assert new['env_id']==old['env_id'] and new['step']==old['step']
                    assert all(new[f]==old[f] for f in ['done','terminated','truncated','fall'])
                    dx=abs(float(new['root_position_x'])-float(old['root_position_x']))
                    dr=abs(float(new['total_reward'])-float(old['step_reward']))
                    max_position_error=max(max_position_error,dx);max_reward_error=max(max_reward_error,dr)
                    assert math.isclose(float(new['root_position_x']),float(old['root_position_x']),rel_tol=0,abs_tol=1e-6)
                    assert math.isclose(float(new['total_reward']),float(old['step_reward']),rel_tol=1e-6,abs_tol=1e-6)
                    matched_steps+=1
            comparisons.extend(checks)
            policies.append(dict(policy=e['label'],key=key,directory=str(directory.relative_to(ROOT)),
                checkpoint_sha256=data['checkpoint_sha256'],actor_input_dim=data['actor_input_dim'],
                trajectory_sha256=digest(directory/'trajectory.csv.gz'),matched_trajectory_rows=matched_steps,
                max_existing_trajectory_x_error=max_position_error,max_existing_step_reward_error=max_reward_error))
            print('REWARD REPRODUCTION PASS '+e['label'],flush=True)
    changed=[n for n,h in protected.items() if digest(ROOT/n)!=h];assert not changed,changed
    save(VAL/'final_unseen_reward_measurement_manifest.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),
        task=manifest['task'],seed=24,terrain_seed=2404,num_envs=100,first_episode_only=True,measurement_only=True,
        consistency_pass=True,tolerance=dict(rtol=1e-6,atol=1e-6),comparisons=comparisons,policies=policies,
        protected_count=len(protected),changed=changed,reward_definition_unchanged=True))


if __name__=='__main__':main()
