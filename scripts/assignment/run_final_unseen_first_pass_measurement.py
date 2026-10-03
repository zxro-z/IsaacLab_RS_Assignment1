"""Sequential fixed 100-env measurement, gated by original-result consistency."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[2]
VAL=ROOT/'validation'
RUNS=VAL/'final_unseen_first_pass_runs'
PYTHON='/home/zxro/anaconda3/envs/env_isaaclab_231/bin/python'
SIM_SITE='/home/zxro/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages'


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    with p.open('x') as s: json.dump(value,s,indent=2,allow_nan=False)


def main():
    assert json.loads((VAL/'official_evaluator_parity.json').read_text())['overall_parity']
    manifest=json.loads((VAL/'final_unseen_eval_manifest.json').read_text())
    prereg=json.loads((VAL/'final_unseen_preregister.json').read_text())
    protected={str(p.relative_to(ROOT)):digest(p) for p in VAL.rglob('*') if p.is_file()}
    protected.update({name:digest(ROOT/name) for name in prereg['source_hashes']})
    for name,h in prereg['source_hashes'].items(): assert protected[name]==h,name
    for e in manifest['policy_set']:
        assert digest(e['checkpoint_path'])==e['checkpoint_integrity']['sha256']
        protected[str(Path(e['checkpoint_path']).relative_to(ROOT))]=digest(e['checkpoint_path'])
    RUNS.mkdir(exist_ok=False)
    save(RUNS/'protected_before.json',protected)
    raw={(r['policy'],int(r['env_id'])):r for r in csv.DictReader((VAL/'final_unseen_raw.csv').open())}
    assert len(raw)==400
    env=os.environ.copy();env.update(ISAAC_SIM_SITE_PACKAGES=SIM_SITE,PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1',CONDA_PREFIX=str(Path(PYTHON).parents[1]))
    comparisons=[]; measurements=[]
    with (RUNS/'commands.log').open('x') as commands:
        for key,e in zip(['base','original_heightscan','heightscan_contact','diverse_heightscan_contact'],manifest['policy_set']):
            directory=RUNS/key
            command=[PYTHON,'scripts/assignment/run.py','scripts/assignment/evaluate_final_unseen_first_pass.py',
                '--trajectory_dir',str(directory),'--task',manifest['task'],'--seed','24','--num_envs','100',
                '--checkpoint',e['checkpoint_path'],'--headless',*e['flags']]
            commands.write('ISAAC_SIM_SITE_PACKAGES='+shlex.quote(SIM_SITE)+' PYTHONDONTWRITEBYTECODE=1 '+shlex.join(command)+'\n');commands.flush()
            print('MEASUREMENT START '+e['label'],flush=True)
            with (RUNS/(key+'.log')).open('x') as log:
                child=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            assert child.returncode==0,f'Runtime failure; preserve {key}.log'
            data=json.loads((directory/'metadata.json').read_text())
            assert data['actor_input_dim']==data['observation_dim']==e['observation_dim']
            checks=[]
            for r in data['episodes']:
                old=raw[(e['label'],r['env_id'])]
                check=dict(policy=e['label'],env_id=r['env_id'],return_difference=r['episode_return']-float(old['episode_return']),
                    length_difference=r['episode_steps']-int(old['episode_length']),
                    legacy_displacement_difference=r['legacy_final_displacement']-float(old['forward_displacement_m']),
                    fall_equal=r['fall']==(old['fall']=='True'),timeout_equal=r['timeout']==(old['time_out']=='True'))
                check['pass']=math.isclose(r['episode_return'],float(old['episode_return']),rel_tol=1e-6,abs_tol=1e-6) and \
                    check['length_difference']==0 and math.isclose(r['legacy_final_displacement'],float(old['forward_displacement_m']),rel_tol=1e-6,abs_tol=1e-6) and check['fall_equal'] and check['timeout_equal']
                checks.append(check)
            save(directory/'consistency.json',checks)
            comparisons.extend(checks)
            assert all(c['pass'] for c in checks),f'Consistency mismatch in {key}; do not analyze'
            measurements.append(dict(policy=e['label'],key=key,directory=str(directory.relative_to(ROOT)),
                checkpoint=e['checkpoint_path'],checkpoint_sha256=data['checkpoint_sha256'],
                actor_input_dim=data['actor_input_dim'],control_dt=data['control_dt'],
                physics_dt=data['physics_dt'],decimation=data['decimation'],terrain_seed=data['terrain_seed'],
                trajectory_sha256=digest(directory/'trajectory.csv.gz')))
            print('CONSISTENCY PASS '+e['label'],flush=True)
    changed=[n for n,h in protected.items() if digest(ROOT/n)!=h]
    assert not changed,changed
    save(VAL/'final_unseen_first_pass_trajectory_manifest.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),
        measurement_only=True,task=manifest['task'],seed=24,num_envs=100,first_episode_only=True,
        official_semantics_parity_preserved=True,original_raw_sha256=digest(VAL/'final_unseen_raw.csv'),
        consistency_tolerance=dict(rtol=1e-6,atol=1e-6),consistency_pass=True,comparisons=comparisons,
        policies=measurements,protected_count=len(protected),protected_changed=changed,
        source_hashes={n:digest(ROOT/n) for n in ['scripts/assignment/evaluate_final_unseen_first_pass.py',
            'scripts/assignment/run_final_unseen_first_pass_measurement.py']},
        declaration='No tuning, retraining, policy selection, or terrain modification was performed. Existing Final results remain unchanged.'))


if __name__=='__main__': main()
