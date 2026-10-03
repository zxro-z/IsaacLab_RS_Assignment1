"""Run only 4-env x four frozen policies; preserve every pre-existing artifact."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / 'validation'
RUNS = VAL / 'official_evaluator_parity_runs'
PYTHON = '/home/zxro/anaconda3/envs/env_isaaclab_231/bin/python'
SIM_SITE = '/home/zxro/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def main():
    manifest = json.loads((VAL / 'final_unseen_eval_manifest.json').read_text())
    prereg = json.loads((VAL / 'final_unseen_preregister.json').read_text())
    protected = {str(p.relative_to(ROOT)): digest(p) for p in VAL.rglob('*')
                 if p.is_file() and RUNS not in p.parents}
    protected.update({name:digest(ROOT/name) for name in prereg['source_hashes']})
    protected.update({str(Path(e['checkpoint_path']).relative_to(ROOT)):digest(e['checkpoint_path'])
                      for e in manifest['policy_set']})
    for name, value in prereg['source_hashes'].items():
        assert protected[name] == value, name
    save(RUNS / 'integrity_before.json', protected)
    env = os.environ.copy()
    env.update(ISAAC_SIM_SITE_PACKAGES=SIM_SITE, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               CONDA_PREFIX=str(Path(PYTHON).parents[1]))
    results = {}
    keys = ['base', 'original_heightscan', 'heightscan_contact', 'diverse_heightscan_contact']
    with (RUNS / 'commands.log').open('x') as commands:
        for key, entry in zip(keys, manifest['policy_set']):
            assert digest(entry['checkpoint_path']) == entry['checkpoint_integrity']['sha256']
            output = RUNS / (key + '.json')
            command = [PYTHON, 'scripts/assignment/run.py', 'scripts/assignment/check_official_evaluator_parity.py',
                       '--parity_output', str(output), '--expected_dim', str(entry['observation_dim']),
                       '--task', manifest['task'], '--seed', '24', '--num_envs', '4',
                       '--checkpoint', entry['checkpoint_path'], '--headless', *entry['flags']]
            commands.write('ISAAC_SIM_SITE_PACKAGES=' + shlex.quote(SIM_SITE) + ' PYTHONDONTWRITEBYTECODE=1 ' + shlex.join(command) + '\n')
            commands.flush()
            print('PARITY START ' + entry['label'], flush=True)
            with (RUNS / (key+'.log')).open('x') as log:
                child = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if child.returncode != 0:
                results[key] = dict(parity=False, status='runtime_failed', exit_code=child.returncode,
                                    log=str(RUNS/(key+'.log')))
                print('PARITY RUNTIME FAILED ' + key, flush=True)
                break
            data = json.loads(output.read_text())
            results[key] = data
            print('PARITY ' + key + ' ' + str(data['parity']), flush=True)
    changed = [name for name, value in protected.items() if digest(ROOT/name) != value]
    save(VAL / 'official_evaluator_parity.json', dict(timestamp=datetime.now(timezone.utc).isoformat(),
         task=manifest['task'], seed=24, num_envs=4,
         official_reference='https://github.com/cailab-hy/IsaacLab_RS/blob/main/scripts/reinforcement_learning/rsl_rl/play_one_episode.py',
         official_reference_snapshot=str((RUNS/'provided_play_one_episode.py').relative_to(ROOT)),
         tolerance=dict(rtol=1e-6, atol=1e-6), policies=results,
         overall_parity=len(results)==4 and all(p['parity'] for p in results.values()) and not changed,
         integrity=dict(protected_file_count=len(protected), changed=changed, original_raw_rows=400),
         full_final_reevaluation_performed=False))
    assert not changed, changed


if __name__ == '__main__':
    main()
