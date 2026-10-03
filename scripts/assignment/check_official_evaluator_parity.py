"""Measurement-only wrapper: run frozen local evaluator with official shadow accounting.

No file is patched. AST instrumentation applies the downloaded official four
accounting statements to the SAME env.step trajectory, retaining all current
policy-specific configuration and inference. Results refuse overwrite.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / 'scripts/reinforcement_learning/rsl_rl/play_one_episode.py'
REFERENCE = ROOT / 'validation/official_evaluator_parity_runs/provided_play_one_episode.py'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statements(code):
    return ast.parse(code).body


def emit(state):
    import numpy as np
    import torch

    env, cli = state['env'], state['args_cli'] if 'args_cli' in state else ARGS
    actual = state['returns'].cpu().numpy()
    expected = state['ref_returns'].cpu().numpy()
    steps = state['lengths'].cpu().numpy()
    ref_steps = state['ref_steps'].cpu().numpy()
    terminal = state['terminal_rewards'].cpu().numpy()
    # Official float64 reference vs existing float32 sequential accumulator.
    close = np.isclose(actual, expected, rtol=1e-6, atol=1e-6)
    rows = []
    for i in range(env.num_envs):
        rows.append(dict(env_id=i, official_episode_return=float(expected[i]),
                         current_episode_return=float(actual[i]),
                         official_episode_steps=int(ref_steps[i]), current_episode_steps=int(steps[i]),
                         absolute_return_difference=float(abs(expected[i]-actual[i])),
                         terminal_reward=float(terminal[i]), terminal_step=int(state['first_done_step'][i]),
                         post_reset_transitions_observed=int(state['post_reset_steps'][i]),
                         post_reset_accounted_steps=0,
                         return_close=bool(close[i]), steps_equal=bool(steps[i] == ref_steps[i])))
    completed = bool(state['ref_finished'].all() and state['finished'].all())
    dim = state['inputs']['actor_input_dim']
    obs_dim = int(state['obs']['policy'].shape[-1])
    result = dict(task=cli.task, seed=cli.seed, num_envs=env.num_envs,
                  checkpoint=cli.checkpoint, checkpoint_sha256=sha256(cli.checkpoint),
                  actor_input_dim=dim, runtime_observation_dim=obs_dim,
                  official_reference_sha256=sha256(REFERENCE), current_source_sha256=sha256(LOCAL),
                  method='official AST accounting on identical current policy observation/action trajectory',
                  official_accumulator_dtype='float64', current_accumulator_dtype=str(state['returns'].dtype),
                  tolerance=dict(rtol=1e-6, atol=1e-6), all_first_episodes_completed=completed,
                  terminal_reward_included=bool(torch.all(state['terminal_checked'])),
                  terminal_step_counted=bool(np.array_equal(steps, state['first_done_step'].cpu().numpy())),
                  post_reset_excluded=bool(state['post_reset_excluded']),
                  official_max_episode_length=env.max_episode_length,
                  official_cap_would_complete=bool(state['tick'] <= env.max_episode_length),
                  episodes=rows)
    result['parity'] = bool(completed and close.all() and np.array_equal(steps, ref_steps)
                            and dim == obs_dim == EXPECTED_DIM and result['terminal_reward_included']
                            and result['terminal_step_counted'] and result['post_reset_excluded']
                            and result['official_cap_would_complete'])
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('PARITY_RESULT ' + json.dumps(result), flush=True)


def main():
    global OUTPUT, EXPECTED_DIM, ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parity_output', required=True, type=Path)
    parser.add_argument('--expected_dim', required=True, type=int)
    own, rest = parser.parse_known_args()
    OUTPUT, EXPECTED_DIM = own.parity_output, own.expected_dim
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if '--num_envs' not in rest or rest[rest.index('--num_envs')+1] != '4':
        raise ValueError('Only the authorized 4-env smoke test is allowed')
    reference = ast.parse(REFERENCE.read_text())
    loop = next(n for n in ast.walk(reference) if isinstance(n, ast.While))
    block = next(n for n in loop.body if isinstance(n, ast.With)).body
    # Take the exact official statements, verifying their order before renaming shadow variables.
    accounting = block[2:]
    expected = statements('active = ~finished\nepisode_rewards[active] += rewards[active]\nepisode_steps[active] += 1\nfinished |= dones.bool()')
    if [ast.dump(n) for n in accounting] != [ast.dump(n) for n in expected]:
        raise ValueError('Official accounting changed; manual review required')

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = {'active':'ref_active', 'finished':'ref_finished',
                       'episode_rewards':'ref_returns', 'episode_steps':'ref_steps'}.get(node.id, node.id)
            return node
    accounting = [Rename().visit(n) for n in accounting]
    tree = ast.parse(LOCAL.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    current_loop = next(n for n in fn.body if isinstance(n, ast.While))
    pos = fn.body.index(current_loop)
    fn.body[pos:pos] = statements('''
ref_finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
ref_returns = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
ref_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
terminal_rewards = torch.zeros_like(ref_returns)
first_done_step = torch.zeros_like(ref_steps)
post_reset_steps = torch.zeros_like(ref_steps)
terminal_checked = torch.zeros_like(ref_finished)
post_reset_excluded = True
tick = 0
''')
    with_block = next(n for n in current_loop.body if isinstance(n, ast.With))
    with_block.body += statements('''
tick += 1
old_ref_finished = ref_finished.clone()
old_ref_returns = ref_returns.clone()
old_ref_steps = ref_steps.clone()
post_reset_steps[old_ref_finished] += 1
terminal_mask = dones.bool() & ~old_ref_finished
terminal_rewards[terminal_mask] = rewards[terminal_mask].double()
first_done_step[terminal_mask] = tick
assert tuple(obs['policy'].shape) == (env.num_envs, inputs['actor_input_dim'])
''') + accounting + statements('''
terminal_checked[terminal_mask] = (ref_returns[terminal_mask] - old_ref_returns[terminal_mask] == rewards[terminal_mask].double()) & (ref_steps[terminal_mask] == old_ref_steps[terminal_mask] + 1)
assert torch.equal(ref_returns[old_ref_finished], old_ref_returns[old_ref_finished])
assert torch.equal(ref_steps[old_ref_finished], old_ref_steps[old_ref_finished])
''')
    # Check actual existing accumulators after their unmodified accounting on every step.
    current_loop.body += statements('''
assert torch.equal(finished, ref_finished)
assert torch.equal(lengths, ref_steps)
assert torch.equal(returns[old_ref_finished], old_current_returns[old_ref_finished]) if tick > 1 else True
old_current_returns = returns.clone()
''')
    fn.body.insert(len(fn.body)-1, statements('_parity_emit(locals())')[0])
    ast.fix_missing_locations(tree)
    sys.path.insert(0, str(LOCAL.parent))
    sys.argv = [str(LOCAL)] + rest
    scope = dict(__name__='__main__', __file__=str(LOCAL), _parity_emit=emit)
    # Emit obtains the argparse object only once the frozen script has parsed it.
    def bound_emit(state):
        global ARGS
        ARGS = scope['args_cli']
        emit(state)
    scope['_parity_emit'] = bound_emit
    exec(compile(tree, str(LOCAL), 'exec'), scope)


if __name__ == '__main__':
    main()
