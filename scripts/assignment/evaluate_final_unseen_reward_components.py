"""Measurement-only reward component buffer logging around the frozen evaluator.

Reuse terminal-safe first-pass telemetry. Never call a reward term or replace
RewardManager.compute. Its existing _step_reward is weight-scaled WITHOUT dt;
restore dt to obtain contributions and quantify round-trip/reduction errors.
"""
import argparse
import ast
import json
from pathlib import Path
import sys

from evaluate_final_unseen_first_pass import Trajectory, ROOT, SOURCE, sha256


class RewardTrajectory(Trajectory):
    def __init__(self, env, inputs, checkpoint, cfg, directory):
        from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
        assert cfg.rewards.to_dict() == AntEnvCfg().rewards.to_dict(), 'Stock reward config changed'
        super().__init__(env, inputs, checkpoint, cfg, directory)
        self.manager = self.base.reward_manager
        self.terms = list(self.manager.active_terms)
        assert self.terms == ['progress','alive','upright','move_to_target','action_l2','energy','joint_pos_limits']
        definitions = []
        for name in self.terms:
            term = self.manager.get_term_cfg(name)
            func = term.func
            owner = func if hasattr(func, '__qualname__') else type(func)
            definitions.append(dict(name=name,function=owner.__module__+'.'+owner.__qualname__,
                weight=term.weight,parameters=json.loads(json.dumps(term.params,default=str))))
        self.metadata.update(reward_terms=definitions, stock_reward_config_match=True,
            reward_capture='Read-only existing RewardManager._step_reward (weighted value/dt); multiply by unchanged control dt',
            reward_raw_functions_recomputed=False, reward_compute_or_weights_modified=False,
            instrumentation_source_sha256=sha256(Path(__file__)),
            reward_source_sha256=sha256(ROOT/'source/isaaclab/isaaclab/managers/reward_manager.py'))
        for row in self.rows:
            row['total_reward'] = 0.0
            for name in self.terms: row['reward_'+name] = 0.0

    def end(self, obs, rewards, dones):
        # reset() clears _episode_sums but leaves _step_reward for this terminal transition intact.
        contribution = (self.manager._step_reward * self.dt).detach().cpu().numpy().copy()
        assert contribution.shape == (self.n,len(self.terms))
        start = len(self.rows)
        super().end(obs,rewards,dones)
        for row in self.rows[start:]:
            row['total_reward'] = row['step_reward']
            for j,name in enumerate(self.terms): row['reward_'+name] = float(contribution[row['env_id'],j])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reward_dir',required=True,type=Path)
    own,rest=parser.parse_known_args()
    for flag,value in [('--task','Isaac-Ant-Final-Unseen-v0'),('--seed','24'),('--num_envs','100')]:
        assert flag in rest and rest[rest.index(flag)+1]==value
    assert '--video' not in rest and '--preflight_steps' not in rest
    if own.reward_dir.exists(): raise FileExistsError(own.reward_dir)
    tree=ast.parse(SOURCE.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in fn.body if isinstance(n,ast.While))
    fn.body.insert(fn.body.index(loop),ast.parse('_capture = _make_capture(env, inputs, resume_path, env_cfg)').body[0])
    loop.body.insert(0,ast.parse('_capture.begin(finished)').body[0])
    block=next(n for n in loop.body if isinstance(n,ast.With))
    block.body.append(ast.parse('_capture.end(obs, rewards, dones)').body[0])
    fn.body.insert(len(fn.body)-1,ast.parse('_capture.finish(locals())').body[0])
    ast.fix_missing_locations(tree)
    sys.path.insert(0,str(SOURCE.parent));sys.argv=[str(SOURCE)]+rest
    exec(compile(tree,str(SOURCE),'exec'),dict(__name__='__main__',__file__=str(SOURCE),
        _make_capture=lambda env,inputs,path,cfg:RewardTrajectory(env,inputs,path,cfg,own.reward_dir)))


if __name__=='__main__':main()
