"""Read-only first-episode trajectories around the frozen parity-checked evaluator.

Instrument in memory only. Capture terminal root pose at the existing recorder
pre-reset hook without adding recorder terms or computing observations/sensors.
Preserve legacy pre-terminal displacement alongside true terminal displacement.
"""
import argparse
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'scripts/reinforcement_learning/rsl_rl/play_one_episode.py'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Trajectory:
    def __init__(self, env, inputs, checkpoint, cfg, directory):
        from isaaclab_tasks.manager_based.classic.ant.ant_final_unseen_env_cfg import COURSE_DESIGN
        self.base, self.directory = env.unwrapped, directory
        self.directory.mkdir(exist_ok=False)
        self.n, self.dt = env.num_envs, self.base.step_dt
        self.origin = self.base.scene.env_origins[:, :2].cpu().numpy().copy()
        self.spawn = COURSE_DESIGN['spawn_local_xyz'][:2]
        self.initial = self.base.scene['robot'].data.root_pos_w[:, :2].cpu().numpy().copy()
        self.tick, self.rows = 0, []
        self.recorder = self.base.recorder_manager
        assert len(self.recorder.active_terms) == 0
        self.original_pre_reset = self.recorder.record_pre_reset
        self.terminal = None
        self.armed = False

        def pre_reset(ids, *args, **kwargs):
            if self.armed:
                self.terminal = (ids.clone(), self.base.scene['robot'].data.root_pos_w[ids, :2].clone())
            return self.original_pre_reset(ids, *args, **kwargs)
        self.recorder.record_pre_reset = pre_reset
        self.metadata = dict(task='Isaac-Ant-Final-Unseen-v0', seed=int(cfg.seed), num_envs=self.n,
            physics_dt=cfg.sim.dt, decimation=cfg.decimation, control_dt=self.dt,
            terrain_seed=cfg.scene.terrain.terrain_generator.seed, course_design=COURSE_DESIGN,
            env_origins_xy=self.origin.tolist(), initial_root_xy=self.initial.tolist(),
            initial_terrain_local_xy=(self.initial-self.origin+self.spawn).tolist(),
            checkpoint=checkpoint, checkpoint_sha256=sha256(checkpoint), **inputs,
            source_sha256=sha256(SOURCE), observation_runtime_checks=True,
            timing='step 0 initial pose; row k true post-control-step k pose; terminal pose before auto-reset',
            coordinate='terrain_local_xy = root_world_xy - env_origin_xy + spawn_local_xy; fixed initial tile, no modulo',
            legacy_displacement='last pre-control-step root x minus initial x, preserved for consistency')
        for i in range(self.n):
            self.rows.append(self.row(i, 0, self.initial[i], False, False, False, False, 0.0))

    def row(self, i, step, xy, terminated, truncated, done, fall, reward):
        local = xy-self.origin[i]+self.spawn
        return dict(env_id=i, step=step, time_s=step*self.dt, root_position_x=float(xy[0]),
            root_position_y=float(xy[1]), terrain_local_x=float(local[0]), terrain_local_y=float(local[1]),
            forward_displacement=float(xy[0]-self.initial[i,0]), terminated=bool(terminated),
            truncated=bool(truncated), done=bool(done), fall=bool(fall), step_reward=float(reward))

    def begin(self, finished):
        self.active = (~finished).cpu().numpy().copy()
        self.terminal = None
        self.armed = True

    def end(self, obs, rewards, dones):
        import numpy as np
        assert tuple(obs['policy'].shape) == (self.n, self.metadata['actor_input_dim'])
        self.tick += 1
        xy = self.base.scene['robot'].data.root_pos_w[:, :2].clone()
        if self.terminal is not None:
            ids, poses = self.terminal
            xy[ids] = poses
        self.armed = False
        xy, rew, done = xy.cpu().numpy(), rewards.cpu().numpy(), dones.bool().cpu().numpy()
        terminated = self.base.reset_terminated.cpu().numpy().copy()
        truncated = self.base.reset_time_outs.cpu().numpy().copy()
        fall = self.base.termination_manager.get_term('torso_height').cpu().numpy().copy()
        assert np.array_equal(done, terminated | truncated)
        if (done & self.active).any():
            assert self.terminal is not None, 'Missing pre-reset terminal pose'
        for i in np.flatnonzero(self.active):
            self.rows.append(self.row(int(i), self.tick, xy[i], terminated[i], truncated[i], done[i], fall[i], rew[i]))

    def finish(self, state):
        self.recorder.record_pre_reset = self.original_pre_reset
        assert state['finished'].all()
        assert self.tick <= self.base.max_episode_length
        self.metadata['all_first_episodes_completed'] = True
        self.metadata['episodes'] = [dict(env_id=i, episode_return=float(state['returns'][i]),
            episode_steps=int(state['lengths'][i]), legacy_final_displacement=float(state['forward_displacement'][i]),
            fall=bool(state['fall'][i]), timeout=bool(state['time_out'][i])) for i in range(self.n)]
        with gzip.open(self.directory/'trajectory.csv.gz', 'wt', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader(); writer.writerows(self.rows)
        with (self.directory/'metadata.json').open('x') as stream:
            json.dump(self.metadata, stream, indent=2, allow_nan=False)
        print(f'FIRST_PASS_TRAJECTORY {self.directory} rows={len(self.rows)}', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory_dir', required=True, type=Path)
    own, rest=parser.parse_known_args()
    for flag,value in [('--task','Isaac-Ant-Final-Unseen-v0'),('--seed','24'),('--num_envs','100')]:
        assert flag in rest and rest[rest.index(flag)+1]==value, (flag,value)
    assert '--video' not in rest and '--preflight_steps' not in rest
    if own.trajectory_dir.exists(): raise FileExistsError(own.trajectory_dir)
    tree=ast.parse(SOURCE.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in fn.body if isinstance(n,ast.While))
    fn.body.insert(fn.body.index(loop),ast.parse('_trajectory = _make_trajectory(env, inputs, resume_path, env_cfg)').body[0])
    loop.body.insert(0,ast.parse('_trajectory.begin(finished)').body[0])
    block=next(n for n in loop.body if isinstance(n,ast.With))
    block.body.append(ast.parse('_trajectory.end(obs, rewards, dones)').body[0])
    fn.body.insert(len(fn.body)-1,ast.parse('_trajectory.finish(locals())').body[0])
    ast.fix_missing_locations(tree)
    sys.path.insert(0,str(SOURCE.parent)); sys.argv=[str(SOURCE)]+rest
    scope=dict(__name__='__main__',__file__=str(SOURCE),
        _make_trajectory=lambda env,inputs,path,cfg:Trajectory(env,inputs,path,cfg,own.trajectory_dir))
    exec(compile(tree,str(SOURCE),'exec'),scope)


if __name__=='__main__': main()
