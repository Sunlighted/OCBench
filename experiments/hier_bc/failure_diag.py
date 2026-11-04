"""Classify how a flat BC prior fails on block-single: roll it out and track the block, gripper and target.

Usage (from impls/): python ../experiments/hier_bc/failure_diag.py --prior_dir=... [--num_episodes=200]

Per episode it records whether the block was ever grasped (gripper contact with the block lifted off the table),
how many separate grasps happened, the closest the block got to the target, and the final state, then buckets
failures into: never grasped, grasped but never carried near the target, carried near the target but not placed,
and placed then disturbed.
"""

import json
import os
import sys
from collections import Counter

import numpy as np
from absl import app, flags

sys.path.insert(0, os.getcwd())

FLAGS = flags.FLAGS
flags.DEFINE_string('prior_dir', '', 'BC prior directory.')
flags.DEFINE_integer('prior_step', 200000, 'Prior checkpoint step.')
flags.DEFINE_string('env_name', 'block-single-task1-v0', 'Environment name.')
flags.DEFINE_integer('num_episodes', 200, 'Episodes.')
flags.DEFINE_integer('num_envs', 50, 'Environments stepped in lockstep.')
flags.DEFINE_string('out', '', 'Output JSON (default: prior_dir/failure_diag.json).')

LIFT_HEIGHT = 0.045  # Block center above this (resting ~0.02-0.03) counts as lifted.
NEAR_TARGET = 0.04  # Horizontal distance to the target counted as "near".


def main(_):
    import gymnasium
    import jax
    import ocbench  # noqa: F401

    from agents import agents
    from envs.env_utils import ActionNormalizer, wrap_env
    from hier_common import cpu_env_name
    from utils.flax_utils import restore_agent

    with open(os.path.join(FLAGS.prior_dir, 'flags.json')) as f:
        prior_flags = json.load(f)
    config = dict(prior_flags['agent'])
    for key in ('actor_hidden_dims', 'betas'):
        if key in config:
            config[key] = tuple(config[key])
    chunk_len = prior_flags['action_chunk_length']
    norm = np.load(os.path.join(FLAGS.prior_dir, 'action_normalizer.npz'))
    normalizer = ActionNormalizer(norm['q_low'], norm['q_high'], float(norm['clip']), norm['gripper_dims'])
    env_name = cpu_env_name(FLAGS.env_name)
    envs = [wrap_env(gymnasium.make(env_name), action_normalizer=normalizer) for _ in range(FLAGS.num_envs)]
    ob_dim = envs[0].observation_space.shape[-1]
    action_dim = envs[0].action_space.shape[-1]
    max_steps = envs[0].spec.max_episode_steps or 1250
    prior = agents[config['agent_name']].create(
        0, np.zeros((1, ob_dim), np.float32), np.zeros((1, chunk_len * action_dim), np.float32), config
    )
    prior = restore_agent(prior, FLAGS.prior_dir, FLAGS.prior_step)

    def target_pos(env):
        u = env.unwrapped
        return np.array(u._data.mocap_pos[u._cube_target_mocap_ids[0]])

    records = []
    rng = jax.random.PRNGKey(0)
    seed = 0
    while len(records) < FLAGS.num_episodes:
        n = min(FLAGS.num_envs, FLAGS.num_episodes - len(records))
        obs, stats = [], []
        for env in envs[:n]:
            ob, _ = env.reset(seed=int(np.random.randint(0, 2**31)))
            seed += 1
            obs.append(ob)
            init = env.unwrapped.compute_ob_info()
            stats.append(dict(target=target_pos(env), init_block=np.asarray(init['privileged/block_0_pos']).tolist(),
                              init_yaw=float(np.asarray(init['privileged/block_0_yaw'])[0]),
                              init_effector=np.asarray(init['proprio/effector_pos']).tolist(), grasps=0, min_eff_block=np.inf,
                              contact_steps=0, close_steps=0, block_moved=0.0, lifted=False, holding=False, min_dist=np.inf,
                              first_grasp=-1, done=False, success=False, steps=0))
        while not all(s['done'] for s in stats):
            rng, key = jax.random.split(rng)
            actions = np.asarray(prior.sample_actions(np.stack(obs), seed=key)).reshape(n, chunk_len, action_dim)
            for i, env in enumerate(envs[:n]):
                s = stats[i]
                if s['done']:
                    continue
                for a in actions[i]:
                    obs[i], _, term, trunc, info = env.step(a)
                    s['steps'] += 1
                    ob_info = env.unwrapped.compute_ob_info()
                    block = np.asarray(ob_info['privileged/block_0_pos'])
                    contact = float(np.asarray(ob_info['proprio/gripper_contact'])[0]) > 0.5
                    holding = contact and block[2] > LIFT_HEIGHT
                    eff = np.asarray(ob_info['proprio/effector_pos'])
                    s['min_eff_block'] = min(s['min_eff_block'], float(np.linalg.norm(eff - block)))
                    s['contact_steps'] += int(contact)
                    s['close_steps'] += int(float(np.asarray(ob_info['proprio/gripper_opening'])[0]) > 0.5)
                    s['block_moved'] = max(s['block_moved'], float(np.linalg.norm(block[:2] - np.asarray(s['init_block'])[:2])))
                    if holding and not s['holding']:
                        s['grasps'] += 1
                        if s['first_grasp'] < 0:
                            s['first_grasp'] = s['steps']
                    s['holding'] = holding
                    s['lifted'] |= holding
                    dist = float(np.linalg.norm(block[:2] - s['target'][:2]))
                    s['min_dist'] = min(s['min_dist'], dist)
                    s['final_dist'] = dist
                    s['final_height'] = float(block[2])
                    if term or trunc or s['steps'] >= max_steps:
                        s['done'] = True
                        s['success'] = bool(info.get('success', False))
                        break
        for s in stats:
            if s['success']:
                s['outcome'] = 'success'
            elif not s['lifted']:
                s['outcome'] = 'never_grasped'
            elif s['min_dist'] > NEAR_TARGET:
                s['outcome'] = 'grasped_not_carried'
            elif s['final_dist'] <= NEAR_TARGET:
                s['outcome'] = 'near_target_not_placed'
            else:
                s['outcome'] = 'placed_then_moved_away'
            records.append({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                            for k, v in s.items()} | dict(target=s['target'].tolist()))
        print(len(records), Counter(r['outcome'] for r in records), flush=True)

    counts = Counter(r['outcome'] for r in records)
    summary = dict(
        episodes=len(records),
        outcomes={k: v / len(records) for k, v in counts.items()},
        grasps_mean={o: float(np.mean([r['grasps'] for r in records if r['outcome'] == o])) for o in counts},
        first_grasp_median=float(np.median([r['first_grasp'] for r in records if r['first_grasp'] >= 0] or [-1])),
    )
    out = FLAGS.out or os.path.join(FLAGS.prior_dir, 'failure_diag.json')
    with open(out, 'w') as f:
        json.dump(dict(summary=summary, episodes=records), f, indent=1, default=lambda o: o.item())
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    app.run(main)
