"""Evaluate a subgoal-conditioned BC checkpoint with the oracle high-level policy.

Environments run in parallel worker processes; the policy runs batched in the main process. Besides the metrics of
subgoal_bc.py, this logs per-subgoal diagnostics (which completion conditions held when a subgoal timed out) and
supports longer episodes and alternative subgoal completion rules.
"""

import json
import multiprocessing as mp
import os
import pickle
import time
from collections import defaultdict

import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt_dir', None, 'Directory with params_<step>.pkl, action_normalizer.npz, and flags.json.')
flags.DEFINE_integer('restore_step', 500000, 'Checkpoint step to restore.')
flags.DEFINE_string('out', None, 'Output JSON path.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_episodes', 500, 'Number of evaluation episodes.')
flags.DEFINE_integer('num_workers', 50, 'Number of environment worker processes.')
flags.DEFINE_integer('max_episode_steps', 0, 'Episode step limit (0: the environment default).')
flags.DEFINE_integer('max_subgoal_steps', 1250, 'Steps before the high-level policy replans a subgoal.')
flags.DEFINE_enum(
    'done_mode',
    'orig',
    ['orig', 'no_effector', 'target_only', 'target_hold'],
    'Subgoal completion rule. orig: block at target, effector far from block, and no gripper contact. '
    'no_effector: drop the effector-distance condition. target_only: block at target. '
    'target_hold: block at target without gripper contact for done_hold_checks consecutive chunk boundaries.',
)
flags.DEFINE_integer('done_hold_checks', 2, 'Consecutive chunk boundaries required by done_mode=target_hold.')
flags.DEFINE_float('done_target_dist', 0.04, 'Max block-target distance for subgoal completion.')
flags.DEFINE_float('done_effector_dist', 0.1, 'Min effector-block distance for subgoal completion.')
flags.DEFINE_float('done_contact', 0.5, 'Max gripper contact for subgoal completion.')

XYZ_CENTER = np.array([0.425, 0.0, 0.0])
XYZ_SCALER = 10.0


def encode_subgoal(block, target_pos, target_yaw, num_blocks):
    one_hot = np.zeros(num_blocks, dtype=np.float32)
    one_hot[block] = 1.0
    return np.concatenate(
        [one_hot, (np.asarray(target_pos) - XYZ_CENTER) * XYZ_SCALER, [np.cos(target_yaw), np.sin(target_yaw)]]
    ).astype(np.float32)


def check_conditions(subgoal, ob_info, cfg):
    block_pos = ob_info[f'privileged/block_{subgoal["block"]}_pos']
    effector_pos = ob_info['proprio/effector_pos']
    target_dist = float(np.linalg.norm(block_pos - subgoal['target_pos']))
    effector_dist = float(np.linalg.norm(effector_pos - block_pos))
    contact = float(ob_info['proprio/gripper_contact'][0])
    return dict(
        target_dist=target_dist,
        effector_dist=effector_dist,
        contact=contact,
        target_ok=target_dist <= cfg['done_target_dist'],
        effector_ok=effector_dist >= cfg['done_effector_dist'],
        contact_ok=contact < cfg['done_contact'],
    )


def is_done(conds, hold_count, cfg):
    mode = cfg['done_mode']
    if mode == 'orig':
        return conds['target_ok'] and conds['effector_ok'] and conds['contact_ok']
    if mode == 'no_effector':
        return conds['target_ok'] and conds['contact_ok']
    if mode == 'target_only':
        return conds['target_ok']
    return hold_count >= cfg['done_hold_checks']


def worker(conn, env_name, normalizer_args, num_envs, seed, cfg):
    """Run num_envs episodes, stepping one action chunk per request."""
    import gymnasium
    import ocbench  # noqa: F401
    from ocbench.oracles.controllers.block import BlockController

    from envs.env_utils import ActionNormalizer, wrap_env

    np.random.seed(seed)
    normalizer = ActionNormalizer(**normalizer_args)
    make_kwargs = {}
    if cfg['max_episode_steps'] > 0:
        make_kwargs['max_episode_steps'] = cfg['max_episode_steps']

    states = []
    for _ in range(num_envs):
        env = wrap_env(gymnasium.make(env_name, **make_kwargs), action_normalizer=normalizer)
        num_blocks = env.unwrapped._num_cubes
        observation, info = env.reset(seed=int(np.random.randint(0, 2**31)))
        controller = BlockController(env=env, p_mistake=0.0)
        controller.reset(observation, env.unwrapped.compute_ob_info())
        states.append(
            dict(
                env=env,
                controller=controller,
                observation=observation,
                info=info,
                subgoal=None,
                subgoal_steps=0,
                hold_count=0,
                ever_target=False,
                first_target_step=None,
                done=False,
                events=[],
            )
        )

    def bookkeeping(state):
        ob_info = state['env'].unwrapped.compute_ob_info()
        subgoal = state['subgoal']
        if subgoal is not None:
            conds = check_conditions(subgoal, ob_info, cfg)
            if conds['target_ok'] and conds['contact_ok']:
                state['hold_count'] += 1
            else:
                state['hold_count'] = 0
            if conds['target_ok'] and not state['ever_target']:
                state['ever_target'] = True
                state['first_target_step'] = state['subgoal_steps']
            outcome = None
            if is_done(conds, state['hold_count'], cfg):
                outcome = 'done'
            elif state['subgoal_steps'] >= cfg['max_subgoal_steps']:
                outcome = 'timeout'
            if outcome is not None:
                state['events'].append(
                    dict(
                        outcome=outcome,
                        block=int(subgoal['block']),
                        is_clear=subgoal.get('name') not in (None, 'goal'),
                        steps=state['subgoal_steps'],
                        ever_target=state['ever_target'],
                        first_target_step=state['first_target_step'],
                        **conds,
                    )
                )
                state['subgoal'] = None
        if state['subgoal'] is None:
            state['subgoal'] = state['controller']._goal_subgoal(ob_info)
            state['subgoal_steps'] = 0
            state['hold_count'] = 0
            state['ever_target'] = False
            state['first_target_step'] = None
        if state['subgoal'] is None:
            state['cond'] = np.zeros(num_blocks + 5, dtype=np.float32)
        else:
            s = state['subgoal']
            state['cond'] = encode_subgoal(s['block'], s['target_pos'], s['target_yaw'], num_blocks)

    def observe():
        obs = np.stack([np.concatenate([s['observation'], s['cond']]) for s in states])
        active = np.array([not s['done'] for s in states])
        return obs, active

    for state in states:
        bookkeeping(state)
    conn.send(observe())

    while True:
        msg, payload = conn.recv()
        if msg == 'step':
            for state, chunk in zip(states, payload):
                if state['done']:
                    continue
                for action in chunk:
                    observation, _, terminated, truncated, info = state['env'].step(action)
                    state['observation'] = observation
                    state['info'] = info
                    state['subgoal_steps'] += 1
                    if terminated or truncated:
                        state['done'] = True
                        break
                if not state['done']:
                    bookkeeping(state)
            conn.send(observe())
        elif msg == 'stats':
            results = []
            for state in states:
                env = state['env'].unwrapped
                # Record the unfinished subgoal at the end of the episode.
                if state['subgoal'] is not None and not state['info'].get('success', False):
                    conds = check_conditions(state['subgoal'], env.compute_ob_info(), cfg)
                    state['events'].append(
                        dict(
                            outcome='episode_end',
                            block=int(state['subgoal']['block']),
                            is_clear=state['subgoal'].get('name') not in (None, 'goal'),
                            steps=state['subgoal_steps'],
                            ever_target=state['ever_target'],
                            first_target_step=state['first_target_step'],
                            **conds,
                        )
                    )
                results.append(
                    dict(
                        success=float(state['info']['success']),
                        blocks_at_target=float(sum(env._compute_successes())),
                        episode_length=float(state['info']['episode']['length']),
                        events=state['events'],
                    )
                )
            conn.send(results)
        elif msg == 'close':
            conn.close()
            return


def main(_):
    import jax

    import ocbench  # noqa: F401
    from agents import agents
    from utils.flax_utils import restore_agent

    with open(os.path.join(FLAGS.ckpt_dir, 'flags.json')) as f:
        train_flags = json.load(f)
    config = train_flags['agent']
    env_name = train_flags['env_name']
    cpu_env_name = env_name.replace('block-', 'block-cpu-', 1)
    chunk_len = train_flags['action_chunk_length']
    norm = np.load(os.path.join(FLAGS.ckpt_dir, 'action_normalizer.npz'))
    normalizer_args = dict(
        q_low=norm['q_low'], q_high=norm['q_high'], clip=float(norm['clip']), gripper_dims=norm['gripper_dims']
    )
    action_dim = len(norm['q_low'])

    import gymnasium

    num_blocks = gymnasium.make(cpu_env_name).unwrapped._num_cubes
    ob_dim = gymnasium.make(cpu_env_name).observation_space.shape[-1] + num_blocks + 5

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed, np.zeros((1, ob_dim), np.float32), np.zeros((1, chunk_len * action_dim), np.float32), config
    )
    agent = restore_agent(agent, FLAGS.ckpt_dir, FLAGS.restore_step)
    def sample(obs, key):
        return agent.sample_actions(observations=obs, seed=key)

    cfg = dict(
        max_episode_steps=FLAGS.max_episode_steps,
        max_subgoal_steps=FLAGS.max_subgoal_steps,
        done_mode=FLAGS.done_mode,
        done_hold_checks=FLAGS.done_hold_checks,
        done_target_dist=FLAGS.done_target_dist,
        done_effector_dist=FLAGS.done_effector_dist,
        done_contact=FLAGS.done_contact,
    )
    counts = np.full(FLAGS.num_workers, FLAGS.num_episodes // FLAGS.num_workers)
    counts[: FLAGS.num_episodes % FLAGS.num_workers] += 1
    counts = counts[counts > 0]

    ctx = mp.get_context('spawn')
    conns, procs = [], []
    for i, n in enumerate(counts):
        parent, child = ctx.Pipe()
        p = ctx.Process(
            target=worker, args=(child, cpu_env_name, normalizer_args, int(n), FLAGS.seed * 1000 + i, cfg), daemon=True
        )
        p.start()
        conns.append(parent)
        procs.append(p)

    start = time.time()
    replies = [c.recv() for c in conns]
    rng = jax.random.PRNGKey(FLAGS.seed)
    num_chunks = 0
    while True:
        obs = np.concatenate([r[0] for r in replies])
        active = np.concatenate([r[1] for r in replies])
        if not active.any():
            break
        rng, key = jax.random.split(rng)
        actions = np.asarray(sample(obs, key))
        actions = actions.reshape(len(obs), chunk_len, action_dim)
        low = np.full(action_dim, -normalizer_args['clip'], np.float32)
        high = np.full(action_dim, normalizer_args['clip'], np.float32)
        low[normalizer_args['gripper_dims']] = -1.0
        high[normalizer_args['gripper_dims']] = 1.0
        actions = np.clip(actions, low, high)
        offset = 0
        for c, n in zip(conns, counts):
            c.send(('step', actions[offset : offset + n]))
            offset += n
        replies = [c.recv() for c in conns]
        num_chunks += 1
        if num_chunks % 20 == 0:
            print(f'chunk {num_chunks}: {int(active.sum())} active, {time.time() - start:.0f}s', flush=True)

    episodes = []
    for c in conns:
        c.send(('stats', None))
        episodes.extend(c.recv())
        c.send(('close', None))
    for p in procs:
        p.join()

    summary = summarize(episodes, num_blocks)
    summary['eval_time'] = time.time() - start
    summary['config'] = dict(cfg, restore_step=FLAGS.restore_step, num_episodes=len(episodes), env_name=env_name)
    print(json.dumps({k: v for k, v in summary.items() if k != 'config'}, indent=1))
    if FLAGS.out:
        os.makedirs(os.path.dirname(os.path.abspath(FLAGS.out)), exist_ok=True)
        with open(FLAGS.out, 'w') as f:
            json.dump(summary, f, indent=1)
        with open(FLAGS.out.removesuffix('.json') + '_episodes.pkl', 'wb') as f:
            pickle.dump(episodes, f)


def summarize(episodes, num_blocks):
    s = {}
    success = np.array([e['success'] for e in episodes])
    blocks = np.array([e['blocks_at_target'] for e in episodes])
    s['success'] = float(success.mean())
    s['success_se'] = float(success.std(ddof=1) / np.sqrt(len(success)))
    s['blocks_at_target'] = float(blocks.mean())
    for k in range(1, num_blocks + 1):
        s[f'blocks_at_target_ge{k}'] = float(np.mean(blocks >= k))
    s['episode_length'] = float(np.mean([e['episode_length'] for e in episodes]))
    lengths = np.array([e['episode_length'] for e in episodes])
    s['success_length_mean'] = float(lengths[success > 0].mean()) if success.any() else None

    events = [ev for e in episodes for ev in e['events']]
    by_outcome = defaultdict(list)
    for ev in events:
        by_outcome[ev['outcome']].append(ev)
    s['subgoals_done_per_ep'] = len(by_outcome['done']) / len(episodes)
    s['timeouts_per_ep'] = len(by_outcome['timeout']) / len(episodes)
    s['clear_subgoals_per_ep'] = sum(ev['is_clear'] for ev in events) / len(episodes)
    if by_outcome['done']:
        s['done_steps_mean'] = float(np.mean([ev['steps'] for ev in by_outcome['done']]))
        s['done_steps_p90'] = float(np.percentile([ev['steps'] for ev in by_outcome['done']], 90))

    # Why did timed-out subgoals not complete?
    timeouts = by_outcome['timeout']
    if timeouts:
        t = {}
        for key in ('target_ok', 'effector_ok', 'contact_ok', 'ever_target'):
            t[key] = float(np.mean([ev[key] for ev in timeouts]))
        patterns = defaultdict(int)
        for ev in timeouts:
            patterns[f"target={int(ev['target_ok'])},eff={int(ev['effector_ok'])},contact={int(ev['contact_ok'])}"] += 1
        t['patterns'] = {k: v / len(timeouts) for k, v in sorted(patterns.items(), key=lambda x: -x[1])}
        reached = [ev['first_target_step'] for ev in timeouts if ev['first_target_step'] is not None]
        t['first_target_step_mean'] = float(np.mean(reached)) if reached else None
        at_target = [ev for ev in timeouts if ev['target_ok']]
        if at_target:
            t['at_target_effector_dist_mean'] = float(np.mean([ev['effector_dist'] for ev in at_target]))
            t['at_target_contact_mean'] = float(np.mean([ev['contact'] for ev in at_target]))
        s['timeout_diagnostics'] = t
    return s


if __name__ == '__main__':
    app.run(main)
