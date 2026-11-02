"""Shared pieces for subgoal-conditioned BC with an oracle high-level policy on all OCBench task families.

Subgoals are encoded from the `target` dicts stored in the dataset segment metadata. At evaluation, the oracle
high-level policy is the family's scripted controller with mistakes disabled, and its subgoals are encoded through the
same `_target_info` method that produced the metadata, so training and evaluation conditions match.
"""

import multiprocessing as mp
import time
from collections import defaultdict

import numpy as np

XYZ_CENTER = np.array([0.425, 0.0, 0.0])
XYZ_SCALER = 10.0
CHAMBER_TASKS = ('cube', 'button', 'drawer', 'window')


def env_family(env_name):
    name = env_name.removeprefix('visual-')
    for family in ('block', 'chamber', 'hanoi', 'switch'):
        if name.startswith(family + '-'):
            return family
    raise ValueError(f'Unsupported environment: {env_name}')


def cpu_env_name(env_name):
    family = env_family(env_name)
    return env_name.replace(f'{family}-', f'{family}-cpu-', 1)


def subgoal_spec(family, env):
    """Return the list of (kind, key, size) fields that make up the subgoal encoding."""
    if family == 'block':
        return [('onehot', 'block', env._num_cubes), ('pos', 'block_pos', 3), ('yaw', 'block_yaw', 2)]
    if family == 'hanoi':
        return [
            ('onehot', 'disk', env._num_disks),
            ('onehot', 'source_peg', env._num_pegs),
            ('onehot', 'peg', env._num_pegs),
            ('pos', 'disk_goal_pos', 3),
            ('yaw', 'disk_goal_yaw', 2),
        ]
    if family == 'switch':
        return [
            ('onehot', 'button', env._num_buttons),
            ('scalar', 'button_state', 1),
            ('pos', 'button_top_pos', 3),
            ('yaw', 'button_top_yaw', 2),
        ]
    return [
        ('task', None, len(CHAMBER_TASKS)),
        ('onehot', 'block', env._num_cubes),
        ('pos', 'block_pos', 3),
        ('yaw', 'block_yaw', 2),
        ('onehot', 'button', env._num_buttons),
        ('scalar', 'button_state', 1),
        ('pos', 'button_top_pos', 3),
        ('yaw', 'button_top_yaw', 2),
        ('pos1', 'drawer_pos', 1),
        ('pos', 'drawer_handle_pos', 3),
        ('pos1', 'window_pos', 1),
        ('pos', 'window_handle_pos', 3),
    ]


def subgoal_dim(spec):
    return sum(size for _, _, size in spec)


def encode_subgoal(spec, subtask_name, target):
    parts = []
    for kind, key, size in spec:
        vec = np.zeros(size, dtype=np.float32)
        if kind == 'task':
            vec[CHAMBER_TASKS.index(subtask_name)] = 1.0
        elif key in target:
            value = target[key]
            if kind == 'onehot':
                vec[int(value)] = 1.0
            elif kind == 'scalar':
                vec[0] = float(np.asarray(value).reshape(-1)[0])
            elif kind == 'pos':
                vec[:] = (np.asarray(value, dtype=np.float64) - XYZ_CENTER) * XYZ_SCALER
            elif kind == 'pos1':
                vec[0] = float(np.asarray(value).reshape(-1)[0]) * XYZ_SCALER
            elif kind == 'yaw':
                yaw = float(np.asarray(value).reshape(-1)[0])
                vec[:] = [np.cos(yaw), np.sin(yaw)]
        parts.append(vec)
    return np.concatenate(parts)


def make_controller(family, env):
    from ocbench.oracles.controllers import BlockController, ChamberController, HanoiController, SwitchController

    if family == 'block':
        return BlockController(env=env, p_mistake=0.0)
    if family == 'chamber':
        return ChamberController(env=env, p_mistake=0.0)
    if family == 'hanoi':
        return HanoiController(env=env, p_mistake=0.0)
    return SwitchController(env=env, p_mistake_range=(0.0, 0.0))


def subtask_name(family, subgoal):
    return dict(block='cube', hanoi='hanoi', switch='button').get(family) or subgoal['task']


def worker(conn, env_name, normalizer_args, num_envs, seed, cfg):
    """Run num_envs episodes with the oracle high-level policy, stepping one action chunk per request."""
    import gymnasium
    import ocbench  # noqa: F401

    from envs.env_utils import ActionNormalizer, wrap_env

    np.random.seed(seed)
    family = env_family(env_name)
    normalizer = ActionNormalizer(**normalizer_args)
    make_kwargs = {}
    if cfg['max_episode_steps'] > 0:
        make_kwargs['max_episode_steps'] = cfg['max_episode_steps']

    states = []
    for _ in range(num_envs):
        env = wrap_env(gymnasium.make(env_name, **make_kwargs), action_normalizer=normalizer)
        spec = subgoal_spec(family, env.unwrapped)
        observation, _ = env.reset(seed=int(np.random.randint(0, 2**31)))
        ob_info = env.unwrapped.compute_ob_info()
        controller = make_controller(family, env)
        controller.reset(observation, ob_info)
        states.append(
            dict(
                env=env,
                controller=controller,
                observation=observation,
                info={},
                subgoal=None,
                subgoal_steps=0,
                release_pos=None,
                done=False,
                events=[],
            )
        )

    def finish_subgoal(state, outcome, ob_info):
        subgoal = state['subgoal']
        state['events'].append(
            dict(
                outcome=outcome,
                task=subtask_name(family, subgoal),
                name=subgoal.get('name'),
                steps=state['subgoal_steps'],
                success_now=bool(state['controller']._subtask_success(subgoal, ob_info)),
                released=state['release_pos'] is not None,
                target_pos=np.asarray(subgoal.get('target_pos', np.full(3, np.nan)), dtype=np.float32),
            )
        )
        if outcome != 'done':
            # End-state snapshot for diagnosing stuck/timeout subgoals.
            state['events'][-1]['diag'] = dict(
                {k: np.array(v).copy() for k, v in ob_info.items() if k.startswith(('privileged/', 'proprio/'))},
                failure=bool(getattr(state['env'].unwrapped, '_failure', False)),
                target=dict(state.get('target') or {}),
            )
        state['subgoal'] = None

    def bookkeeping(state):
        if cfg.get('flat'):
            state['cond'] = np.zeros(0, dtype=np.float32)
            return
        env = state['env'].unwrapped
        ob_info = env.compute_ob_info()
        controller = state['controller']
        if family == 'switch':
            controller._update_plan_from_info(ob_info)
        subgoal = state['subgoal']
        if subgoal is not None:
            success = controller._subtask_success(subgoal, ob_info)
            contact_ok = ob_info['proprio/gripper_contact'][0] < cfg['done_contact'] or family == 'switch'
            effector_pos = ob_info['proprio/effector_pos']
            if not (success and contact_ok):
                state['release_pos'] = None
            elif state['release_pos'] is None:
                state['release_pos'] = effector_pos.copy()
            if (
                state['release_pos'] is not None
                and np.linalg.norm(effector_pos - state['release_pos']) >= cfg['done_away_dist']
            ):
                finish_subgoal(state, 'done', ob_info)
            elif state['subgoal_steps'] >= cfg['max_subgoal_steps']:
                finish_subgoal(state, 'timeout', ob_info)
        if state['subgoal'] is None:
            state['subgoal'] = controller._choose_subgoal(ob_info)
            state['subgoal_steps'] = 0
            state['release_pos'] = None
            if state['subgoal'] is not None:
                state['target'] = controller._target_info(state['subgoal'], ob_info)
        if state['subgoal'] is None:
            state['cond'] = np.zeros(subgoal_dim(spec), dtype=np.float32)
        else:
            state['cond'] = encode_subgoal(spec, subtask_name(family, state['subgoal']), state['target'])

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
                if state['subgoal'] is not None and not state['info'].get('success', False):
                    finish_subgoal(state, 'episode_end', state['env'].unwrapped.compute_ob_info())
                results.append(
                    dict(
                        success=float(state['info'].get('success', 0.0)),
                        episode_length=float(state['info']['episode']['length']),
                        events=state['events'],
                    )
                )
            conn.send(results)
        elif msg == 'close':
            conn.close()
            return


def evaluate_parallel(
    sample_fn, env_name, normalizer_args, chunk_len, action_dim, num_episodes, num_workers, cfg, seed, wave_size=50
):
    """Evaluate in waves of at most wave_size episodes to bound host memory (each environment takes ~150MB)."""
    start = time.time()
    episodes = []
    for wave, wave_start in enumerate(range(0, num_episodes, wave_size)):
        n = min(wave_size, num_episodes - wave_start)
        episodes.extend(
            _evaluate_wave(
                sample_fn, env_name, normalizer_args, chunk_len, action_dim, n, min(num_workers, n), cfg,
                seed * 100 + wave,
            )
        )
    summary = summarize(episodes)
    summary['eval_time'] = time.time() - start
    return summary, episodes


def _evaluate_wave(sample_fn, env_name, normalizer_args, chunk_len, action_dim, num_episodes, num_workers, cfg, seed):
    """Evaluate with environments in worker processes and batched policy inference in this process."""
    counts = np.full(num_workers, num_episodes // num_workers)
    counts[: num_episodes % num_workers] += 1
    counts = counts[counts > 0]

    ctx = mp.get_context('spawn')
    conns, procs = [], []
    for i, n in enumerate(counts):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=worker, args=(child, env_name, normalizer_args, int(n), seed * 1000 + i, cfg), daemon=True)
        p.start()
        conns.append(parent)
        procs.append(p)

    low = np.full(action_dim, -normalizer_args['clip'], np.float32)
    high = np.full(action_dim, normalizer_args['clip'], np.float32)
    low[normalizer_args['gripper_dims']] = -1.0
    high[normalizer_args['gripper_dims']] = 1.0

    replies = [c.recv() for c in conns]
    while True:
        obs = np.concatenate([r[0] for r in replies])
        active = np.concatenate([r[1] for r in replies])
        if not active.any():
            break
        actions = np.asarray(sample_fn(obs)).reshape(len(obs), chunk_len, action_dim)
        actions = np.clip(actions, low, high)
        offset = 0
        for c, n in zip(conns, counts):
            c.send(('step', actions[offset : offset + n]))
            offset += n
        replies = [c.recv() for c in conns]

    episodes = []
    for c in conns:
        c.send(('stats', None))
        episodes.extend(c.recv())
        c.send(('close', None))
    for p in procs:
        p.join()
    return episodes


def summarize(episodes):
    s = {}
    success = np.array([e['success'] for e in episodes])
    lengths = np.array([e['episode_length'] for e in episodes])
    s['success'] = float(success.mean())
    s['success_se'] = float(success.std(ddof=1) / np.sqrt(len(success))) if len(success) > 1 else 0.0
    s['episode_length'] = float(lengths.mean())
    s['success_length_median'] = float(np.median(lengths[success > 0])) if success.any() else -1.0

    events = [ev for e in episodes for ev in e['events']]
    by_outcome = defaultdict(list)
    for ev in events:
        by_outcome[ev['outcome']].append(ev)
    s['subgoals_done_per_ep'] = len(by_outcome['done']) / len(episodes)
    s['timeouts_per_ep'] = len(by_outcome['timeout']) / len(episodes)
    if by_outcome['done']:
        steps = [ev['steps'] for ev in by_outcome['done']]
        s['done_steps_median'] = float(np.median(steps))
        s['done_steps_p90'] = float(np.percentile(steps, 90))
    if by_outcome['timeout']:
        s['timeout_success_now'] = float(np.mean([ev['success_now'] for ev in by_outcome['timeout']]))
    for task in sorted({ev['task'] for ev in events}):
        task_events = [ev for ev in events if ev['task'] == task and ev['outcome'] in ('done', 'timeout')]
        if task_events:
            s[f'task/{task}/done_rate'] = float(np.mean([ev['outcome'] == 'done' for ev in task_events]))
            done = [ev['steps'] for ev in task_events if ev['outcome'] == 'done']
            if done:
                s[f'task/{task}/done_steps_median'] = float(np.median(done))
    return s
