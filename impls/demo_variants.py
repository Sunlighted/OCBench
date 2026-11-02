"""Collect block demonstrations with controllable "human-like" properties.

The OCBench scripted block controller makes demonstrations human-like through three sources of randomness:
- trajectory randomness inside the pick-and-place primitive (heights, yaw modes, carry side, curves, timing jitter),
  plus the controller's random block order, scaled here by `traj_noise` in [0, 1] (0: one fixed, unimodal trajectory
  per initial state, i.e. planner-like; 1: OCBench default). Continuous draws shrink toward their midpoint and
  discrete draws fall back to the most likely option, so traj_noise=0 removes multimodality too;
- per-episode speed (`speed_profile`, the range of segment durations; a single value fixes the speed);
- high-level mistakes followed by recovery (`p_mistake`).

The output is an OCBench-format .npz (loadable with `ocbench.load_dataset`) plus a `-metadata.pkl` with per-episode
start/length/success/speed, so the same training code can consume it.
"""

import os
import pickle
import time
import types

import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'block-cpu-single-task1-v0', 'CPU environment name.')
flags.DEFINE_integer('num_episodes', 100, 'Number of successful episodes to keep.')
flags.DEFINE_float('traj_noise', 1.0, 'Scale of primitive/controller trajectory randomness (0: deterministic).')
flags.DEFINE_string('speed_profile', '0.7,1.5', 'Segment duration range "lo,hi" (equal values fix the speed).')
flags.DEFINE_float('p_mistake', 0.0, 'Probability of a high-level mistake per subgoal.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('save_path', '', 'Output .npz path.')
flags.DEFINE_integer('fixed_init', 0, 'If 1, reset every episode with the same seed (to measure demo diversity).')


class ScaledRandom:
    """Stand-in for `np.random` that shrinks every random draw toward a fixed value by a factor `scale`.

    uniform -> midpoint + scale * (sample - midpoint); discrete draws (choice/randint/permutation) are random with
    probability `scale` and otherwise take the most likely (or first) option.
    """

    def __init__(self, scale, rng, scaled=('uniform', 'choice', 'randint', 'permutation')):
        self.scale = scale
        self.rng = rng
        self.scaled = set(scaled)

    def uniform(self, low=0.0, high=1.0, size=None):
        if 'uniform' not in self.scaled:
            return self.rng.uniform(low, high, size)
        low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
        mid = (low + high) / 2
        value = mid + self.scale * (self.rng.uniform(low, high, size) - mid)
        return float(value) if np.ndim(value) == 0 else value

    def _random_draw(self, method):
        return method not in self.scaled or (self.scale > 0 and self.rng.uniform() < self.scale)

    def choice(self, a, size=None, replace=True, p=None):
        if self._random_draw('choice'):
            return self.rng.choice(a, size=size, replace=replace, p=p)
        options = np.arange(a) if np.ndim(a) == 0 else np.asarray(a)
        value = options[int(np.argmax(p))] if p is not None else options[0]
        return value if size is None else np.full(size, value)

    def randint(self, low, high=None, size=None):
        if self._random_draw('randint'):
            return self.rng.randint(low, high, size)
        value = 0 if high is None else low
        return value if size is None else np.full(size, value)

    def permutation(self, x):
        if self._random_draw('permutation'):
            return self.rng.permutation(x)
        return np.arange(x) if np.ndim(x) == 0 else np.asarray(x).copy()

    def __getattr__(self, name):
        return getattr(np.random, name)


def set_traj_noise(scale, seed):
    """Route the demonstrator's style randomness through ScaledRandom.

    All draws in the pick-and-place primitive are scaled. In the block controller only the block order (permutation)
    is: its other draws define the task (stack-anywhere goals) or the mistakes, which `p_mistake` controls.
    """
    from ocbench.oracles.controllers import block as block_controller
    from ocbench.oracles.primitives import cube, primitive

    def proxy(random):
        namespace = types.SimpleNamespace(**{k: getattr(np, k) for k in dir(np) if not k.startswith('__')})
        namespace.random = random
        return namespace

    rng = np.random.RandomState(seed)
    primitive.np = cube.np = proxy(ScaledRandom(scale, rng))
    block_controller.np = proxy(ScaledRandom(scale, rng, scaled=('permutation',)))


def collect_episode(env, controller, reset_seed):
    observation, _ = env.reset(seed=reset_seed)
    info = env.unwrapped.compute_ob_info()
    controller.reset(observation, info)
    observations, actions, qposes, qvels = [observation], [], [], []
    step_info = {}
    while True:
        qposes.append(env.unwrapped._data.qpos.copy())
        qvels.append(env.unwrapped._data.qvel.copy())
        action = controller.select_action(observation, info)
        observation, _, terminated, truncated, step_info = env.step(action)
        info = env.unwrapped.compute_ob_info()
        observations.append(observation)
        actions.append(action)
        if terminated or truncated or controller.done:
            break
    return dict(
        observations=np.asarray(observations, np.float32),
        actions=np.asarray(actions, np.float32),
        qpos=np.asarray(qposes, np.float32),
        qvel=np.asarray(qvels, np.float32),
        success=bool(step_info.get('success', False)),
        speed=controller.speed,
    )


def main(_):
    import gymnasium
    import ocbench  # noqa: F401
    from ocbench.oracles.controllers import BlockController

    np.random.seed(FLAGS.seed)
    set_traj_noise(FLAGS.traj_noise, FLAGS.seed)
    lo, hi = (float(x) for x in FLAGS.speed_profile.split(','))
    env = gymnasium.make(FLAGS.env_name)
    controller = BlockController(env=env, p_mistake=FLAGS.p_mistake, speed_profile=(lo, hi))

    episodes, attempts = [], 0
    start = time.time()
    while len(episodes) < FLAGS.num_episodes:
        reset_seed = FLAGS.seed if FLAGS.fixed_init else FLAGS.seed * 1_000_000 + attempts
        attempts += 1
        episode = collect_episode(env, controller, reset_seed)
        if episode['success']:
            episodes.append(episode)
    lengths = np.array([len(e['actions']) for e in episodes])
    print(
        f'{len(episodes)}/{attempts} successful, length mean {lengths.mean():.0f} std {lengths.std():.0f}, '
        f'{time.time() - start:.0f}s',
        flush=True,
    )

    terminals = np.concatenate([np.eye(1, n, n - 1, dtype=bool)[0] for n in lengths])
    data = dict(
        observations=np.concatenate([e['observations'] for e in episodes]),
        actions=np.concatenate([e['actions'] for e in episodes]),
        rewards=np.concatenate([np.eye(1, n, n - 1, dtype=np.float32)[0] for n in lengths]),
        masks=(~terminals).astype(np.float32),
        terminals=terminals,
        qpos=np.concatenate([e['qpos'] for e in episodes]),
        qvel=np.concatenate([e['qvel'] for e in episodes]),
        observation_interval=np.asarray(1, dtype=np.int32),
    )
    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.save_path)), exist_ok=True)
    np.savez(FLAGS.save_path, **data)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    metadata = dict(
        episodes=[
            dict(episode_idx=i, start=int(s), end=int(s + n - 1), length=int(n), success=True, speed=e['speed'],
                 segments=[])
            for i, (s, n, e) in enumerate(zip(starts, lengths, episodes))
        ],
        config=dict(
            env_name=FLAGS.env_name,
            traj_noise=FLAGS.traj_noise,
            speed_profile=(lo, hi),
            p_mistake=FLAGS.p_mistake,
            seed=FLAGS.seed,
            attempts=attempts,
        ),
    )
    with open(FLAGS.save_path.removesuffix('.npz') + '-metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)


if __name__ == '__main__':
    app.run(main)
