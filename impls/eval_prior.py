"""Evaluate a flat BC prior (bc, fbc or csfbc) with full-chunk execution, optionally with scaled flow noise.

`--noise_scale=0` integrates the flow from zero noise (a deterministic, mode-like policy); 1 is ordinary sampling.
`--exec_len` executes only the first steps of each chunk before replanning (receding horizon).
`--train_inits` resets with seeds 0..N-1 (the demonstrations' initial states) instead of random seeds.
"""

import json
import os

import numpy as np
from absl import app, flags

from hier_common import cpu_env_name

FLAGS = flags.FLAGS
flags.DEFINE_string('prior_dir', '', 'Prior directory.')
flags.DEFINE_integer('prior_step', 200000, 'Prior checkpoint step.')
flags.DEFINE_string('env_name', 'block-single-task1-v0', 'Environment name.')
flags.DEFINE_float('noise_scale', 1.0, 'Scale of the flow noise (flow priors).')
flags.DEFINE_bool('train_inits', False, 'Reset with seeds 0..N-1 instead of random seeds.')
flags.DEFINE_integer('num_episodes', 500, 'Episodes.')
flags.DEFINE_integer('num_envs', 50, 'Environments stepped in lockstep.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('exec_len', 0, 'Steps executed per chunk before replanning (0: the whole chunk).')


def main(_):
    import gymnasium
    import jax
    import jax.numpy as jnp
    import ocbench  # noqa: F401

    from agents import agents
    from envs.env_utils import ActionNormalizer, wrap_env
    from utils.flax_utils import restore_agent

    np.random.seed(FLAGS.seed)
    with open(os.path.join(FLAGS.prior_dir, 'flags.json')) as f:
        prior_flags = json.load(f)
    config = dict(prior_flags['agent'])
    for key in ('actor_hidden_dims', 'betas'):
        if key in config:
            config[key] = tuple(config[key])
    chunk_len = prior_flags['action_chunk_length']
    env_name = cpu_env_name(FLAGS.env_name)
    norm = np.load(os.path.join(FLAGS.prior_dir, 'action_normalizer.npz'))
    normalizer = ActionNormalizer(norm['q_low'], norm['q_high'], float(norm['clip']), norm['gripper_dims'])
    envs = [wrap_env(gymnasium.make(env_name), action_normalizer=normalizer) for _ in range(FLAGS.num_envs)]
    ob_dim, action_dim = envs[0].observation_space.shape[-1], envs[0].action_space.shape[-1]
    name = config['agent_name']
    prior = agents[name].create(0, np.zeros((1, ob_dim), np.float32), np.zeros((1, chunk_len * action_dim), np.float32), config)
    prior = restore_agent(prior, FLAGS.prior_dir, FLAGS.prior_step)
    low, high = np.tile(normalizer.action_low, chunk_len), np.tile(normalizer.action_high, chunk_len)

    @jax.jit
    def act(obs, key):
        if name == 'bc':
            return prior.sample_actions(obs, seed=key)
        noise = FLAGS.noise_scale * jax.random.normal(key, (obs.shape[0], chunk_len * action_dim))
        if name == 'csfbc':
            return prior.decode(obs, noise, jnp.zeros_like(obs), jnp.zeros(obs.shape[0]))
        return prior.compute_flow_actions(obs, noise)

    key = jax.random.PRNGKey(FLAGS.seed)
    successes, seed = [], 0
    while len(successes) < FLAGS.num_episodes:
        obs = []
        for env in envs:
            s = seed if FLAGS.train_inits else int(np.random.randint(0, 2**31))
            seed += 1
            obs.append(env.reset(seed=s)[0])
        obs = np.stack(obs).astype(np.float32)
        done = np.zeros(len(envs), bool)
        succ = np.zeros(len(envs), bool)
        while not done.all():
            key, sub = jax.random.split(key)
            chunks = np.clip(np.asarray(act(obs, sub)), low, high).reshape(len(envs), chunk_len, action_dim)
            for i, env in enumerate(envs):
                if done[i]:
                    continue
                for a in chunks[i][: FLAGS.exec_len or chunk_len]:
                    ob, _, terminated, truncated, info = env.step(a)
                    if terminated or truncated:
                        done[i], succ[i] = True, bool(info.get('success', False))
                        break
                obs[i] = ob
        successes += succ.tolist()
    s = np.array(successes[: FLAGS.num_episodes], float)
    print(json.dumps(dict(prior=os.path.basename(FLAGS.prior_dir.rstrip('/')), noise_scale=FLAGS.noise_scale,
                          exec_len=FLAGS.exec_len,
                          train_inits=FLAGS.train_inits, success=s.mean(), se=s.std() / np.sqrt(len(s)))), flush=True)


if __name__ == '__main__':
    app.run(main)
