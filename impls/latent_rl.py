"""Online latent-space RL on a frozen flow BC prior: DSRL (`--method=dsrl`, a flow BC prior from agents/fbc.py) or TMRL
(`--method=tmrl`, a context-smoothed prior from agents/csfbc.py).

SAC (agents/latent_sac.py) picks, per decision, the flow noise (one action_dim vector repeated over the chunk, scaled by
`--noise_bound`) and, for TMRL, the context noise (scaled by `--context_noise_bound`) and the context diffusion time (in
[0, `--timestep_max`]). The prior decodes them into a chunk that is executed in full. Environments run in lockstep and
reset independently; every decision is one transition with the chunk's discounted sparse reward (1 on success). No
demonstrations are used online (TMRL's default). With `--truncation_terminal`, the time limit ends an episode as a
failure (no bootstrapping).
"""

import json
import os
import random
import time

import numpy as np
from absl import app, flags

from hier_common import cpu_env_name

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'block-single-task1-v0', 'Environment name.')
flags.DEFINE_enum('method', 'tmrl', ['dsrl', 'tmrl'], 'Latent RL method.')
flags.DEFINE_string('prior_dir', '', 'Prior directory (fbc for dsrl, csfbc for tmrl).')
flags.DEFINE_integer('prior_step', 200000, 'Prior checkpoint step.')
flags.DEFINE_string('save_dir', '../exp/latent_rl', 'Save directory.')
flags.DEFINE_string('run_name', '', 'Run name.')
flags.DEFINE_integer('online_env_steps', 1000000, 'Online interaction budget in control steps.')
flags.DEFINE_integer('num_envs', 25, 'Environments stepped in lockstep.')
flags.DEFINE_float('utd', 4.0, 'Updates per collected decision (chunk) transition.')
flags.DEFINE_integer('learning_starts', 2000, 'Transitions collected with uniform random latents before training.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('discount', 0.9998, 'Discount per control step (0.995 per 25-step chunk).')
flags.DEFINE_bool('truncation_terminal', True, 'Treat the time limit as a terminal failure.')
flags.DEFINE_float('noise_bound', 1.0, 'Bound of the flow noise.')
flags.DEFINE_float('context_noise_bound', 1.0, 'Bound of the context noise (tmrl).')
flags.DEFINE_float('timestep_max', 1.0, 'Maximum context diffusion time (tmrl).')
flags.DEFINE_bool('backup_entropy', True, 'Include the entropy bonus in the TD target.')
flags.DEFINE_integer('eval_every', 50000, 'Evaluate every this many env steps.')
flags.DEFINE_integer('eval_episodes', 100, 'Evaluation episodes.')
flags.DEFINE_integer('final_eval_episodes', 500, 'Final evaluation episodes.')
flags.DEFINE_integer('eval_envs', 50, 'Evaluation environments stepped in lockstep.')


def main(_):
    import gymnasium
    import jax
    import jax.numpy as jnp
    import ocbench  # noqa: F401

    from agents import agents
    from agents.latent_sac import LatentSACAgent
    from envs.env_utils import ActionNormalizer, wrap_env
    from utils.flax_utils import restore_agent
    from utils.log_utils import CsvLogger, get_flag_dict

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    run_name = FLAGS.run_name or f'{os.path.basename(FLAGS.prior_dir.rstrip("/"))}-{FLAGS.method}'
    save_dir = os.path.join(FLAGS.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        json.dump(get_flag_dict(), f)

    with open(os.path.join(FLAGS.prior_dir, 'flags.json')) as f:
        prior_flags = json.load(f)
    chunk_len = prior_flags['action_chunk_length']
    env_name = cpu_env_name(FLAGS.env_name)
    probe_env = gymnasium.make(env_name)
    ob_dim = probe_env.observation_space.shape[-1]
    action_dim = probe_env.action_space.shape[-1]
    del probe_env

    prior_config = dict(prior_flags['agent'])
    for key in ('actor_hidden_dims', 'betas'):
        if key in prior_config:
            prior_config[key] = tuple(prior_config[key])
    expected = 'csfbc' if FLAGS.method == 'tmrl' else 'fbc'
    assert prior_config['agent_name'] == expected, f'{FLAGS.method} needs a {expected} prior.'
    prior = agents[expected].create(
        FLAGS.seed, np.zeros((1, ob_dim), np.float32), np.zeros((1, chunk_len * action_dim), np.float32), prior_config
    )
    prior = restore_agent(prior, FLAGS.prior_dir, FLAGS.prior_step)
    norm = np.load(os.path.join(FLAGS.prior_dir, 'action_normalizer.npz'))
    normalizer = ActionNormalizer(norm['q_low'], norm['q_high'], float(norm['clip']), norm['gripper_dims'])
    act_low = np.tile(normalizer.action_low, chunk_len)
    act_high = np.tile(normalizer.action_high, chunk_len)

    segments = [('noise', action_dim)]
    if FLAGS.method == 'tmrl':
        segments += [('context', ob_dim), ('time', 1)]

    @jax.jit
    def decode(observations, u):
        """Chunks [n, chunk_len * action_dim] from latents u in [-1, 1]."""
        noise = jnp.tile(FLAGS.noise_bound * u[:, :action_dim], (1, chunk_len))
        if FLAGS.method == 'tmrl':
            context = FLAGS.context_noise_bound * u[:, action_dim : action_dim + ob_dim]
            t_context = FLAGS.timestep_max * (u[:, -1] + 1) / 2
            chunks = prior.decode(observations, noise, context, t_context)
        else:
            chunks = prior.compute_flow_actions(observations, noise)
        return jnp.clip(chunks, act_low, act_high)

    agent = LatentSACAgent.create(
        FLAGS.seed, np.zeros((1, ob_dim), np.float32), segments,
        dict(num_qs=5, hidden_dims=(256, 256, 256), lr=3e-4, tau=0.005, init_alpha=0.1,
             backup_entropy=FLAGS.backup_entropy),
    )
    latent_dim = sum(size for _, size in segments)

    def make_env():
        return wrap_env(gymnasium.make(env_name), action_normalizer=normalizer)

    # Replay of decision transitions.
    capacity = FLAGS.online_env_steps // chunk_len + 10 * FLAGS.num_envs
    buf = dict(
        observations=np.zeros((capacity, ob_dim), np.float32),
        latents=np.zeros((capacity, latent_dim), np.float32),
        rewards=np.zeros(capacity, np.float32),
        discounts=np.zeros(capacity, np.float32),
        next_observations=np.zeros((capacity, ob_dim), np.float32),
    )
    size = [0]

    def add(**transition):
        for k, v in transition.items():
            buf[k][size[0]] = v
        size[0] += 1

    def run_chunk(env, chunk):
        """Execute one chunk; returns the next observation, discounted reward, discount and done/success."""
        reward, disc, done, success = 0.0, 1.0, False, False
        ob = None
        for a in chunk.reshape(chunk_len, action_dim):
            ob, r, terminated, truncated, info = env.step(a)
            reward += disc * r
            disc *= FLAGS.discount
            success = bool(info.get('success', False))
            if terminated or truncated:
                done = True
                if terminated or FLAGS.truncation_terminal:
                    disc = 0.0
                break
        return ob, reward, disc, done, success

    rng = [jax.random.PRNGKey(FLAGS.seed + 1)]

    def next_key():
        rng[0], key = jax.random.split(rng[0])
        return key

    def evaluate(num_episodes, envs):
        successes, lengths = [], []
        while len(successes) < num_episodes:
            obs = np.stack([env.reset(seed=int(np.random.randint(0, 2**31)))[0] for env in envs]).astype(np.float32)
            done = np.zeros(len(envs), bool)
            steps = np.zeros(len(envs), int)
            succ = np.zeros(len(envs), bool)
            while not done.all():
                u = agent.sample_latents(obs, next_key(), deterministic=True)
                chunks = np.asarray(decode(obs, u))
                for i, env in enumerate(envs):
                    if done[i]:
                        continue
                    ob, _, _, d, s = run_chunk(env, chunks[i])
                    obs[i] = ob
                    steps[i] += chunk_len
                    done[i], succ[i] = d, s
            successes += succ.tolist()
            lengths += steps.tolist()
        successes = np.array(successes[:num_episodes], float)
        return dict(success=successes.mean(), success_se=successes.std() / np.sqrt(len(successes)),
                    episode_length=float(np.mean(lengths[:num_episodes])))

    logger = CsvLogger(os.path.join(save_dir, 'log.csv'))
    start = time.time()

    def log(env_steps, row):
        row = dict(env_steps=env_steps, time=time.time() - start, **row)
        logger.log(row, step=env_steps)
        print(json.dumps({k: round(float(v), 4) for k, v in row.items()}), flush=True)

    eval_envs = [make_env() for _ in range(FLAGS.eval_envs)]
    log(0, {f'eval/{k}': v for k, v in evaluate(FLAGS.eval_episodes, eval_envs).items()})

    envs = [make_env() for _ in range(FLAGS.num_envs)]
    obs = np.stack([env.reset(seed=int(np.random.randint(0, 2**31)))[0] for env in envs]).astype(np.float32)
    env_steps, next_eval, pending, info = 0, FLAGS.eval_every, 0.0, {}
    ep_success, ep_count = [], 0
    while env_steps < FLAGS.online_env_steps:
        if size[0] < FLAGS.learning_starts:
            u = np.random.uniform(-1, 1, (FLAGS.num_envs, latent_dim)).astype(np.float32)
        else:
            u = np.asarray(agent.sample_latents(obs, next_key()))
        chunks = np.asarray(decode(obs, u))
        for i, env in enumerate(envs):
            ob, reward, disc, done, success = run_chunk(env, chunks[i])
            add(observations=obs[i], latents=u[i], rewards=reward, discounts=disc, next_observations=ob)
            env_steps += chunk_len
            if done:
                ep_success.append(success)
                ep_count += 1
                ob, _ = env.reset(seed=int(np.random.randint(0, 2**31)))
            obs[i] = ob
        if size[0] >= FLAGS.learning_starts:
            pending += FLAGS.utd * FLAGS.num_envs
            n = int(pending)
            pending -= n
            if n:
                idx = np.random.randint(0, size[0], (n, FLAGS.batch_size))
                batches = {k: v[idx] for k, v in buf.items()}
                agent, info = agent.update_many(batches)
        if env_steps >= next_eval or env_steps >= FLAGS.online_env_steps:
            next_eval += FLAGS.eval_every
            row = {f'train/{k}': float(v) for k, v in info.items()}
            recent = ep_success[-100:]
            row['exploration/success'] = float(np.mean(recent)) if recent else 0.0
            row['exploration/episodes'] = ep_count
            row.update({f'eval/{k}': v for k, v in evaluate(FLAGS.eval_episodes, eval_envs).items()})
            log(env_steps, row)

    final = evaluate(FLAGS.final_eval_episodes, eval_envs)
    final['env_steps'] = env_steps
    with open(os.path.join(save_dir, 'final_eval.json'), 'w') as f:
        json.dump(final, f, indent=1)
    print(f'final: {json.dumps(final)}', flush=True)


if __name__ == '__main__':
    app.run(main)
