"""Sanity check of the chunk-level TD machinery used by rl_prior.py, by policy evaluation of a fixed BC prior.

Rolls out the prior with full-chunk execution into the same ReplayBuffer and n-step sampling used for RL (sparse
reward, the time limit as a terminal failure), checks the sampled n-step rewards/discounts against values computed by
hand from the episodes, trains a critic ensemble by SARSA-style TD on the logged next chunks, and compares its
predictions on held-out episodes with the Monte-Carlo discounted returns (gamma^(steps to success), or 0 on failure).
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
flags.DEFINE_integer('train_rounds', 8, 'Collection rounds (one episode per env) for critic training.')
flags.DEFINE_integer('test_rounds', 2, 'Collection rounds for held-out evaluation.')
flags.DEFINE_integer('num_envs', 50, 'Environments per round.')
flags.DEFINE_integer('n_step', 200, 'TD steps.')
flags.DEFINE_float('discount', 0.999, 'Discount per control step.')
flags.DEFINE_integer('updates', 30000, 'Critic updates.')
flags.DEFINE_string('out', '', 'Output JSON.')


def main(_):
    import flax
    import gymnasium
    import jax
    import jax.numpy as jnp
    import ocbench  # noqa: F401
    import optax

    from agents import agents
    from agents.bon import QNetwork
    from envs.env_utils import ActionNormalizer, wrap_env
    from envs.online import collect_online_episodes
    from utils.datasets import ReplayBuffer
    from utils.flax_utils import ModuleDict, TrainState, restore_agent
    from utils.networks import ensemblize

    with open(os.path.join(FLAGS.prior_dir, 'flags.json')) as f:
        prior_flags = json.load(f)
    config = dict(prior_flags['agent'])
    for key in ('actor_hidden_dims', 'betas'):
        if key in config:
            config[key] = tuple(config[key])
    chunk_len = prior_flags['action_chunk_length']
    norm = np.load(os.path.join(FLAGS.prior_dir, 'action_normalizer.npz'))
    normalizer = ActionNormalizer(norm['q_low'], norm['q_high'], float(norm['clip']), norm['gripper_dims'])

    class TruncationAsTermination(gymnasium.Wrapper):
        def step(self, action):
            ob, reward, terminated, truncated, info = self.env.step(action)
            return ob, reward, terminated or truncated, False, info

    env_name = cpu_env_name(FLAGS.env_name)
    envs = [
        TruncationAsTermination(wrap_env(gymnasium.make(env_name), action_normalizer=normalizer))
        for _ in range(FLAGS.num_envs)
    ]
    ob_dim, action_dim = envs[0].observation_space.shape[-1], envs[0].action_space.shape[-1]
    prior = agents[config['agent_name']].create(
        0, np.zeros((1, ob_dim), np.float32), np.zeros((1, chunk_len * action_dim), np.float32), config
    )
    prior = restore_agent(prior, FLAGS.prior_dir, FLAGS.prior_step)

    class Sampler:
        def sample_actions(self, observations, seed):
            return prior.sample_actions(observations, seed=seed)

    example = dict(
        observations=np.zeros(ob_dim, np.float32), actions=np.zeros(action_dim, np.float32), rewards=0.0,
        terminals=0.0, masks=1.0, next_observations=np.zeros(ob_dim, np.float32),
    )
    max_steps = envs[0].spec.max_episode_steps or 1250
    buffers = {}
    rng = jax.random.PRNGKey(0)
    for split, rounds in (('train', FLAGS.train_rounds), ('test', FLAGS.test_rounds)):
        buf = ReplayBuffer.create(example, size=rounds * FLAGS.num_envs * max_steps + 10)
        for _ in range(rounds):
            # Seed each env's RNG differently; collect_online_episodes then resets without a seed (random inits).
            for env in envs:
                env.reset(seed=int(np.random.randint(0, 2**31)))
            rng, stats = collect_online_episodes(Sampler(), envs, buf, rng, {}, chunk_len)
        buf.action_chunk_length = chunk_len
        buf.n_step = FLAGS.n_step
        buf.discount = FLAGS.discount
        buffers[split] = buf

    def episodes(buf):
        """(start, length, success) of every episode in the buffer."""
        out = []
        for start, end in zip(buf.initial_locs, buf.terminal_locs):
            rewards = buf['rewards'][start : end + 1]
            out.append((int(start), int(end - start + 1), bool(rewards[-1] > 0)))
        return out

    # 1. n-step quantities against hand computation.
    train = buffers['train']
    eps = episodes(train)
    max_err = 0.0
    for start, length, success in eps[:50]:
        for t in range(0, length, chunk_len):
            idx = np.array([start + t])
            b = train.sample(1, idxs=idx, keys=('rewards', 'discounts', 'steps'))
            steps = min(FLAGS.n_step, length - t)
            reward = FLAGS.discount ** (length - 1 - t) if (success and length - t <= FLAGS.n_step) else 0.0
            # The episode always ends terminally (success or time limit), so no bootstrapping past its end.
            discount = 0.0 if length - t <= FLAGS.n_step else FLAGS.discount**steps
            max_err = max(max_err, abs(float(b['rewards'][0]) - reward), abs(float(b['discounts'][0]) - discount))
    print(json.dumps(dict(check='n_step', episodes=len(eps), success_rate=float(np.mean([e[2] for e in eps])),
                          max_abs_error=max_err)), flush=True)

    # 2. Policy evaluation with SARSA targets on the logged next chunks.
    critic_def = ensemblize(QNetwork, 2)(hidden_dims=(512, 512, 512, 512))
    module = ModuleDict(dict(critic=critic_def, target_critic=critic_def))
    ex = (np.zeros((1, ob_dim), np.float32), np.zeros((1, chunk_len * action_dim), np.float32))
    params = module.init(jax.random.PRNGKey(1), critic=ex, target_critic=ex)['params']
    params['modules_target_critic'] = params['modules_critic']
    net = TrainState.create(module, params, tx=optax.adam(3e-4))

    @jax.jit
    def update(net, batch):
        next_q = net.select('target_critic')(batch['next_observations'], batch['next_actions']).min(axis=0)
        target = batch['rewards'] + batch['discounts'] * next_q

        def loss_fn(p):
            qs = net.select('critic')(batch['observations'], batch['actions'], params=p)
            loss = ((qs - target[None]) ** 2).mean()
            return loss, dict(loss=loss, q=qs.mean())

        net, info = net.apply_loss_fn(loss_fn)
        net.params['modules_target_critic'] = jax.tree_util.tree_map(
            lambda a, b: 0.005 * a + 0.995 * b, net.params['modules_critic'], net.params['modules_target_critic']
        )
        return net, info

    keys = ('observations', 'actions', 'rewards', 'discounts', 'next_observations', 'steps')
    for i in range(FLAGS.updates):
        idx = train.get_random_idxs(256)
        b = train.sample(256, idxs=idx, keys=keys)
        next_idx = np.minimum(idx + b['steps'], train.size - 1)
        b['next_actions'] = train.get_action_chunks(next_idx)
        b = {k: v for k, v in b.items() if k != 'steps'}
        net, info = update(net, b)
        if i % 5000 == 0:
            print(json.dumps(dict(update=i, loss=float(info['loss']), q=float(info['q']))), flush=True)

    # 3. Held-out comparison with Monte-Carlo returns at every decision point.
    test = buffers['test']
    preds, returns, first_q, first_success = [], [], [], []
    for start, length, success in episodes(test):
        ts = np.arange(0, length, chunk_len)
        idx = start + ts
        obs = test.get_observations(idx)
        acts = test.get_action_chunks(idx)
        q = np.asarray(net.select('critic')(obs, acts).mean(axis=0))
        g = np.where(success, FLAGS.discount ** (length - 1 - ts), 0.0)
        preds += q.tolist()
        returns += g.tolist()
        first_q.append(float(q[0]))
        first_success.append(success)
    preds, returns = np.array(preds), np.array(returns)
    first_q, first_success = np.array(first_q), np.array(first_success)
    pos, neg = first_q[first_success], first_q[~first_success]
    auc = float((pos[:, None] > neg[None, :]).mean()) if len(pos) and len(neg) else float('nan')
    result = dict(
        check='policy_evaluation',
        test_episodes=len(first_q),
        test_success=float(first_success.mean()),
        corr_q_mc=float(np.corrcoef(preds, returns)[0, 1]),
        mae_q_mc=float(np.abs(preds - returns).mean()),
        mean_q=float(preds.mean()),
        mean_mc=float(returns.mean()),
        auc_first_q_success=auc,
        first_q_success=float(pos.mean()) if len(pos) else None,
        first_q_failure=float(neg.mean()) if len(neg) else None,
    )
    print(json.dumps(result), flush=True)
    if FLAGS.out:
        with open(FLAGS.out, 'w') as f:
            json.dump(result, f, indent=1)


if __name__ == '__main__':
    app.run(main)
