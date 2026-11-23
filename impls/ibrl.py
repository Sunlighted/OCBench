"""IBRL (agents/ibrl.py) or ResFiT (agents/resfit.py, `--method=resfit`) with a frozen chunked BC prior (a `hier_bc.py --flat=1` checkpoint), at the control-step level.

The prior is executed open loop as in BC evaluation: it is queried every `chunk_len` steps and the step-t BC proposal
is entry t mod chunk_len of the current chunk. The agent sees the augmented observation [env observation, one-hot
phase within the chunk, BC proposal], which makes the BC proposal a function of the state. The prior's directory
supplies the demonstrations (preloaded into replay, as in IBRL) and the action normalizer.

Reward: 1 on the step that reaches success (episodes terminate on success), else 0. Training follows the IBRL
robomimic state configs: n-step TD (`n_step`), `num_critic_updates` critic updates and one actor update every
`update_every` env steps, an ensemble of 5 LayerNorm critics, a dropout actor, Gaussian exploration noise 0.1.
ResFiT follows the repo's TwoArmCoffee command: `random_steps` of base action plus uniform noise, then
`critic_warmup_steps` of critic-only updates, then 4 critic updates and 1 actor update per env step, 10 critics,
n-step 5, discount 0.995, residual scale 0.2, noise 0.025, actor learning rate 1e-6, half of each batch from the demos.
"""

import json
import os
import random
import time

import numpy as np
from absl import app, flags

from hier_common import cpu_env_name, env_family, load_segment_file, subgoal_spec

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'block-single-task1-v0', 'Environment name.')
flags.DEFINE_string('prior_dir', '', 'Directory of the BC prior (params, action_normalizer.npz, flags.json).')
flags.DEFINE_integer('prior_step', 200000, 'Prior checkpoint step.')
flags.DEFINE_string('save_dir', '../exp/ibrl', 'Save directory.')
flags.DEFINE_string('run_name', '', 'Run name (default: prior name and method).')
flags.DEFINE_enum('method', 'ibrl', ['ibrl', 'resfit'], 'Fine-tuning method.')
flags.DEFINE_float('action_scale', 0.2, 'Residual bound in normalized action units (resfit).')
flags.DEFINE_float('residual_reg', 0.0, 'L2 penalty on the residual in units of its bound, summed over dims (resfit).')
flags.DEFINE_bool('gripper_residual', True, 'Let the residual act on the gripper dimension (resfit).')
flags.DEFINE_float('actor_lr', 1e-6, 'Actor learning rate (resfit).')
flags.DEFINE_integer('random_steps', 10000, 'Initial env steps with uniform residual noise and no updates (resfit).')
flags.DEFINE_float('random_scale', 0.2, 'Uniform residual noise bound during the random steps (resfit).')
flags.DEFINE_integer('critic_warmup_steps', 10000, 'Env steps of critic-only updates after the random steps (resfit).')
flags.DEFINE_integer('online_env_steps', 500000, 'Online interaction budget in control steps.')
flags.DEFINE_integer('collection_episodes', 25, 'Episodes collected (in lockstep) per collection round.')
flags.DEFINE_integer('update_every', 2, 'Env steps per update group.')
flags.DEFINE_integer('num_critic_updates', 5, 'Critic updates per update group (followed by one actor update).')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('online_ratio', -1.0, 'Fraction of each batch drawn from online data (-1: uniform over demos and '
                   'online data, as with IBRL preloading).')
flags.DEFINE_float('discount', 0.995, 'Discount per control step.')
flags.DEFINE_integer('n_step', 3, 'TD steps.')
flags.DEFINE_float('beta', 10.0, 'Softmax temperature of the actor/BC choice on Q.')
flags.DEFINE_float('expl_noise', 0.1, 'Exploration noise std on the actor output (in [-1, 1]).')
flags.DEFINE_float('dropout', 0.5, 'Actor dropout.')
flags.DEFINE_integer('num_qs', 5, 'Critic ensemble size.')
flags.DEFINE_integer('hidden_dim', 1024, 'Actor and critic width (3 hidden layers).')
flags.DEFINE_float('lr', 1e-4, 'Learning rate.')
flags.DEFINE_float('actor_box', 1.5, 'Bound of the actor output in normalized action units (the normalizer clips '
                   'at 5, which would make the 0.1 exploration noise, applied to the tanh output, half the action range).')
flags.DEFINE_float('q_max', 1.0, 'Upper clip of the TD target (returns are at most 1).')
flags.DEFINE_integer('pretrain_groups', 0, 'Update groups on demonstrations before going online, with the actor '
                     'regressed onto the BC proposal (0: none, as in IBRL).')
flags.DEFINE_integer('eval_every', 4, 'Evaluate every this many collection rounds.')
flags.DEFINE_integer('eval_episodes', 100, 'Evaluation episodes.')
flags.DEFINE_integer('final_eval_episodes', 500, 'Final evaluation episodes.')
flags.DEFINE_integer('eval_envs', 50, 'Evaluation environments stepped in lockstep.')


# ResFiT settings (from the repo's TwoArmCoffee command and residual_td3.py) for flags left at their IBRL defaults.
RESFIT_DEFAULTS = dict(
    num_qs=10, n_step=5, discount=0.995, expl_noise=0.025, update_every=1, num_critic_updates=4, online_ratio=0.5,
    lr=1e-4, q_max=0.0, collection_episodes=10,
)


def main(_):
    import gymnasium
    import jax
    import ocbench  # noqa: F401

    from agents import agents
    from agents.ibrl import IBRLAgent
    from agents.resfit import ResFiTAgent
    from envs.env_utils import ActionNormalizer, wrap_env
    from utils.datasets import ReplayBuffer
    from utils.flax_utils import restore_agent
    from utils.log_utils import CsvLogger, get_flag_dict

    if FLAGS.method == 'resfit':
        for name, value in RESFIT_DEFAULTS.items():
            if not FLAGS[name].present:
                FLAGS[name].value = value
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
    spec = subgoal_spec(env_family(FLAGS.env_name), probe_env.unwrapped)
    del probe_env

    prior_config = dict(prior_flags['agent'])
    for key in ('actor_hidden_dims', 'betas'):
        if key in prior_config:
            prior_config[key] = tuple(prior_config[key])
    prior = agents[prior_config['agent_name']].create(
        FLAGS.seed, np.zeros((1, ob_dim), np.float32), np.zeros((1, chunk_len * action_dim), np.float32), prior_config
    )
    prior = restore_agent(prior, FLAGS.prior_dir, FLAGS.prior_step)
    norm = np.load(os.path.join(FLAGS.prior_dir, 'action_normalizer.npz'))
    action_normalizer = ActionNormalizer(norm['q_low'], norm['q_high'], float(norm['clip']), norm['gripper_dims'])
    act_low, act_high = action_normalizer.action_low, action_normalizer.action_high
    actor_low, actor_high = np.maximum(act_low, -FLAGS.actor_box), np.minimum(act_high, FLAGS.actor_box)
    prior_rng = [jax.random.PRNGKey(FLAGS.seed + 3)]

    def plan(observations):
        """Prior chunks [n, chunk_len, action_dim] for a padded batch of env observations."""
        prior_rng[0], key = jax.random.split(prior_rng[0])
        chunks = np.asarray(prior.sample_actions(observations, seed=key)).reshape(len(observations), chunk_len, -1)
        return np.clip(chunks, act_low, act_high)

    phases = np.eye(chunk_len, dtype=np.float32)

    def augment(ob, chunk, step):
        return np.concatenate([ob, phases[step % chunk_len], chunk[step % chunk_len]]).astype(np.float32)

    aug_dim = ob_dim + chunk_len + action_dim
    example = dict(
        observations=np.zeros(aug_dim, np.float32),
        actions=np.zeros(action_dim, np.float32),
        rewards=0.0,
        terminals=0.0,
        masks=1.0,
        next_observations=np.zeros(aug_dim, np.float32),
    )

    # Demonstrations, with the BC proposals the prior makes along them when run open loop from each episode start.
    demos = load_segment_file(prior_flags['dataset_path'], spec, True, ob_dim, action_dim, prior_flags['max_episodes'])
    demo_actions = action_normalizer.normalize(demos['actions'])
    ends = np.flatnonzero(demos['terminals']) + 1
    demo_buffer = ReplayBuffer.create(example, size=len(demo_actions))
    start = 0
    for episode, end in enumerate(ends):
        obs = demos['observations'][start + episode : end + episode + 1, :ob_dim]
        anchors = obs[::chunk_len]
        chunks = np.concatenate([plan(anchors[i : i + 256]) for i in range(0, len(anchors), 256)])
        aug = np.stack([augment(obs[t], chunks[t // chunk_len], t) for t in range(len(obs))])
        for t in range(len(obs) - 1):
            last = t == len(obs) - 2
            demo_buffer.add_transition(dict(
                observations=aug[t], actions=demo_actions[start + t], rewards=float(last), terminals=float(last),
                masks=float(not last), next_observations=aug[t + 1],
            ))
        start = end
    demo_buffer.update_episode_locs()

    replay_buffer = ReplayBuffer.create(example, size=FLAGS.online_env_steps + 2 * FLAGS.collection_episodes * 2000)
    for dataset in (demo_buffer, replay_buffer):
        dataset.n_step = FLAGS.n_step
        dataset.discount = FLAGS.discount

    ex_aug, ex_act = np.zeros((1, aug_dim), np.float32), np.zeros((1, action_dim), np.float32)
    if FLAGS.method == 'resfit':
        # Standardize the env observation by the demonstration statistics (std floored at 0.1, as in ResFiT); the
        # phase and base-action entries are already in unit range.
        demo_obs = demo_buffer['observations'][: demo_buffer.size]
        obs_mean, obs_std = np.zeros(aug_dim, np.float32), np.ones(aug_dim, np.float32)
        obs_mean[:ob_dim] = demo_obs[:, :ob_dim].mean(0)
        obs_std[:ob_dim] = np.maximum(demo_obs[:, :ob_dim].std(0), 0.1)
        agent = ResFiTAgent.create(
            FLAGS.seed, ex_aug, ex_act, act_low, act_high, obs_mean, obs_std,
            dict(
                num_qs=FLAGS.num_qs, num_min_qs=2, critic_hidden_dims=(FLAGS.hidden_dim,) * 2,
                actor_hidden_dims=(FLAGS.hidden_dim,) * 2, action_scale=FLAGS.action_scale, critic_lr=FLAGS.lr,
                actor_lr=FLAGS.actor_lr, tau=0.005, expl_noise=FLAGS.expl_noise, noise_clip=0.3,
                random_scale=FLAGS.random_scale, q_max=FLAGS.q_max, residual_reg=FLAGS.residual_reg,
                residual_mask=tuple(
                    0.0 if (g and not FLAGS.gripper_residual) else 1.0 for g in action_normalizer.gripper_dims
                ),
            ),
        )
    else:
        agent = IBRLAgent.create(
            FLAGS.seed, ex_aug, ex_act, actor_low, actor_high,
            dict(
                num_qs=FLAGS.num_qs, num_min_qs=2, critic_hidden_dims=(FLAGS.hidden_dim,) * 3,
                actor_hidden_dims=(FLAGS.hidden_dim,) * 3, dropout=FLAGS.dropout, lr=FLAGS.lr, tau=0.01,
                beta=FLAGS.beta, expl_noise=FLAGS.expl_noise, noise_std=0.1, noise_clip=0.3, q_max=FLAGS.q_max,
            ),
        )
    # Per-step statistic of the second output of `sample_actions`: the BC choice (IBRL) or the |residual| (ResFiT).
    aux_name = 'residual_abs' if FLAGS.method == 'resfit' else 'bc_fraction'

    def make_envs(n):
        return [wrap_env(gymnasium.make(env_name), action_normalizer=action_normalizer) for _ in range(n)]

    online_envs = make_envs(FLAGS.collection_episodes)
    eval_envs = make_envs(FLAGS.eval_envs)

    def run_episodes(envs, rng, explore, buffer=None, random_actions=False):
        """One episode per env in lockstep from random initial states; returns the rng and episode statistics."""
        n = len(envs)
        obs = np.stack([env.reset(seed=int(np.random.randint(0, 2**31)))[0] for env in envs]).astype(np.float32)
        chunks = plan(obs)
        steps = np.zeros(n, int)
        done = np.zeros(n, bool)
        success = np.zeros(n, bool)
        bc_chosen = np.zeros(n)
        trajs = [[] for _ in range(n)]
        aug = np.stack([augment(obs[i], chunks[i], 0) for i in range(n)])
        while not done.all():
            rng, key = jax.random.split(rng)
            act_kwargs = dict(random=True) if random_actions else {}
            actions, chose_bc = agent.sample_actions(aug, key, explore=explore, **act_kwargs)
            actions, chose_bc = np.asarray(actions), np.asarray(chose_bc)
            active = np.flatnonzero(~done)
            for i in active:
                ob, _, terminated, truncated, info = envs[i].step(actions[i].copy())
                obs[i] = ob
                steps[i] += 1
                bc_chosen[i] += chose_bc[i]
                success[i] = bool(info.get('success', False))
                done[i] = terminated or truncated
                trajs[i].append(dict(observations=aug[i], actions=actions[i], rewards=float(terminated and success[i]),
                                     terminals=float(done[i]), masks=float(not terminated)))
            replan = active[(steps[active] % chunk_len == 0) & ~done[active]]
            if len(replan):
                padded = np.concatenate([obs[replan], np.repeat(obs[replan[:1]], n - len(replan), axis=0)])
                chunks[replan] = plan(padded)[: len(replan)]
            for i in active:
                aug[i] = augment(obs[i], chunks[i], steps[i])
                trajs[i][-1]['next_observations'] = aug[i].copy()
        if buffer is not None:
            for traj in trajs:
                for transition in traj:
                    buffer.add_transition(transition)
            buffer.update_episode_locs()
        stats = dict(success=success.mean(), episode_length=steps.mean(), **{aux_name: bc_chosen.sum() / steps.sum()})
        return rng, stats, int(steps.sum())

    eval_rng = [jax.random.PRNGKey(FLAGS.seed + 1)]

    def evaluate(num_episodes):
        results = []
        for _ in range(-(-num_episodes // len(eval_envs))):
            eval_rng[0], stats, _ = run_episodes(eval_envs, eval_rng[0], explore=False)
            results.append(stats)
        out = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
        out['success_se'] = float(np.sqrt(out['success'] * (1 - out['success']) / (len(results) * len(eval_envs))))
        return out

    batch_keys = ('observations', 'actions', 'rewards', 'discounts', 'next_observations')

    def sample(num):
        if FLAGS.online_ratio < 0:
            ratio = replay_buffer.size / (replay_buffer.size + demo_buffer.size)
        else:
            ratio = FLAGS.online_ratio
        num_online = int(round(num * ratio))
        parts = [demo_buffer.sample(num - num_online, keys=batch_keys)]
        if num_online:
            parts.append(replay_buffer.sample(num_online, keys=batch_keys))
        batch = {k: np.concatenate([p[k] for p in parts]) for k in batch_keys}
        perm = np.random.permutation(num)
        return {k: v[perm] for k, v in batch.items()}

    def train(num_groups, bc=False, train_actor=True):
        nonlocal agent
        info = {}
        chunk = 100  # Groups per device call; batches are sampled together and reshaped.
        k, b = FLAGS.num_critic_updates, FLAGS.batch_size
        for i in range(0, num_groups, chunk):
            g = min(chunk, num_groups - i)
            batch = sample(g * k * b)
            batch = {key: v.reshape(g, k, b, *v.shape[1:]) for key, v in batch.items()}
            if FLAGS.method == 'resfit':
                agent, info = agent.update_many(batch, train_actor=train_actor)
            else:
                agent, info = agent.update_many(batch, bc=bc)
        return {f'train/{key}': float(v) for key, v in info.items()}

    logger = CsvLogger(os.path.join(save_dir, 'log.csv'))
    start_time = time.time()

    def log(env_steps, row):
        row = dict(env_steps=env_steps, time=time.time() - start_time, **row)
        logger.log(row, step=env_steps)
        print(json.dumps({k: round(float(v), 4) for k, v in row.items()}), flush=True)

    row = train(FLAGS.pretrain_groups, bc=True) if FLAGS.pretrain_groups else {}
    log(0, row | {f'eval/{k}': v for k, v in evaluate(FLAGS.eval_episodes).items()} | {'demos/transitions': demo_buffer.size})

    env_steps, round_idx = 0, 0
    collect_rng = jax.random.PRNGKey(FLAGS.seed + 2)
    while env_steps < FLAGS.online_env_steps:
        random_round = FLAGS.method == 'resfit' and env_steps < FLAGS.random_steps
        collect_rng, stats, new_steps = run_episodes(
            online_envs, collect_rng, explore=True, buffer=replay_buffer, random_actions=random_round
        )
        env_steps += new_steps
        round_idx += 1
        row = {f'exploration/{k}': v for k, v in stats.items()}
        if not random_round:
            # Steps collected before this round: the critic must have trained for `critic_warmup_steps` already.
            train_actor = (
                FLAGS.method != 'resfit' or env_steps - new_steps >= FLAGS.random_steps + FLAGS.critic_warmup_steps
            )
            row.update(train(new_steps // FLAGS.update_every, train_actor=train_actor))
        if round_idx % FLAGS.eval_every == 0 or env_steps >= FLAGS.online_env_steps:
            row.update({f'eval/{k}': v for k, v in evaluate(FLAGS.eval_episodes).items()})
        log(env_steps, row)

    final = evaluate(FLAGS.final_eval_episodes)
    final['env_steps'] = env_steps
    with open(os.path.join(save_dir, 'final_eval.json'), 'w') as f:
        json.dump(final, f, indent=1)
    print(f'final: {json.dumps(final)}', flush=True)


if __name__ == '__main__':
    app.run(main)
