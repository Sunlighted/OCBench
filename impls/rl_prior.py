"""Online RL fine-tuning on top of a frozen BC prior (a `hier_bc.py --flat=1` checkpoint, BC or flow BC).

The prior is never updated. A chunk-level critic Q(s, A) is trained with n-step TD (n = chunk length) on a 50/50 mix
of the prior's demonstrations and online data. `--method` picks how the policy uses the prior:
- `bon` (agents/bon.py, flow prior): execute the best of N prior samples, so the policy can only reweight actions
  inside the prior's support. `--num_samples=1` runs the prior itself through the same loop (control).
- `residual` (agents/residual.py): execute the prior's chunk plus a learned residual bounded by `--residual_scale`.

Reward: -`step_penalty` per control step and 0 on the step that reaches success (episodes terminate on success), so
the return is minus the discounted time to success: the critic prefers succeeding, and succeeding fast.
"""

import json
import os
import random
import time

import numpy as np
from absl import app, flags

from hier_common import cpu_env_name, env_family, evaluate_parallel, load_segment_file, subgoal_spec

# Heavy imports live in main(): evaluation workers are spawned processes that re-import this module.

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'block-single-task1-v0', 'Environment name.')
flags.DEFINE_string('prior_dir', '', 'Directory of the flow BC prior (params, action_normalizer.npz, flags.json).')
flags.DEFINE_integer('prior_step', 200000, 'Prior checkpoint step.')
flags.DEFINE_string('save_dir', '../exp/rl_prior', 'Save directory.')
flags.DEFINE_enum('method', 'bon', ['bon', 'residual'], 'Fine-tuning method.')
flags.DEFINE_string('run_name', '', 'Run name (default: prior name and method).')
flags.DEFINE_integer('num_samples', 32, 'N: prior samples per state (1: run the prior itself, no training).')
flags.DEFINE_integer('online_env_steps', 500000, 'Online interaction budget in control steps.')
flags.DEFINE_integer('collection_episodes', 50, 'Episodes collected per collection round.')
flags.DEFINE_float('utd', 1.0, 'Updates per collected chunk transition.')
flags.DEFINE_integer('warmup_updates', 5000, 'Critic updates on demonstrations before going online.')
flags.DEFINE_integer('batch_size', 256, 'Critic batch size.')
flags.DEFINE_float('online_ratio', 0.5, 'Fraction of each batch drawn from online data.')
flags.DEFINE_float('discount', 0.98, 'Discount per action chunk.')
flags.DEFINE_float('step_penalty', 0.04, 'Reward per control step before success.')
flags.DEFINE_integer('eval_every', 2, 'Evaluate every this many collection rounds.')
flags.DEFINE_integer('eval_episodes', 100, 'Evaluation episodes.')
flags.DEFINE_integer('final_eval_episodes', 500, 'Final evaluation episodes.')
flags.DEFINE_integer('eval_workers', 25, 'Evaluation worker processes.')
flags.DEFINE_float('residual_scale', 0.2, 'Residual bound in normalized action units (residual).')
flags.DEFINE_float('expl_noise', 0.5, 'Exploration noise std on the residual, in units of the bound (residual).')
flags.DEFINE_float('residual_reg', 0.03, 'L2 penalty on the residual (in units of the bound), summed over the chunk (residual).')


def main(_):
    import gymnasium
    import jax
    import ocbench  # noqa: F401

    from agents import agents
    from agents.bon import BoNAgent
    from agents.residual import ResidualAgent
    from envs.env_utils import ActionNormalizer, wrap_env
    from envs.online import collect_online_episodes, sample_online_batch
    from utils.datasets import Dataset, ReplayBuffer
    from utils.flax_utils import restore_agent
    from utils.log_utils import CsvLogger, get_flag_dict

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    method_tag = f'bon{FLAGS.num_samples}' if FLAGS.method == 'bon' else f'res{FLAGS.residual_scale:g}'
    run_name = FLAGS.run_name or f'{os.path.basename(FLAGS.prior_dir.rstrip("/"))}-{method_tag}'
    save_dir = os.path.join(FLAGS.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        json.dump(get_flag_dict(), f)

    # Prior and its action normalizer.
    with open(os.path.join(FLAGS.prior_dir, 'flags.json')) as f:
        prior_flags = json.load(f)
    chunk_len = prior_flags['action_chunk_length']
    env_name = cpu_env_name(FLAGS.env_name)
    probe_env = gymnasium.make(env_name)
    ob_dim = probe_env.observation_space.shape[-1]
    action_dim = probe_env.action_space.shape[-1]
    max_episode_steps = probe_env.spec.max_episode_steps
    spec = subgoal_spec(env_family(FLAGS.env_name), probe_env.unwrapped)
    del probe_env

    prior_config = dict(prior_flags['agent'])
    for key in ('actor_hidden_dims', 'betas'):
        if key in prior_config:
            prior_config[key] = tuple(prior_config[key])
    ex_obs = np.zeros((1, ob_dim), np.float32)
    ex_act = np.zeros((1, chunk_len * action_dim), np.float32)
    prior = agents[prior_config['agent_name']].create(FLAGS.seed, ex_obs, ex_act, prior_config)
    prior = restore_agent(prior, FLAGS.prior_dir, FLAGS.prior_step)
    norm = np.load(os.path.join(FLAGS.prior_dir, 'action_normalizer.npz'))
    action_normalizer = ActionNormalizer(norm['q_low'], norm['q_high'], float(norm['clip']), norm['gripper_dims'])
    normalizer_args = dict(
        q_low=action_normalizer.q_low,
        q_high=action_normalizer.q_high,
        clip=action_normalizer.clip,
        gripper_dims=np.flatnonzero(action_normalizer.gripper_dims),
    )

    # The prior's demonstrations (with the shaped reward) and the online buffer, sampled as chunk transitions.
    demos = load_segment_file(prior_flags['dataset_path'], spec, True, ob_dim, action_dim, prior_flags['max_episodes'])
    demos['actions'] = action_normalizer.normalize(demos['actions'])
    # Every demonstration is a successful episode that terminates on its last step (the segment loader does not keep
    # rewards or masks).
    demos['rewards'] = FLAGS.step_penalty * (demos['terminals'].astype(np.float32) - 1.0)
    demos['masks'] = 1.0 - demos['terminals'].astype(np.float32)
    demo_dataset = Dataset.create(**demos)

    class ShapedReward(gymnasium.RewardWrapper):
        def reward(self, reward):
            return FLAGS.step_penalty * (reward - 1.0)

    online_envs = [
        wrap_env(ShapedReward(gymnasium.make(env_name)), action_normalizer=action_normalizer)
        for _ in range(FLAGS.collection_episodes)
    ]
    example = dict(
        observations=np.zeros(ob_dim, np.float32),
        actions=np.zeros(action_dim, np.float32),
        rewards=0.0,
        terminals=0.0,
        masks=1.0,
        next_observations=np.zeros(ob_dim, np.float32),
    )
    replay_buffer = ReplayBuffer.create(example, size=FLAGS.online_env_steps + FLAGS.collection_episodes * max_episode_steps)
    for dataset in (demo_dataset, replay_buffer):
        dataset.action_chunk_length = chunk_len
        dataset.n_step = chunk_len
        dataset.discount = FLAGS.discount ** (1.0 / chunk_len)

    act_low = np.tile(action_normalizer.action_low, chunk_len).astype(np.float32)
    act_high = np.tile(action_normalizer.action_high, chunk_len).astype(np.float32)
    critic_config = dict(num_qs=2, critic_hidden_dims=(512, 512, 512, 512), lr=3e-4, tau=0.005)
    if FLAGS.method == 'bon':
        agent = BoNAgent.create(
            FLAGS.seed, prior, ex_obs, ex_act, act_low, act_high, dict(num_samples=FLAGS.num_samples, **critic_config)
        )
    else:
        agent = ResidualAgent.create(
            FLAGS.seed, prior, ex_obs, ex_act, act_low, act_high,
            dict(
                actor_hidden_dims=(512, 512, 512),
                scale=FLAGS.residual_scale,
                expl_noise=FLAGS.expl_noise,
                residual_reg=FLAGS.residual_reg,
                target_noise=0.2,
                target_noise_clip=0.5,
                **critic_config,
            ),
        )
    trains = FLAGS.method == 'residual' or FLAGS.num_samples > 1

    eval_rng = [jax.random.PRNGKey(FLAGS.seed + 1)]
    eval_cfg = dict(max_episode_steps=max_episode_steps, max_subgoal_steps=0, done_away_dist=0.1, done_contact=0.5,
                    flat=True)

    def evaluate(num_episodes):
        def sample_fn(obs):
            eval_rng[0], key = jax.random.split(eval_rng[0])
            if FLAGS.method == 'residual':
                return agent.sample_actions(obs, key, explore=False)
            return agent.sample_actions(obs, key)

        metrics, _ = evaluate_parallel(
            sample_fn, env_name, normalizer_args, chunk_len, action_dim, num_episodes,
            min(FLAGS.eval_workers, num_episodes), eval_cfg, FLAGS.seed,
        )
        return {k: metrics[k] for k in ('success', 'success_se', 'episode_length', 'success_length_median')}

    def train(num_updates, online):
        nonlocal agent
        info = {}
        for _ in range(num_updates):
            if online:
                batch = sample_online_batch(demo_dataset, replay_buffer, FLAGS.batch_size, FLAGS.online_ratio)
            else:
                batch = demo_dataset.sample(FLAGS.batch_size)
            batch = {k: batch[k] for k in ('observations', 'actions', 'rewards', 'discounts', 'next_observations')}
            agent, info = agent.update(batch) if FLAGS.method == 'bon' else agent.update(batch, train_actor=online)
        return {f'train/{k}': float(v) for k, v in info.items()}

    logger = CsvLogger(os.path.join(save_dir, 'log.csv'))
    start = time.time()

    def log(env_steps, row):
        row = dict(env_steps=env_steps, time=time.time() - start, **row)
        logger.log(row, step=env_steps)
        print(json.dumps({k: round(float(v), 4) for k, v in row.items()}), flush=True)

    row = {}
    if trains:
        row.update(train(FLAGS.warmup_updates, online=False))
    row.update({f'eval/{k}': v for k, v in evaluate(FLAGS.eval_episodes).items()})
    log(0, row)

    env_steps, round_idx = 0, 0
    collect_rng = jax.random.PRNGKey(FLAGS.seed + 2)
    while env_steps < FLAGS.online_env_steps:
        collect_rng, expl = collect_online_episodes(agent, online_envs, replay_buffer, collect_rng, {}, chunk_len)
        new_steps = int(expl['exploration/transitions'])
        env_steps += new_steps
        round_idx += 1
        row = {k: expl[k] for k in ('exploration/success', 'exploration/episode.length') if k in expl}
        if trains:
            row.update(train(int(FLAGS.utd * new_steps / chunk_len), online=True))
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
