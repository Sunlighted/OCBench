"""Online RL from a small demonstration set, with or without a frozen BC prior (a `hier_bc.py --flat=1` checkpoint).

The prior's directory always supplies the demonstrations and the action normalizer; the prior is never updated. A chunk-level critic Q(s, A) is trained with n-step TD (n = chunk length) on a 50/50 mix
of the prior's demonstrations and online data. `--method` picks how the policy uses the prior:
- `bon` (agents/bon.py, flow prior): execute the best of N prior samples, so the policy can only reweight actions
  inside the prior's support. `--num_samples=1` runs the prior itself through the same loop (control).
- `residual` (agents/residual.py): execute the prior's chunk plus a learned residual bounded by `--residual_scale`.
- `rlpd` (agents/rlpd.py): RLPD over action chunks, with a policy trained from scratch (the prior network is unused);
  `--bc_steps` first trains the actor on the demonstrations by maximum likelihood (BC initialization).
- `td3bc` (agents/td3bc.py, deterministic BC prior): fine-tune the prior network itself with TD3+BC (`--bc_weight`),
  after a critic warmup on the demonstrations and `--actor_freeze_steps` of online data with the actor frozen.
  `--q_weight=0 --awr_temp=T` gives AWR instead.
`--self_imitation` appends every successful online episode to the demonstration set (whose half of each batch the
BC term regresses onto), so with `--method=td3bc --q_weight=0` the policy is iterated success-filtered BC.

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
flags.DEFINE_enum('method', 'bon', ['bon', 'residual', 'rlpd', 'td3bc'], 'Fine-tuning method.')
flags.DEFINE_string('run_name', '', 'Run name (default: prior name and method).')
flags.DEFINE_integer('num_samples', 32, 'N: prior samples per state (1: run the prior itself, no training).')
flags.DEFINE_integer('online_env_steps', 500000, 'Online interaction budget in control steps.')
flags.DEFINE_integer('collection_episodes', 50, 'Episodes collected per collection round.')
flags.DEFINE_float('utd', 1.0, 'Updates per collected chunk transition.')
flags.DEFINE_integer('warmup_updates', 5000, 'Critic updates on demonstrations before going online (bon, residual).')
flags.DEFINE_integer('batch_size', 256, 'Critic batch size.')
flags.DEFINE_float('online_ratio', 0.5, 'Fraction of each batch drawn from online data.')
flags.DEFINE_float('discount', 0.9992, 'Discount per control step (0.98 per 25 steps).')
flags.DEFINE_integer('exec_len', 0, 'Control steps executed per decision: the first exec_len steps of the prior chunk '
                     '(0: the whole chunk). The critic and the residual act on chunks of this length.')
flags.DEFINE_float('step_penalty', 0.04, 'Reward per control step before success.')
flags.DEFINE_integer('eval_every', 2, 'Evaluate every this many collection rounds.')
flags.DEFINE_integer('eval_episodes', 100, 'Evaluation episodes.')
flags.DEFINE_integer('final_eval_episodes', 500, 'Final evaluation episodes.')
flags.DEFINE_integer('eval_workers', 25, 'Evaluation worker processes.')
flags.DEFINE_float('residual_scale', 0.2, 'Residual bound in normalized action units (residual).')
flags.DEFINE_float('expl_noise', 0.5, 'Exploration noise std on the residual, in units of the bound (residual).')
flags.DEFINE_integer('num_qs', 0, 'Critic ensemble size (0: 2 for bon/residual, 10 for rlpd).')
flags.DEFINE_string('actor_dims', '512,512,512', 'RLPD actor hidden dims (an even number of equal widths uses a residual MLP).')
flags.DEFINE_integer('bc_steps', 0, 'Actor BC (maximum likelihood) steps on the demonstrations before going online (rlpd).')
flags.DEFINE_float('bc_weight', 1.0, 'Weight of the squared error to demonstration chunks in the actor loss (td3bc).')
flags.DEFINE_float('actor_lr', 3e-5, 'Actor learning rate (td3bc).')
flags.DEFINE_bool('self_imitation', False, 'Append successful online episodes to the demonstrations.')
flags.DEFINE_float('awr_temp', 0.0, 'AWR temperature on batch-normalized advantages (td3bc; 0: off).')
flags.DEFINE_float('q_weight', 1.0, 'Weight of the normalized Q term in the actor loss (td3bc; 0: BC only).')
flags.DEFINE_float('action_noise', 0.05, 'Exploration noise std in normalized action units (td3bc).')
flags.DEFINE_integer('actor_freeze_steps', 50000, 'Online env steps before actor updates start (td3bc).')
flags.DEFINE_float('residual_reg', 0.03, 'L2 penalty on the residual (in units of the bound), summed over the chunk (residual).')


def main(_):
    import gymnasium
    import jax
    import ocbench  # noqa: F401

    from agents import agents
    from agents.bon import BoNAgent, TruncatedPrior
    from agents.residual import ResidualAgent
    from agents.rlpd import RLPDAgent
    from agents.td3bc import TD3BCAgent
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

    successes = []  # Successful online episodes as (observations [T+1], normalized actions [T]).

    class SuccessRecorder(gymnasium.Wrapper):
        def reset(self, **kwargs):
            ob, info = self.env.reset(**kwargs)
            self.obs, self.acts = [ob], []
            return ob, info

        def step(self, action):
            ob, reward, terminated, truncated, info = self.env.step(action)
            self.obs.append(ob)
            self.acts.append(np.asarray(action, np.float32))
            if terminated and info.get('success', False):
                successes.append((np.stack(self.obs), np.stack(self.acts)))
            return ob, reward, terminated, truncated, info

    def with_successes(base, episodes):
        """The demonstration fields with `episodes` appended (each as one terminating successful episode)."""
        fields = {k: [v] for k, v in base.items() if k != 'observation_interval'}
        for obs, acts in episodes:
            terminals = np.zeros(len(acts), bool)
            terminals[-1] = True
            fields['observations'].append(obs.astype(np.float32))
            fields['actions'].append(acts)
            fields['terminals'].append(terminals)
            fields['rewards'].append(FLAGS.step_penalty * (terminals.astype(np.float32) - 1.0))
            fields['masks'].append(1.0 - terminals.astype(np.float32))
        out = {k: np.concatenate(v) for k, v in fields.items()}
        out['observation_interval'] = base['observation_interval']
        return out

    online_envs = [
        wrap_env(ShapedReward(gymnasium.make(env_name)), action_normalizer=action_normalizer)
        for _ in range(FLAGS.collection_episodes)
    ]
    if FLAGS.self_imitation:
        online_envs = [SuccessRecorder(env) for env in online_envs]
    example = dict(
        observations=np.zeros(ob_dim, np.float32),
        actions=np.zeros(action_dim, np.float32),
        rewards=0.0,
        terminals=0.0,
        masks=1.0,
        next_observations=np.zeros(ob_dim, np.float32),
    )
    replay_buffer = ReplayBuffer.create(example, size=FLAGS.online_env_steps + FLAGS.collection_episodes * max_episode_steps)
    exec_len = FLAGS.exec_len or chunk_len
    for dataset in (demo_dataset, replay_buffer):
        dataset.action_chunk_length = exec_len
        dataset.n_step = exec_len
        dataset.discount = FLAGS.discount

    act_low = np.tile(action_normalizer.action_low, exec_len).astype(np.float32)
    act_high = np.tile(action_normalizer.action_high, exec_len).astype(np.float32)
    bc_prior = prior
    prior = TruncatedPrior(prior, exec_len * action_dim)
    ex_act = np.zeros((1, exec_len * action_dim), np.float32)
    num_qs = FLAGS.num_qs or (10 if FLAGS.method in ('rlpd', 'td3bc') else 2)
    critic_config = dict(num_qs=num_qs, critic_hidden_dims=(512, 512, 512, 512), lr=3e-4, tau=0.005)
    if FLAGS.method == 'bon':
        agent = BoNAgent.create(
            FLAGS.seed, prior, ex_obs, ex_act, act_low, act_high, dict(num_samples=FLAGS.num_samples, **critic_config)
        )
    elif FLAGS.method == 'td3bc':
        agent = TD3BCAgent.create(
            FLAGS.seed, bc_prior, ex_obs, ex_act, act_low, act_high,
            dict(
                num_min_qs=2,
                q_max=0.0,
                bc_weight=FLAGS.bc_weight,
                q_weight=FLAGS.q_weight,
                awr_temp=FLAGS.awr_temp,
                awr_max_weight=20.0,
                actor_lr=FLAGS.actor_lr,
                expl_noise=FLAGS.action_noise,
                target_noise=0.1,
                target_noise_clip=0.3,
                num_demo=FLAGS.batch_size - int(FLAGS.batch_size * FLAGS.online_ratio),
                **critic_config,
            ),
        )
    elif FLAGS.method == 'rlpd':
        actor_dims = tuple(int(d) for d in FLAGS.actor_dims.split(','))
        agent = RLPDAgent.create(
            FLAGS.seed, ex_obs, ex_act, act_low, act_high,
            dict(actor_hidden_dims=actor_dims, actor_resmlp=len(actor_dims) % 2 == 0 and len(set(actor_dims)) == 1, num_min_qs=2, backup_entropy=False, init_alpha=0.01, **critic_config),
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
    trains = FLAGS.method != 'bon' or FLAGS.num_samples > 1

    eval_rng = [jax.random.PRNGKey(FLAGS.seed + 1)]
    eval_cfg = dict(max_episode_steps=max_episode_steps, max_subgoal_steps=0, done_away_dist=0.1, done_contact=0.5,
                    flat=True)

    def evaluate(num_episodes):
        def sample_fn(obs):
            eval_rng[0], key = jax.random.split(eval_rng[0])
            if FLAGS.method != 'bon':
                return agent.sample_actions(obs, key, explore=False)
            return agent.sample_actions(obs, key)

        metrics, _ = evaluate_parallel(
            sample_fn, env_name, normalizer_args, exec_len, action_dim, num_episodes,
            min(FLAGS.eval_workers, num_episodes), eval_cfg, FLAGS.seed,
        )
        return {k: metrics[k] for k in ('success', 'success_se', 'episode_length', 'success_length_median')}

    batch_keys = ('observations', 'actions', 'rewards', 'discounts', 'next_observations')

    def sample_batch(online):
        if online:
            batch = sample_online_batch(demo_dataset, replay_buffer, FLAGS.batch_size, FLAGS.online_ratio)
        else:
            batch = demo_dataset.sample(FLAGS.batch_size)
        return {k: batch[k] for k in batch_keys}

    def train(num_updates, online, bc=False, actor_on=1.0):
        nonlocal agent
        info = {}
        if FLAGS.method in ('rlpd', 'td3bc'):
            # Scan over groups of stacked batches to amortize dispatch at high UTD.
            group = 50
            for i in range(0, num_updates, group):
                batches = [sample_batch(online) for _ in range(min(group, num_updates - i))]
                batches = {k: np.stack([b[k] for b in batches]) for k in batch_keys}
                if FLAGS.method == 'td3bc':
                    agent, info = agent.update_many(batches, np.float32(actor_on))
                elif bc:
                    agent, info = agent.bc_update_many(batches)
                else:
                    agent, info = agent.update_many(batches)
        else:
            for _ in range(num_updates):
                batch = sample_batch(online)
                agent, info = agent.update(batch) if FLAGS.method == 'bon' else agent.update(batch, train_actor=online)
        return {f'train/{k}': float(v) for k, v in info.items()}

    logger = CsvLogger(os.path.join(save_dir, 'log.csv'))
    start = time.time()

    def log(env_steps, row):
        row = dict(env_steps=env_steps, time=time.time() - start, **row)
        logger.log(row, step=env_steps)
        print(json.dumps({k: round(float(v), 4) for k, v in row.items()}), flush=True)

    row = {}
    if FLAGS.method == 'rlpd':
        row.update(train(FLAGS.bc_steps, online=False, bc=True))
    elif FLAGS.method == 'td3bc':
        row.update(train(FLAGS.warmup_updates, online=False, actor_on=0.0))
    elif trains:
        row.update(train(FLAGS.warmup_updates, online=False))
    row.update({f'eval/{k}': v for k, v in evaluate(FLAGS.eval_episodes).items()})
    log(0, row)

    env_steps, round_idx = 0, 0
    collect_rng = jax.random.PRNGKey(FLAGS.seed + 2)
    while env_steps < FLAGS.online_env_steps:
        collect_rng, expl = collect_online_episodes(agent, online_envs, replay_buffer, collect_rng, {}, exec_len)
        new_steps = int(expl['exploration/transitions'])
        env_steps += new_steps
        round_idx += 1
        row = {k: expl[k] for k in ('exploration/success', 'exploration/episode.length') if k in expl}
        if FLAGS.self_imitation and successes:
            demo_dataset = Dataset.create(**with_successes(demos, successes))
            demo_dataset.action_chunk_length = exec_len
            demo_dataset.n_step = exec_len
            demo_dataset.discount = FLAGS.discount
            row['demos/episodes'] = len(demo_dataset.initial_locs)
        if trains:
            actor_on = float(FLAGS.method != 'td3bc' or env_steps > FLAGS.actor_freeze_steps)
            row.update(train(int(FLAGS.utd * new_steps / exec_len), online=True, actor_on=actor_on))
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
