"""Subgoal-conditioned BC with an oracle high-level policy for block tasks.

The low-level policy is trained on the scripted-policy segments stored in the dataset metadata. Each successful
segment is relabeled with its actual target (block one-hot, target position, and target yaw), including segments
from failed episodes and potential high-level mistakes. At evaluation, the oracle high-level policy picks subgoals
with the scripted controller's goal logic (no mistakes) and switches when the current subgoal is complete.
"""

import json
import os
import pickle
import random
import time
from collections import defaultdict

import gymnasium
import jax
import numpy as np
import ocbench
import tqdm
from absl import app, flags
from ml_collections import config_flags
from ocbench.oracles.controllers.block import BlockController

from agents import agents
from envs.env_utils import ActionNormalizer, make_eval_env
from utils.datasets import Dataset
from utils.evaluation import clip_actions, supply_rng
from utils.flax_utils import save_agent
from utils.log_utils import CsvLogger, get_flag_dict

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'block-quadruple-task1-v0', 'Environment name (block tasks only).')
flags.DEFINE_string('dataset_root', None, 'Root directory for downloaded datasets.')
flags.DEFINE_integer('num_shards', 1, 'Number of training shards to use.')
flags.DEFINE_string('save_dir', '../exp/subgoal_bc', 'Save directory.')
flags.DEFINE_integer('offline_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('eval_episodes', 100, 'Number of evaluation episodes.')
flags.DEFINE_integer('final_eval_episodes', 500, 'Number of evaluation episodes at the end of training.')
flags.DEFINE_integer('save_interval', 500000, 'Saving interval. Set to 0 to disable saving.')
flags.DEFINE_integer('action_chunk_length', 25, 'Action chunk length.')
flags.DEFINE_float('action_norm_percentile', 1.0, 'Action normalization lower percentile.')
flags.DEFINE_float('action_norm_clip', 5.0, 'Action normalization clip value.')
flags.DEFINE_integer('max_subgoal_steps', 1250, 'Steps before the high-level policy replans a subgoal.')
flags.DEFINE_float('done_target_dist', 0.04, 'Max block-target distance for subgoal completion.')
flags.DEFINE_float('done_effector_dist', 0.1, 'Min effector-block distance for subgoal completion.')
flags.DEFINE_float('done_contact', 0.5, 'Max gripper contact for subgoal completion.')

config_flags.DEFINE_config_file('agent', os.path.join(os.path.dirname(__file__), 'agents/bc.py'), lock_config=False)

XYZ_CENTER = np.array([0.425, 0.0, 0.0])
XYZ_SCALER = 10.0


def encode_subgoal(block, target_pos, target_yaw, num_blocks):
    one_hot = np.zeros(num_blocks, dtype=np.float32)
    one_hot[block] = 1.0
    return np.concatenate(
        [one_hot, (np.asarray(target_pos) - XYZ_CENTER) * XYZ_SCALER, [np.cos(target_yaw), np.sin(target_yaw)]]
    ).astype(np.float32)


def load_subgoal_dataset(data_path, metadata_path, num_blocks):
    """Load a dataset file and split its successful segments into subgoal-conditioned episodes."""
    data = ocbench.load_dataset([data_path])
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)['episodes']
    offsets, observation_offsets = ocbench.get_episode_offsets(data)
    assert len(metadata) == len(offsets) - 1

    starts, lengths, obs_starts, conds = [], [], [], []
    for episode_idx, episode in enumerate(metadata):
        assert episode['start'] == offsets[episode_idx]
        for segment in episode['segments']:
            if not segment['subtask_success']:
                continue
            target = segment['target']
            starts.append(segment['start'])
            lengths.append(segment['end'] - segment['start'] + 1)
            obs_starts.append(observation_offsets[episode_idx] + segment['start'] - offsets[episode_idx])
            conds.append(encode_subgoal(target['block'], target['block_pos'], target['block_yaw'], num_blocks))
    starts, lengths, obs_starts = np.asarray(starts), np.asarray(lengths), np.asarray(obs_starts)
    conds = np.stack(conds)

    # Each segment keeps its observation after the last action, so it becomes a valid episode.
    transition_idxs = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, lengths)])
    observation_idxs = np.concatenate([np.arange(s, s + n + 1) for s, n in zip(obs_starts, lengths)])
    observations = np.empty((len(observation_idxs), data['observations'].shape[-1] + conds.shape[-1]), np.float32)
    observations[:, : data['observations'].shape[-1]] = data['observations'][observation_idxs]
    observations[:, data['observations'].shape[-1] :] = np.repeat(conds, lengths + 1, axis=0)
    terminals = np.zeros(len(transition_idxs), dtype=bool)
    terminals[np.cumsum(lengths) - 1] = True

    dataset = dict(
        observations=observations,
        actions=data['actions'][transition_idxs],
        rewards=np.zeros(len(transition_idxs), dtype=np.float32),
        masks=np.ones(len(transition_idxs), dtype=np.float32),
        terminals=terminals,
        observation_interval=np.asarray(1, dtype=np.int32),
    )
    stats = dict(
        num_segments=len(lengths),
        num_transitions=int(lengths.sum()),
        total_transitions=len(data['actions']),
        mean_segment_length=float(lengths.mean()),
    )
    return dataset, data['actions'], stats


def subgoal_done(subgoal, ob_info):
    block_pos = ob_info[f'privileged/block_{subgoal["block"]}_pos']
    effector_pos = ob_info['proprio/effector_pos']
    return (
        np.linalg.norm(block_pos - subgoal['target_pos']) <= FLAGS.done_target_dist
        and np.linalg.norm(effector_pos - block_pos) >= FLAGS.done_effector_dist
        and ob_info['proprio/gripper_contact'][0] < FLAGS.done_contact
    )


def evaluate_hierarchical(agent, envs, num_blocks, action_chunk_length):
    """Evaluate the low-level policy with the oracle high-level policy."""
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    num_episodes = len(envs)
    action_dim = envs[0].action_space.shape[-1]
    states = []
    for env in envs:
        observation, info = env.reset()
        controller = BlockController(env=env, p_mistake=0.0)
        controller.reset(observation, env.unwrapped.compute_ob_info())
        states.append(
            dict(
                env=env,
                controller=controller,
                observation=observation,
                subgoal=None,
                subgoal_steps=0,
                completed=0,
                timeouts=0,
                done=False,
                info=info,
            )
        )

    while any(not state['done'] for state in states):
        active = [state for state in states if not state['done']]
        for state in active:
            ob_info = state['env'].unwrapped.compute_ob_info()
            if state['subgoal'] is not None and subgoal_done(state['subgoal'], ob_info):
                state['completed'] += 1
                state['subgoal'] = None
            elif state['subgoal'] is not None and state['subgoal_steps'] >= FLAGS.max_subgoal_steps:
                state['timeouts'] += 1
                state['subgoal'] = None
            if state['subgoal'] is None:
                state['subgoal'] = state['controller']._goal_subgoal(ob_info)
                state['subgoal_steps'] = 0
            if state['subgoal'] is None:
                # All blocks are at their targets; the episode terminates on the next step.
                state['cond'] = np.zeros(num_blocks + 5, dtype=np.float32)
            else:
                subgoal = state['subgoal']
                state['cond'] = encode_subgoal(
                    subgoal['block'], subgoal['target_pos'], subgoal['target_yaw'], num_blocks
                )

        observations = [np.concatenate([state['observation'], state['cond']]) for state in active]
        observations += [observations[-1]] * (num_episodes - len(observations))
        actions = np.array(actor_fn(observations=np.stack(observations)))
        actions = clip_actions(actions, envs[0], action_chunk_length)
        action_chunks = actions.reshape(num_episodes, action_chunk_length, action_dim)

        for state, action_chunk in zip(active, action_chunks[: len(active)]):
            for action in action_chunk:
                observation, _, terminated, truncated, info = state['env'].step(action)
                state['observation'] = observation
                state['info'] = info
                state['subgoal_steps'] += 1
                if terminated or truncated:
                    state['done'] = True
                    break

    stats = defaultdict(list)
    for state in states:
        stats['success'].append(float(state['info']['success']))
        stats['blocks_at_target'].append(float(sum(state['env'].unwrapped._compute_successes())))
        stats['subgoals_completed'].append(float(state['completed']))
        stats['subgoal_timeouts'].append(float(state['timeouts']))
        stats['episode_length'].append(float(state['info']['episode']['length']))
    blocks = np.asarray(stats['blocks_at_target'])
    summary = {k: float(np.mean(v)) for k, v in stats.items()}
    for k in range(1, num_blocks + 1):
        summary[f'blocks_at_target_ge{k}'] = float(np.mean(blocks >= k))
    return summary


def main(_):
    env_name = FLAGS.env_name
    assert env_name.startswith('block-') and '-cpu-' not in env_name, 'Use an MJWarp block environment name.'
    cpu_env_name = env_name.replace('block-', 'block-cpu-', 1)

    exp_name = f'sd{FLAGS.seed:03d}_{time.strftime("%Y%m%d_%H%M%S")}'
    save_dir = os.path.join(FLAGS.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        json.dump(get_flag_dict(), f)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    num_blocks = gymnasium.make(cpu_env_name).unwrapped._num_cubes

    # Load segment datasets.
    train_paths, val_paths = ocbench.download_datasets(env_name, FLAGS.dataset_root, FLAGS.num_shards)
    train_parts, raw_actions = [], []
    for path in train_paths:
        dataset, actions, stats = load_subgoal_dataset(path, path.removesuffix('.npz') + '-metadata.pkl', num_blocks)
        print(f'Loaded {path}: {stats}')
        train_parts.append(dataset)
        raw_actions.append(actions)
    val_dataset, _, val_stats = load_subgoal_dataset(
        val_paths[0], val_paths[0].removesuffix('.npz') + '-metadata.pkl', num_blocks
    )
    print(f'Loaded {val_paths[0]}: {val_stats}')
    train_dataset = {
        k: train_parts[0][k] if k == 'observation_interval' else np.concatenate([d[k] for d in train_parts])
        for k in train_parts[0]
    }
    del train_parts

    # Fit on all training actions, before filtering successful segments.
    action_normalizer = ActionNormalizer.fit(
        np.concatenate(raw_actions), FLAGS.action_norm_percentile, FLAGS.action_norm_clip
    )
    del raw_actions
    np.savez(
        os.path.join(save_dir, 'action_normalizer.npz'),
        q_low=action_normalizer.q_low,
        q_high=action_normalizer.q_high,
        clip=action_normalizer.clip,
        gripper_dims=np.flatnonzero(action_normalizer.gripper_dims),
    )
    for dataset in (train_dataset, val_dataset):
        dataset['actions'] = action_normalizer.normalize(dataset['actions'])

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)
    for dataset in (train_dataset, val_dataset):
        dataset.action_chunk_length = FLAGS.action_chunk_length
    print(f'Train transitions: {train_dataset.size}, epoch = {train_dataset.size / FLAGS.agent.batch_size:.0f} steps')

    # Create agent.
    config = FLAGS.agent
    agent_class = agents[config['agent_name']]
    sample_kwargs = getattr(agent_class, 'sample_kwargs', {})
    example_batch = train_dataset.sample(1, **sample_kwargs)
    agent = agent_class.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], config)

    def run_eval(num_episodes):
        envs = [make_eval_env(cpu_env_name, action_normalizer=action_normalizer) for _ in range(num_episodes)]
        start = time.time()
        metrics = evaluate_hierarchical(agent, envs, num_blocks, FLAGS.action_chunk_length)
        metrics['eval_time'] = time.time() - start
        return metrics

    train_logger = CsvLogger(os.path.join(save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(save_dir, 'eval.csv'))
    first_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(config['batch_size'], **sample_kwargs)
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            _, val_info = agent.total_loss(val_dataset.sample(config['batch_size'], **sample_kwargs), grad_params=None)
            train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['epoch'] = i * config['batch_size'] / train_dataset.size
            train_metrics['time/total_time'] = time.time() - first_time
            train_logger.log(train_metrics, step=i)

        is_last = i == FLAGS.offline_steps
        if (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0) or is_last:
            num_episodes = FLAGS.final_eval_episodes if is_last else FLAGS.eval_episodes
            eval_metrics = {f'evaluation/{k}': v for k, v in run_eval(num_episodes).items()}
            eval_metrics['epoch'] = i * config['batch_size'] / train_dataset.size
            eval_metrics['evaluation/num_episodes'] = num_episodes
            eval_logger.log(eval_metrics, step=i)
            print(f'step {i}: {eval_metrics}', flush=True)

        if FLAGS.save_interval != 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
