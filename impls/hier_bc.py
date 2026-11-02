"""Subgoal-conditioned BC with an oracle high-level policy on block, chamber, hanoi, and switch tasks.

Generalizes subgoal_bc.py beyond block tasks (see hier_common.py). The final evaluation also runs with longer episode
limits (`final_eval_horizons`) to measure how much of the failure comes from running out of time. With `--flat=1`, the
same pipeline trains and evaluates the original flat BC instead.
"""

import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
from absl import app, flags
from ml_collections import config_flags

from hier_common import cpu_env_name, encode_subgoal, env_family, evaluate_parallel, subgoal_dim, subgoal_spec

# Heavy imports (JAX, agents, environments) live in main(): evaluation workers are spawned processes that re-import
# this module, and importing JAX in each of them exhausts host memory.

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'hanoi-triple-task1-v0', 'Environment name.')
flags.DEFINE_string('dataset_root', None, 'Root directory for downloaded datasets.')
flags.DEFINE_integer('num_shards', 1, 'Number of training shards to use (0: all).')
flags.DEFINE_integer('delete_shards', 0, 'Delete each raw training shard after loading it (for full-data runs).')
flags.DEFINE_integer('flat', 0, 'Train and evaluate the original flat BC (successful episodes, no subgoals).')
flags.DEFINE_string('save_dir', '../exp/hier_bc', 'Save directory.')
flags.DEFINE_string('run_name', '', 'Run directory name (default: env name and time).')
flags.DEFINE_integer('offline_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('eval_episodes', 100, 'Number of evaluation episodes.')
flags.DEFINE_integer('final_eval_episodes', 500, 'Number of evaluation episodes at the end of training.')
flags.DEFINE_string('final_eval_horizons', '1,2,4', 'Episode limits for the final evaluation, as env-limit multiples.')
flags.DEFINE_integer('eval_workers', 25, 'Environment worker processes for evaluation.')
flags.DEFINE_integer('eval_wave_size', 50, 'Max concurrent evaluation episodes (bounds host memory).')
flags.DEFINE_integer('save_interval', 250000, 'Saving interval. Set to 0 to disable saving.')
flags.DEFINE_integer('restore_step', 0, 'If > 0, skip training and only run the final evaluation of this checkpoint.')
flags.DEFINE_integer('action_chunk_length', 25, 'Action chunk length.')
flags.DEFINE_float('action_norm_percentile', 1.0, 'Action normalization lower percentile.')
flags.DEFINE_float('action_norm_clip', 5.0, 'Action normalization clip value.')
flags.DEFINE_integer('max_subgoal_steps', 0, 'Steps before the oracle replans a subgoal (0: family default).')
flags.DEFINE_float('done_away_dist', 0.1, 'Effector distance moved after subtask success and release.')
flags.DEFINE_float('done_contact', 0.5, 'Max gripper contact to count as released.')

DEFAULT_MAX_SUBGOAL_STEPS = dict(block=1250, chamber=1250, hanoi=1875, switch=600)


def select_segments(metadata, spec, flat):
    """Pick the training segments of one dataset file from its metadata alone.

    Hierarchical: every successful subtask segment, conditioned on its encoded target. Flat: every successful episode,
    unconditioned (the original BC setup).
    """
    episode_starts, starts, lengths, episode_idxs, conds = [], [], [], [], []
    for episode_idx, episode in enumerate(metadata):
        episode_starts.append(episode['start'])
        if flat:
            if episode['success']:
                starts.append(episode['start'])
                lengths.append(episode['length'])
                episode_idxs.append(episode_idx)
            continue
        for segment in episode['segments']:
            if not segment['subtask_success'] or segment['end'] is None or segment['end'] < segment['start']:
                continue
            starts.append(segment['start'])
            lengths.append(segment['end'] - segment['start'] + 1)
            episode_idxs.append(episode_idx)
            conds.append(encode_subgoal(spec, segment['subtask_name'], segment['target']))
    cond_dim = 0 if flat else subgoal_dim(spec)
    return dict(
        episode_starts=np.asarray(episode_starts),
        starts=np.asarray(starts),
        lengths=np.asarray(lengths),
        episode_idxs=np.asarray(episode_idxs),
        conds=np.stack(conds) if conds else np.zeros((len(starts), cond_dim), np.float32),
    )


def num_rows(selection):
    """(observations, transitions) that a selection occupies: each segment keeps its observation after the last action."""
    return int((selection['lengths'] + 1).sum()), int(selection['lengths'].sum())


def fill_segments(data, selection, out, obs_pos, act_pos):
    """Copy the selected segments of a loaded dataset file into the preallocated arrays in `out`."""
    import ocbench

    offsets, observation_offsets = ocbench.get_episode_offsets(data)
    assert np.array_equal(offsets[:-1], selection['episode_starts'])
    starts, lengths, episode_idxs = selection['starts'], selection['lengths'], selection['episode_idxs']
    obs_starts = observation_offsets[episode_idxs] + starts - offsets[episode_idxs]
    transition_idxs = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, lengths)])
    observation_idxs = np.concatenate([np.arange(s, s + n + 1) for s, n in zip(obs_starts, lengths)])

    n_obs, n_act = len(observation_idxs), len(transition_idxs)
    ob_dim = data['observations'].shape[-1]
    out['observations'][obs_pos : obs_pos + n_obs, :ob_dim] = data['observations'][observation_idxs]
    out['observations'][obs_pos : obs_pos + n_obs, ob_dim:] = np.repeat(selection['conds'], lengths + 1, axis=0)
    out['actions'][act_pos : act_pos + n_act] = data['actions'][transition_idxs]
    out['terminals'][act_pos + np.cumsum(lengths) - 1] = True
    return obs_pos + n_obs, act_pos + n_act


def allocate(n_obs, n_act, ob_dim, action_dim):
    return dict(
        observations=np.empty((n_obs, ob_dim), np.float32),
        actions=np.empty((n_act, action_dim), np.float32),
        rewards=np.zeros(n_act, dtype=np.float32),
        masks=np.ones(n_act, dtype=np.float32),
        terminals=np.zeros(n_act, dtype=bool),
        observation_interval=np.asarray(1, dtype=np.int32),
    )


def load_metadata(path):
    with open(path.removesuffix('.npz') + '-metadata.pkl', 'rb') as f:
        return pickle.load(f)['episodes']


def load_segment_file(path, spec, flat, ob_dim, action_dim):
    """Load the training segments of a single dataset file (used for the validation set)."""
    import ocbench

    selection = select_segments(load_metadata(path), spec, flat)
    out = allocate(*num_rows(selection), ob_dim, action_dim)
    fill_segments(ocbench.load_dataset([path]), selection, out, 0, 0)
    return out


def load_segment_shards(env_name, dataset_root, num_shards, spec, flat, ob_dim, action_dim, delete_shards):
    """Load the training segments of many shards one at a time into arrays sized up front from the metadata.

    Only one raw shard is on disk and in memory at a time when `delete_shards` is set, so the full dataset fits on one
    host. Returns the dataset and the first shard's raw actions (for fitting the action normalizer).
    """
    import ocbench
    from huggingface_hub import hf_hub_download
    from ocbench.dataset_utils import DATASET_REPO, DATASET_REVISION

    # The first shard goes through the regular download path, which also fetches the release manifest and the
    # validation file.
    first_paths, val_paths = ocbench.download_datasets(env_name, dataset_root, 1)
    root = Path(dataset_root).expanduser().resolve()
    manifest_path = root / '.cache' / 'ocbench' / DATASET_REVISION / 'manifest.jsonl'
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    subdir = os.path.relpath(os.path.dirname(first_paths[0]), root)
    rel_paths = sorted(
        e['path'] for e in manifest if e['path'].rsplit('/', 1)[0] == subdir and e['kind'] == 'data' and e['split'] == 'train'
    )
    if num_shards > 0:
        rel_paths = rel_paths[:num_shards]

    def fetch(rel):
        if not (root / rel).is_file():
            hf_hub_download(DATASET_REPO, rel, repo_type='dataset', revision=DATASET_REVISION, local_dir=root)
        return str(root / rel)

    selections = []
    for rel in rel_paths:
        fetch(rel.removesuffix('.npz') + '-metadata.pkl')
        selections.append(select_segments(load_metadata(str(root / rel)), spec, flat))
    sizes = [num_rows(s) for s in selections]
    out = allocate(sum(n for n, _ in sizes), sum(n for _, n in sizes), ob_dim, action_dim)
    print(f'Allocated {len(rel_paths)} shards: {sum(n for _, n in sizes)} transitions', flush=True)

    obs_pos = act_pos = 0
    raw_actions = None
    for rel, selection in zip(rel_paths, selections):
        start = time.time()
        path = fetch(rel)
        data = ocbench.load_dataset([path])
        if raw_actions is None:
            raw_actions = data['actions'].copy()
        obs_pos, act_pos = fill_segments(data, selection, out, obs_pos, act_pos)
        del data
        if delete_shards:
            os.remove(path)
        print(f'Loaded {rel}: {len(selection["lengths"])} segments, {time.time() - start:.0f}s', flush=True)
    assert obs_pos == len(out['observations']) and act_pos == len(out['actions'])
    return out, raw_actions, val_paths


def main(_):
    import gymnasium
    import jax
    import ocbench
    import tqdm

    from agents import agents
    from envs.env_utils import ActionNormalizer
    from utils.datasets import Dataset
    from utils.flax_utils import restore_agent, save_agent
    from utils.log_utils import CsvLogger, get_flag_dict

    env_name = FLAGS.env_name
    family = env_family(env_name)
    eval_env_name = cpu_env_name(env_name)
    run_name = FLAGS.run_name or f'{env_name}_sd{FLAGS.seed:03d}_{time.strftime("%Y%m%d_%H%M%S")}'
    save_dir = os.path.join(FLAGS.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    if FLAGS.restore_step == 0:
        with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
            json.dump(get_flag_dict(), f)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    probe_env = gymnasium.make(eval_env_name)
    spec = subgoal_spec(family, probe_env.unwrapped)
    max_episode_steps = probe_env.spec.max_episode_steps
    ob_dim = probe_env.observation_space.shape[-1] + (0 if FLAGS.flat else subgoal_dim(spec))
    action_dim = probe_env.action_space.shape[-1]
    del probe_env

    config = FLAGS.agent
    agent_class = agents[config['agent_name']]
    sample_kwargs = getattr(agent_class, 'sample_kwargs', {})
    chunk_len = FLAGS.action_chunk_length

    if FLAGS.restore_step > 0:
        norm = np.load(os.path.join(save_dir, 'action_normalizer.npz'))
        action_normalizer = ActionNormalizer(norm['q_low'], norm['q_high'], float(norm['clip']), norm['gripper_dims'])
        agent = agent_class.create(
            FLAGS.seed, np.zeros((1, ob_dim), np.float32), np.zeros((1, chunk_len * action_dim), np.float32), config
        )
        agent = restore_agent(agent, save_dir, FLAGS.restore_step)
    else:
        # Load segment datasets.
        train_dataset, raw_actions, val_paths = load_segment_shards(
            env_name, FLAGS.dataset_root, FLAGS.num_shards, spec, FLAGS.flat, ob_dim, action_dim, FLAGS.delete_shards
        )
        val_dataset = load_segment_file(val_paths[0], spec, FLAGS.flat, ob_dim, action_dim)

        # Fit on the first shard's training actions (all of them, before filtering successful segments).
        action_normalizer = ActionNormalizer.fit(raw_actions, FLAGS.action_norm_percentile, FLAGS.action_norm_clip)
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
            dataset.action_chunk_length = chunk_len
        print(
            f'Train transitions: {train_dataset.size}, epoch = {train_dataset.size / config["batch_size"]:.0f} steps',
            flush=True,
        )
        example_batch = train_dataset.sample(1, **sample_kwargs)
        agent = agent_class.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], config)

    normalizer_args = dict(
        q_low=action_normalizer.q_low,
        q_high=action_normalizer.q_high,
        clip=action_normalizer.clip,
        gripper_dims=np.flatnonzero(action_normalizer.gripper_dims),
    )
    eval_rng = [jax.random.PRNGKey(FLAGS.seed)]

    def sample_fn(obs):
        eval_rng[0], key = jax.random.split(eval_rng[0])
        return agent.sample_actions(observations=obs, seed=key)

    def run_eval(num_episodes, horizon_mult=1):
        cfg = dict(
            max_episode_steps=max_episode_steps * horizon_mult,
            max_subgoal_steps=FLAGS.max_subgoal_steps or DEFAULT_MAX_SUBGOAL_STEPS[family],
            done_away_dist=FLAGS.done_away_dist,
            done_contact=FLAGS.done_contact,
            flat=bool(FLAGS.flat),
        )
        return evaluate_parallel(
            sample_fn,
            eval_env_name,
            normalizer_args,
            chunk_len,
            action_dim,
            num_episodes,
            min(FLAGS.eval_workers, num_episodes),
            cfg,
            FLAGS.seed,
            wave_size=FLAGS.eval_wave_size,
        )

    def final_eval(step):
        results = {}
        for mult in [int(m) for m in FLAGS.final_eval_horizons.split(',')]:
            metrics, episodes = run_eval(FLAGS.final_eval_episodes, mult)
            metrics['max_episode_steps'] = max_episode_steps * mult
            results[f'x{mult}'] = metrics
            print(f'final eval x{mult}: {json.dumps(metrics)}', flush=True)
            with open(os.path.join(save_dir, f'final_eval_{step}_x{mult}_episodes.pkl'), 'wb') as f:
                pickle.dump(episodes, f)
        with open(os.path.join(save_dir, f'final_eval_{step}.json'), 'w') as f:
            json.dump(results, f, indent=1)

    if FLAGS.restore_step > 0:
        final_eval(FLAGS.restore_step)
        return

    train_logger = CsvLogger(os.path.join(save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(save_dir, 'eval.csv'))
    first_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True, mininterval=60):
        batch = train_dataset.sample(config['batch_size'], **sample_kwargs)
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            _, val_info = agent.total_loss(val_dataset.sample(config['batch_size'], **sample_kwargs), grad_params=None)
            train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['epoch'] = i * config['batch_size'] / train_dataset.size
            train_metrics['time/total_time'] = time.time() - first_time
            train_logger.log(train_metrics, step=i)

        if FLAGS.save_interval != 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, save_dir, i)

        if FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0 and i != FLAGS.offline_steps:
            metrics, _ = run_eval(FLAGS.eval_episodes)
            eval_metrics = {f'evaluation/{k}': v for k, v in metrics.items() if not k.startswith('task/')}
            eval_metrics['epoch'] = i * config['batch_size'] / train_dataset.size
            eval_logger.log(eval_metrics, step=i)
            print(f'step {i}: {json.dumps(metrics)}', flush=True)

    train_logger.close()
    eval_logger.close()
    del train_dataset, val_dataset
    final_eval(FLAGS.offline_steps)


if __name__ == '__main__':
    # Defined here because loading the agent config imports JAX, which spawned evaluation workers must not do.
    config_flags.DEFINE_config_file('agent', os.path.join(os.path.dirname(__file__), 'agents/bc.py'), lock_config=False)
    app.run(main)
