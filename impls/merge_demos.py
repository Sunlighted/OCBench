"""Merge demo_variants.py shards (.npz + -metadata.pkl) into one dataset, optionally capped at a transition count.

Usage: python merge_demos.py --out=merged.npz --max_transitions=5801574 shard0.npz shard1.npz ...
Episodes are taken in shard order until adding the next one would exceed `--max_transitions` (0: keep all).
"""

import pickle

import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('out', '', 'Output .npz path.')
flags.DEFINE_integer('max_transitions', 0, 'Cap on the total number of transitions (0: no cap).')


def main(argv):
    shards = argv[1:]
    fields = {k: [] for k in ('observations', 'actions', 'rewards', 'masks', 'terminals', 'qpos', 'qvel')}
    episodes, total, configs = [], 0, []
    for path in shards:
        # Materialize each array once: indexing an NpzFile re-reads the whole array on every access.
        with np.load(path) as npz:
            data = {k: npz[k] for k in npz.files}
        with open(path.removesuffix('.npz') + '-metadata.pkl', 'rb') as f:
            meta = pickle.load(f)
        configs.append(meta['config'])
        obs_start = 0
        for ep in meta['episodes']:
            n = ep['length']
            if FLAGS.max_transitions and total + n > FLAGS.max_transitions:
                break
            # Each episode stores n actions and n + 1 observations.
            s = ep['start']
            fields['observations'].append(data['observations'][obs_start : obs_start + n + 1])
            for k in ('actions', 'rewards', 'masks', 'terminals', 'qpos', 'qvel'):
                fields[k].append(data[k][s : s + n])
            episodes.append(dict(ep, episode_idx=len(episodes), start=total, end=total + n - 1))
            total += n
            obs_start += n + 1
        else:
            continue
        break
    out = {k: np.concatenate(v) for k, v in fields.items()}
    out['observation_interval'] = np.asarray(1, dtype=np.int32)
    np.savez(FLAGS.out, **out)
    with open(FLAGS.out.removesuffix('.npz') + '-metadata.pkl', 'wb') as f:
        pickle.dump(dict(episodes=episodes, config=dict(shards=shards, shard_configs=configs)), f)
    print(f'{len(episodes)} episodes, {total} transitions -> {FLAGS.out}')


if __name__ == '__main__':
    app.run(main)
