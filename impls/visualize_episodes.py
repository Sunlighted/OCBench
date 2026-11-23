"""Roll out a flat BC prior with a longer time limit and render typical, slow and failed episodes.

Run with MUJOCO_GL=osmesa on headless machines. Episodes are rolled out in lockstep from random initial states with
full-chunk execution, recording the MuJoCo state and the proprioceptive/privileged quantities at every step. Each episode
is classified as `normal` (success within the standard limit), `slow` (success only after it) or `fail`. For a few of
each, it writes an mp4 (front and wrist views, with the step, chunk index, gripper and contact overlaid) and a timeline
plot (effector-block distance, heights, gripper opening, contact, chunk boundaries), plus a JSON summary.
"""

import json
import os

import numpy as np
from absl import app, flags

from hier_common import cpu_env_name

FLAGS = flags.FLAGS
flags.DEFINE_string('prior_dir', '', 'Prior directory.')
flags.DEFINE_integer('prior_step', 1000000, 'Prior checkpoint step.')
flags.DEFINE_string('env_name', 'block-single-task1-v0', 'Environment name.')
flags.DEFINE_integer('max_episode_steps', 2500, 'Time limit of the rollouts.')
flags.DEFINE_integer('standard_limit', 1250, 'Standard time limit (for the normal/slow split).')
flags.DEFINE_integer('num_episodes', 200, 'Episodes to roll out.')
flags.DEFINE_integer('num_envs', 50, 'Environments stepped in lockstep.')
flags.DEFINE_integer('num_normal', 2, 'Normal episodes to render.')
flags.DEFINE_integer('num_slow', 5, 'Slow episodes to render.')
flags.DEFINE_integer('num_fail', 3, 'Failed episodes to render.')
flags.DEFINE_integer('frame_skip', 4, 'Render every this many control steps.')
flags.DEFINE_integer('fps', 25, 'Video frames per second.')
flags.DEFINE_string('out_dir', '', 'Output directory.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def main(_):
    import gymnasium
    import imageio
    import jax
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mujoco
    import ocbench  # noqa: F401
    from PIL import Image, ImageDraw

    from agents import agents
    from envs.env_utils import ActionNormalizer, wrap_env
    from utils.flax_utils import restore_agent

    np.random.seed(FLAGS.seed)
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    with open(os.path.join(FLAGS.prior_dir, 'flags.json')) as f:
        prior_flags = json.load(f)
    config = dict(prior_flags['agent'])
    for key in ('actor_hidden_dims', 'betas'):
        if key in config:
            config[key] = tuple(config[key])
    chunk_len = prior_flags['action_chunk_length']
    norm = np.load(os.path.join(FLAGS.prior_dir, 'action_normalizer.npz'))
    normalizer = ActionNormalizer(norm['q_low'], norm['q_high'], float(norm['clip']), norm['gripper_dims'])
    env_name = cpu_env_name(FLAGS.env_name)

    def make(render=False):
        kwargs = dict(max_episode_steps=FLAGS.max_episode_steps)
        if render:
            kwargs.update(render_mode='rgb_array', width=320, height=240)
        return wrap_env(gymnasium.make(env_name, **kwargs), action_normalizer=normalizer)

    envs = [make() for _ in range(FLAGS.num_envs)]
    ob_dim, action_dim = envs[0].observation_space.shape[-1], envs[0].action_space.shape[-1]
    prior = agents[config['agent_name']].create(
        0, np.zeros((1, ob_dim), np.float32), np.zeros((1, chunk_len * action_dim), np.float32), config
    )
    prior = restore_agent(prior, FLAGS.prior_dir, FLAGS.prior_step)
    low, high = np.tile(normalizer.action_low, chunk_len), np.tile(normalizer.action_high, chunk_len)

    def record(env):
        u = env.unwrapped
        info = u.compute_ob_info()
        return dict(
            qpos=u._data.qpos.copy(), qvel=u._data.qvel.copy(),
            eff=np.asarray(info['proprio/effector_pos']).copy(),
            block=np.asarray(info['privileged/block_0_pos']).copy(),
            grip=float(np.asarray(info['proprio/gripper_opening'])[0]),
            contact=float(np.asarray(info['proprio/gripper_contact'])[0]),
        )

    episodes = []
    key = jax.random.PRNGKey(FLAGS.seed)
    while len(episodes) < FLAGS.num_episodes:
        seeds = [int(np.random.randint(0, 2**31)) for _ in envs]
        obs = np.stack([env.reset(seed=s)[0] for env, s in zip(envs, seeds)]).astype(np.float32)
        trajs = [[record(env)] for env in envs]
        done = np.zeros(len(envs), bool)
        success_step = np.full(len(envs), -1)
        while not done.all():
            key, sub = jax.random.split(key)
            chunks = np.clip(np.asarray(prior.sample_actions(obs, seed=sub)), low, high)
            chunks = chunks.reshape(len(envs), chunk_len, action_dim)
            for i, env in enumerate(envs):
                if done[i]:
                    continue
                for a in chunks[i]:
                    ob, _, terminated, truncated, info = env.step(a)
                    trajs[i].append(record(env))
                    if terminated or truncated:
                        done[i] = True
                        if info.get('success', False):
                            success_step[i] = len(trajs[i]) - 1
                        break
                obs[i] = ob
        for i in range(len(envs)):
            episodes.append(dict(seed=seeds[i], traj=trajs[i], success_step=int(success_step[i])))
        print(f'{len(episodes)} episodes rolled out', flush=True)

    def kind(ep):
        if ep['success_step'] < 0:
            return 'fail'
        return 'normal' if ep['success_step'] <= FLAGS.standard_limit else 'slow'

    def first_grasp(traj):
        for t, s in enumerate(traj):
            if s['contact'] > 0.5 and s['block'][2] > 0.045:
                return t
        return -1

    summary = dict(episodes=len(episodes), counts={}, rendered=[])
    for k in ('normal', 'slow', 'fail'):
        summary['counts'][k] = sum(kind(e) == k for e in episodes)
    print(json.dumps(summary['counts']), flush=True)

    target = np.array(envs[0].unwrapped._data.mocap_pos[envs[0].unwrapped._cube_target_mocap_ids[0]])
    render_env = make(render=True)
    render_env.reset(seed=0)
    u = render_env.unwrapped
    picks = []
    for k, n in (('normal', FLAGS.num_normal), ('slow', FLAGS.num_slow), ('fail', FLAGS.num_fail)):
        group = [e for e in episodes if kind(e) == k]
        if k == 'slow':
            group = sorted(group, key=lambda e: -e['success_step'])
        picks += [(k, e) for e in group[:n]]

    for idx, (k, ep) in enumerate(picks):
        traj = ep['traj']
        name = f'{idx:02d}_{k}_seed{ep["seed"]}'
        steps = np.arange(len(traj))
        eff = np.stack([s['eff'] for s in traj])
        block = np.stack([s['block'] for s in traj])
        grip = np.array([s['grip'] for s in traj])
        contact = np.array([s['contact'] for s in traj])
        fg = first_grasp(traj)

        # Timeline plot.
        fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(steps, np.linalg.norm(eff[:, :2] - block[:, :2], axis=1), label='effector-block (xy)')
        axes[0].plot(steps, np.linalg.norm(block[:, :2] - target[:2], axis=1), label='block-target (xy)')
        axes[0].set_ylabel('distance (m)')
        axes[1].plot(steps, eff[:, 2], label='effector z')
        axes[1].plot(steps, block[:, 2], label='block z')
        axes[1].set_ylabel('height (m)')
        axes[2].plot(steps, grip, label='gripper opening')
        axes[2].plot(steps, contact, label='contact', alpha=0.7)
        axes[2].set_ylabel('gripper')
        axes[3].plot(eff[:, 0], color='C0', label='effector x')
        axes[3].plot(eff[:, 1], color='C1', label='effector y')
        axes[3].plot(block[:, 0], color='C0', ls='--', label='block x')
        axes[3].plot(block[:, 1], color='C1', ls='--', label='block y')
        axes[3].set_ylabel('position (m)')
        axes[3].set_xlabel('control step')
        for ax in axes:
            for c in range(0, len(traj), chunk_len):
                ax.axvline(c, color='0.9', lw=0.5, zorder=0)
            ax.axvline(FLAGS.standard_limit, color='red', ls=':', lw=1)
            if fg >= 0:
                ax.axvline(fg, color='green', ls='--', lw=1)
            ax.legend(loc='upper right', fontsize=7)
        title = f'{k}: success at step {ep["success_step"]}' if ep['success_step'] >= 0 else f'{k}: no success'
        fig.suptitle(f'{title}, first grasp at {fg} (green), standard limit 1250 (red), chunks every {chunk_len} steps')
        fig.tight_layout()
        fig.savefig(os.path.join(FLAGS.out_dir, name + '.png'), dpi=90)
        plt.close(fig)

        # Video: front and wrist views side by side.
        writer = imageio.get_writer(os.path.join(FLAGS.out_dir, name + '.mp4'), fps=FLAGS.fps, codec='libx264',
                                    quality=7, macro_block_size=8)
        for t in range(0, len(traj), FLAGS.frame_skip):
            u._data.qpos[:] = traj[t]['qpos']
            u._data.qvel[:] = traj[t]['qvel']
            mujoco.mj_forward(u._model, u._data)
            front = u.render(camera='front_pixels')
            wrist = u.render(camera='ur5e/wrist')
            frame = Image.fromarray(np.concatenate([front, wrist], axis=1))
            draw = ImageDraw.Draw(frame)
            label = (f'{k}  step {t}  chunk {t // chunk_len}  grip {traj[t]["grip"]:.2f}  '
                     f'contact {int(traj[t]["contact"] > 0.5)}' + ('  >1250' if t > FLAGS.standard_limit else ''))
            draw.rectangle([0, 0, frame.width, 14], fill=(0, 0, 0))
            draw.text((4, 1), label, fill=(255, 255, 0))
            writer.append_data(np.asarray(frame))
        writer.close()
        summary['rendered'].append(dict(name=name, kind=k, seed=ep['seed'], success_step=ep['success_step'],
                                        first_grasp=fg, length=len(traj) - 1))
        print(f'rendered {name}', flush=True)

    with open(os.path.join(FLAGS.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    app.run(main)
