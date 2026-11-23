"""EXPO (Dong et al., 2026, "Stable Reinforcement Learning with Expressive Policies") over action chunks.

A frozen base policy (the BC prior) proposes `num_samples` chunks per state; a small tanh-Gaussian edit policy
pi(delta | s, a_base) adds a bounded correction delta = `edit_scale` * u, u in [-1, 1]^d. The on-the-fly policy executes
the candidate (base or edited) with the highest ensemble-mean Q, and the TD target uses the same selection at the next
state with the target critics (min over a random pair, clipped to [0, `q_max`]). The edit policy is trained SAC-style to
maximize Q of the edited chunk with automatic entropy tuning; the base policy is never trained.
`batch['rewards']`/`batch['discounts']` are the n-step quantities.
"""

from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from agents.bon import QNetwork
from agents.rlpd import ChunkActor
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import LogParam, ensemblize


class EXPOAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    prior: Any
    act_low: Any
    act_high: Any
    config: Any = nonpytree_field()

    def base_actions(self, observations, rng):
        return jnp.clip(self.prior.sample_actions(observations, seed=rng), self.act_low, self.act_high)

    def edit(self, observations, base, rng, deterministic=False, params=None):
        """Edited chunks and log-probabilities of the (unscaled) edit."""
        dist = self.network.select('actor')(jnp.concatenate([observations, base], axis=-1), params=params)
        if deterministic:
            u, log_probs = dist.mode(), jnp.zeros(base.shape[:-1])
        else:
            u, log_probs = dist.sample_and_log_prob(seed=rng)
        return jnp.clip(base + self.config['edit_scale'] * u, self.act_low, self.act_high), log_probs

    def on_the_fly(self, observations, rng, module, deterministic=False):
        """Best of the base samples and their edits by ensemble-mean Q of `module`; also the fraction of edits chosen."""
        base_rng, edit_rng = jax.random.split(rng)
        batch, n = observations.shape[0], self.config['num_samples']
        tiled = jnp.repeat(observations, n, axis=0)
        base = self.base_actions(tiled, base_rng)
        edited, _ = self.edit(tiled, base, edit_rng, deterministic)
        candidates = jnp.concatenate([base.reshape(batch, n, -1), edited.reshape(batch, n, -1)], axis=1)
        obs2 = jnp.repeat(observations, 2 * n, axis=0)
        qs = self.network.select(module)(obs2, candidates.reshape(batch * 2 * n, -1)).mean(axis=0).reshape(batch, 2 * n)
        best = jnp.argmax(qs, axis=1)
        return candidates[jnp.arange(batch), best], (best >= n).mean()

    @partial(jax.jit, static_argnames=('explore',))
    def sample_actions(self, observations, seed, explore=True):
        actions, _ = self.on_the_fly(observations, seed, 'critic', deterministic=not explore)
        return actions

    @jax.jit
    def update(self, batch):
        rng, next_rng, subset_rng, base_rng, edit_rng = jax.random.split(self.rng, 5)
        next_obs = batch['next_observations']
        next_actions, edit_frac = self.on_the_fly(next_obs, next_rng, 'target_critic')
        next_qs = self.network.select('target_critic')(next_obs, next_actions)
        subset = jax.random.choice(subset_rng, self.config['num_qs'], (self.config['num_min_qs'],), replace=False)
        target = batch['rewards'] + batch['discounts'] * next_qs[subset].min(axis=0)
        target = jnp.clip(target, 0.0, self.config['q_max'])
        obs = batch['observations']
        base = self.base_actions(obs, base_rng)

        def loss_fn(params):
            qs = self.network.select('critic')(obs, batch['actions'], params=params)
            critic_loss = ((qs - target[None]) ** 2).mean()

            edited, log_probs = self.edit(obs, base, edit_rng, params=params)
            q_edit = self.network.select('critic')(obs, edited).mean(axis=0)
            alpha = self.network.select('alpha')()
            actor_loss = (alpha * log_probs - q_edit).mean()
            entropy = -jax.lax.stop_gradient(log_probs).mean()
            alpha_loss = self.network.select('alpha')(params=params) * (entropy - self.config['target_entropy'])
            return critic_loss + actor_loss + alpha_loss, dict(
                critic_loss=critic_loss, q_mean=qs.mean(), target_mean=target.mean(), actor_loss=actor_loss,
                q_edit=q_edit.mean(), entropy=entropy, alpha=alpha, edit_frac=edit_frac,
            )

        network, info = self.network.apply_loss_fn(loss_fn)
        tau = self.config['tau']
        network.params['modules_target_critic'] = jax.tree_util.tree_map(
            lambda p, tp: tau * p + (1 - tau) * tp, network.params['modules_critic'], network.params['modules_target_critic']
        )
        return self.replace(rng=rng, network=network), info

    @classmethod
    def create(cls, seed, prior, ex_observations, ex_actions, act_low, act_high, config):
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        action_dim = ex_actions.shape[-1]
        critic_def = ensemblize(QNetwork, config['num_qs'])(hidden_dims=tuple(config['critic_hidden_dims']))
        actor_def = ChunkActor(hidden_dims=tuple(config['actor_hidden_dims']), action_dim=action_dim)
        module = ModuleDict(dict(
            critic=critic_def, target_critic=critic_def, actor=actor_def, alpha=LogParam(init_value=config['init_alpha'])
        ))
        args = (ex_observations, ex_actions)
        params = module.init(
            init_rng, critic=args, target_critic=args, actor=(jnp.concatenate(args, axis=-1),), alpha=()
        )['params']
        params['modules_target_critic'] = params['modules_critic']
        network = TrainState.create(module, params, tx=optax.adam(config['lr']))
        config = dict(config, target_entropy=config.get('target_entropy') or -0.5 * action_dim)
        return cls(
            rng=rng, network=network, prior=prior, act_low=jnp.asarray(act_low), act_high=jnp.asarray(act_high),
            config=flax.core.FrozenDict(config),
        )
