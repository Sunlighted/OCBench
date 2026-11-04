"""Best-of-N action-chunk selection over a frozen flow BC prior, with a chunk-level critic.

The policy samples N action chunks from the prior and executes the one with the highest mean ensemble Q. The critic is
trained by TD on chunk transitions (`batch['rewards']`/`batch['discounts']` are the n-step quantities over one chunk),
bootstrapping from the Best-of-N chunk at the next state. Only the critic is trained.
"""

from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, ensemblize


class QNetwork(nn.Module):
    """Q(s, A) for a flattened action chunk A."""

    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)
        return MLP((*self.hidden_dims, 1), activate_final=False, layer_norm=True)(x).squeeze(-1)


class TruncatedPrior(flax.struct.PyTreeNode):
    """A prior that executes only the first `exec_dim` entries of its flattened action chunk (receding horizon)."""

    prior: Any
    exec_dim: int = nonpytree_field()

    def sample_actions(self, observations, seed=None):
        return self.prior.sample_actions(observations, seed=seed)[..., : self.exec_dim]


class BoNAgent(flax.struct.PyTreeNode):
    rng: Any
    critic: Any
    prior: Any
    act_low: Any
    act_high: Any
    config: Any = nonpytree_field()

    def best_of_n(self, observations, rng, num_samples):
        """Return the prior sample with the highest mean Q for each state, and all sample Q values."""
        batch = observations.shape[0]
        tiled = jnp.repeat(observations, num_samples, axis=0)
        actions = jnp.clip(self.prior.sample_actions(tiled, seed=rng), self.act_low, self.act_high)
        qs = self.critic.select('critic')(tiled, actions).mean(axis=0).reshape(batch, num_samples)
        best = jnp.argmax(qs, axis=1)
        actions = actions.reshape(batch, num_samples, -1)
        return actions[jnp.arange(batch), best], qs

    @partial(jax.jit, static_argnames=('num_samples',))
    def sample_actions(self, observations, seed, num_samples=None):
        actions, _ = self.best_of_n(observations, seed, num_samples or self.config['num_samples'])
        return actions

    @jax.jit
    def update(self, batch):
        rng, next_rng = jax.random.split(self.rng)
        next_actions, _ = self.best_of_n(batch['next_observations'], next_rng, self.config['num_samples'])
        next_q = self.critic.select('target_critic')(batch['next_observations'], next_actions).min(axis=0)
        target = batch['rewards'] + batch['discounts'] * next_q

        def loss_fn(params):
            qs = self.critic.select('critic')(batch['observations'], batch['actions'], params=params)
            loss = ((qs - target[None]) ** 2).mean()
            return loss, dict(critic_loss=loss, q_mean=qs.mean(), q_min=qs.min(), target_mean=target.mean())

        critic, info = self.critic.apply_loss_fn(loss_fn)
        tau = self.config['tau']
        critic.params['modules_target_critic'] = jax.tree_util.tree_map(
            lambda p, tp: tau * p + (1 - tau) * tp,
            critic.params['modules_critic'],
            critic.params['modules_target_critic'],
        )
        return self.replace(rng=rng, critic=critic), info

    @classmethod
    def create(cls, seed, prior, ex_observations, ex_actions, act_low, act_high, config):
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        critic_def = ensemblize(QNetwork, config['num_qs'])(hidden_dims=tuple(config['critic_hidden_dims']))
        module = ModuleDict(dict(critic=critic_def, target_critic=critic_def))
        args = (ex_observations, ex_actions)
        params = module.init(init_rng, critic=args, target_critic=args)['params']
        params['modules_target_critic'] = params['modules_critic']
        critic = TrainState.create(module, params, tx=optax.adam(config['lr']))
        return cls(
            rng=rng,
            critic=critic,
            prior=prior,
            act_low=jnp.asarray(act_low),
            act_high=jnp.asarray(act_high),
            config=flax.core.FrozenDict(config),
        )
