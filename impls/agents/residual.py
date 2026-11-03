"""Residual RL on a frozen BC prior (deterministic BC or flow BC), at the action-chunk level.

The executed chunk is A = clip(A0 + scale * tanh(f(s, A0))), where A0 is the prior's chunk at state s (one sample for
a flow prior) and f is a residual actor whose output layer is zero-initialized, so training starts from the prior.
The residual actor is trained TD3-style to maximize a chunk-level critic Q(s, A) with an L2 penalty on f, which is trained by TD on chunk
transitions (`batch['rewards']`/`batch['discounts']` are the n-step quantities over one chunk) with target policy
smoothing. Exploration adds Gaussian noise to the residual during data collection. Unlike Best-of-N, the policy can
leave the prior's support, but only within `scale` of the prior's chunk.
"""

from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.bon import QNetwork
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, ensemblize


class ResidualActor(nn.Module):
    """Residual f(s, A0) in [-1, 1]^d, zero at initialization."""

    hidden_dims: Sequence[int]
    action_dim: int

    @nn.compact
    def __call__(self, observations, base_actions):
        x = jnp.concatenate([observations, base_actions], axis=-1)
        x = MLP(self.hidden_dims, activate_final=True, layer_norm=True)(x)
        return jnp.tanh(nn.Dense(self.action_dim, kernel_init=nn.initializers.zeros)(x))


class ResidualAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    prior: Any
    act_low: Any
    act_high: Any
    config: Any = nonpytree_field()

    def base_actions(self, observations, rng):
        return jnp.clip(self.prior.sample_actions(observations, seed=rng), self.act_low, self.act_high)

    def residual_actions(self, observations, base, noise_rng=None, noise_std=0.0, noise_clip=None, params=None):
        delta = self.network.select('actor')(observations, base, params=params)
        if noise_rng is not None:
            noise = noise_std * jax.random.normal(noise_rng, delta.shape)
            if noise_clip is not None:
                noise = jnp.clip(noise, -noise_clip, noise_clip)
            delta = jnp.clip(delta + noise, -1.0, 1.0)
        return jnp.clip(base + self.config['scale'] * delta, self.act_low, self.act_high)

    @partial(jax.jit, static_argnames=('explore',))
    def sample_actions(self, observations, seed, explore=True):
        base_rng, noise_rng = jax.random.split(seed)
        base = self.base_actions(observations, base_rng)
        if not explore:
            return self.residual_actions(observations, base)
        return self.residual_actions(observations, base, noise_rng, self.config['expl_noise'])

    @partial(jax.jit, static_argnames=('train_actor',))
    def update(self, batch, train_actor=True):
        rng, base_rng, noise_rng, actor_base_rng = jax.random.split(self.rng, 4)
        next_obs = batch['next_observations']
        next_base = self.base_actions(next_obs, base_rng)
        next_actions = self.residual_actions(
            next_obs, next_base, noise_rng, self.config['target_noise'], self.config['target_noise_clip']
        )
        next_q = self.network.select('target_critic')(next_obs, next_actions).min(axis=0)
        target = batch['rewards'] + batch['discounts'] * next_q
        base = self.base_actions(batch['observations'], actor_base_rng)

        def loss_fn(params):
            qs = self.network.select('critic')(batch['observations'], batch['actions'], params=params)
            critic_loss = ((qs - target[None]) ** 2).mean()
            delta = self.network.select('actor')(batch['observations'], base, params=params)
            actions = jnp.clip(base + self.config['scale'] * delta, self.act_low, self.act_high)
            # The stored critic params (no gradient): the actor loss only trains the actor.
            q_pi = self.network.select('critic')(batch['observations'], actions).mean(axis=0)
            q_loss = -(q_pi / jax.lax.stop_gradient(jnp.abs(q_pi).mean() + 1e-6)).mean()
            # L2 penalty on the residual keeps it from saturating at the bound where the chunk critic extrapolates.
            reg_loss = self.config['residual_reg'] * (delta**2).sum(axis=-1).mean()
            actor_loss = q_loss + reg_loss
            if not train_actor:
                actor_loss = 0.0 * actor_loss
            info = dict(
                critic_loss=critic_loss,
                actor_loss=actor_loss,
                q_mean=qs.mean(),
                target_mean=target.mean(),
                q_pi=q_pi.mean(),
                q_loss=q_loss,
                reg_loss=reg_loss,
                residual_abs=jnp.abs(delta).mean(),
            )
            return critic_loss + actor_loss, info

        network, info = self.network.apply_loss_fn(loss_fn)
        tau = self.config['tau']
        network.params['modules_target_critic'] = jax.tree_util.tree_map(
            lambda p, tp: tau * p + (1 - tau) * tp,
            network.params['modules_critic'],
            network.params['modules_target_critic'],
        )
        return self.replace(rng=rng, network=network), info

    @classmethod
    def create(cls, seed, prior, ex_observations, ex_actions, act_low, act_high, config):
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        critic_def = ensemblize(QNetwork, config['num_qs'])(hidden_dims=tuple(config['critic_hidden_dims']))
        actor_def = ResidualActor(hidden_dims=tuple(config['actor_hidden_dims']), action_dim=ex_actions.shape[-1])
        module = ModuleDict(dict(critic=critic_def, target_critic=critic_def, actor=actor_def))
        critic_args = (ex_observations, ex_actions)
        params = module.init(init_rng, critic=critic_args, target_critic=critic_args, actor=critic_args)['params']
        params['modules_target_critic'] = params['modules_critic']
        network = TrainState.create(module, params, tx=optax.adam(config['lr']))
        return cls(
            rng=rng,
            network=network,
            prior=prior,
            act_low=jnp.asarray(act_low),
            act_high=jnp.asarray(act_high),
            config=flax.core.FrozenDict(config),
        )
