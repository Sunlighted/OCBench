"""Reversal Q-Learning (RQL; Oberai et al., 2026; github.com/aoberai/rql) over action chunks, from a flow BC prior.

The critic V(s, x, t) scores intermediate states of the policy's flow (x at flow time t). It is trained on the data
chunk reversed through the current flow by a random time d (to x_{1-d} at time 1-d), with an expectile TD target
r + gamma^n V'(s', x_0 ~ N(0, I), 0), where V' is the target ensemble's mean minus `rho` times its std. The flow actor
(the prior's velocity network) maximizes V one Euler step ahead of interpolated points, plus `alpha` times the flow
matching (BC) loss, computed on the first `num_bc` rows of the batch (the demonstrations; the whole batch when 0).
Actions are sampled from an EMA copy of the actor. `batch['rewards']`/`batch['discounts']` are the n-step quantities.
"""

import copy
from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, ensemblize


class FlowStateValue(nn.Module):
    """V(s, x, t) on a flow state (scalar output)."""

    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations, actions, times):
        x = jnp.concatenate([observations, actions, times], axis=-1)
        return MLP((*self.hidden_dims, 1), activate_final=False, layer_norm=True)(x).squeeze(-1)


class RQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def values(self, observations, actions, times, module='value', params=None):
        return self.network.select(module)(observations, actions, times, params=params)

    def total_loss(self, batch, grad_params, rng):
        rng, noise_rng, u_rng, k_rng, x_rng, t_rng = jax.random.split(rng, 6)
        obs, actions = batch['observations'], batch['actions']
        batch_size, action_dim = actions.shape
        steps = self.config['flow_steps']

        # TD target: the value of the next state at pure noise (the start of the flow).
        next_obs = batch['next_observations']
        next_vs = self.values(
            next_obs, jax.random.normal(noise_rng, (batch_size, action_dim)), jnp.zeros((batch_size, 1)),
            module='target_value',
        )
        next_v = next_vs.mean(axis=0) - self.config['rho'] * next_vs.std(axis=0)
        target = batch['rewards'] + batch['discounts'] * next_v
        if self.config['v_max'] > 0:
            # Sparse 0/1 rewards bound returns to [0, v_max]; clipping stops overestimation from compounding.
            target = jnp.clip(target, 0.0, self.config['v_max'])

        # Reverse the data chunk through the current flow by d (half uniform, half on the Euler grid).
        d = jnp.concatenate([
            jax.random.uniform(u_rng, (batch_size // 2,)),
            jax.random.randint(k_rng, (batch_size - batch_size // 2,), 0, steps + 1) / steps,
        ])[:, None]
        x_f, f = actions, jnp.ones((batch_size, 1))
        for _ in range(steps):
            vel = self.network.select('actor')(obs, x_f, f)
            x_f = x_f - vel * d / steps
            f = f - d / steps
        vs = self.values(obs, jax.lax.stop_gradient(x_f), f, params=grad_params)
        diff = target[None] - vs
        weight = jnp.where(diff >= 0, self.config['expectile'], 1 - self.config['expectile'])
        critic_loss = (weight * diff**2).mean()

        # Actor: flow matching (BC) plus the value one Euler step ahead of interpolated points.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * actions
        pred = self.network.select('actor')(obs, x_t, t, params=grad_params)
        num_bc = self.config['num_bc'] or batch_size
        bc_loss = ((pred - (actions - x_0)) ** 2)[:num_bc].mean()
        dt = jnp.minimum(1.0 / steps, 1.0 - t)
        q_pe = self.values(obs, x_t + pred * dt, jnp.minimum(t + 1.0 / steps, 1.0)).mean(axis=0)
        actor_loss = -q_pe.mean()

        loss = critic_loss + actor_loss + self.config['alpha'] * bc_loss
        return loss, dict(
            critic_loss=critic_loss, actor_loss=actor_loss, bc_loss=bc_loss, v_mean=vs.mean(), target_mean=target.mean(),
            q_pe=q_pe.mean(),
        )

    @jax.jit
    def update(self, batch):
        rng, loss_rng = jax.random.split(self.rng)
        network, info = self.network.apply_loss_fn(lambda p: self.total_loss(batch, p, loss_rng))
        for name, rate in (('value', self.config['tau']), ('actor', 1.0 - self.config['ema'])):
            network.params[f'modules_target_{name}'] = jax.tree_util.tree_map(
                lambda p, tp: rate * p + (1 - rate) * tp,
                network.params[f'modules_{name}'],
                network.params[f'modules_target_{name}'],
            )
        return self.replace(rng=rng, network=network), info

    @partial(jax.jit, static_argnames=('explore',))
    def sample_actions(self, observations, seed, explore=True):
        """Integrate the EMA actor's flow from Gaussian noise (the flow noise is the only exploration). Unclipped, as
        for the FBC prior: the caller clips to the action box."""
        actions = jax.random.normal(seed, (*observations.shape[:-1], self.config['action_dim']))
        steps = self.config['flow_steps']
        for i in range(steps):
            t = jnp.full((*observations.shape[:-1], 1), i / steps)
            actions = actions + self.network.select('target_actor')(observations, actions, t) / steps
        return actions

    @classmethod
    def create(cls, seed, prior, ex_observations, ex_actions, config):
        """`prior` is a restored FBCAgent: its velocity network and parameters initialize the actor."""
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        actor_def = prior.network.model_def.modules['actor_flow']
        value_def = ensemblize(FlowStateValue, config['num_qs'])(hidden_dims=tuple(config['value_hidden_dims']))
        ex_times = ex_actions[..., :1]
        module = ModuleDict(dict(
            actor=actor_def, target_actor=copy.deepcopy(actor_def), value=value_def, target_value=copy.deepcopy(value_def)
        ))
        args = (ex_observations, ex_actions, ex_times)
        params = dict(module.init(init_rng, actor=args, target_actor=args, value=args, target_value=args)['params'])
        params['modules_actor'] = prior.network.params['modules_actor_flow']
        params['modules_target_actor'] = prior.network.params['modules_actor_flow']
        params['modules_target_value'] = params['modules_value']
        network = TrainState.create(module, params, tx=optax.adam(config['lr']))
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(config, action_dim=ex_actions.shape[-1]))
