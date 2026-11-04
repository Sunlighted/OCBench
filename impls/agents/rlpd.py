"""RLPD (Ball et al., 2023) over action chunks (as in Q-chunking, Li et al., 2025).

SAC with a tanh-Gaussian actor over flattened action chunks (no entropy term in the TD target by default: with a
long effective horizon it swamps the task reward), a LayerNorm critic ensemble whose TD target takes the min
over a random subset of `num_min_qs` members, and automatic entropy tuning. Batches mix demonstrations and online data
(done by the caller). `batch['rewards']`/`batch['discounts']` are the n-step quantities over one chunk.
`bc_update` trains the actor alone by maximum likelihood on demonstration chunks (optional BC initialization).
"""

from functools import partial
from typing import Any, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.bon import QNetwork
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, LogParam, ResidualMLP, TransformedWithMode, ensemblize


class ChunkActor(nn.Module):
    """Tanh-Gaussian policy over a flattened action chunk, in [-1, 1]^d (mapped to the action box by the agent)."""

    hidden_dims: Sequence[int]
    action_dim: int
    residual: bool = False
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, observations):
        if self.residual:
            out = ResidualMLP(self.hidden_dims, 2 * self.action_dim)(observations)
        else:
            out = MLP((*self.hidden_dims, 2 * self.action_dim), layer_norm=True)(observations)
        means, log_stds = jnp.split(out, 2, axis=-1)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        dist = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return TransformedWithMode(dist, distrax.Block(distrax.Tanh(), ndims=1))


class RLPDAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    act_low: Any
    act_high: Any
    config: Any = nonpytree_field()

    def to_box(self, u):
        return self.act_low + (u + 1.0) * 0.5 * (self.act_high - self.act_low)

    def from_box(self, actions):
        u = 2.0 * (actions - self.act_low) / (self.act_high - self.act_low) - 1.0
        return jnp.clip(u, -0.999, 0.999)

    @partial(jax.jit, static_argnames=('explore',))
    def sample_actions(self, observations, seed, explore=True):
        dist = self.network.select('actor')(observations)
        u = dist.sample(seed=seed) if explore else dist.mode()
        return self.to_box(u)

    def critic_loss(self, batch, params, rng):
        rng, sample_rng, subset_rng = jax.random.split(rng, 3)
        next_dist = self.network.select('actor')(batch['next_observations'])
        next_u, next_log_probs = next_dist.sample_and_log_prob(seed=sample_rng)
        next_qs = self.network.select('target_critic')(batch['next_observations'], self.to_box(next_u))
        subset = jax.random.choice(subset_rng, self.config['num_qs'], (self.config['num_min_qs'],), replace=False)
        next_q = next_qs[subset].min(axis=0)
        if self.config['backup_entropy']:
            next_q = next_q - self.network.select('alpha')() * next_log_probs
        target = batch['rewards'] + batch['discounts'] * next_q

        qs = self.network.select('critic')(batch['observations'], batch['actions'], params=params)
        loss = ((qs - target[None]) ** 2).mean()
        return loss, dict(critic_loss=loss, q_mean=qs.mean(), target_mean=target.mean())

    def actor_loss(self, batch, params, rng):
        dist = self.network.select('actor')(batch['observations'], params=params)
        u, log_probs = dist.sample_and_log_prob(seed=rng)
        q = self.network.select('critic')(batch['observations'], self.to_box(u)).mean(axis=0)
        alpha = self.network.select('alpha')()
        actor_loss = (alpha * log_probs - q).mean()

        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_param = self.network.select('alpha')(params=params)
        alpha_loss = alpha_param * (entropy - self.config['target_entropy'])
        return actor_loss + alpha_loss, dict(actor_loss=actor_loss, entropy=entropy, alpha=alpha)

    def soft_update_target(self, network):
        tau = self.config['tau']
        network.params['modules_target_critic'] = jax.tree_util.tree_map(
            lambda p, tp: tau * p + (1 - tau) * tp,
            network.params['modules_critic'],
            network.params['modules_target_critic'],
        )
        return network

    def update(self, batch):
        rng, critic_rng, actor_rng = jax.random.split(self.rng, 3)

        def loss_fn(params):
            critic_loss, critic_info = self.critic_loss(batch, params, critic_rng)
            actor_loss, actor_info = self.actor_loss(batch, params, actor_rng)
            return critic_loss + actor_loss, {**critic_info, **actor_info}

        network, info = self.network.apply_loss_fn(loss_fn)
        return self.replace(rng=rng, network=self.soft_update_target(network)), info

    @jax.jit
    def update_many(self, batches):
        """Run one update per leading index of `batches` (stacked batches) inside a scan."""

        def step(agent, batch):
            return agent.update(batch)

        agent, infos = jax.lax.scan(step, self, batches)
        return agent, jax.tree_util.tree_map(lambda x: x[-1], infos)

    @jax.jit
    def bc_update_many(self, batches):
        """Maximum-likelihood actor updates on demonstration chunks (stacked batches)."""

        def step(agent, batch):
            def loss_fn(params):
                dist = agent.network.select('actor')(batch['observations'], params=params)
                loss = -dist.log_prob(agent.from_box(batch['actions'])).mean()
                return loss, dict(bc_loss=loss)

            network, info = agent.network.apply_loss_fn(loss_fn)
            return agent.replace(network=network), info

        agent, infos = jax.lax.scan(step, self, batches)
        return agent, jax.tree_util.tree_map(lambda x: x[-1], infos)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, act_low, act_high, config):
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        action_dim = ex_actions.shape[-1]
        critic_def = ensemblize(QNetwork, config['num_qs'])(hidden_dims=tuple(config['critic_hidden_dims']))
        actor_def = ChunkActor(
            hidden_dims=tuple(config['actor_hidden_dims']), action_dim=action_dim, residual=config['actor_resmlp']
        )
        alpha_def = LogParam(init_value=config['init_alpha'])
        module = ModuleDict(dict(critic=critic_def, target_critic=critic_def, actor=actor_def, alpha=alpha_def))
        critic_args = (ex_observations, ex_actions)
        params = module.init(
            init_rng, critic=critic_args, target_critic=critic_args, actor=(ex_observations,), alpha=()
        )['params']
        params['modules_target_critic'] = params['modules_critic']
        network = TrainState.create(module, params, tx=optax.adam(config['lr']))
        config = dict(config, target_entropy=config.get('target_entropy') or -0.5 * action_dim)
        return cls(
            rng=rng,
            network=network,
            act_low=jnp.asarray(act_low),
            act_high=jnp.asarray(act_high),
            config=flax.core.FrozenDict(config),
        )
