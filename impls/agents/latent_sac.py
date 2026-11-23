"""SAC over the latent inputs of a frozen flow BC prior: DSRL (Wagenmaker et al., 2025) and TMRL (Hu et al., 2026).

The actor outputs a tanh-squashed latent u in [-1, 1]^d made of named segments: the flow noise (`noise`, action_dim
entries, repeated over the chunk as in the TMRL code), and for TMRL also the context noise (`context`, ob_dim entries)
and the context diffusion time (`time`, 1 entry). The caller maps u to the prior's inputs and decodes a chunk. The
critic Q(s, u) is a LayerNorm ensemble; its TD target takes the min over a random pair and, by default, the SAC entropy
bonus. Each segment has its own automatically tuned temperature with target entropy -(segment size), as in TMRL.
`batch['rewards']`/`batch['discounts']` are the per-chunk quantities.
"""

from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, LogParam, ensemblize


class LatentCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations, latents):
        x = jnp.concatenate([observations, latents], axis=-1)
        return MLP((*self.hidden_dims, 1), activate_final=False, layer_norm=True)(x).squeeze(-1)


class LatentActor(nn.Module):
    """Tanh-Gaussian over the latent; returns (u, per-dimension log-probabilities)."""

    hidden_dims: Sequence[int]
    latent_dim: int

    @nn.compact
    def __call__(self, observations, rng, deterministic=False):
        x = MLP(self.hidden_dims, activate_final=True, layer_norm=True)(observations)
        mean = nn.Dense(self.latent_dim)(x)
        log_std = jnp.clip(nn.Dense(self.latent_dim)(x), -5.0, 2.0)
        if deterministic:
            return jnp.tanh(mean), jnp.zeros_like(mean)
        eps = jax.random.normal(rng, mean.shape)
        pre = mean + jnp.exp(log_std) * eps
        u = jnp.tanh(pre)
        log_probs = -0.5 * eps**2 - log_std - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(1 - u**2 + 1e-6)
        return u, log_probs


class LatentSACAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def segment_log_probs(self, log_probs):
        """Per-segment summed log-probabilities, shape [num_segments, batch]."""
        out, start = [], 0
        for size in self.config['segment_sizes']:
            out.append(log_probs[..., start : start + size].sum(-1))
            start += size
        return jnp.stack(out)

    def alphas(self, params=None):
        return jnp.stack([self.network.select(f'alpha_{i}')(params=params) for i in range(len(self.config['segment_sizes']))])

    @partial(jax.jit, static_argnames=('deterministic',))
    def sample_latents(self, observations, seed, deterministic=False):
        u, _ = self.network.select('actor')(observations, seed, deterministic)
        return u

    def update(self, batch, rng):
        next_rng, actor_rng, subset_rng = jax.random.split(rng, 3)
        next_u, next_lp = self.network.select('actor')(batch['next_observations'], next_rng)
        next_qs = self.network.select('target_critic')(batch['next_observations'], next_u)
        subset = jax.random.choice(subset_rng, self.config['num_qs'], (2,), replace=False)
        next_q = next_qs[subset].min(axis=0)
        if self.config['backup_entropy']:
            next_q = next_q - (self.alphas()[:, None] * self.segment_log_probs(next_lp)).sum(0)
        target = batch['rewards'] + batch['discounts'] * next_q

        def loss_fn(params):
            qs = self.network.select('critic')(batch['observations'], batch['latents'], params=params)
            critic_loss = ((qs - target[None]) ** 2).mean()
            u, lp = self.network.select('actor')(batch['observations'], actor_rng, params=params)
            q_pi = self.network.select('critic')(batch['observations'], u).mean(axis=0)
            seg_lp = self.segment_log_probs(lp)
            actor_loss = ((self.alphas()[:, None] * seg_lp).sum(0) - q_pi).mean()
            entropies = -jax.lax.stop_gradient(seg_lp).mean(axis=1)
            targets = -jnp.asarray(self.config['segment_sizes'], jnp.float32)
            alpha_loss = (self.alphas(params=params) * (entropies - targets)).sum()
            info = dict(critic_loss=critic_loss, q_mean=qs.mean(), target_mean=target.mean(), actor_loss=actor_loss)
            for i, name in enumerate(self.config['segment_names']):
                info[f'entropy_{name}'] = entropies[i]
                info[f'alpha_{name}'] = self.alphas()[i]
            return critic_loss + actor_loss + alpha_loss, info

        network, info = self.network.apply_loss_fn(loss_fn)
        tau = self.config['tau']
        network.params['modules_target_critic'] = jax.tree_util.tree_map(
            lambda p, tp: tau * p + (1 - tau) * tp, network.params['modules_critic'], network.params['modules_target_critic']
        )
        return self.replace(network=network), info

    @jax.jit
    def update_many(self, batches):
        """One update per leading index of `batches` inside a scan."""

        def step(agent, batch):
            rng, key = jax.random.split(agent.rng)
            return agent.replace(rng=rng).update(batch, key)

        agent, infos = jax.lax.scan(step, self, batches)
        return agent, jax.tree_util.tree_map(lambda x: x[-1], infos)

    @classmethod
    def create(cls, seed, ex_observations, segments, config):
        """`segments` is a list of (name, size) pairs that make up the latent."""
        rng, init_rng, actor_rng = jax.random.split(jax.random.PRNGKey(seed), 3)
        latent_dim = sum(size for _, size in segments)
        ex_latents = jnp.zeros((1, latent_dim))
        critic_def = ensemblize(LatentCritic, config['num_qs'])(hidden_dims=tuple(config['hidden_dims']))
        modules = dict(
            critic=critic_def, target_critic=critic_def,
            actor=LatentActor(hidden_dims=tuple(config['hidden_dims']), latent_dim=latent_dim),
        )
        modules.update({f'alpha_{i}': LogParam(init_value=config['init_alpha']) for i in range(len(segments))})
        module = ModuleDict(modules)
        init_args = dict(
            critic=(ex_observations, ex_latents), target_critic=(ex_observations, ex_latents),
            actor=(ex_observations, actor_rng),
        )
        init_args.update({f'alpha_{i}': () for i in range(len(segments))})
        params = module.init(init_rng, **init_args)['params']
        params['modules_target_critic'] = params['modules_critic']
        network = TrainState.create(module, params, tx=optax.adam(config['lr']))
        config = dict(
            config, segment_names=tuple(n for n, _ in segments), segment_sizes=tuple(s for _, s in segments)
        )
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(config))
