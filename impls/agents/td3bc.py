"""Online fine-tuning of a deterministic BC prior with TD3+BC over action chunks.

The actor is the BC prior's network (agents/bc.py), initialized from its checkpoint and trained to maximize a chunk-level
critic, normalized by the batch mean |Q| as in TD3+BC, plus `bc_weight` times the squared error to the demonstration
chunks in the batch (the first `num_demo` rows). The critic is a LayerNorm ensemble whose TD target takes the min over
a random subset of `num_min_qs` members at the target actor's smoothed action. Exploration adds fixed Gaussian noise.
With `awr_temp` > 0 the actor also regresses onto all batch chunks weighted by exp(normalized advantage / awr_temp)
(AWR), which uses the critic only to weight data. `batch['rewards']`/`batch['discounts']` are the n-step quantities
over one chunk. Actor updates are scaled by
`actor_on` (0 during critic warmup).
"""

from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from agents.bon import QNetwork
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ensemblize


class TD3BCAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    act_low: Any
    act_high: Any
    config: Any = nonpytree_field()

    def actions(self, observations, module='actor', params=None):
        actions = self.network.select(module)(observations, params=params)[..., : self.config['exec_dim']]
        return jnp.clip(actions, self.act_low, self.act_high)

    @partial(jax.jit, static_argnames=('explore',))
    def sample_actions(self, observations, seed, explore=True):
        actions = self.actions(observations)
        if explore:
            actions = actions + self.config['expl_noise'] * jax.random.normal(seed, actions.shape)
        return jnp.clip(actions, self.act_low, self.act_high)

    def update(self, batch, actor_on):
        rng, noise_rng, subset_rng = jax.random.split(self.rng, 3)
        next_obs = batch['next_observations']
        noise = jnp.clip(
            self.config['target_noise'] * jax.random.normal(noise_rng, batch['actions'].shape),
            -self.config['target_noise_clip'],
            self.config['target_noise_clip'],
        )
        next_actions = jnp.clip(self.actions(next_obs, 'target_actor') + noise, self.act_low, self.act_high)
        next_qs = self.network.select('target_critic')(next_obs, next_actions)
        subset = jax.random.choice(subset_rng, self.config['num_qs'], (self.config['num_min_qs'],), replace=False)
        target = batch['rewards'] + batch['discounts'] * next_qs[subset].min(axis=0)
        # Returns are bounded above by `q_max` (all rewards <= 0): clipping the target stops the overestimation of
        # out-of-distribution actor chunks from propagating through bootstrapping.
        target = jnp.minimum(target, self.config['q_max'])

        def loss_fn(params):
            qs = self.network.select('critic')(batch['observations'], batch['actions'], params=params)
            critic_loss = ((qs - target[None]) ** 2).mean()

            actions = self.actions(batch['observations'], params=params)
            # The stored critic params (no gradient): the actor loss only trains the actor.
            q_pi = self.network.select('critic')(batch['observations'], actions).mean(axis=0)
            q_loss = -(q_pi / jax.lax.stop_gradient(jnp.abs(q_pi).mean() + 1e-6)).mean()
            num_demo = self.config['num_demo']
            bc_loss = ((actions[:num_demo] - batch['actions'][:num_demo]) ** 2).mean()
            actor_loss = self.config['q_weight'] * q_loss + self.config['bc_weight'] * bc_loss
            awr_loss, weights = 0.0, jnp.zeros(())
            if self.config['awr_temp'] > 0:
                # Advantage-weighted regression onto the batch chunks: the critic only weights the data, so the actor
                # never follows the critic's action gradient.
                q_data = jax.lax.stop_gradient(qs).mean(axis=0)
                adv = q_data - jax.lax.stop_gradient(q_pi)
                adv = adv / (adv.std() + 1e-6)
                weights = jnp.minimum(jnp.exp(adv / self.config['awr_temp']), self.config['awr_max_weight'])
                awr_loss = (weights * ((actions - batch['actions']) ** 2).mean(axis=-1)).mean()
                actor_loss = actor_loss + awr_loss
            info = dict(
                critic_loss=critic_loss,
                q_mean=qs.mean(),
                target_mean=target.mean(),
                q_pi=q_pi.mean(),
                q_loss=q_loss,
                bc_loss=bc_loss,
                awr_loss=awr_loss,
                awr_weight_max=weights.max(),
            )
            return critic_loss + actor_on * actor_loss, info

        network, info = self.network.apply_loss_fn(loss_fn)
        tau = self.config['tau']
        for name in ('critic', 'actor'):
            network.params[f'modules_target_{name}'] = jax.tree_util.tree_map(
                lambda p, tp: tau * p + (1 - tau) * tp,
                network.params[f'modules_{name}'],
                network.params[f'modules_target_{name}'],
            )
        return self.replace(rng=rng, network=network), info

    @jax.jit
    def update_many(self, batches, actor_on):
        """Run one update per leading index of `batches` (stacked batches) inside a scan."""

        def step(agent, batch):
            return agent.update(batch, actor_on)

        agent, infos = jax.lax.scan(step, self, batches)
        return agent, jax.tree_util.tree_map(lambda x: x[-1], infos)

    @classmethod
    def create(cls, seed, prior, ex_observations, ex_actions, act_low, act_high, config):
        """`prior` is a restored BCAgent; its actor network and parameters initialize the actor."""
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        critic_def = ensemblize(QNetwork, config['num_qs'])(hidden_dims=tuple(config['critic_hidden_dims']))
        actor_def = prior.network.model_def.modules['actor']
        module = ModuleDict(dict(critic=critic_def, target_critic=critic_def, actor=actor_def, target_actor=actor_def))
        critic_args = (ex_observations, ex_actions)
        params = module.init(
            init_rng, critic=critic_args, target_critic=critic_args, actor=(ex_observations,),
            target_actor=(ex_observations,),
        )['params']
        params = flax.core.unfreeze(params)
        params['modules_target_critic'] = params['modules_critic']
        params['modules_actor'] = prior.network.params['modules_actor']
        params['modules_target_actor'] = prior.network.params['modules_actor']
        labels = {k: 'actor' if k == 'modules_actor' else 'critic' for k in params}
        tx = optax.multi_transform(
            dict(actor=optax.adam(config['actor_lr']), critic=optax.adam(config['lr'])),
            lambda p: {k: jax.tree_util.tree_map(lambda _: labels[k], v) for k, v in p.items()},
        )
        network = TrainState.create(module, params, tx=tx)
        return cls(
            rng=rng,
            network=network,
            act_low=jnp.asarray(act_low),
            act_high=jnp.asarray(act_high),
            config=flax.core.FrozenDict(config, exec_dim=ex_actions.shape[-1]),
        )
