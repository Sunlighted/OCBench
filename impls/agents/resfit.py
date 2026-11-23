"""ResFiT (Ankile et al., 2025; github.com/amazon-far/residual-offpolicy-rl): residual TD3 at the control-step level.

The executed action is clip(a_base + r), where a_base is the frozen base policy's action (read from the observation:
the caller appends it as the last `action_dim` entries, so a chunked base policy can run open loop) and
r = `action_scale` * tanh(f(s)) comes from a residual actor whose output layer is zero-initialized. The critic Q(s, a)
scores the combined action; the TD target uses the target actor with clipped Gaussian noise and the min over a random
pair of a LayerNorm ensemble. The actor maximizes the ensemble-mean Q with a very small learning rate, and both
networks clip gradients. Observations are standardized inside the networks. `batch['rewards']`/`batch['discounts']`
are the n-step quantities.
"""

from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.ibrl import soft_update
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ensemblize


class LNMLP(nn.Module):
    """Dense -> LayerNorm -> ReLU blocks (the ResFiT actor and critic trunks)."""

    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_dims:
            x = nn.relu(nn.LayerNorm()(nn.Dense(size, kernel_init=nn.initializers.orthogonal())(x)))
        return x


class Critic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations, actions):
        x = LNMLP(self.hidden_dims)(jnp.concatenate([observations, actions], axis=-1))
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal())(x).squeeze(-1)


class ResidualActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    action_scale: float
    residual_mask: Sequence[float]

    @nn.compact
    def __call__(self, observations):
        x = LNMLP(self.hidden_dims)(observations)
        r = self.action_scale * jnp.tanh(nn.Dense(self.action_dim, kernel_init=nn.initializers.zeros)(x))
        return r * jnp.asarray(self.residual_mask)


class ResFiTAgent(flax.struct.PyTreeNode):
    rng: Any
    critic: Any
    actor: Any
    obs_mean: Any
    obs_std: Any
    act_low: Any
    act_high: Any
    config: Any = nonpytree_field()

    def norm(self, observations):
        return (observations - self.obs_mean) / self.obs_std

    def combine(self, observations, residual):
        base = observations[..., -self.config['action_dim'] :]
        return jnp.clip(base + residual, self.act_low, self.act_high)

    def residual(self, observations, rng, noise_std, noise_clip=None, module='actor', params=None):
        r = self.actor.select(module)(self.norm(observations), params=params)
        noise = noise_std * jax.random.normal(rng, r.shape)
        if noise_clip is not None:
            noise = jnp.clip(noise, -noise_clip, noise_clip)
        return r + noise

    @partial(jax.jit, static_argnames=('explore', 'random'))
    def sample_actions(self, observations, seed, explore=True, random=False):
        """Returns the actions and the mean |residual| per row."""
        if random:
            # Warmup exploration: the base action plus uniform noise of `random_scale`.
            r = jax.random.uniform(
                seed, (*observations.shape[:-1], self.config['action_dim']),
                minval=-self.config['random_scale'], maxval=self.config['random_scale'],
            )
        else:
            r = self.residual(observations, seed, self.config['expl_noise'] if explore else 0.0)
            r = r * jnp.asarray(self.config['residual_mask'])
        return self.combine(observations, r), jnp.abs(r).mean(axis=-1)

    def critic_update(self, batch, rng):
        noise_rng, subset_rng = jax.random.split(rng)
        next_obs = batch['next_observations']
        next_r = self.residual(next_obs, noise_rng, self.config['expl_noise'], self.config['noise_clip'], 'target_actor')
        next_qs = self.critic.select('target_critic')(self.norm(next_obs), self.combine(next_obs, next_r))
        subset = jax.random.choice(subset_rng, self.config['num_qs'], (self.config['num_min_qs'],), replace=False)
        target = batch['rewards'] + batch['discounts'] * next_qs[subset].min(axis=0)
        if self.config['q_max'] > 0:
            target = jnp.clip(target, 0.0, self.config['q_max'])

        def loss_fn(params):
            qs = self.critic.select('critic')(self.norm(batch['observations']), batch['actions'], params=params)
            loss = ((qs - target[None]) ** 2).mean()
            return loss, dict(critic_loss=loss, q_mean=qs.mean(), target_mean=target.mean())

        critic, info = self.critic.apply_loss_fn(loss_fn)
        critic = critic.replace(params=soft_update(dict(critic.params), 'critic', 'target_critic', self.config['tau']))
        return self.replace(critic=critic), info

    def actor_update(self, batch, rng):
        obs = batch['observations']

        def loss_fn(params):
            r = self.residual(obs, rng, 0.0, params=params)
            q = self.critic.select('critic')(self.norm(obs), self.combine(obs, r)).mean(axis=0)
            # L2 penalty on the residual in units of its bound (ResFiT's `action_l2_reg_weight`).
            reg = self.config['residual_reg'] * ((r / self.config['action_scale']) ** 2).sum(axis=-1).mean()
            loss = -q.mean() + reg
            return loss, dict(actor_loss=loss, reg_loss=reg, residual_abs=jnp.abs(r).mean())

        actor, info = self.actor.apply_loss_fn(loss_fn)
        actor = actor.replace(params=soft_update(dict(actor.params), 'actor', 'target_actor', self.config['tau']))
        return self.replace(actor=actor), info

    @partial(jax.jit, static_argnames=('train_actor',))
    def update_many(self, batches, train_actor=True):
        """`batches` has leading axes [groups, num_critic_updates]: per group, that many critic updates and (with
        `train_actor`) one actor update on the group's last batch."""

        def group_step(agent, group):
            def critic_step(agent, batch):
                rng, key = jax.random.split(agent.rng)
                return agent.replace(rng=rng).critic_update(batch, key)

            agent, infos = jax.lax.scan(critic_step, agent, group)
            infos = jax.tree_util.tree_map(lambda x: x[-1], infos)
            if train_actor:
                rng, key = jax.random.split(agent.rng)
                agent, actor_info = agent.replace(rng=rng).actor_update(jax.tree_util.tree_map(lambda x: x[-1], group), key)
                infos = {**infos, **actor_info}
            return agent, infos

        agent, infos = jax.lax.scan(group_step, self, batches)
        return agent, jax.tree_util.tree_map(lambda x: x[-1], infos)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, act_low, act_high, obs_mean, obs_std, config):
        rng, critic_rng, actor_rng = jax.random.split(jax.random.PRNGKey(seed), 3)
        critic_def = ensemblize(Critic, config['num_qs'])(hidden_dims=tuple(config['critic_hidden_dims']))
        critic_module = ModuleDict(dict(critic=critic_def, target_critic=critic_def))
        args = (ex_observations, ex_actions)
        params = dict(critic_module.init(critic_rng, critic=args, target_critic=args)['params'])
        params['modules_target_critic'] = params['modules_critic']
        critic_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(config['critic_lr']))
        critic = TrainState.create(critic_module, params, tx=critic_tx)

        actor_def = ResidualActor(
            hidden_dims=tuple(config['actor_hidden_dims']), action_dim=ex_actions.shape[-1],
            action_scale=config['action_scale'], residual_mask=tuple(config['residual_mask']),
        )
        actor_module = ModuleDict(dict(actor=actor_def, target_actor=actor_def))
        params = dict(actor_module.init(actor_rng, actor=(ex_observations,), target_actor=(ex_observations,))['params'])
        params['modules_target_actor'] = params['modules_actor']
        actor_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(config['actor_lr']))
        actor = TrainState.create(actor_module, params, tx=actor_tx)
        return cls(
            rng=rng,
            critic=critic,
            actor=actor,
            obs_mean=jnp.asarray(obs_mean),
            obs_std=jnp.asarray(obs_std),
            act_low=jnp.asarray(act_low),
            act_high=jnp.asarray(act_high),
            config=flax.core.FrozenDict(config, action_dim=ex_actions.shape[-1]),
        )
