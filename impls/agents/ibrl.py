"""IBRL (Hu et al., 2023; github.com/hengyuan-hu/ibrl, `ibrl_soft`) at the control-step level.

A TD3 actor trained from scratch (with dropout) acts next to a frozen BC policy. At every step the agent proposes the
actor's action and the BC policy's action and samples one of the two from softmax(`beta` * Q), with Q the min over a
random subset of `num_min_qs` target critics; the TD target uses the same rule at the next state with the target actor.
The BC proposal is read from the observation: the caller appends it as the last `action_dim` entries (agents never
call the BC policy themselves, so a chunked BC policy can be executed open loop). The actor maximizes the min over all
critics and is updated once per `num_critic_updates` critic updates. `batch['rewards']`/`batch['discounts']` are the
n-step quantities.
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
from utils.networks import default_init, ensemblize


class DropoutActor(nn.Module):
    """Deterministic actor in [-1, 1]^d with dropout after every hidden layer (as in the IBRL state actor)."""

    hidden_dims: Sequence[int]
    action_dim: int
    dropout: float

    @nn.compact
    def __call__(self, observations, train=False):
        x = observations
        for size in self.hidden_dims:
            x = nn.Dense(size, kernel_init=default_init())(x)
            x = nn.Dropout(self.dropout, deterministic=not train)(x)
            x = nn.relu(x)
        return jnp.tanh(nn.Dense(self.action_dim, kernel_init=default_init())(x))


def soft_update(params, source, target, tau):
    params[f'modules_{target}'] = jax.tree_util.tree_map(
        lambda p, tp: tau * p + (1 - tau) * tp, params[f'modules_{source}'], params[f'modules_{target}']
    )
    return params


class IBRLAgent(flax.struct.PyTreeNode):
    rng: Any
    critic: Any
    actor: Any
    act_low: Any
    act_high: Any
    config: Any = nonpytree_field()

    def rl_actions(self, observations, rng, noise_std, noise_clip=None, module='actor', train=False, params=None):
        dropout_rng, noise_rng = jax.random.split(rng)
        u = self.actor.select(module)(observations, train, params=params, rngs={'dropout': dropout_rng})
        noise = noise_std * jax.random.normal(noise_rng, u.shape)
        if noise_clip is not None:
            noise = jnp.clip(noise, -noise_clip, noise_clip)
        u = jnp.clip(u + noise, -1.0, 1.0)
        return self.act_low + (u + 1.0) * 0.5 * (self.act_high - self.act_low)

    def choose(self, observations, rl_actions, rng):
        """Sample the actor's or the BC action from softmax(beta * Q); returns the actions and 1 where BC was chosen."""
        subset_rng, choice_rng = jax.random.split(rng)
        bc_actions = observations[..., -self.config['action_dim'] :]
        candidates = jnp.stack([rl_actions, bc_actions], axis=1)
        batch = observations.shape[0]
        qs = self.critic.select('target_critic')(jnp.repeat(observations, 2, axis=0), candidates.reshape(2 * batch, -1))
        subset = jax.random.choice(subset_rng, self.config['num_qs'], (self.config['num_min_qs'],), replace=False)
        q = qs[subset].min(axis=0).reshape(batch, 2)
        idx = jax.random.categorical(choice_rng, self.config['beta'] * q, axis=1)
        return candidates[jnp.arange(batch), idx], idx

    @partial(jax.jit, static_argnames=('explore',))
    def sample_actions(self, observations, seed, explore=True):
        actor_rng, choice_rng = jax.random.split(seed)
        noise = self.config['expl_noise'] if explore else 0.0
        return self.choose(observations, self.rl_actions(observations, actor_rng, noise), choice_rng)

    def critic_update(self, batch, rng):
        actor_rng, choice_rng, subset_rng = jax.random.split(rng, 3)
        next_obs = batch['next_observations']
        next_rl = self.rl_actions(
            next_obs, actor_rng, self.config['noise_std'], self.config['noise_clip'], 'target_actor', train=True
        )
        next_actions, next_bc = self.choose(next_obs, next_rl, choice_rng)
        next_qs = self.critic.select('target_critic')(next_obs, next_actions)
        subset = jax.random.choice(subset_rng, self.config['num_qs'], (self.config['num_min_qs'],), replace=False)
        target = batch['rewards'] + batch['discounts'] * next_qs[subset].min(axis=0)
        # Returns lie in [0, 1] (reward 1 once, on success): clipping the target keeps the overestimation of the actor's
        # out-of-distribution actions from compounding through bootstrapping.
        target = jnp.clip(target, 0.0, self.config['q_max'])

        def loss_fn(params):
            qs = self.critic.select('critic')(batch['observations'], batch['actions'], params=params)
            loss = ((qs - target[None]) ** 2).mean(axis=1).sum()
            return loss, dict(critic_loss=loss, q_mean=qs.mean(), target_mean=target.mean(), bootstrap_bc=next_bc.mean())

        critic, info = self.critic.apply_loss_fn(loss_fn)
        critic = critic.replace(params=soft_update(dict(critic.params), 'critic', 'target_critic', self.config['tau']))
        return self.replace(critic=critic), info

    def actor_update(self, batch, rng, bc):
        def loss_fn(params):
            if bc:
                # Imitate the BC proposal (the observation's last entries): the actor starts as a copy of the prior.
                actions = self.rl_actions(batch['observations'], rng, 0.0, train=True, params=params)
                loss = ((actions - batch['observations'][..., -self.config['action_dim'] :]) ** 2).mean()
                return loss, dict(actor_bc_loss=loss)
            actions = self.rl_actions(
                batch['observations'], rng, self.config['noise_std'], self.config['noise_clip'], train=True,
                params=params,
            )
            q = self.critic.select('critic')(batch['observations'], actions).min(axis=0)
            loss = -q.mean()
            return loss, dict(actor_loss=loss)

        actor, info = self.actor.apply_loss_fn(loss_fn)
        actor = actor.replace(params=soft_update(dict(actor.params), 'actor', 'target_actor', self.config['tau']))
        return self.replace(actor=actor), info

    @partial(jax.jit, static_argnames=('bc',))
    def update_many(self, batches, bc=False):
        """`batches` has leading axes [groups, num_critic_updates]: per group, that many critic updates and one actor
        update (on the group's last batch; with `bc`, regression onto the BC proposal instead of maximizing Q)."""

        def group_step(agent, group):
            def critic_step(agent, batch):
                rng, key = jax.random.split(agent.rng)
                agent, info = agent.replace(rng=rng).critic_update(batch, key)
                return agent, info

            agent, critic_infos = jax.lax.scan(critic_step, agent, group)
            rng, key = jax.random.split(agent.rng)
            last = jax.tree_util.tree_map(lambda x: x[-1], group)
            agent, actor_info = agent.replace(rng=rng).actor_update(last, key, bc)
            return agent, {**jax.tree_util.tree_map(lambda x: x[-1], critic_infos), **actor_info}

        agent, infos = jax.lax.scan(group_step, self, batches)
        return agent, jax.tree_util.tree_map(lambda x: x[-1], infos)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, act_low, act_high, config):
        rng, critic_rng, actor_rng = jax.random.split(jax.random.PRNGKey(seed), 3)
        critic_def = ensemblize(QNetwork, config['num_qs'])(hidden_dims=tuple(config['critic_hidden_dims']))
        critic_module = ModuleDict(dict(critic=critic_def, target_critic=critic_def))
        args = (ex_observations, ex_actions)
        params = dict(critic_module.init(critic_rng, critic=args, target_critic=args)['params'])
        params['modules_target_critic'] = params['modules_critic']
        critic = TrainState.create(critic_module, params, tx=optax.adam(config['lr']))

        actor_def = DropoutActor(
            hidden_dims=tuple(config['actor_hidden_dims']), action_dim=ex_actions.shape[-1], dropout=config['dropout']
        )
        actor_module = ModuleDict(dict(actor=actor_def, target_actor=actor_def))
        params = dict(actor_module.init(actor_rng, actor=(ex_observations,), target_actor=(ex_observations,))['params'])
        params['modules_target_actor'] = params['modules_actor']
        actor = TrainState.create(actor_module, params, tx=optax.adam(config['lr']))
        return cls(
            rng=rng,
            critic=critic,
            actor=actor,
            act_low=jnp.asarray(act_low),
            act_high=jnp.asarray(act_high),
            config=flax.core.FrozenDict(config, action_dim=ex_actions.shape[-1]),
        )
