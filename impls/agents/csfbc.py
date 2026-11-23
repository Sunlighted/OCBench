from typing import Any, ClassVar

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.loss_utils import sample_beta_times
from utils.networks import MLP_CLASSES, ActorVectorField


class CSFBCAgent(flax.struct.PyTreeNode):
    """Context-smoothed flow BC (the state-noising pretraining of TMRL; Hu et al., 2026, github.com/matthewh6/tmrl).

    The flow is conditioned only on a diffused copy of the standardized observation and its diffusion timestep t_c
    (DDPM schedule with betas linspace(1e-4, 0.02, `num_context_steps`)): at t_c = 0 it is ordinary flow BC, and larger
    t_c makes it act as at nearby states. `decode` lets a high-level policy choose the flow noise, the context noise and
    t_c (TMRL); `sample_actions` uses a clean context. Observation statistics are taken from the example batch at
    creation and stored with the agent.
    """

    sample_kwargs: ClassVar[dict] = dict(keys=('observations', 'actions'))

    rng: Any
    network: Any
    obs_mean: Any
    obs_std: Any
    config: Any = nonpytree_field()

    def alpha_bars(self):
        steps = self.config['num_context_steps']
        return jnp.cumprod(1.0 - jnp.linspace(1e-4, 0.02, steps))

    def context(self, observations, context_noise, t_context):
        """The diffused standardized observation and t_c, as the flow's conditioning input."""
        steps = self.config['num_context_steps']
        idx = jnp.clip((t_context * steps).astype(jnp.int32), 0, steps - 1)
        ab = self.alpha_bars()[idx][..., None]
        z = (observations - self.obs_mean) / self.obs_std
        # t_c enters with sinusoidal features so the network can switch behavior on the noise level.
        freqs = jnp.arange(1, 5) * jnp.pi
        t = t_context[..., None]
        t_feats = jnp.concatenate([t, jnp.sin(freqs * t), jnp.cos(freqs * t)], axis=-1)
        return jnp.concatenate([jnp.sqrt(ab) * z + jnp.sqrt(1 - ab) * context_noise, t_feats], axis=-1)

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        x_rng, t_rng, c_rng, tc_rng = jax.random.split(rng, 4)
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = sample_beta_times(t_rng, (batch_size, 1), self.config['betas'])
        x_t = (1 - t) * x_0 + t * x_1
        obs = batch['observations']
        tc_rng, clean_rng = jax.random.split(tc_rng)
        t_context = jax.random.uniform(tc_rng, (batch_size,))
        # A `clean_frac` share of samples uses the clean context (t_c = 0), the one used when acting without RL.
        t_context = jnp.where(jax.random.uniform(clean_rng, (batch_size,)) < self.config['clean_frac'], 0.0, t_context)
        cond = self.context(obs, jax.random.normal(c_rng, obs.shape), t_context)
        pred = self.network.select('actor_flow')(cond, x_t, t, params=grad_params)
        actor_loss = jnp.mean((pred - (x_1 - x_0)) ** 2)
        return actor_loss, {'actor_loss': actor_loss}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        actor_loss, actor_info = self.actor_loss(batch, grad_params, rng)
        return actor_loss, {f'actor/{k}': v for k, v in actor_info.items()}

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=lambda p: self.total_loss(batch, p, rng=rng))
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def compute_metrics(self, batch, rng):
        errors = self.sample_actions(batch['observations'], seed=rng) - batch['actions']
        return {'actor/mse': jnp.mean(errors**2), 'actor/mae': jnp.mean(jnp.abs(errors))}

    @jax.jit
    def decode(self, observations, action_noise, context_noise, t_context):
        """Integrate the flow from `action_noise` conditioned on the context diffused to `t_context`."""
        cond = self.context(observations, context_noise, t_context)
        actions = action_noise
        steps = self.config['flow_steps']
        for i in range(steps):
            t = jnp.full((*observations.shape[:-1], 1), i / steps)
            actions = actions + self.network.select('actor_flow')(cond, actions, t) / steps
        return actions

    @jax.jit
    def sample_actions(self, observations, seed=None):
        """Flow BC with a clean context (t_c = 0)."""
        batch_shape = observations.shape[:-1]
        noises = jax.random.normal(seed, (*batch_shape, self.config['action_dim']))
        return self.decode(observations, noises, jnp.zeros_like(observations), jnp.zeros(batch_shape))

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        action_dim = ex_actions.shape[-1]
        actor_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            mlp_class=MLP_CLASSES[config['mlp_class']],
            layer_norm=config['actor_layer_norm'],
        )
        ex_cond = jnp.concatenate([ex_observations, jnp.zeros((*ex_observations.shape[:-1], 9))], axis=-1)
        network_def = ModuleDict(dict(actor_flow=actor_def))
        params = network_def.init(init_rng, actor_flow=(ex_cond, ex_actions, ex_actions[..., :1]))['params']
        network = TrainState.create(network_def, params, tx=optax.adam(learning_rate=config['lr']))
        # Standardization statistics from the example batch (std floored at 0.1); restored with the agent.
        obs_mean = ex_observations.mean(axis=0)
        obs_std = jnp.maximum(ex_observations.std(axis=0), 0.1)
        if not config['standardize']:
            # Raw observations (already roughly unit-scaled by the environment), as the flow BC prior uses.
            obs_mean, obs_std = jnp.zeros_like(obs_mean), jnp.ones_like(obs_std)
        config['ob_dims'] = ex_observations.shape[1:]
        config['action_dim'] = action_dim
        return cls(rng, network=network, obs_mean=obs_mean, obs_std=obs_std, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='csfbc',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=1e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor network hidden dimensions.
            mlp_class='resmlp',  # MLP class ('mlp' or 'resmlp').
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            flow_steps=10,  # Number of flow steps.
            betas=(1.0, 1.5),  # Beta parameters for time sampling.
            num_context_steps=100,  # Diffusion steps of the context-noise schedule.
            clean_frac=0.0,  # Fraction of training samples with the clean context (0 in TMRL).
            standardize=True,  # Whether to standardize observations by the example batch statistics before noising.
        )
    )
    return config
