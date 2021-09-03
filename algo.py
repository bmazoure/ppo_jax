from abc import abstractmethod
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.random import PRNGKey
import optax

import flax.linen as nn
import flax

from policies import *
from buffer import RolloutBuffer
from models import TwinHeadModel

from gym.spaces import Box, Discrete

from utils import fake_state, clip_gradient_norm, replace_params
# from tensorflow_probability.substrates import jax as tfp
# tfd = tfp.distributions
# tfb = tfp.bijectors

class PPO():
    name = "PPO"

    def __init__(self,
                 num_agent_steps,
                 state_space,
                 action_space,
                 seed,
                 max_grad_norm=10.0,
                 gamma=0.995,
                 n_steps=256,
                 num_envs=64,
                 n_minibatch=8,
                 lr=3e-4,
                 epoch_ppo=10,
                 clip_eps=0.2,
                 lambd=0.97,
                 entropy_coeff=0.1,
                 critic_coeff=1):

        batch_size = num_envs // n_minibatch
        np.random.seed(seed)
        self.key = PRNGKey(seed)

        self.agent_step = 0
        self.episode_step = 0
        self.learning_step = 0
        self.num_agent_steps = num_agent_steps
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.discrete_action = False if type(action_space) == Box else True

        self.buffer = RolloutBuffer(
            n_steps=n_steps,
            num_envs=num_envs,
            batch_size=batch_size,
            state_space=state_space,
            action_space=action_space,
        )
        self.discount = gamma
        self.batch_size = batch_size

        self.n_steps = n_steps
        self.num_envs = num_envs

        # Critic.
        self.model = TwinHeadModel(action_space=action_space,
                                   prefix_critic='vfunction',
                                   prefix_actor="policy")
        self.fake_args_model = jnp.zeros((1, 64, 64, 3))
        self.params_model = self.model.init(self.key, self.fake_args_model)

        opt_init, self.opt_update = optax.chain(
            optax.clip_by_global_norm(
                self.max_grad_norm
            ),  # Clip by the gradient by the global norm.
            optax.adam(lr))

        self.opt_state = opt_init(self.params_model)

        # Other parameters.
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.max_grad_norm = max_grad_norm
        self.entropy_coeff = entropy_coeff
        self.critic_coeff = critic_coeff
        self.n_minibatch = n_minibatch

        self.n_steps = n_steps
        self.num_envs = num_envs

        self.idxes = np.arange(num_envs * n_steps)

        self.update_step = 0

    @partial(jax.jit, static_argnums=0)
    def select_action(
        self,
        state: np.ndarray,
    ) -> jnp.ndarray:
        _, pi_s, log_pi_s = self.model.apply(self.params_model, state)
        return log_pi_s.argmax(1)

    @partial(jax.jit, static_argnums=0)
    def explore(
            self, params_model: flax.core.frozen_dict.FrozenDict,
            state: np.ndarray,
            rng: PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        rng, key = jax.random.split(rng)
        value, pi_s, log_pi_s = self.model.apply(params_model, state)
        action = jax.random.categorical(key, logits=log_pi_s)
        return action, jnp.take(log_pi_s, action), value[:, 0], rng

    def is_update(self):
        return self.agent_step % (self.n_steps * self.num_envs) == 0

    def get_mask(self, env, done):
        return done

    def step(self, env, state):
        self.agent_step += self.num_envs
        self.episode_step += 1

        action, log_pi, value, new_key = self.explore(self.params_model, state, self.key)
        self.key = new_key
        next_state, reward, done, _ = env.step(action)
        mask = self.get_mask(env, done)

        self.buffer.append((state * 255.).astype(np.uint8), action, reward,
                           mask, log_pi, value, next_state)

        return next_state

    def update(self):

        self.update_step += 1
        state, action, reward, done, log_pi_old, value, next_state = self.buffer.get(
        )
        state = state.astype(np.float32) / 255.
        next_state = next_state.astype(np.float32) / 255.

        # Calculate gamma-returns and GAEs.
        gae, target = self.calculate_gae(
            params_model=self.params_model,
            state=state,
            reward=reward,
            done=done,
            next_state=next_state,
        )

        avg_loss_actor, avg_loss_critic = 0., 0.

        def flatten_dims(x):
            return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])

        size_batch = self.num_envs * self.n_steps
        size_minibatch = size_batch // self.n_minibatch

        for e in range(self.epoch_ppo):
            np.random.shuffle(self.idxes)
            for start in range(0, size_batch, size_minibatch):
                self.learning_step += 1
                idx = self.idxes[start:start + size_minibatch]

                total_loss, params_model = self._update_models(
                    self.params_model,
                    flatten_dims(state)[idx],
                    flatten_dims(target)[idx].reshape(-1, 1),
                    flatten_dims(action)[idx].reshape(-1, 1),
                    flatten_dims(log_pi_old)[idx].reshape(-1, 1),
                    flatten_dims(value)[idx].reshape(-1, 1),
                    flatten_dims(gae)[idx].reshape(-1, 1))
                avg_loss_actor += total_loss[1][0]
                avg_loss_critic += total_loss[1][1]

        avg_loss_actor /= (self.epoch_ppo * (self.num_envs // self.batch_size))
        avg_loss_critic /= (self.epoch_ppo *
                            (self.num_envs // self.batch_size))

        self.params_model = params_model
        return avg_loss_actor, avg_loss_critic

    @partial(jax.jit, static_argnums=0)
    def _update_models(self, params_model: flax.core.frozen_dict.FrozenDict,
                       state: jnp.ndarray, target: jnp.ndarray,
                       action: jnp.ndarray, log_pi_old: jnp.ndarray,
                       value: jnp.ndarray, gae: jnp.ndarray):

        total_loss, grad = jax.value_and_grad(self._loss_actor_and_critic,
                                              has_aux=True)(
                                                  params_model,
                                                  state=state,
                                                  target=target,
                                                  value_old=value,
                                                  log_pi_old=log_pi_old,
                                                  gae=gae,
                                                  action=action)

        updates, self.opt_state = self.opt_update(grad, self.opt_state,
                                                  params_model)
        params_model = optax.apply_updates(params_model, updates)

        return total_loss, params_model

    @partial(jax.jit, static_argnums=0)
    def _loss_actor_and_critic(
        self,
        params_model: flax.core.frozen_dict.FrozenDict,
        state: np.ndarray,
        target: np.ndarray,
        value_old: np.ndarray,
        log_pi_old: np.ndarray,
        gae: jnp.ndarray,
        action: np.ndarray,
    ) -> jnp.ndarray:

        value_pred, pi_s, log_pi_s = self.model.apply(params_model, state)
        value_pred = value_pred.squeeze()
        value_pred_clipped = value_old + (value_pred - value_old).clip(
            -self.clip_eps, self.clip_eps)
        value_losses = jnp.square(value_pred - target)
        value_losses_clipped = jnp.square(value_pred_clipped - target)
        value_loss = 0.5 * jnp.maximum(value_losses,
                                       value_losses_clipped).mean()

        log_pi = log_pi_s.take(action)
        ratio = jnp.exp(log_pi - log_pi_old)
        loss_actor1 = ratio * gae
        loss_actor2 = jnp.clip(ratio, 1.0 - self.clip_eps,
                               1.0 + self.clip_eps) * gae
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()

        total_loss = loss_actor + self.critic_coeff * value_loss - self.entropy_coeff * (
            -jnp.sum(log_pi_s * pi_s, -1)).mean()
        return total_loss, (loss_actor, value_loss)

    @partial(jax.jit, static_argnums=0)
    def calculate_gae(
        self,
        params_model: flax.core.frozen_dict.FrozenDict,
        state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        obs_shape = state.shape

        state = self.model.apply(params_model,
                                 state.reshape(-1, *obs_shape[2:]))[0]

        # Current and next value estimates.
        value = jax.lax.stop_gradient(state).reshape(*obs_shape[:2])

        gae = []
        lastgaelam = 0.
        for t in jnp.arange(self.n_steps - 1, -1, -1):
            if t == self.n_steps - 1:
                nextnonterminal = 1.0 - done[-1]
                nextvalues = value[-1]
            else:
                nextnonterminal = 1.0 - done[t + 1]
                nextvalues = value[t + 1]

            delta = reward[
                t] + self.gamma * nextvalues * nextnonterminal - value[t]

            lastgaelam = delta + self.gamma * self.lambd * nextnonterminal * lastgaelam

            gae.insert(0, lastgaelam)

        # delta = reward + self.gamma * next_value * (1.0 - done) - value
        # gae = [delta[-1]]
        # for t in jnp.arange(self.n_steps - 2, -1, -1):
        #     gae.insert(0, delta[t] + self.gamma * self.lambd * (1 - done[t]) * gae[0])

        gae = jnp.array(gae)
        return (gae - gae.mean()) / (gae.std() + 1e-8), gae + value