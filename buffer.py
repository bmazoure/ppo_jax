from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit)
def calculate_gae(
    n_steps: int,
    discount: float,
    gae_lambda: float,
    value: np.ndarray,
    reward: np.ndarray,
    done: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = []
    gae = 0.
    for t in reversed(range(len(reward)-1)):
        value_diff = discount * value[t + 1] * (1-done[t]) - value[t]
        delta = reward[t] + value_diff
        gae = delta + discount * gae_lambda * (1-done[t]) * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    advantages = jnp.array(advantages)
    return advantages, advantages + value[:-1]

class Batch:
    """
    Batch of data.

    Inspired by: https://github.com/ku2482/rljax/tree/master/rljax/algorithm .
    """

    def __init__(
        self,
        discount: float,
        gae_lambda: float,
        n_steps: int,
        num_envs: int,
        state_space,
    ):
        self._n = 0
        self._p = 0
        self.num_envs = num_envs
        self.buffer_size = num_envs * n_steps
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.discount = discount
        self.gae_lambda = gae_lambda

        self.state_shape = [state_space.shape[1],state_space.shape[2],state_space.shape[0]]

        self.reset()

    def reset(self):
        self.states = np.empty((self.n_steps, self.num_envs, *self.state_shape), dtype=np.uint8)
        self.actions = np.empty((self.n_steps, self.num_envs), dtype=np.int32)
        self.rewards = np.empty((self.n_steps, self.num_envs), dtype=np.float32)
        self.dones = np.empty((self.n_steps, self.num_envs), dtype=np.uint8)
        self.log_pis_old = np.empty((self.n_steps, self.num_envs), dtype=np.float32)
        self.values_old = np.empty((self.n_steps, self.num_envs), dtype=np.float32)

    def append(self, state, action, reward, done, log_pi, value):
        self.states[self._p] = state
        self.actions[self._p] = action
        self.rewards[self._p] = reward
        self.dones[self._p] = done
        self.log_pis_old[self._p] = log_pi
        self.values_old[self._p] = value

        self._p = (self._p + 1) % self.n_steps
        self._n = min(self._n + 1, self.n_steps)

    def get(self):
        gae, target = calculate_gae(
            n_steps=self.n_steps,
            discount=self.discount,
            gae_lambda=self.gae_lambda,
            value=self.values_old,
            reward=self.rewards,
            done=self.dones
        )
        batch = (
            self.states[:-1],
            self.actions[:-1],
            self.log_pis_old[:-1],
            self.values_old[:-1],
            target,
            gae
        )
        return batch
