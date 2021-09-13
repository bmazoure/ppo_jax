import numpy as np
from gym.spaces import Box, Discrete
import jax
from functools import partial
from typing import Any, Callable, Tuple, List
import jax.numpy as jnp

@partial(jax.jit)
def calculate_gae(
    value: np.ndarray,
    reward: np.ndarray,
    done: np.ndarray,
    discount: float,
    gae_lambda: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    advantages = []
    gae = 0.
    for t in reversed(range(len(value)-1)):
        value_diff = discount * value[t + 1] * done[t] - value[t]
        delta = reward[t] + value_diff
        gae = delta + discount * gae_lambda * done[t] * gae
        advantages.append(gae)
    advantages = jnp.array(advantages)
    advantages = advantages[::-1]
    return advantages, advantages + value[:-1]

class Batch:
    """
    Batch of data.
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
        # Calculate gamma-returns and GAEs.
        gae, target = calculate_gae(
            value=self.values_old,
            reward=self.rewards,
            done=self.dones,
            discount=self.discount,
            gae_lambda=self.gae_lambda
        )
        batch = (
            self.states,
            self.actions,
            self.log_pis_old,
            self.values_old,
            target,
            gae
        )
        return batch