import numpy as np
from gym.spaces import Box, Discrete

class RolloutBuffer:
    """
    Rollout Buffer.
    """

    def __init__(
        self,
        n_steps,
        num_envs,
        batch_size,
        state_space,
        action_space,
    ):
        self._n = 0
        self._p = 0
        self.buffer_size = num_envs * n_steps
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.n_steps = n_steps

        state_shape = [state_space.shape[1],state_space.shape[2],state_space.shape[0]]

        self.state = np.empty((n_steps, num_envs, *state_shape), dtype=np.uint8)
        self.reward = np.empty((n_steps, num_envs), dtype=np.float32)
        self.done = np.empty((n_steps, num_envs), dtype=np.float32)
        self.log_pi = np.empty((n_steps, num_envs), dtype=np.float32)
        self.value = np.empty((n_steps, num_envs), dtype=np.float32)
        self.next_state = np.empty((n_steps, num_envs, *state_shape), dtype=np.uint8)

        if type(action_space) == Box:
            self.action = np.empty((n_steps, num_envs, *action_space.shape), dtype=np.float32)
        elif type(action_space) == Discrete:
            self.action = np.empty((n_steps, num_envs), dtype=np.int32)
        else:
            raise NotImplementedError

        self.batch_idx = 0

    def append(self, state, action, reward, done, log_pi, value, next_state):
        self.state[self._p] = state
        self.action[self._p] = action
        self.reward[self._p] = reward
        self.done[self._p] = done
        self.log_pi[self._p] = log_pi
        self.value[self._p] = value
        self.next_state[self._p] = next_state

        self._p = (self._p + 1) % self.n_steps
        self._n = min(self._n + 1, self.n_steps)

    def get(self):
        batch = (
            self.state,
            self.action,
            self.reward,
            self.done,
            self.log_pi,
            self.value,
            self.next_state,
        )
        return batch