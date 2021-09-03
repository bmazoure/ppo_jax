import jax.numpy as jnp
import flax.linen as nn

from typing import Any, Tuple

from models import MLP, Impala

class CategoricalPolicy(nn.Module):
    """
    Policy for PPO-Discrete.
    """

    action_space: Any
    prefix: str = 'policy'

    @nn.compact
    def __call__(self, x):
        out = x
        out = Impala(prefix='shared_encoder')(out)
        out = nn.Dense(256, name=self.prefix + '_mlp_1')(out)
        out = nn.relu(out)
        out = nn.Dense(256, name=self.prefix + '_mlp_2')(out)
        out = nn.relu(out)
        out = nn.Dense(self.action_space.n, name=self.prefix + '_mlp_3')(out)

        pi_s = nn.softmax(out, axis=1)
        log_pi_s = jnp.log(pi_s + (pi_s == 0.0) * 1e-8)
        return pi_s, log_pi_s