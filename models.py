import flax.linen as nn
import jax.numpy as jnp

from typing import Any, Tuple, Optional

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

class MLP(nn.Module):
    """Simple MLP """
    output_dim: int
    hidden_units: Tuple
    prefix = 'MLP'

    @nn.compact
    def __call__(self, x):
        for i, unit in enumerate(self.hidden_units):
            x = nn.Dense(unit, kernel_init=default_init(), name=self.prefix + '_mlp_%d' % i)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=default_init(),
                     name=self.prefix + '_mlp_%d' % (i + 1))(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block."""
    num_channels: int
    prefix: str

    @nn.compact
    def __call__(self, x):
        # Conv branch
        y = nn.relu(x)
        y = nn.Conv(self.num_channels,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=default_init(),
                    name=self.prefix + '/conv2d_1')(y)
        y = nn.relu(y)
        y = nn.Conv(self.num_channels,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=default_init(),
                    name=self.prefix + '/conv2d_2')(y)

        return y + x


class Impala(nn.Module):
    """IMPALA architecture."""
    prefix: str

    @nn.compact
    def __call__(self, x):
        out = x
        for i, (num_channels, num_blocks) in enumerate([(16, 2), (32, 2),
                                                        (32, 2)]):
            conv = nn.Conv(num_channels,
                           kernel_size=[3, 3],
                           strides=(1, 1),
                           padding='SAME',
                           kernel_init=default_init(),
                           name=self.prefix + '/conv2d_%d' % i)
            out = conv(out)

            out = nn.max_pool(out,
                              window_shape=(3, 3),
                              strides=(2, 2),
                              padding='SAME')
            for j in range(num_blocks):
                block = ResidualBlock(num_channels,
                                      prefix='residual_{}_{}'.format(i, j))
                out = block(out)

        out = out.reshape(out.shape[0], -1)
        out = nn.relu(out)
        out = nn.Dense(256, name=self.prefix + '/representation')(out)
        out = nn.relu(out)
        return out


class TwinHeadModel(nn.Module):
    """Critic+Actor for PPO."""
    action_dim: int
    prefix_critic: str = "critic"
    prefix_actor: str = "policy"

    @nn.compact
    def __call__(self, x):
        z = Impala(prefix='shared_encoder')(x)
        # Linear critic
        v = nn.Dense(1, kernel_init=default_init(), name=self.prefix_critic + '/fc_v')(z)

        out2 = nn.Dense(self.action_dim,
                        kernel_init=default_init(),
                        name=self.prefix_actor + '/fc_pi')(z)

        pi_s = nn.softmax(out2, axis=1)
        log_pi_s = jnp.log(pi_s)

        return v, pi_s, log_pi_s
