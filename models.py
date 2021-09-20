from typing import Any, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp


def default_conv_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.xavier_uniform()

def default_mlp_init(scale: Optional[float] = 0.01):
    return nn.initializers.orthogonal(scale)

def default_logits_init(scale: Optional[float] = 0.01):
    return nn.initializers.orthogonal(scale)


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
                    kernel_init=default_conv_init(),
                    name=self.prefix + '/conv2d_1')(y)
        y = nn.relu(y)
        y = nn.Conv(self.num_channels,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=default_conv_init(),
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
                           kernel_init=default_conv_init(),
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
        out = nn.Dense(256, kernel_init=default_mlp_init(), name=self.prefix + '/representation')(out)
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
        v = nn.Dense(1, kernel_init=default_mlp_init(), name=self.prefix_critic + '/fc_v')(z)

        logits = nn.Dense(self.action_dim,
                        kernel_init=default_logits_init(),
                        name=self.prefix_actor + '/fc_pi')(z)
        
        return v, logits
