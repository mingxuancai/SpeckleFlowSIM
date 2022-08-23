# Description:
#  
# Written by Ruiming Cao on September 10, 2021
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

import functools
import numpy as np
from flax import linen as nn
import jax.numpy as jnp
import jax
from typing import Callable, Any
import calcil as cc

from flax.struct import dataclass
import utils


def generate_dense_yx_coords(dim_yx):
    xlin = np.arange(dim_yx[1]) / dim_yx[1] * 2 - 1
    ylin = np.arange(dim_yx[0]) / dim_yx[0] * 2 - 1
    y, x = np.meshgrid(ylin, xlin, indexing='ij')
    yx = np.concatenate((y[:, :, None], x[:, :, None]), axis=2)

    return yx.reshape([-1, 2])


def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
  """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

  Instead of computing [sin(x), cos(x)], we use the trig identity
  cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

  From https://github.com/google/mipnerf

  Args:
    x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
    min_deg: int, the minimum (inclusive) degree of the encoding.
    max_deg: int, the maximum (exclusive) degree of the encoding.
    legacy_posenc_order: bool, keep the same ordering as the original tf code.

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if min_deg == max_deg:
    return x
  scales = jnp.array([2**i for i in range(min_deg, max_deg)])
  if legacy_posenc_order:
    xb = x[Ellipsis, None, :] * scales[:, None]
    four_feat = jnp.reshape(
        jnp.sin(jnp.stack([xb, xb + 0.5 * jnp.pi], -2)),
        list(x.shape[:-1]) + [-1])
  else:
    xb = jnp.reshape((x[Ellipsis, None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
    four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
  return jnp.concatenate([x] + [four_feat], axis=-1)


def annealed_posenc(x, min_deg, max_deg, num_freqs, alpha):
  """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

  Instead of computing [sin(x), cos(x)], we use the trig identity
  cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

  From https://github.com/google/mipnerf

  Args:
    x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
    min_deg: int, the minimum (inclusive) degree of the encoding.
    max_deg: int, the maximum (exclusive) degree of the encoding.

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if min_deg == max_deg:
    return x

  freq_bands = 2.0 ** jnp.linspace(min_deg, max_deg - 1, num_freqs, endpoint=False)

  xb = jnp.pi * x[..., None, :] * freq_bands[:, None]

  alpha *= num_freqs
  bands = jnp.linspace(min_deg, max_deg - 1, num_freqs, endpoint=False)
  coef = jnp.clip(alpha - bands, 0.0, 1.0)
  window = 0.5 * (1 + jnp.cos(jnp.pi * coef + jnp.pi))

  features = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))

  features = window[None, :, None] * features
  features = jnp.reshape(features, list(x.shape[:-1]) + [-1])
  return jnp.concatenate([x, features], axis=-1)


@dataclass
class MLPParameters:
    net_depth: int  # The depth of the first part of MLP.
    net_width: int  # The width of the first part of MLP.
    net_activation: Callable[..., Any] # The activation function.
    skip_layer: int  # The layer to add skip layers to.
    kernel_init: Callable = jax.nn.initializers.glorot_uniform()


@dataclass
class SpaceTimeParameters:
    mlp_out_activation: Callable[..., Any]  # The activation function for MLP output
    posenc_min: int = 0
    posenc_max: int = 10
    time_posenc_min: int = 0
    time_posenc_max: int = 4
    n_step_coarse_to_fine: int = 5000
    include_padding: bool = False


class MLP(cc.forward.Model):
    """A simple MLP with a condition term option adding after the first layer of the network."""
    net_depth: int = 8  # The depth of the MLP.
    net_width: int = 256  # The width of the MLP.
    net_activation: Callable = nn.relu  # The activation function.
    skip_layer: int = 4  # The layer to add skip layers to.
    num_output_channels: int = 1  # The number of sigma channels.
    kernel_init: Callable = jax.nn.initializers.glorot_uniform()  # kernel weight initializer

    @nn.compact
    def __call__(self, x):
        """Evaluate the MLP.

        Args:
          x: jnp.ndarray(float32), [batch, num_samples, feature], points.

        Returns:
          output: jnp.ndarray(float32), with a shape of
               [batch, num_samples, num_rgb_channels].
        """
        input_dim = x.shape[:-1]
        feature_dim = x.shape[-1]
        x = x.reshape([-1, feature_dim])
        dense_layer = functools.partial(
            nn.Dense, kernel_init=self.kernel_init)
        inputs = x

        for i in range(self.net_depth):
            x = dense_layer(self.net_width)(x)
            x = self.net_activation(x)
            if i % self.skip_layer == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis=-1)

        output = dense_layer(self.num_output_channels)(x).reshape(
            input_dim + (self.num_output_channels, ))
        return output


class SpaceTimeMLP(cc.forward.Model):

    optical_param: utils.SystemParameters
    space_time_param: SpaceTimeParameters
    motion_mlp_param: MLPParameters
    object_mlp_param: MLPParameters
    num_output_channels: int

    def setup(self):
        # motion MLP
        self.motion_mlp = MLP(net_depth=self.motion_mlp_param.net_depth,
                              net_width=self.motion_mlp_param.net_width,
                              net_activation=self.motion_mlp_param.net_activation,
                              skip_layer=self.motion_mlp_param.skip_layer,
                              num_output_channels=2,
                              kernel_init=self.motion_mlp_param.kernel_init)

        # object MLP
        self.object_mlp = MLP(net_depth=self.object_mlp_param.net_depth,
                              net_width=self.object_mlp_param.net_width,
                              net_activation=self.object_mlp_param.net_activation,
                              skip_layer=self.object_mlp_param.skip_layer,
                              num_output_channels=self.num_output_channels,
                              kernel_init=self.object_mlp_param.kernel_init)
        
        if self.space_time_param.include_padding == False:
            self.dim_yx = self.optical_param.dim_yx
        else:
            self.dim_yx = (self.optical_param.dim_yx[0] + self.optical_param.padding_yx[0] * 2,
                               self.optical_param.dim_yx[1] + self.optical_param.padding_yx[1] * 2)

        self.list_yx = generate_dense_yx_coords(self.dim_yx)[jnp.newaxis, :, :]

    def __call__(self, t, step=1e7):
        # generate a list of all xy coordinates
        list_yx = jnp.tile(self.list_yx, (t.shape[0], 1, 1))

        # positional encoding
        t_posenc = posenc(jnp.tile(t[:, jnp.newaxis, jnp.newaxis], (1, list_yx.shape[1], 1)),
                          self.space_time_param.time_posenc_min, self.space_time_param.time_posenc_max)

        yx_time_adjusted = self.motion_mlp(jnp.concatenate([list_yx, t_posenc], axis=-1)) + list_yx

        # coarse to fine posenc
        yx_posenc_time_adjusted = annealed_posenc(
            yx_time_adjusted,
            self.space_time_param.posenc_min,
            self.space_time_param.posenc_max,
            num_freqs=self.space_time_param.posenc_max - self.space_time_param.posenc_min,
            alpha=(1.0 * step / self.space_time_param.n_step_coarse_to_fine))

        output = self.object_mlp(yx_posenc_time_adjusted)

        # reshape MLP's output
        output = jnp.reshape(output, (-1, ) + self.dim_yx + (self.num_output_channels, ))
        output = self.space_time_param.mlp_out_activation(output)

        return output
