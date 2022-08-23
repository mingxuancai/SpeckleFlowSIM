# Description:
#  
# Created by Ruiming Cao on December 18, 2020
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from flax.struct import dataclass

import calcil as cc
from calcil.physics.wave_optics import genPupil, zernikePolynomial
import utils
import spacetime


@dataclass
class SpeckleFlowSIMParameters:
    space_time_param: spacetime.SpaceTimeParameters
    motion_mlp_param: spacetime.MLPParameters
    mlp_param: spacetime.MLPParameters


class SpeckleSIMCoherent(cc.forward.Model):
    optical_param: utils.SystemParameters
    pupil_aber_type: int = 1

    def setup(self):
        self.dim_yx_incl_padding = (self.optical_param.dim_yx[0] + self.optical_param.padding_yx[0] * 2,
                                    self.optical_param.dim_yx[1] + self.optical_param.padding_yx[1] * 2)
        self.pupil = genPupil(self.dim_yx_incl_padding,
                              self.optical_param.pixel_size, self.optical_param.na,
                              self.optical_param.wavelength).astype(jnp.complex64)[jnp.newaxis, :, :]

        if self.pupil_aber_type == 1:
            self.pupil_phase_aberration = self.param('pupil_phase_aberration', nn.initializers.zeros,
                                                     self.dim_yx_incl_padding)[jnp.newaxis, :, :]
            self.pupil_amp_aberration = self.param('pupil_amp_aberration', nn.initializers.ones,
                                                   self.dim_yx_incl_padding)[jnp.newaxis, :, :]
            self.aberrated_pupil = jnp.exp(1.0j * self.pupil_phase_aberration) * self.pupil_amp_aberration * self.pupil
        elif self.pupil_aber_type == 2:
            self.pupil_phase_aberration = self.param('pupil_phase_aberration', nn.initializers.zeros,
                                                     self.dim_yx_incl_padding)[jnp.newaxis, :, :]
            self.aberrated_pupil = jnp.exp(1.0j * self.pupil_phase_aberration) * self.pupil
        elif self.pupil_aber_type == 3:
            zern_indices = np.array([0, ]) + np.arange(3, 21)
            self.zern_poly = jnp.array([zernikePolynomial(i, self.dim_yx_incl_padding, self.optical_param.pixel_size,
                                                          self.optical_param.na, self.optical_param.wavelength)
                                        for i in zern_indices])
            self.zern_coef = self.param('zern_coef', nn.initializers.zeros, (len(zern_indices), 1, 1))
            self.aberrated_pupil = jnp.exp(1.0j * jnp.sum(self.zern_poly * self.zern_coef, axis=0))[jnp.newaxis, :,
                                   :] * self.pupil
        else:
            self.aberrated_pupil = self.pupil

    def __call__(self, U_speckle, phase_obj, absorption_obj):
        phase_obj = jnp.pad(phase_obj, [(0, 0), (self.optical_param.padding_yx[0], self.optical_param.padding_yx[0]),
                                        (self.optical_param.padding_yx[1], self.optical_param.padding_yx[1])],
                            constant_values=0.0)
        absorption_obj = jnp.pad(absorption_obj,
                                 [(0, 0), (self.optical_param.padding_yx[0], self.optical_param.padding_yx[0]),
                                  (self.optical_param.padding_yx[1], self.optical_param.padding_yx[1])],
                                 constant_values=0.0)
        o = U_speckle[jnp.newaxis, :, :] * jnp.exp(1.0j * phase_obj - absorption_obj)
        U_out = jnp.fft.ifft2(jnp.fft.fft2(o, axes=(1, 2)) * self.aberrated_pupil, axes=(1, 2))[:,
                self.optical_param.padding_yx[0]: self.optical_param.dim_yx[0] + self.optical_param.padding_yx[0],
                self.optical_param.padding_yx[1]: self.optical_param.dim_yx[1] + self.optical_param.padding_yx[1]]

        I_out = jnp.abs(U_out)
        return I_out


class SpeckleSIMCoherentNoPadding(cc.forward.Model):
    optical_param: utils.SystemParameters
    zern_poly_max_index: int = 21

    def setup(self):
        self.shape_incl_padding = (self.optical_param.dim_yx[0] + self.optical_param.padding_yx[0] * 2,
                                   self.optical_param.dim_yx[1] + self.optical_param.padding_yx[1] * 2)
        self.pupil = genPupil(self.shape_incl_padding,
                              self.optical_param.pixel_size, self.optical_param.na,
                              self.optical_param.wavelength).astype(jnp.complex64)[jnp.newaxis, :, :]
        zern_indices = np.array([0, ]) + np.arange(3, self.zern_poly_max_index)
        self.zern_poly = jnp.array([zernikePolynomial(i, self.shape_incl_padding, self.optical_param.pixel_size,
                                                      self.optical_param.na, self.optical_param.wavelength)
                                    for i in zern_indices])
        self.zern_coef = self.param('zern_coef', nn.initializers.zeros, (len(zern_indices), 1, 1))
        self.pupil_aberration = jnp.sum(self.zern_poly * self.zern_coef, axis=0) * self.pupil

    def __call__(self, U_speckle, phase_obj, absorption_obj):
        o = U_speckle[jnp.newaxis, :, :] * jnp.exp(1.0j * phase_obj - absorption_obj)
        U_out = jnp.fft.ifft2(jnp.fft.fft2(o, axes=(1, 2)) * self.pupil * jnp.exp(1.0j * self.pupil_aberration),
                              axes=(1, 2))[:,
                self.optical_param.padding_yx[0]: self.optical_param.dim_yx[0] + self.optical_param.padding_yx[0],
                self.optical_param.padding_yx[1]: self.optical_param.dim_yx[1] + self.optical_param.padding_yx[1]]

        I_out = jnp.abs(U_out)
        return I_out


class SpeckleFlowSIMCoherent(cc.forward.Model):
    optical_param: utils.SystemParameters
    speckle_flow_SIM_param: SpeckleFlowSIMParameters

    def setup(self):
        self.spacetime = spacetime.SpaceTimeMLP(self.optical_param,
                                                self.speckle_flow_SIM_param.space_time_param,
                                                self.speckle_flow_SIM_param.motion_mlp_param,
                                                self.speckle_flow_SIM_param.mlp_param,
                                                num_output_channels=2)
        if self.speckle_flow_SIM_param.space_time_param.include_padding:
            self.forward = SpeckleSIMCoherentNoPadding(self.optical_param)
        else:
            self.forward = SpeckleSIMCoherent(self.optical_param)
        speckle_dim_yx = (self.optical_param.dim_yx[0] + self.optical_param.padding_yx[0] * 2,
                          self.optical_param.dim_yx[1] + self.optical_param.padding_yx[1] * 2)
        self.U_speckle = self.param('speckle_amp', nn.initializers.ones, speckle_dim_yx) * jnp.exp(
            1.0j * self.param('speckle_phase', nn.initializers.zeros, speckle_dim_yx))

    def __call__(self, input_dict):
        obj = self.spacetime(input_dict['t'], input_dict['step'])
        phase_obj = obj[..., 0]
        absorption_obj = nn.relu(obj[..., 1])  # non-negative
        I = self.forward(self.U_speckle, phase_obj, absorption_obj)
        return I, obj


class SpeckleFlowSIMCoherentAbsorrptionObj(cc.forward.Model):
    optical_param: utils.SystemParameters
    speckle_flow_SIM_param: SpeckleFlowSIMParameters

    def setup(self):
        self.spacetime = spacetime.SpaceTimeMLP(self.optical_param,
                                                self.speckle_flow_SIM_param.space_time_param,
                                                self.speckle_flow_SIM_param.motion_mlp_param,
                                                self.speckle_flow_SIM_param.mlp_param,
                                                num_output_channels=2)

        if self.speckle_flow_SIM_param.space_time_param.include_padding:
            self.forward = SpeckleSIMCoherentNoPadding(self.optical_param)
        else:
            self.forward = SpeckleSIMCoherent(self.optical_param, 1)

        speckle_dim_yx = (self.optical_param.dim_yx[0] + self.optical_param.padding_yx[0] * 2,
                          self.optical_param.dim_yx[1] + self.optical_param.padding_yx[1] * 2)
        self.U_speckle = self.param('speckle_amp', nn.initializers.ones, speckle_dim_yx) * jnp.exp(
            1.0j * self.param('speckle_phase', nn.initializers.zeros, speckle_dim_yx))

    def __call__(self, input_dict):
        obj = self.spacetime(input_dict['t'], input_dict['step'])
        absorption_obj = obj[..., 1]
        obj = jnp.stack((jnp.zeros_like(absorption_obj), absorption_obj), axis=-1)
        I = self.forward(self.U_speckle, jnp.zeros_like(absorption_obj), absorption_obj)
        return I, obj


def loss_fn(variables, input_dict, forward_fn, reg_highfreq=0.0, pupil=None):
    I, obj = forward_fn(variables, input_dict)

    loss_l2 = ((input_dict['img'] - I) ** 2).mean()

    loss_highfreq = 0.
    if reg_highfreq > 0 and pupil is not None:
        loss_highfreq = reg_highfreq * jnp.abs(jnp.fft.fftn(jnp.exp(1.0j * obj[..., 0] - obj[..., 1]), axes=(-2, -1)) *
                                               (1 - pupil)[jnp.newaxis, :, :]).mean()

    loss = loss_l2 + loss_highfreq

    aux = {'loss_total': loss, 'loss_l2': loss_l2, 'loss_highfreq': loss_highfreq}
    return loss, aux
