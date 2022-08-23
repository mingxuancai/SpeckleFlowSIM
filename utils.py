# Description:
#  Supporting modules for speckle flow SIM
# Written by Ruiming Cao on September 13, 2021
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

from typing import Tuple, Union, List

import numpy as np
from PIL import Image
from flax.struct import dataclass
from skimage import data, transform
import imageio

from calcil.physics.wave_optics import propKernelNumpy, genGrid


@dataclass
class SystemParameters:
    dim_yx: Tuple[int, int]
    wavelength: float
    na: float
    pixel_size: float
    RI_medium: float
    padding_yx: Tuple[int, int]= (0, 0)
    mean_background_amp: float = 1.0


def generate_speckle(param: SystemParameters,
                     speckle_size_coef: float = 1.0,
                     strength_coef: float = 1.0,
                     band_limited: bool = False,
                     padding: bool = False,
                     prop_distance: Union[float, List[float]] = 0.0,
                     seed: int = 219) -> np.ndarray:
    if padding:
        dim_yx = (param.padding_yx[0] * 2 + param.dim_yx[0], param.padding_yx[1] * 2 + param.dim_yx[1])
    else:
        dim_yx = param.dim_yx

    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2.*np.pi*strength_coef, size=(int(dim_yx[0]*speckle_size_coef),
                                                         int(dim_yx[1]*speckle_size_coef)))
    phase = transform.resize(phase, dim_yx)

    if isinstance(prop_distance, list):
        U_speckle = np.array([np.fft.ifft2(np.fft.fft2(np.exp(1.0j * phase)) *
                                           propKernelNumpy(phase.shape, param.pixel_size, param.wavelength,
                                                      prop_distance=d, RI=param.RI_medium, NA=param.na,
                                                      band_limited=band_limited)) for d in prop_distance])
    else:
        U_speckle = np.fft.ifft2(np.fft.fft2(np.exp(1.0j * phase)) *
                                 propKernelNumpy(phase.shape, param.pixel_size, param.wavelength,
                                            prop_distance=prop_distance, RI=param.RI_medium, NA=param.na,
                                            band_limited=band_limited))

    if padding and isinstance(prop_distance, list):
        U_speckle = U_speckle[:, param.padding_yx[0]:param.padding_yx[0]+param.dim_yx[0],
                    param.padding_yx[1]:param.padding_yx[1]+param.dim_yx[1]]
    elif padding:
        U_speckle = U_speckle[param.padding_yx[0]:param.padding_yx[0]+param.dim_yx[0],
                    param.padding_yx[1]:param.padding_yx[1]+param.dim_yx[1]]

    return U_speckle


def generate_linear_motion(t, start_pos_yx, end_pos_yx, rot_start=0, rot_end=0):
    rot = (1 - t) * rot_start + t * rot_end
    y = (1 - t) * start_pos_yx[0] + t * end_pos_yx[0]
    x = (1 - t) * start_pos_yx[1] + t * end_pos_yx[1]
    return np.array([x, y, rot]).transpose()


def generate_affine_motion(t, start_pos_yx, end_pos_yx, rot_start=0, rot_end=0, scale_start=1, scale_end=1,
                           shear_start=0, shear_end=0):
    rot = (1 - t) * rot_start + t * rot_end
    scale = (1 - t) * scale_start + t * scale_end
    shear = (1 - t) * shear_start + t * shear_end
    y = (1 - t) * start_pos_yx[0] + t * end_pos_yx[0]
    x = (1 - t) * start_pos_yx[1] + t * end_pos_yx[1]
    return np.array([x, y, rot, scale, shear]).transpose()


def object_transform(obj: np.ndarray, target_dim_yx: Tuple[int, int],
                     coord: Union[List[float], Tuple[float, float, float, float]]) -> np.ndarray:
    """
    Linear transformation of a given object.

    :param obj: original object to be transformed
    :param target_dim_yx: output matrix dimension
    :param coord: tuple or list to specify the transformation as (x, y, orientation, scale)
    :return obj_transformed: transformed object
    """
    scaling_factor = min(target_dim_yx[0] / obj.shape[0], target_dim_yx[1] / obj.shape[1])

    obj = transform.resize(obj, (int(obj.shape[0] * scaling_factor), int(obj.shape[1] * scaling_factor)),
                           anti_aliasing=True)
    obj = np.pad(obj, (((target_dim_yx[0] - obj.shape[0]) // 2,
                        target_dim_yx[0] - (target_dim_yx[0] - obj.shape[0]) // 2 - obj.shape[0]),
                       ((target_dim_yx[1] - obj.shape[1]) // 2,
                        target_dim_yx[1] - (target_dim_yx[1] - obj.shape[1]) // 2 - obj.shape[1])))

    # shrink, rotate, translate
    trans_shrink = transform.SimilarityTransform(scale=coord[3])
    trans_shift1 = transform.SimilarityTransform(translation=[target_dim_yx[1] * (1 - coord[3]) * 0.5,
                                                              target_dim_yx[0] * (1 - coord[3]) * 0.5])
    trans_rotate = transform.SimilarityTransform(rotation=np.deg2rad(coord[2]))
    trans_shift2 = transform.SimilarityTransform(translation=[-target_dim_yx[1] * 0.5, -target_dim_yx[0] * 0.5])
    trans_shift2_inv = transform.SimilarityTransform(translation=[target_dim_yx[1] * 0.5, target_dim_yx[0] * 0.5])
    trans_shift3 = transform.SimilarityTransform(translation=[coord[0], coord[1]])

    obj_transformed = transform.warp(obj, (trans_shrink + trans_shift1 + trans_shift2 + trans_rotate +
                                                   trans_shift2_inv + trans_shift3).inverse)
    return obj_transformed


def object_transform_affine(obj: np.ndarray, target_dim_yx: Tuple[int, int],
                            coord: Union[List[float], Tuple[float, float, float, float, float]]) -> np.ndarray:
    """
    Affine transformation of a given object.

    :param obj: original object to be transformed
    :param target_dim_yx: output matrix dimension
    :param coord: tuple or list to specify the transformation as (x, y, orientation, scale, shear)
    :return obj_transformed: transformed object
    """
    scaling_factor = min(target_dim_yx[0] / obj.shape[0], target_dim_yx[1] / obj.shape[1])

    obj = transform.resize(obj, (int(obj.shape[0] * scaling_factor), int(obj.shape[1] * scaling_factor)),
                           anti_aliasing=True)
    obj = np.pad(obj, (((target_dim_yx[0] - obj.shape[0]) // 2,
                        target_dim_yx[0] - (target_dim_yx[0] - obj.shape[0]) // 2 - obj.shape[0]),
                       ((target_dim_yx[1] - obj.shape[1]) // 2,
                        target_dim_yx[1] - (target_dim_yx[1] - obj.shape[1]) // 2 - obj.shape[1])))

    # shrink, rotate, translate
    trans = transform.AffineTransform(scale=1/coord[3], rotation=np.deg2rad(coord[2]), shear=np.deg2rad(coord[4]),
                                      translation=[coord[0]/coord[3], coord[1]/coord[3]])
    trans_shift1 = transform.SimilarityTransform(translation=[target_dim_yx[1] * (1 - 1/coord[3]) * 0.5,
                                                              target_dim_yx[0] * (1 - 1/coord[3]) * 0.5])
    obj_transformed = transform.warp(obj, trans + trans_shift1)
    return obj_transformed


def object_transform_swirl(obj: np.ndarray, target_dim_yx: Tuple[int, int], scale, strength, radius):

    scaling_factor = min(target_dim_yx[0] / obj.shape[0], target_dim_yx[1] / obj.shape[1])

    obj = transform.resize(obj, (int(obj.shape[0] * scaling_factor), int(obj.shape[1] * scaling_factor)),
                           anti_aliasing=True)
    obj = np.pad(obj, (((target_dim_yx[0] - obj.shape[0]) // 2,
                        target_dim_yx[0] - (target_dim_yx[0] - obj.shape[0]) // 2 - obj.shape[0]),
                       ((target_dim_yx[1] - obj.shape[1]) // 2,
                        target_dim_yx[1] - (target_dim_yx[1] - obj.shape[1]) // 2 - obj.shape[1])))
    trans_shrink = transform.SimilarityTransform(scale=scale)
    trans_shift1 = transform.SimilarityTransform(translation=[target_dim_yx[1] * (1 - scale) * 0.5,
                                                              target_dim_yx[0] * (1 - scale) * 0.5])
    obj = transform.warp(obj, (trans_shrink + trans_shift1).inverse)

    obj_transformed = transform.swirl(obj, rotation=0, strength=strength, radius=radius)
    return obj_transformed


class PhantomTemporal:
    def __init__(self, param):
        self.param = param
        self.xlin          = genGrid(self.param.dim_yx[1], self.param.pixel_size)
        self.ylin          = genGrid(self.param.dim_yx[0], self.param.pixel_size)

    def generate_bead_phantom(self, coordinates, phase=1.0):
        # coordinates: [(x, y, orientation, scale)] orientation doesn't matter for now

        obj_phase = np.zeros(self.param.dim_yx, dtype=np.complex128)
        obj_fluo = np.zeros(self.param.dim_yx)
        for coord in coordinates:
            additive_phase = (np.maximum(coord[3]**2 - (self.ylin[:, np.newaxis] - coord[1])**2 -
                                         (self.xlin[np.newaxis, :] - coord[0])**2, 0.0))**0.5 * 2 * phase
            obj_phase += additive_phase

            additive_fluo = np.abs(np.maximum((1.5 * self.param.pixel_size)**2 - (self.ylin[:, np.newaxis] - coord[1])**2 -
                                         (self.xlin[np.newaxis, :] - coord[0])**2, 0.0))**0.5
            obj_fluo += additive_fluo
        return obj_phase, obj_fluo

    def generate_shepp_logan(self, coordinates, max_value=1.0):
        # coordinates: [(x, y, orientation, scale), ...]

        obj_phase = np.zeros(self.param.dim_yx, dtype=np.complex128)
        obj_fluo = np.zeros(self.param.dim_yx)

        for coord in coordinates:
            phantom = data.shepp_logan_phantom()
            phantom_fluo = (phantom >= 0.7).astype(np.float)
            phantom[phantom >= 0.7] = 0
            phantom_transformed = object_transform(
                phantom, self.param.dim_yx,
                (coord[0]/self.param.pixel_size, coord[1]/self.param.pixel_size, coord[2], coord[3]))
            phantom_fluo_transformed = object_transform(
                phantom_fluo, self.param.dim_yx,
                (coord[0]/self.param.pixel_size, coord[1]/self.param.pixel_size, coord[2], coord[3]))

            phantom_transformed = phantom_transformed / np.max(phantom_transformed) * max_value
            obj_phase +=  phantom_transformed
            obj_fluo += phantom_fluo_transformed

        return obj_phase, obj_fluo

    def generate_shepp_logan_affine(self, coordinates, max_value=1.0):
        # coordinates: [(x, y, orientation, scale, shear), ...]

        obj_phase = np.zeros(self.param.dim_yx, dtype=np.complex128)
        obj_fluo = np.zeros(self.param.dim_yx)

        for coord in coordinates:
            phantom = data.shepp_logan_phantom()
            phantom_fluo = (phantom >= 0.7).astype(np.float)
            phantom[phantom >= 0.7] = 0
            phantom_transformed = object_transform_affine(
                phantom, self.param.dim_yx,
                (coord[0]/self.param.pixel_size, coord[1]/self.param.pixel_size, coord[2], coord[3], coord[4]))
            phantom_fluo_transformed = object_transform_affine(
                phantom_fluo, self.param.dim_yx,
                (coord[0]/self.param.pixel_size, coord[1]/self.param.pixel_size, coord[2], coord[3], coord[4]))

            phantom_transformed = phantom_transformed / np.max(phantom_transformed) * max_value
            obj_phase +=  phantom_transformed
            obj_fluo += phantom_fluo_transformed

        return obj_phase, obj_fluo

    def generate_shepp_logan_swirl(self, coordinates, max_value=1.0):
        # coordinates: [(x, y, strengh, radius), ...]
        # x, y are dummy var here
        obj_phase = np.zeros(self.param.dim_yx, dtype=np.complex128)
        obj_fluo = np.zeros(self.param.dim_yx)

        for coord in coordinates:
            phantom = data.shepp_logan_phantom()
            phantom_fluo = (phantom >= 0.7).astype(np.float)
            phantom[phantom >= 0.7] = 0
            phantom_transformed = object_transform_swirl(
                phantom, self.param.dim_yx, coord[2], coord[3], coord[4])
            phantom_fluo_transformed = object_transform_swirl(
                phantom_fluo, self.param.dim_yx, coord[2], coord[3], coord[4])

            phantom_transformed = phantom_transformed / np.max(phantom_transformed) * max_value
            obj_phase +=  phantom_transformed
            obj_fluo += phantom_fluo_transformed

        return obj_phase, obj_fluo

    def generate_usaf_target(self, coordinates, max_value=1.0):
        # coordinates: [(x, y, orientation, scale), ...]
        filepath = 'experiment/USAF-1951.png'
        img_dim_hw = [1550, 1550]
        im_frame = Image.open(filepath).convert('L')
        phantom = 255 - np.asarray(im_frame)
        phantom = transform.resize(phantom, (400, 400), anti_aliasing=True)

        obj_phase = np.zeros(self.param.dim_yx, dtype=np.float64)
        for coord in coordinates:
            phantom_transformed = object_transform(
                phantom, self.param.dim_yx,
                (coord[0]/self.param.pixel_size, coord[1]/self.param.pixel_size, coord[2], coord[3]))

            phantom_transformed = phantom_transformed.astype(np.float64) / np.max(phantom_transformed) * max_value
            obj_phase += phantom_transformed
        return obj_phase

    def generate_usaf_target_affine(self, coordinates, max_value=1.0):
        # coordinates: [(x, y, orientation, scale, shear), ...]
        filepath = 'experiment/USAF-1951.png'
        img_dim_hw = [1550, 1550]
        im_frame = Image.open(filepath).convert('L')
        phantom = 255 - np.asarray(im_frame)
        phantom = transform.resize(phantom, (400, 400), anti_aliasing=True)

        obj_phase = np.zeros(self.param.dim_yx, dtype=np.float64)
        for coord in coordinates:
            phantom_transformed = object_transform_affine(
                phantom, self.param.dim_yx,
                (coord[0] / self.param.pixel_size, coord[1] / self.param.pixel_size, coord[2], coord[3], coord[4]))

            phantom_transformed = phantom_transformed.astype(np.float64) / np.max(phantom_transformed) * max_value
            obj_phase += phantom_transformed
        return obj_phase

    def generate_usaf_target_swirl(self, coordinates, max_value=1.0):
        # coordinates: [(x, y, strengh, radius), ...]
        # x, y are dummy var here
        filepath = 'experiment/USAF-1951.png'
        img_dim_hw = [1550, 1550]
        im_frame = Image.open(filepath).convert('L')
        phantom = 255 - np.asarray(im_frame)
        phantom = transform.resize(phantom, (400, 400), anti_aliasing=True)

        obj_phase = np.zeros(self.param.dim_yx, dtype=np.float64)
        for coord in coordinates:
            phantom_transformed = object_transform_swirl(
                phantom, self.param.dim_yx, coord[2], coord[3], coord[4])

            phantom_transformed = phantom_transformed.astype(np.float64) / np.max(phantom_transformed) * max_value
            obj_phase += phantom_transformed
        return obj_phase

def load_video(filename, fov=None, single_channel=False, target_dim=None):
    vid = imageio.get_reader(filename, 'ffmpeg')
    ret = []
    for image in vid.iter_data():
        if single_channel:
            image = np.mean(image, axis=-1)
        if fov:
            image = image[fov[0]:fov[1], fov[2]:fov[3]]
        if target_dim:
            image = transform.resize(image, target_dim)
        ret.append(image)

    return np.array(ret)
