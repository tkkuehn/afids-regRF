"""General purpose methods."""
from __future__ import annotations 

import itertools as it
from collections.abc import Sequence 
from os import PathLike
from typing import overload, NoReturn, Optional
from typing_extensions import Literal

import nibabel as nib
import numpy as np
import pandas as pd 
from numpy.typing import NDArray

def read_nii_metadata(nii_path: PathLike | str) -> tuple[NDArray, NDArray]:
    """Load nifti data and header information and normalize MRI volume"""
    nii = nib.loadsave.load(nii_path)
    nii_affine = nii.affine
    nii_data = nii.get_fdata()
    nii_data = (nii_data - nii_data.min()) / (nii_data.max() - nii_data.min())
    return nii_affine, nii_data


def get_fid(fcsv_path: PathLike | str, fid_num: int) -> NDArray[np.single]:
    """Extract specific fiducial's spatial coordinates"""
    fcsv_df = pd.read_csv(fcsv_path, sep=",", header=2)

    return fcsv_df.loc[fid_num, ["x", "y", "z"]].to_numpy(dtype="single")


def fid_world2voxel(
    fid_world: NDArray,
    nii_affine: NDArray,
    resample_size: float = 1,
    padding: Optional[Sequence[int] | NDArray | int] = None,
) -> NDArray[np.int_]:
    """Transform fiducials in world coordinates to voxel coordinates

    Optionally, resample to match resampled image
    """

    # Translation
    fid_voxel = fid_world.T - nii_affine[:3, 3:4]
    # Rotation
    fid_voxel = np.dot(fid_voxel, np.linalg.inv(nii_affine[:3, :3]))

    # Round to nearest voxel
    fid_voxel = np.rint(np.diag(fid_voxel) * resample_size)

    if padding:
        fid_voxel = np.pad(fid_voxel, padding, mode="constant")

    return fid_voxel.astype(int)


def integral_volume(resampled_image: NDArray) -> NDArray:
    """Compute zero-padded resampled volume"""
    iv_image = resampled_image.cumsum(0).cumsum(1).cumsum(2)
    iv_zeropad = np.zeros(
        (iv_image.shape[0] + 1, iv_image.shape[1] + 1, iv_image.shape[2] + 1)
    )
    iv_zeropad[1:, 1:, 1:] = iv_image

    return iv_zeropad


def sample_coord_region(
    coord: NDArray, sampling_rate: int, multiplier: int = 1
) -> list[tuple[float, float, float]]:
    """Generate a list of voxels in the neighbourhood of a coordinate.

    Parameters
    ----------
    coord: NDArray
        Centre coordinate.
    sampling_rate: int
        "Radius" of the neighbourhood.
    multiplier: int
        Multiplier of neighbourhood's size (as defined by sampling_rate)
    """
    return list(
        it.product(
            *[
                range(
                    coord[0] - sampling_rate * multiplier,
                    coord[0] + (sampling_rate * multiplier) + 1,
                    multiplier,
                ),
                range(
                    coord[1] - sampling_rate * multiplier,
                    coord[1] + (sampling_rate * multiplier) + 1,
                    multiplier,
                ),
                range(
                    coord[2] - sampling_rate * multiplier,
                    coord[2] + (sampling_rate * multiplier) + 1,
                    multiplier,
                ),
            ]
        )
    )


def gen_offset_corners(
    num_offsets: int = 4000,
    lower_range: range = range(-17, 15),
    size_range: range = range(1, 4),
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray, NDArray]:
    """Generate the lower and upper corners of n boxes relative to the origin.

    "Lower" here refers to corners that are closer to the origin.

    Parameters
    ----------
    num_offsets:
        Number of offset corners to generate
    lower_range
        Distance along each axis the lower corner is allowed to be.
    size_range
        Range of allowed sizes of the box along each dimension.
    """
    rng = rng if rng else np.random.default_rng()
    upper_corners = (
        lower_corners := rng.choice(lower_range, size=(num_offsets, 3))
    ) + rng.choice(size_range, size=(num_offsets, 3))
    return lower_corners, upper_corners


def gen_box_averages(img: NDArray, corner_list: NDArray) -> NDArray:
    """For every box defined by corner_list, compute the average voxel value.

    Parameters
    ----------
    img
        An image containing all the boxes defined by corner_list.
    corner_list
        An n x 6 array of pairs of corners defining boxes within img.
    """
    # compute the integral image for more efficient generation of haar-like features
    iv_image = integral_volume(img)

    # n x 1 array of the sum of the voxel values in each box
    # See Cui et al. Fig. 2
    voxel_sums = (
        iv_image[corner_list[:, 3] + 1, corner_list[:, 4] + 1, corner_list[:, 5] + 1]
        - iv_image[corner_list[:, 0], corner_list[:, 4] + 1, corner_list[:, 5] + 1]
        - iv_image[corner_list[:, 3] + 1, corner_list[:, 4] + 1, corner_list[:, 2]]
        - iv_image[corner_list[:, 3] + 1, corner_list[:, 1], corner_list[:, 5] + 1]
        - iv_image[corner_list[:, 0], corner_list[:, 1], corner_list[:, 2]]
        + iv_image[corner_list[:, 3] + 1, corner_list[:, 1], corner_list[:, 2]]
        + iv_image[corner_list[:, 0], corner_list[:, 1], corner_list[:, 5] + 1]
        + iv_image[corner_list[:, 0], corner_list[:, 4] + 1, corner_list[:, 2]]
    )

    # n x 1 array of the number of voxels in each box
    nums_box_voxel = (
        (corner_list[:, 3] - corner_list[:, 0] + 1)
        * (corner_list[:, 4] - corner_list[:, 1] + 1)
        * (corner_list[:, 5] - corner_list[:, 2] + 1)
    )

    # n x 1 array of the average voxel value in each box
    return voxel_sums / nums_box_voxel


def gen_feature_boxes(
    feature_offset_corners: tuple[NDArray, NDArray], all_samples: NDArray
):
    """Generate a set of boxes from which to compute Haar-like features.

    Parameters
    ----------
    feature_offset_corners
        The lower and upper corner offsets for each box
    all_samples
        The sample points relative to which the boxes will be defined
    """
    lower_offsets, higher_offsets = feature_offset_corners

    # Generate 4000 bounding boxes near each feature
    min_corner_list = np.zeros((4000 * all_samples.shape[0], 3)).astype("uint8")
    max_corner_list = np.zeros((4000 * all_samples.shape[0], 3)).astype("uint8")
    for idx in range(all_samples.shape[0]):
        min_corner_list[idx * 4000 : (idx + 1) * 4000] = (
            all_samples[idx] + lower_offsets
        )
        max_corner_list[idx * 4000 : (idx + 1) * 4000] = (
            all_samples[idx] + higher_offsets
        )
    return np.hstack((min_corner_list, max_corner_list))


@overload
def gen_features(
    img_path: ...,
    fiducial: ...,
    feature_offset_corners: ...,
    padding: ...,
    sampling_rate: ...,
    size: ...,
    predict: Literal[False] = ...,
) -> list[NDArray]:
    ...


@overload
def gen_features(
    img_path: ...,
    fiducial: ...,
    feature_offset_corners: ...,
    padding: ...,
    sampling_rate: ...,
    size: ...,
    predict: Literal[True] = ...,
) -> tuple[NDArray, NDArray, NDArray]:
    ...


def gen_features(
    img_path: PathLike | str,
    fiducial: NDArray,
    feature_offset_corners: tuple[NDArray, NDArray],
    padding: int,
    sampling_rate: int,
    size: int,
    predict: bool = False
):
    """Generate features for one image and fiducial."""
    aff, img = read_nii_metadata(img_path)

    # Get and compute new fiducial location
    resampled_fid = fid_world2voxel(fiducial, aff, resample_size=size, padding=padding)

    # Get image samples (sample more closer to target)
    # Concatenate and retain unique samples and
    all_samples = np.unique(
        np.array(
            sample_coord_region(resampled_fid, sampling_rate)
            + sample_coord_region(resampled_fid, sampling_rate, multiplier=2)
        ),
        axis=0,
    )

    box_averages = gen_box_averages(
        img, gen_feature_boxes(feature_offset_corners, all_samples)
    )

    # Generate the indices for the first and second halves of the boxes per sample
    first_half_indices = np.zeros(2000 * all_samples.shape[0])
    second_half_indices = np.zeros(2000 * all_samples.shape[0])
    for index in range(all_samples.shape[0]):
        first_half_indices[index * 2000 : (index + 1) * 2000] = range(
            index * 4000, index * 4000 + 2000
        )
        second_half_indices[index * 2000: (index + 1) * 2000] = range(
            index * 4000 + 2000, (index + 1) * 4000
        )
    first_half_indices = first_half_indices.astype(int)
    second_half_indices = second_half_indices.astype(int)

    # These are the Haar-like features, the difference in average intensity between
    # paired boxes
    diff = box_averages[first_half_indices] - box_averages[second_half_indices]
    diff = np.reshape(diff, (all_samples.shape[0], 2000))

    # If features for training
    if not predict:
        # The position of each sample relative to the fiducial.
        dist = np.linalg.norm(all_samples - resampled_fid[0:3], axis=1)
        prob = np.exp(-0.5 * dist)  # weighted more towards closer distances

        return [np.hstack((diff[index], prob[index])) for index in range(prob.shape[0])]
    # Features for prediction
    else:
        return aff, diff, all_samples


def afids_to_fcsv(
    afid_coords: NDArray, 
    fcsv_template: str, 
    fcsv_output: PathLike | str
) -> NoReturn:
    """AFIDS to Slicer-compatible .fcsv file"""
    # Read in fcsv template
    with open(fcsv_template, "r") as f:
        fcsv = [line.strip() for line in f]

    # Loop over fiducials
    for fid in range(1, 33):
        # Update fcsv, skipping header
        line_idx, fid_idx = fid + 2, fid - 1
        
        fcsv[line_idx] = fcsv[line_idx].replace(
            f"afid{fid}_x", str(afid_coords[fid_idx][0])
        )
        fcsv[line_idx] = fcsv[line_idx].replace(
            f"afid{fid}_y", str(afid_coords[fid_idx][1])
        )
        fcsv[line_idx] = fcsv[line_idx].replace(
            f"afid{fid}_z", str(afid_coords[fid_idx][2])
        )

    # Write output fcsv
    with open(fcsv_output, "w") as f:
        f.write("\n".join(line for line in fcsv))

