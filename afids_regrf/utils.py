"""General purpose methods."""
from __future__ import annotations

import csv
import itertools as it
from collections.abc import Sequence
from os import PathLike
from typing import Optional, overload

import nibabel as nib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import zoom
from typing_extensions import Literal

AFIDS_FIELDNAMES = [
    "id",
    "x",
    "y",
    "z",
    "ow",
    "ox",
    "oy",
    "oz",
    "vis",
    "sel",
    "lock",
    "label",
    "desc",
    "associatedNodeID",
]


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
) -> pd.DataFrame:
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
    return pd.DataFrame(
        np.array(
            list(
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
        ),
        columns=["x", "y", "z"],
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


def gen_box_averages(img: NDArray, corner_list: pd.DataFrame) -> pd.DataFrame:
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

    def img_idx(type_: Literal["min", "max"], coord: Literal["x", "y", "z"]) -> NDArray:
        return corner_list.loc[:, f"{coord}{type_}"].to_numpy(dtype="uint32")

    # n x 1 array of the sum of the voxel values in each box
    # See Cui et al. Fig. 2
    voxel_sums = (
        iv_image[
            img_idx("max", "x") + 1,
            img_idx("max", "y") + 1,
            img_idx("max", "z") + 1,
        ]
        - iv_image[
            img_idx("min", "x"),
            img_idx("max", "y") + 1,
            img_idx("max", "z") + 1,
        ]
        - iv_image[
            img_idx("max", "x") + 1,
            img_idx("max", "y") + 1,
            img_idx("min", "z"),
        ]
        - iv_image[
            img_idx("max", "x") + 1,
            img_idx("min", "y"),
            img_idx("max", "z") + 1,
        ]
        - iv_image[
            img_idx("min", "x"),
            img_idx("min", "y"),
            img_idx("min", "z"),
        ]
        + iv_image[
            img_idx("max", "x") + 1,
            img_idx("min", "y"),
            img_idx("min", "z"),
        ]
        + iv_image[
            img_idx("min", "x"),
            img_idx("min", "y"),
            img_idx("max", "z") + 1,
        ]
        + iv_image[
            img_idx("min", "x"),
            img_idx("max", "y") + 1,
            img_idx("min", "z"),
        ]
    )

    # n x 1 array of the number of voxels in each box
    nums_box_voxel = (
        (img_idx("max", "x") - img_idx("min", "x") + 1)
        * (img_idx("max", "y") - img_idx("min", "y") + 1)
        * (img_idx("max", "z") - img_idx("min", "z") + 1)
    )

    # n x 1 array of the average voxel value in each box
    return pd.DataFrame(
        voxel_sums / nums_box_voxel, index=corner_list.index, columns=["box_avg"]
    )


def gen_feature_boxes(
    feature_offset_corners: tuple[NDArray, NDArray], all_samples: pd.DataFrame
) -> pd.DataFrame:
    """Generate a set of boxes from which to compute Haar-like features.

    Parameters
    ----------
    feature_offset_corners
        The lower and upper corner offsets for each box
    all_samples
        The sample points relative to which the boxes will be defined
    """
    lower_offsets, higher_offsets = feature_offset_corners
    sample_arr = all_samples.to_numpy(dtype="uint32")
    num_offsets = lower_offsets.shape[0]
    num_samples = sample_arr.shape[0]

    index = pd.MultiIndex.from_product(
        [range(num_samples), range(num_offsets)],
        names=["sample", "offset"],
    )
    min_corner_list = np.zeros((num_samples * num_offsets, 3)).astype("uint32")
    max_corner_list = np.zeros((num_samples * num_offsets, 3)).astype("uint32")

    for idx in range(num_samples):
        min_corner_list[idx * num_offsets : (idx + 1) * num_offsets] = (
            sample_arr[idx] + lower_offsets
        )
        max_corner_list[idx * num_offsets : (idx + 1) * num_offsets] = (
            sample_arr[idx] + higher_offsets
        )

    corner_list = pd.DataFrame(
        np.hstack((min_corner_list, max_corner_list)),
        index=index,
        columns=["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"],
    )
    return corner_list


def is_in_array(
    array_df: pd.DataFrame, array_shape: tuple[int, int, int]
) -> pd.DataFrame:
    """Filter an index dataframe for values within an array shape."""
    return (
        (array_df["x"] > 0)
        & (array_df["x"] < array_shape[0])
        & (array_df["y"] > 0)
        & (array_df["y"] < array_shape[1])
        & (array_df["z"] > 0)
        & (array_df["z"] < array_shape[2])
    )


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
    predict: bool = False,
):
    """Generate features for one image and fiducial."""
    aff, img = read_nii_metadata(img_path)

    # Get and compute new fiducial location
    resampled_fid = fid_world2voxel(fiducial, aff, resample_size=size, padding=padding)
    img = zoom(img, size)

    # Get image samples (sample more closer to target)
    # Concatenate and retain unique samples
    all_samples = (
        pd.concat(
            [
                sample_coord_region(resampled_fid, sampling_rate),
                sample_coord_region(resampled_fid, sampling_rate, multiplier=2),
            ]
        )
        .drop_duplicates(ignore_index=True)
        .sort_values(by=["x", "y", "z"], ignore_index=True)  # Just to match old way
        .loc[lambda df: is_in_array(df, img.shape), :]
    )

    box_averages = gen_box_averages(
        img, gen_feature_boxes(feature_offset_corners, all_samples)
    )

    # These are the Haar-like features, the difference in average intensity between
    # paired boxes
    diff = pd.DataFrame(
        box_averages.loc[(slice(None), slice(0, 1999)), :].to_numpy(dtype=np.float_)
        - box_averages.loc[(slice(None), slice(2000, 3999)), :].to_numpy(
            dtype=np.float_
        ),
        index=pd.MultiIndex.from_product(
            [range(all_samples.shape[0]), zip(range(0, 2000), range(2000, 4000))]
        ),
    )

    diff = np.reshape(diff.to_numpy(dtype=np.float_), (all_samples.shape[0], 2000))

    # If features for training
    if not predict:
        # The position of each sample relative to the fiducial.
        dist = np.linalg.norm(all_samples - resampled_fid[0:3], axis=1)
        prob = np.exp(-0.5 * dist)  # weighted more towards closer distances

        return [np.hstack((diff[index], prob[index])) for index in range(prob.shape[0])]
    # Features for prediction
    return aff, diff, all_samples.to_numpy(dtype=np.int_)


def afids_to_fcsv(
    afid_coords: NDArray, fcsv_template: str, fcsv_output: PathLike | str
) -> None:
    """AFIDS to Slicer-compatible .fcsv file"""
    # Read in fcsv template
    with open(fcsv_template, "r", encoding="utf-8", newline="") as fcsv_file:
        header = [fcsv_file.readline() for _ in range(3)]
        reader = csv.DictReader(fcsv_file, fieldnames=AFIDS_FIELDNAMES)
        fcsv = list(reader)

    # Loop over fiducials
    for idx, row in enumerate(fcsv):
        # Update fcsv, skipping header
        row["x"] = afid_coords[idx][0]
        row["y"] = afid_coords[idx][1]
        row["z"] = afid_coords[idx][2]

    # Write output fcsv
    with open(fcsv_output, "w", encoding="utf-8", newline="") as out_fcsv_file:
        for line in header:
            out_fcsv_file.write(line)
        writer = csv.DictWriter(out_fcsv_file, fieldnames=AFIDS_FIELDNAMES)
        for row in fcsv:
            writer.writerow(row)
