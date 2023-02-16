"""Methods for training a regRF model."""
from __future__ import annotations

import itertools as it
from os import PathLike
from typing import Optional, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import dump
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

# Constants -- Should parameterize our functions instead I think
PAD_FLAG = False
PADDING = 0
SIZE = 1
SAMPLING_RATE = 5


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


def gen_features(
    img_path: PathLike | str,
    fcsv_path: PathLike | str,
    feature_offsets_path: PathLike | str,
    fiducial_num: int,
) -> list[NDArray]:
    """Generate features for one image and fiducial."""
    # Load image -- assumes correct bids spec
    aff, img = read_nii_metadata(img_path)

    # Get and compute new fiducial location
    fid_world = get_fid(fcsv_path, fiducial_num - 1)
    resampled_fid = fid_world2voxel(fid_world, aff, resample_size=SIZE, padding=PADDING)

    # Get image samples (sample more closer to target)
    inner_its = [
        range(resampled_fid[0] - SAMPLING_RATE, resampled_fid[0] + (SAMPLING_RATE + 1)),
        range(resampled_fid[1] - SAMPLING_RATE, resampled_fid[1] + (SAMPLING_RATE + 1)),
        range(resampled_fid[2] - SAMPLING_RATE, resampled_fid[2] + (SAMPLING_RATE + 1)),
    ]
    inner_samples = list(it.product(*inner_its))

    outer_its = [
        range(
            resampled_fid[0] - SAMPLING_RATE * 2,
            resampled_fid[0] + SAMPLING_RATE * 2 + 1,
            2,
        ),
        range(
            resampled_fid[1] - SAMPLING_RATE * 2,
            resampled_fid[1] + SAMPLING_RATE * 2 + 1,
            2,
        ),
        range(
            resampled_fid[2] - SAMPLING_RATE * 2,
            resampled_fid[2] + SAMPLING_RATE * 2 + 1,
            2,
        ),
    ]
    outer_samples = list(it.product(*outer_its))

    # Concatenate and retain unique samples and
    all_samples = np.unique(np.array(inner_samples + outer_samples), axis=0)

    # Compute Haar-like features features
    # Make this optional to load or create
    feature_offsets_data = np.load(feature_offsets_path)
    smin, smax = feature_offsets_data["arr_0"], feature_offsets_data["arr_1"]

    # Generate bounding cube surrounding features
    min_corner_list = np.zeros((4000 * all_samples.shape[0], 3)).astype("uint8")
    max_corner_list = np.zeros((4000 * all_samples.shape[0], 3)).astype("uint8")

    for idx in range(all_samples.shape[0]):
        min_corner_list[idx * 4000 : (idx + 1) * 4000] = all_samples[idx] + smin
        max_corner_list[idx * 4000 : (idx + 1) * 4000] = all_samples[idx] + smax

    corner_list = np.hstack((min_corner_list, max_corner_list))

    # compute the integral image for more efficient generation of haar-like features
    iv_image = integral_volume(img)

    # intialize a numpy array to store features
    testerarr = np.zeros((4000 * all_samples.shape[0]))

    numerator = (
        iv_image[corner_list[:, 3] + 1, corner_list[:, 4] + 1, corner_list[:, 5] + 1]
        - iv_image[corner_list[:, 0], corner_list[:, 4] + 1, corner_list[:, 5] + 1]
        - iv_image[corner_list[:, 3] + 1, corner_list[:, 4] + 1, corner_list[:, 2]]
        - iv_image[corner_list[:, 3] + 1, corner_list[:, 1], corner_list[:, 5] + 1]
        - iv_image[corner_list[:, 0], corner_list[:, 1], corner_list[:, 2]]
        + iv_image[corner_list[:, 3] + 1, corner_list[:, 1], corner_list[:, 2]]
        + iv_image[corner_list[:, 0], corner_list[:, 1], corner_list[:, 5] + 1]
        + iv_image[corner_list[:, 0], corner_list[:, 4] + 1, corner_list[:, 2]]
    )

    denominator = (
        (corner_list[:, 3] - corner_list[:, 0] + 1)
        * (corner_list[:, 4] - corner_list[:, 1] + 1)
        * (corner_list[:, 5] - corner_list[:, 2] + 1)
    )

    # dump features into intialized variable
    testerarr = numerator / denominator
    vector1arr = np.zeros((4000 * all_samples.shape[0]))
    vector2arr = np.zeros((4000 * all_samples.shape[0]))

    for index in range(all_samples.shape[0]):
        vector = range(index * 4000, index * 4000 + 2000)
        vector1arr[index * 4000 : (index + 1) * 4000 - 2000] = vector

    for index in range(all_samples.shape[0]):
        vector = range(index * 4000 + 2000, index * 4000 + 4000)
        vector2arr[index * 4000 + 2000 : (index + 1) * 4000] = vector

    vector1arr[0] = 1
    vector1arr = vector1arr[vector1arr != 0]
    vector1arr[0] = 0
    vector2arr = vector2arr[vector2arr != 0]
    vector1arr = vector1arr.astype(int)
    vector2arr = vector2arr.astype(int)

    diff = testerarr[vector1arr] - testerarr[vector2arr]
    diff = np.reshape(diff, (all_samples.shape[0], 2000))
    dist = all_samples - resampled_fid[0:3]
    prob = np.sqrt(dist[:, 0] ** 2 + dist[:, 1] ** 2 + dist[:, 2] ** 2)
    prob = np.exp(-0.5 * prob)  # weighted more towards closer distances

    return [np.hstack((diff[index], prob[index])) for index in range(prob.shape[0])]


def train_afid_model(
    afid_num: int,
    subject_paths: Sequence[PathLike[str] | str],
    fcsv_paths: Sequence[PathLike[str] | str],
    feature_offsets_path: PathLike | str,
):
    """Train a regRF model for a fiducial."""
    finalpred = np.asarray(
        it.chain.from_iterable(
            gen_features(subject_path, fcsv_path, feature_offsets_path, afid_num)
            for subject_path, fcsv_path in zip(subject_paths, fcsv_paths, strict=True)
        )
    )
    print(finalpred.shape)
    finalpredarr = np.concatenate((np.zeros((1, 2001))), finalpred)[1:, :]
    regr_rf = RandomForestRegressor(
        n_estimators=20,
        max_features=0.33,
        min_samples_leaf=5,
        random_state=2,
        n_jobs=-1,
    )
    x_train = finalpredarr[:, :-1]
    y_train = finalpredarr[:, -1]

    print("training start")
    mdl = regr_rf.fit(x_train, y_train)
    dump(mdl, f"{afid_num}_{SAMPLING_RATE}x{SAMPLING_RATE}x{SAMPLING_RATE}.joblib")
    print("training ended")


def train_all_afid_models(
    subject_paths: Sequence[PathLike[str] | str],
    fcsv_paths: Sequence[PathLike[str] | str],
    feature_offsets_path: PathLike | str,
):
    """Train a regRF fiducial for each of the 32 AFIDs."""
    for afid_num in range(1, 33):
        train_afid_model(afid_num, subject_paths, fcsv_paths, feature_offsets_path)
