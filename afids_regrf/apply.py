"""Methods for applying trained regRF models."""
from __future__ import annotations

import itertools as it
from collections.abc import Iterable, Sequence
from os import PathLike

import numpy as np
import pandas as pd
from joblib import load
from numpy.typing import NDArray

from .utils import afids_to_fcsv, get_fid, gen_features

def apply_afid_model(
    afid_num: int,
    subject_paths: Iterable[PathLike[str] | str],
    fcsv_paths: Iterable[PathLike[str] | str],
    feature_offsets: tuple[NDArray, NDArray],
    padding: int,
    sampling_rate: int,
    size: int
) -> NDArray:
    """Apply a trained regRF model for a fiducial"""
    aff, diff, samples = it.chain.from_iterable(
        gen_features(
            subject_path,
            get_fid(fcsv_path, afid_num- -1),
            feature_offsets,
            padding,
            sampling_rate,
            size,
            predict=True
        )
        for subject_path, fcsv_path in zip(subject_paths, fcsv_paths)
    )

    # NOTE: Load from appropriate location
    # Load trained model and predict distances of coordinates
    regr_rf = load(
        f"afid-{afid_num.zfill(2)}_desc-rf_sampleRate-iso{sampling_rate}vox_model.joblib"
        )
    dist_predict = regr_rf.predict(diff)

    # Extract smallest Euclidean distance from predictions
    dist_df = pd.DataFrame(dist_predict)
    print(dist_df[0].max())
    idx = dist_df[0].idxmax()

    # Reverse look up to determine voxel with lowest distances
    print(f'Voxel coordinates with greatest liklihood of being AFID #{afid_num} are: {samples[idx]}')

    afid_coords = aff[:3, :3].dot(samples[idx]) + aff[:3, 3]

    return afid_coords


def apply_all_afid_models(
    subject_paths: Sequence[PathLike[str] | str],
    fcsv_paths: Sequence[PathLike[str] | str],
    feature_offsets_path: PathLike | str,
    padding: int = 0,
    size: int = 1,
    sampling_rate: int = 5,
):
    """Apply a trained regRF fiducial for each of the 32 AFIDs."""
    all_afids_coords = np.empty((3, ), dtype=float) 

    feature_offsets = np.load(feature_offsets_path)
    for afid_num in range(1, 33):
        afid_coords = apply_afid_model(
            afid_num,
            subject_paths,
            fcsv_paths,
            (feature_offsets["arr_0"], feature_offsets["arr_1"]),
            padding,
            size,
            sampling_rate,
        )
        all_afids_coords = np.vstack((all_afids_coords, afid_coords))
    
    afids_to_fcsv(all_afids_coords[1:].astype(int))