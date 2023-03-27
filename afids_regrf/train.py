"""Methods for training a regRF model."""
from __future__ import annotations

import itertools as it
from collections.abc import Iterable, Sequence
from os import PathLike
from typing import NoReturn

import numpy as np
from joblib import dump
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

from .utils import get_fid, gen_features

def train_afid_model(
    afid_num: int,
    subject_paths: Iterable[PathLike[str] | str],
    fcsv_paths: Iterable[PathLike[str] | str],
    feature_offsets: tuple[NDArray, NDArray],
    padding: int,
    sampling_rate: int,
    size: int
) -> NoReturn:
    """Train a regRF model for a fiducial."""
    finalpred = np.asarray(
        it.chain.from_iterable(
            gen_features(
                subject_path, 
                get_fid(fcsv_path, afid_num - 1), 
                feature_offsets,
                padding,
                sampling_rate,
                size,
            )
            for subject_path, fcsv_path in zip(subject_paths, fcsv_paths)
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

    # NOTE: Should dump the model into appropriate location
    print("training start")
    mdl = regr_rf.fit(x_train, y_train)
    dump(
        mdl, 
        f"afid-{str(afid_num).zfill(2)}_desc-rf_sampleRate-iso{sampling_rate}vox_model.joblib"
    )
    print("training ended")


def train_all_afid_models(
    subject_paths: Sequence[PathLike[str] | str],
    fcsv_paths: Sequence[PathLike[str] | str],
    feature_offsets_path: PathLike | str,
    padding: int = 0,
    size: int = 1,
    sampling_rate: int = 5,
) -> NoReturn:
    """Train a regRF fiducial for each of the 32 AFIDs."""
    feature_offsets = np.load(feature_offsets_path)
    for afid_num in range(1, 33):
        train_afid_model(
            afid_num,
            subject_paths,
            fcsv_paths,
            (feature_offsets["arr_0"], feature_offsets["arr_1"]),
            padding,
            size,
            sampling_rate,
        )