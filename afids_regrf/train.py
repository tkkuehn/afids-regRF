#!/usr/bin/env python
"""Methods for training a regRF model."""
from __future__ import annotations

import itertools as it
from argparse import ArgumentParser
from collections.abc import Iterable, Sequence
from os import PathLike
from pathlib import Path

import numpy as np
from joblib import dump
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

from afids_regrf.utils import gen_features, get_fid


def train_afid_model(
    afid_num: int,
    subject_paths: Iterable[PathLike[str] | str],
    fcsv_paths: Iterable[PathLike[str] | str],
    feature_offsets: tuple[NDArray, NDArray],
    padding: int,
    sampling_rate: int,
    size: int,
) -> RandomForestRegressor:
    """Train a regRF model for a fiducial."""
    finalpred = np.asarray(
        list(
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
    )
    finalpredarr = np.concatenate((np.zeros((1, 2001)), finalpred))[1:, :]
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
    print(f"training start - afid-{str(afid_num).zfill(2)}")
    model = regr_rf.fit(x_train, y_train)
    print("training ended")

    return model


def train_all_afid_models(
    subject_paths: Sequence[PathLike[str] | str],
    fcsv_paths: Sequence[PathLike[str] | str],
    feature_offsets_path: PathLike | str,
    model_dir_path: PathLike | str,
    padding: int = 0,
    sampling_rate: int = 5,
    size: int = 1,
) -> None:
    """Train a regRF fiducial for each of the 32 AFIDs."""
    feature_offsets = np.load(feature_offsets_path)
    for afid_num in range(1, 33):
        model = train_afid_model(
            afid_num,
            subject_paths,
            fcsv_paths,
            (feature_offsets["arr_0"], feature_offsets["arr_1"]),
            padding,
            sampling_rate,
            size,
        )

        # Save model
        model_fname = f"afid-{str(afid_num).zfill(2)}_desc-rf_sampleRate-iso{sampling_rate}vox_model.joblib"
        dump(model, Path(model_dir_path) / model_fname)


def gen_parser() -> ArgumentParser:
    """Generate CLI parser for script"""
    parser = ArgumentParser()

    parser.add_argument(
        "--subject_paths",
        nargs="+",
        type=str,
        help=(
            "Path to subject nifti images. If more than 1 subject, pass paths "
            "as space-separated list."
        ),
    )
    parser.add_argument(
        "--fcsv_paths",
        nargs="+",
        type=str,
        help=(
            "Path to subject fcsv files. If more than 1 subject, pass paths as "
            "space-separated list."
        ),
    )
    parser.add_argument(
        "--feature_offsets_path", type=str, help=("Path to featuers_offsets.npz file")
    )
    parser.add_argument(
        "--model_dir_path",
        type=str,
        help=("Path to directory for saving fitted models."),
    )
    parser.add_argument(
        "--padding",
        nargs="?",
        type=int,
        default=0,
        required=False,
        help=("Number of voxels to add when zero-padding nifti images. " "Default: 0"),
    )
    parser.add_argument(
        "--size",
        nargs="?",
        type=int,
        default=1,
        required=False,
        help=("Factor to resample nifti image by. Default: 1"),
    )
    parser.add_argument(
        "--sampling_rate",
        nargs="?",
        type=int,
        default=5,
        required=False,
        help=(
            "Number of voxels in both directions along each axis to sample as "
            "part of the training Default: 5"
        ),
    )

    return parser


def main():
    parser = gen_parser()
    args = parser.parse_args()

    train_all_afid_models(
        subject_paths=args.subject_paths,
        fcsv_paths=args.fcsv_paths,
        feature_offsets_path=args.feature_offsets_path,
        model_dir_path=args.model_dir_path,
        padding=args.padding,
        sampling_rate=args.sampling_rate,
        size=args.size,
    )


if __name__ == "__main__":
    main()
