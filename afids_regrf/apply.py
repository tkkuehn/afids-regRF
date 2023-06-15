#!/usr/bin/env python
"""Methods for applying trained regRF models."""
from __future__ import annotations

import itertools as it
from argparse import ArgumentParser
from collections.abc import Iterable, Sequence
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
from importlib_resources import files
from joblib import load
from numpy.typing import NDArray

from afids_regrf.utils import afids_to_fcsv, gen_features, get_fid


def apply_afid_model(
    afid_num: int,
    subject_paths: Iterable[PathLike[str] | str],
    fcsv_paths: Iterable[PathLike[str] | str],
    model_dir_path: PathLike[str] | str,
    feature_offsets: tuple[NDArray, NDArray],
    padding: int,
    sampling_rate: int,
    size: int,
) -> NDArray:
    """Apply a trained regRF model for a fiducial"""
    aff, diff, samples = it.chain.from_iterable(
        gen_features(
            subject_path,
            get_fid(fcsv_path, afid_num - 1),
            feature_offsets,
            padding,
            sampling_rate,
            size,
            predict=True,
        )
        for subject_path, fcsv_path in zip(subject_paths, fcsv_paths)
    )

    # NOTE: Load from appropriate location
    # Load trained model and predict distances of coordinates
    model_fname = f"afid-{str(afid_num).zfill(2)}_desc-rf_sampleRate-iso{sampling_rate}vox_model.joblib"
    regr_rf = load(Path(model_dir_path) / model_fname)
    dist_predict = regr_rf.predict(diff)

    # Extract smallest Euclidean distance from predictions
    dist_df = pd.DataFrame(dist_predict)
    print(dist_df[0].max())
    idx = dist_df[0].idxmax()

    # Reverse look up to determine voxel with lowest distances
    print(
        f"Voxel coordinates with greatest likelihood of being AFID #{afid_num} are: {samples[idx]}"
    )

    afid_coords = aff[:3, :3].dot(samples[idx]) + aff[:3, 3]

    return afid_coords


def apply_all_afid_models(
    subject_paths: Sequence[PathLike[str] | str],
    fcsv_paths: Sequence[PathLike[str] | str],
    out_paths: Iterable[PathLike[str] | str],
    feature_offsets_path: PathLike | str,
    model_dir_path: PathLike | str,
    padding: int = 0,
    sampling_rate: int = 5,
    size: int = 1,
) -> None:
    """Apply a trained regRF fiducial for each of the 32 AFIDs."""
    all_afids_coords = np.empty((3,), dtype=float)

    feature_offsets = np.load(feature_offsets_path)
    for afid_num in range(1, 33):
        afid_coords = apply_afid_model(
            afid_num,
            subject_paths,
            fcsv_paths,
            model_dir_path,
            (feature_offsets["arr_0"], feature_offsets["arr_1"]),
            padding,
            sampling_rate,
            size,
        )
        all_afids_coords = np.vstack((all_afids_coords, afid_coords))

    for idx, out_path in enumerate(out_paths):
        afids_to_fcsv(
            all_afids_coords[(idx * 32) + 1 : (idx * 32) + 33],
            "afids_regrf/modelling/resources/dummy.fcsv",
            out_path,
        )


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
        "--model_dir_path",
        type=str,
        help="Path to directory containing trained models.",
    )
    parser.add_argument(
        "--output_fcsv_paths", nargs="+", type=str, help="Path to output FCSVs."
    )
    parser.add_argument(
        "--feature_offsets_path",
        type=str,
        help="Path to features_offsets.npz file",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--padding",
        nargs="?",
        type=int,
        default=0,
        required=False,
        help=("Number of voxels to add when zero-padding nifti images. Default: 0"),
    )
    parser.add_argument(
        "--size",
        nargs="?",
        type=int,
        default=1,
        required=False,
        help="Factor to resample nifti image by. Default: 1",
    )
    parser.add_argument(
        "--sampling_rate",
        nargs="?",
        type=int,
        default=5,
        required=False,
        help=(
            "Number of voxels in both directions along each axis to sample as "
            "part of the training. Default: 5"
        ),
    )

    return parser


def main():
    parser = gen_parser()
    args = parser.parse_args()

    apply_all_afid_models(
        subject_paths=args.subject_paths,
        fcsv_paths=[
            files("afids_regrf.resources").joinpath(
                "tpl-MNI152NLin2009cAsym_res-01_desc-groundtruth_afids.fcsv"
            )
            for _ in args.subject_paths
        ],
        out_paths=args.output_fcsv_paths,
        feature_offsets_path=args.feature_offsets_path
        if args.feature_offsets_path is not None
        else files("afids_regrf.resources").joinpath("feature_offsets.npz"),
        model_dir_path=args.model_dir_path,
        padding=args.padding,
        sampling_rate=args.sampling_rate,
        size=args.size,
    )


if __name__ == "__main__":
    main()
