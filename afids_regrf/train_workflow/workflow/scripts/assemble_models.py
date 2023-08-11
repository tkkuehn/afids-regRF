from __future__ import annotations

import json
import tarfile
from argparse import ArgumentParser
from collections.abc import Mapping
from io import BytesIO
from os import PathLike


def assemble_models(
    models: Mapping[int, PathLike[str] | str],
    padding: int,
    size: int,
    sampling_rate: int,
    out_path: PathLike | str,
    feature_offsets_path: PathLike | str,
) -> None:
    config_info = tarfile.TarInfo("config.json")
    config_content = json.dumps(
        {
            "padding": padding,
            "size": size,
            "sampling_rate": sampling_rate,
            "version": "1.0.0",
        }
    ).encode("utf-8")
    config_info.size = len(config_content)
    with tarfile.open(out_path, "x:gz") as tar_file:
        tar_file.addfile(
            config_info,
            BytesIO(config_content),
        )
        for afid_label in range(1, 33):
            tar_file.add(models[afid_label], arcname=f"afid-{afid_label:02}.model")
        tar_file.add(feature_offsets_path, arcname="feature_offsets.npz")


def gen_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("models", nargs=32)
    parser.add_argument("padding", type=int)
    parser.add_argument("size", type=int)
    parser.add_argument("sampling_rate", type=int)
    parser.add_argument("feature_offsets")
    parser.add_argument("out_path")

    return parser


def main() -> None:
    args = gen_parser().parse_args()

    assemble_models(
        {idx + 1: model_path for idx, model_path in enumerate(args.models)},
        args.padding,
        args.size,
        args.sampling_rate,
        args.out_path,
        args.feature_offsets,
    )


if __name__ == "__main__":
    main()
