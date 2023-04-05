#!/usr/bin/env python3

import timeit

import numpy as np
import pandas as pd

import afids_regrf.utils as utils

IMG_PATH = "/localscratch/afids-data/data/datasets/AFIDs-HCP/sub-103111/anat/sub-103111_acq-MP2RAGE_T1w.nii.gz"
SIZE = 1
PADDING = 0
SAMPLING_RATE = 5

feature_offsets = (
    (file_ := np.load("afids_regrf/modelling/resources/feature_offsets.npz"))["arr_0"],
    file_["arr_1"],
)
fiducial = np.array([-0.6118652714866496, 3.3806264314898784, -2.491723726723984])
aff, img = utils.read_nii_metadata(IMG_PATH)
img = utils.zoom(img, SIZE)
resampled_fid = utils.fid_world2voxel(
    fiducial, aff, resample_size=SIZE, padding=PADDING
)


def sample_coords():
    return utils.sample_coord_region(resampled_fid, SAMPLING_RATE)


sample_time = timeit.timeit(sample_coords, number=5)
print(f"Sample time: {sample_time}")


def gen_all_samples():
    return (
        pd.concat(
            [
                sample_coords(),
                utils.sample_coord_region(resampled_fid, SAMPLING_RATE, multiplier=2),
            ]
        )
        .drop_duplicates(ignore_index=True)
        .sort_values(by=["x", "y", "z"], ignore_index=True)
    )


all_sample_time = timeit.timeit(gen_all_samples, number=5)
print(f"All sample time: {all_sample_time}")
all_samples = gen_all_samples()


def gen_feature_boxes():
    return utils.gen_feature_boxes(feature_offsets, all_samples)


boxes_time = timeit.timeit(gen_feature_boxes, number=5)
print(f"Boxes time: {boxes_time}")
boxes = gen_feature_boxes()


def gen_box_averages():
    return utils.gen_box_averages(img, boxes)


averages_time = timeit.timeit(gen_box_averages, number=5)
print("averages time: {averages_time}")
averages = gen_box_averages()


gen_features_time = timeit.timeit(
    lambda: utils.gen_features(
        IMG_PATH,
        fiducial,
        feature_offsets,
        0,
        5,
        1,
        predict=False,
    ),
    number=5,
)

with open("pandas_times.txt", "w", encoding="utf-8") as time_file:
    time_file.write(f"sample time: {sample_time}\n")
    time_file.write(f"all sample time: {all_sample_time}\n")
    time_file.write(f"boxes time: {boxes_time}\n")
    time_file.write(f"box averages time: {averages_time}\n")
    time_file.write(f"total time: {gen_features_time}\n")
