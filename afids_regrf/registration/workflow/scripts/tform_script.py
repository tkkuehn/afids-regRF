#!/usr/bin/env python
# coding: utf-8

import csv
import numpy as np

fcsv_source = snakemake.params[0]
xfm_txt = snakemake.input[0]
template = snakemake.params[1]
fcsv_new = snakemake.output[0]

with open(fcsv_source, 'r') as file:
    reader = csv.reader(file)
    next(reader)
    next(reader)
    next(reader)
    arr = np.empty((0,3))
    for row in reader:
        x = row[1:4]
        arr = np.vstack([arr,x])
    arr = np.asarray(arr,dtype='float64')


f = open(xfm_txt, 'r')
contents = f.readlines()
list_of_lists = []
for line in contents:
    stripped_line = line.strip()
    line_list = stripped_line.split()
    list_of_lists.append(line_list)

tform = np.empty((0,4))
for row in list_of_lists:
    x = row[:]
    tform = np.vstack([tform,x])

tform = np.asarray(tform,dtype='float64')
tform = np.linalg.inv(tform)
ones = np.ones((32,1))
arr = np.hstack((arr,ones))

'''
factor = np.array([[1, 1, 1, 1],[1, 1, 1, 1],[1, 1, 1, 1],[1, 1, 1, 1]])
tform = np.multiply(tform,factor)
'''

tform_applied = np.empty((0,4))
for i in range(32):
    x = np.matmul(tform,arr[i].transpose())
    tform_applied = np.vstack([tform_applied,x])


with open(template, 'r') as file:
    list_of_lists = []
    reader = csv.reader(file)
    for i in range(3):
        list_of_lists.append(next(reader))
    for idx, val in enumerate(reader):
        val[1:4] = tform_applied[idx][:3]
        list_of_lists.append(val)


with open(fcsv_new, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(list_of_lists)





