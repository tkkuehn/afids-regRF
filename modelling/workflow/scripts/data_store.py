#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 22:26:50 2020

@author: greydon
"""
import numpy as np
import nibabel as nib
import csv
from imresize import imresize
import itertools
import hickle as hkl
import os

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


args = Namespace(output_dir=snakemake.output[0], 
                 iafid=snakemake.params[0],
                 model_params=snakemake.params[1],
                 combined_files=snakemake.params[2],
                 train_level=snakemake.params[3])

space = 'space-'+os.path.basename(args.output_dir).split('space-')[0].split('_')[0]
finalpredarr_all = []
for inifti, ifcsv in args.combined_files.items():
    # Loading image
    niimeta = nib.load(inifti)
    hdr = niimeta.header
    img_source = niimeta.get_fdata()
    img_source = np.transpose(img_source,(2,0,1))
    
    finalpredarr = []
    # Loading and processing .fcsv file.
    with open(ifcsv) as file:
        csv_reader = csv.reader(file, delimiter=',')
        for skip in range(3):
            next(csv_reader)
            
        arr = np.empty((0,3))
        for row in csv_reader:
            x = row[1:4]
            x = np.asarray(x,dtype='float64')
            arr = np.vstack([arr,x])
        
        if (hdr['qform_code'] > 0 and hdr['sform_code'] == 0):
            newarr = []
            B = hdr['quatern_b']
            C = hdr['quatern_c']
            D = hdr['quatern_d']
            A = np.sqrt(1 - B**2 - C**2 - D**2)
            
            R = [A**2+B**2-C**2-D**2, 2*(B*C-A*D), 2*(B*D+A*C)], [2*(B*C+A*D), A**2+C**2-B**2-D**2,2*(C*D+A*B)], [2*(B*D-A*C), 2*(C*D+A*B), A**2+D**2-C**2-B**2]
            R = np.array(R)
            
            ijk = arr[int(args.iafid)-1].reshape(-1,1)
            ijk[2] = ijk[2]*hdr['pixdim'][0]
            pixdim = hdr['pixdim'][1],hdr['pixdim'][2],hdr['pixdim'][3]
            pixdim = np.array(pixdim).reshape(-1,1)
            fill = np.matmul(R,ijk)*pixdim+np.vstack([hdr['qoffset_x'],hdr['qoffset_y'],hdr['qoffset_z']])
            fill = fill.reshape(3)
            newarr.append(fill)
            
            arr = np.array(newarr)
            
            arr = arr-1
            
        elif hdr['sform_code'] > 0:
            
            newarr = []
            four = np.vstack([hdr['srow_x'],hdr['srow_y'],hdr['srow_z'],[0,0,0,1]])
            four = np.linalg.inv(four)
            trying = np.hstack([arr,np.ones((32,1))])
            fill = np.matmul(four,trying[int(args.iafid)-1].reshape(-1,1))
            fill = fill.reshape(4)
            newarr.append(fill)
            
            arr = np.array(newarr)
            arr = arr-1
            
        else:
            print('Error in sform_code or qform_code, cannot obtain coordinates.')
            
    img = np.single(img_source)
    img = (img-np.amin(img))/(np.amax(img)-np.amin(img))
    
    skip = False
    if args.train_level == 'fine':
        ##FINE
        arr = np.rint(arr)
        arr = arr.astype(int)
        perm = [2,0,1]
        arr = arr[:,perm]
        
        patch = img[arr[0,0]-30:arr[0,0]+31,arr[0,1]-30:arr[0,1]+31,arr[0,2]-30:arr[0,2]+31]
        
        # Upsampled image patch (only using patch because other parts of image are not relevant).
        patch = imresize(patch,2)
    
        if arr[0,0] < 30 or arr[0,1] < 30 or arr[0,2] < 30:
            print('skip')
            skip=True
        
        inner = []
        outer = []
    
        iterables = [ range(60-5,60+6), range(60-5,60+6), range(60-5,60+6) ]
        for t in itertools.product(*iterables):
            inner.append(t)
    
        iterables = [ range(60-10,60+11,2), range(60-10,60+11,2), range(60-10,60+11,2) ]
        for t in itertools.product(*iterables):
            outer.append(t)
    
    
    elif args.train_level =='medium':
        # Image at normal resolution.
        img_new = imresize(img,1)
    
        arr = np.rint(arr)
        img_pad = np.pad(img_new, 50, mode='constant')
        arr = arr + 50
        arr = arr.astype(int)
        perm = [2,0,1]
        arr = arr[:,perm]
        
        patch = img_pad[arr[0,0]-60:arr[0,0]+61,arr[0,1]-60:arr[0,1]+61,arr[0,2]-60:arr[0,2]+61]
        patch = (patch-np.amin(patch))/(np.amax(patch)-np.amin(patch))
        
        inner = []
        outer = []
    
        iterables = [ range(60-5,60+6), range(60-5,60+6), range(60-5,60+6) ]
        for t in itertools.product(*iterables):
            inner.append(t)
        
        iterables = [ range(60-10,60+11,2), range(60-10,60+11,2), range(60-10,60+11,2) ]
        for t in itertools.product(*iterables):
            outer.append(t)
    
    elif args.train_level == 'coarse':
        # Downsampled image.
        img_new = imresize(img,0.25)
    
        arr = np.rint(arr/4)
        patch = np.pad(img_new, 50, mode='constant')
        arr = arr + 50
        arr = arr.astype(int)
        
        inner = []
        outer = []
        
        iterables = [ range(arr[0][0]-5,arr[0][0]+6), range(arr[0][1]-5,arr[0][1]+6), range(arr[0][2]-5,arr[0][2]+6) ]
        for t in itertools.product(*iterables):
            inner.append(t)
    
        iterables = [ range(arr[0][0]-10,arr[0][0]+11,2), range(arr[0][1]-10,arr[0][1]+11,2), range(arr[0][2]-10,arr[0][2]+11,2) ]
        for t in itertools.product(*iterables):
            outer.append(t)
    
    if not skip:
        J = patch.cumsum(0).cumsum(1).cumsum(2)
        
        inner = np.array(inner)
        outer = np.array(outer)
        
        full = np.concatenate((inner,outer))
        full = np.unique(full,axis=0)
        
        # Loads offset file that specifies where to extract features.
        file = np.load(args.model_params['feature_offsets'])
        smin = file['arr_0']
        smax = file['arr_1']
        
        perm = [2,0,1]
        full = full[:,perm]
        smin = smin[:,perm]
        smax = smax[:,perm]
        
        mincornerlist = np.zeros((4000*full.shape[0], 3)).astype('uint8')
        maxcornerlist = np.zeros((4000*full.shape[0], 3)).astype('uint8')
        
        for index in range(full.shape[0]):
            mincorner = full[index] + smin
            maxcorner = full[index] + smax
            mincornerlist[index*4000:(index+1)*4000] = mincorner
            maxcornerlist[index*4000:(index+1)*4000] = maxcorner
        
        cornerlist = np.hstack((mincornerlist,maxcornerlist))
        cornerlist = cornerlist.astype(int)
        
        Jnew = np.zeros((J.shape[0]+1,J.shape[1]+1,J.shape[2]+1))
        Jnew[1:,1:,1:] = J
        J = Jnew
        
        testerarr = np.zeros((4000*full.shape[0]))
        
        # Generation of features (random blocks of intensity around fiducial)
        numerator = J[cornerlist[:,3] + 1, cornerlist[:,4]+1, cornerlist[:,5]+1] - J[cornerlist[:,3]+1,cornerlist[:,4]+1,cornerlist[:,2]] - \
        J[cornerlist[:,3]+1,cornerlist[:,1],cornerlist[:,5]+1] - J[cornerlist[:,0],cornerlist[:,4]+1,cornerlist[:,5]+1] + \
        J[cornerlist[:,0],cornerlist[:,1],cornerlist[:,5]+1] + J[cornerlist[:,0],cornerlist[:,4]+1,cornerlist[:,2]] + \
        J[cornerlist[:,3]+1,cornerlist[:,1],cornerlist[:,2]] - J[cornerlist[:,0],cornerlist[:,1],cornerlist[:,2]]
    
        denominator = (cornerlist[:,3]-cornerlist[:,0]+1)*(cornerlist[:,4]-cornerlist[:,1]+1)*(cornerlist[:,5]-cornerlist[:,2]+1)
        
        testerarr = numerator/denominator
        
        vector1arr = np.zeros((4000*full.shape[0]))
        vector2arr = np.zeros((4000*full.shape[0]))
        
        for index in range(full.shape[0]):
            vector = range(index*4000,index*4000+2000)
            vector1arr[index*4000:(index+1)*4000-2000] = vector
        
        for index in range(full.shape[0]):
            vector = range(index*4000+2000,index*4000+4000)
            vector2arr[index*4000+2000:(index+1)*4000] = vector
        
        vector1arr[0] = 1
        vector1arr = vector1arr[vector1arr != 0]
        vector1arr[0] = 0
        vector2arr = vector2arr[vector2arr != 0]
        vector1arr = vector1arr.astype(int)
        vector2arr = vector2arr.astype(int)
        
        diff = testerarr[vector1arr] - testerarr[vector2arr]
        diff = np.reshape(diff,(full.shape[0],2000))
        dist = full - 60
        p = np.sqrt(dist[:,0]**2 + dist[:,1]**2 + dist[:,2]**2)
        
        finalpred = []
        for index in range(p.shape[0]):
            finalpred.append(np.hstack((diff[index],p[index])))
        
        # Concatenate to array of feature vectors.
        finalpredarr.append(np.asarray(finalpred, dtype=np.float32))
    
    finalpredarr_all.append({ 'name' : os.path.basename(inifti).split('_')[0], 'data_arr' : np.asarray(finalpredarr, dtype=np.float32)})

# Save to file
data = { 'name' : '_'.join([args.iafid, space,args.train_level]), 'data_arr' : finalpredarr_all}

# Dump data to file
with open(args.output_dir,'w') as f:
    hkl.dump( data, f)






