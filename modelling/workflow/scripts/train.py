#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 20:32:42 2020

@author: greydon
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import hickle as hkl

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


args = Namespace(input_fname=snakemake.input[0], 
                 model_params=snakemake.params[0],
                 out_name=snakemake.output[0])

data_all = hkl.load(args.input_fname)

finalpredarr = []
for idata in data_all['data_arr']:
    finalpredarr.append(idata['data_arr'][0][1:,:])
    
# Model training
finalpredarr = np.asarray(finalpredarr, dtype=np.float32)
finalpredarr=finalpredarr.reshape(finalpredarr.shape[0]*finalpredarr.shape[1],finalpredarr.shape[2])

regr_rf = RandomForestRegressor(n_estimators=args.model_params['randomforest']['n_estimators'], 
                                max_features=args.model_params['randomforest']['max_features'], 
                                min_samples_leaf=args.model_params['randomforest']['min_samples_leaf'],
                                random_state=args.model_params['randomforest']['random_state'], 
                                n_jobs=args.model_params['randomforest']['n_jobs'])

X_train = finalpredarr[:,:-1]
y_train = finalpredarr[:,-1]

Mdl = regr_rf.fit(X_train, y_train)

with open(args.out_name, 'wb') as f:
    joblib.dump(Mdl, f)
    