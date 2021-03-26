#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:16:47 2021

@author: margauxtornqvist
"""
"""
This script executes our best results

"""
# imports

import numpy as np
import os
from utils import check_DNA, load_train_data, load_test_data, get_gram, kaggle_submit
from kernels import GaussianKernel, SpectrumKernel, MismatchKernel
from preprocessing import center_gram, scale_gram
from GridSearch import CvSearch_Spectrum, CvSearch_Sum, get_best_sum
from SumKernels import compute_gram
from KRR import KRR


# loading datasets
data_folder = 'data/'
experiment_path = './submissions'

# load string data
X_train, Y_train = load_train_data(data_folder, 'raw')
X_test = load_test_data(data_folder, 'raw')

# map the target vectors to do binary classification
Y_train_ = 2 * Y_train - 1

##### DATASET 0
kernels = [['mismatch', ii] for ii in [8,9,10]] 
lambd_0 = .0007
K_train_0, K_test_0 = get_gram(kernels=kernels, dataset_idx=0,
                               X_train=X_train[0], X_test=X_test[0], scale=True, center=True, sum_gram= True)
model_0 = KRR(lambd=lambd_0, threshold=0)
model_0.fit(K_train_0, Y_train_[0])
y_pred_0 = model_0.predict(K_test_0)

##### DATASET 1
kernels = [['mismatch', ii] for ii in [4,8,10]] 
lambd_1 = 0.0004
K_train_1, K_test_1 = get_gram(kernels=kernels, dataset_idx=1,
                               X_train=X_train[1], X_test=X_test[1], scale=True, center=True, sum_gram=True)
model_1 = KRR(lambd=lambd_1, threshold=0)
model_1.fit(K_train_1, Y_train_[1])
y_pred_1 = model_1.predict(K_test_1)

##### DATASET 2
kernels = [['mismatch', ii] for ii in [7,8,9]] 
lambd_2 = 0.00015
K_train_2, K_test_2 = get_gram(kernels=kernels, dataset_idx=2,
                               X_train=X_train[2], X_test=X_test[2], scale=True, center=True, sum_gram=True)
model_2 = KRR(lambd=lambd_2, threshold=0)
model_2.fit(K_train_2, Y_train_[2])
y_pred_2 = model_2.predict(K_test_2)


# concatenate the predictions
y_pred = np.concatenate((y_pred_0,
                         y_pred_1,
                         y_pred_2), axis=0)

df_submit = kaggle_submit(y_pred, experiment_path, num_submit=7)