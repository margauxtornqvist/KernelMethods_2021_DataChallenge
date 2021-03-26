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
from grid_search import CvSearch_Spectrum, CvSearch_Sum, get_best_sum
from sum_kernels import compute_gram
from KRR import KRR
from KSVM import KernelSVM


# loading datasets
data_folder = 'data/'
experiment_path = './submissions'

# load string data
X_train, Y_train = load_train_data(data_folder, 'raw')
X_test = load_test_data(data_folder, 'raw')

# map the target vectors to do binary classification
Y_train_ = 2 * Y_train - 1

##### DATASET 0
kernels = [['mismatch', ii] for ii in [7,8]]
lambd_0 = 1.4
K_train_0, K_test_0 = get_gram(kernels=kernels, dataset_idx=0,
                               X_train=X_train[0], X_test=X_test[0], scale=False, center=False, sum_gram= True)
model_0 = KernelSVM(lambd=lambd_0)
model_0.fit(K_train_0, Y_train_[0])
y_pred_0 = model_0.predict(K_test_0)

##### DATASET 1
kernels = [['mismatch', ii] for ii in [4,8,10,11]]
lambd_1 = 2.3
K_train_1, K_test_1 = get_gram(kernels=kernels, dataset_idx=1,
                               X_train=X_train[1], X_test=X_test[1], scale=False, center=False, sum_gram=True)
model_1 = KernelSVM(lambd=lambd_1)
model_1.fit(K_train_1, Y_train_[1])
y_pred_1 = model_1.predict(K_test_1)

##### DATASET 2
kernels = [['mismatch', ii] for ii in [4,6,9,11]] 
lambd_2 = 1.6
K_train_2, K_test_2 = get_gram(kernels=kernels, dataset_idx=2,
                               X_train=X_train[2], X_test=X_test[2], scale=False, center=False, sum_gram=True)
model_2 = KernelSVM(lambd=lambd_2)
model_2.fit(K_train_2, Y_train_[2])
y_pred_2 = model_2.predict(K_test_2)


# concatenate the predictions
y_pred = np.concatenate((y_pred_0,
                         y_pred_1,
                         y_pred_2), axis=0)

df_submit = kaggle_submit(y_pred, experiment_path, num_submit=7)