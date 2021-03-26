#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:42:55 2021

@author: margauxtornqvist
"""
import numpy as np
import pandas as pd
from copy import deepcopy
from kernels import GaussianKernel, SpectrumKernel
from kernels import MismatchKernel
from utils import check_DNA, load_train_data, load_test_data
from preprocessing import center_gram, scale_gram


data_folder = 'data/'
# load string data
X_train_DNA, Y_train = load_train_data(data_folder, 'raw')
X_test_DNA = load_test_data(data_folder, 'raw')

# map the target vectors to do binary classification
Y_train_ = 2 * Y_train - 1

# compute and save spectrum kernel on dataset 0 for kmer_size=3,...,10
launch = False
if launch:
    X_DNA_0 = np.concatenate((X_train_DNA[0], X_test_DNA[0]))
    for ii in range(3, 11):
        K_ = SpectrumKernel(X=X_DNA_0, Y=X_DNA_0, kmer_size=ii)
        folder_dir = './mykernels/spectrum/'
        save_kernel(K_, folder_dir, dataset_idx=0, kmer_size=ii)
        
# compute and save spectrum kernel on dataset 1 for kmer_size=5,...,10
launch = False

if launch:
    X_DNA_1 = np.concatenate((X_train_DNA[1], X_test_DNA[1]))
    for ii in range(3, 11):
        K_ = SpectrumKernel(X=X_DNA_1, Y=X_DNA_1, kmer_size=ii)
        folder_dir = './mykernels/spectrum/'
        save_kernel(K_, folder_dir, dataset_idx=1, kmer_size=ii)
        
# compute and save spectrum kernel on dataset 2 for kmer_size=5,...,10
launch = False
if launch:
    X_DNA_2 = np.concatenate((X_train_DNA[2], X_test_DNA[2]))
    for ii in range(3, 11):
        K_ = SpectrumKernel(X=X_DNA_2, Y=X_DNA_2, kmer_size=ii)
        folder_dir = './mykernels/spectrum/'
        save_kernel(K_, folder_dir, dataset_idx=2, kmer_size=ii)
        
# saving mismatch kernels on dataset 0 for kmer_size=3,...,10
launch = False
if launch:
    folder_dir = './mykernels/mismatch/'
    dataset0 = pd.read_csv(data_folder + '/Xtr0.csv')
    X0 = dataset0['seq']
    test0 = pd.read_csv(data_folder + '/Xte0.csv')
    X0 = pd.concat([X0, test0['seq']], axis = 0, ignore_index = True)

    for k in range(3,11):
        K_ = gram_mismatch(X0,k)
        save_kernel(K_, folder_dir, 0 ,'mismatch',kmer_size= k )
        
# saving mismatch kernels on dataset 1 for kmer_size=3,...,10
launch = False
if launch:
    folder_dir = './mykernels/mismatch/'
    dataset0 = pd.read_csv(data_folder + '/Xtr1.csv')
    X0 = dataset0['seq']
    test0 = pd.read_csv(data_folder + '/Xte1.csv')
    X0 = pd.concat([X0, test0['seq']], axis = 0, ignore_index = True)

    for k in range(3,11):
        K_ = gram_mismatch(X0,k)
        save_kernel(K_, folder_dir, 1 ,'mismatch',kmer_size= k )

# saving mismatch kernels on dataset 2 for kmer_size=3,...,10
launch = False
if launch:
    folder_dir = './mykernels/mismatch/'
    dataset0 = pd.read_csv(data_folder + '/Xtr2.csv')
    X0 = dataset0['seq']
    test0 = pd.read_csv(data_folder + '/Xte2.csv')
    X0 = pd.concat([X0, test0['seq']], axis = 0, ignore_index = True)

    for k in range(3,11):
        K_ = gram_mismatch(X0,k)
        save_kernel(K_, folder_dir, 2 ,'mismatch',kmer_size= k )
