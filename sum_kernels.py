#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:52:44 2021

@author: margauxtornqvist
"""
import numpy as np


def compute_gram(kernels, X_train, X_test, scale=True, center=True, sum_gram=False):
    """
    Build the Gram matrices considered to learn the polynomial combination of kernels.
    Parameters:
      kernels: list of list, list s.t. each element kernels[ii] corresponds to a gram matrix to build
                             (e.g if the first gram matrix we want to build is the Gaussian Gram matrix with parameter gamma=10:
                              kernels[ii][0]=='rbf', kernels[ii][1]=='gamma')
      X_train: np.array, traning set
      X_test: np.array, testing set
      scale: boolean, if True each gram matrix is scaled
      center: boolean, if True each gram matrix is centered in the RKHS
      sum_kernels: boolean, if True returns the sum of the Gram matrices else returns the array with all Gram matrices
    Returns:
      K_train, np.array, the gram matrices on the training set (used to learn the parameters)
      K_test, np.array, the gram matrices on the testing set (used to predict for new data)
    """
    l, n, m = len(kernels), len(X_train), len(X_test)
    K_train, K_test = np.zeros((l, n, n)), np.zeros((l, m, n))
    
    # concatenate training and testing set to build Gram matrices
    X_ = np.concatenate((X_train, X_test), axis=0)
    
    for ii in range(l):
        
        # RBF KERNEL
        if kernels[ii][0]=='rbf':
            
            # build Gram matrix on the whole dataset 
            K_ = GaussianKernel(X_, X_, gamma=kernels[ii][1])
            
            # scale and center Gram matrix if needed
            if center:
                K_ = center_gram(K_)
            if scale:
                K_ = scale_gram(K_)
                
            # select training Gram matrix (to learn parameters) and testing Gram matrix (to do predictions)
            K_train[ii] = K_[:n, :n]
            K_test[ii] = K_[n:, :n]
        
        # SPECTRUM KERNEL
        if kernels[ii][0]=='spectrum':
            
            # build Gram matrix on the whole dataset
            K_ = SpectrumKernel(X_, X_, kmer_size=kernels[ii][1])
            
            # scale and center Gram matrix if needed
            if center:
                K_ = center_gram(K_)
            if scale:
                K_ = scale_gram(K_)
                
            # select training Gram matrix (to learn parameters) and testing Gram matrix (to do predictions)
            K_train[ii] = K_[:n, :n]
            K_test[ii] = K_[n:, :n]
        
        # EXPONENTIAL SPECTRUM KERNEL
        if kernels[ii][0]=='exp_spectrum':
            
            # build Gram matrix on the whole dataset
            K_ = ExponentialSpectrumKernel(X_, X_, kmer_size=kernels[ii][1], 
                                                   lambd_exp=kernels[ii][2])
            
            # scale and center Gram matrix if needed
            if center:
                K_ = center_gram(K_)
            if scale:
                K_ = scale_gram(K_)
                
            # select training Gram matrix (to learn parameters) and testing Gram matrix (to do predictions)
            K_train[ii] = K_[:n, :n]
            K_test[ii] = K_[n:, :n]
            
    if sum_gram:        
        return K_train.sum(0), K_test.sum(0)
    else:
        return K_train, K_test
    
