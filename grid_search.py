#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:49:53 2021

@author: margauxtornqvist
"""
"""
The main functions in this library are :

        - CvSearch_Spectrum
        - CvSearch_RBF
        - CvSearch_Sum
"""
import pandas as pd
import numpy as np
import sys
import time
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from itertools import combinations


def CvSearch_RBF(X_train, y_train, lambds, gammas, method='svm', cv=5, n_iter=5, verbose=True):
    """
    Perform cross validation on the following hyperparamters:
       - regularization parameter
       - degree of the polynomial combination
    Parameters:
      X_train: np.array, the training set
      y_train: np.array, the training labels
      lambds: list or np.array, regularization parameters to test
      gammas: np.array, parameter of rbf kernel to test
      cv: int, number of folds on which we perform cross validation (5 iterations by default)
    Returns:
      df: pd.DataFrame, dataframe storing the averaged perfomances over the folds for each combination of parameters
    """
    tt = time.time()
    n_iters = cv * len(lambds) * len(gammas)
    n_samples = y_train.shape[0]
    LAMBD, GAM, TRAIN, VAL = [], [], [], []
    i=0
    
    for gamma in gammas:
        
        # compute Gram matrix for current gamma value
        K_train = GaussianKernel(X=X_train, Y=X_train, gamma=gamma)
        
        for lambd in lambds:
            LAMBD.append(lambd)
            GAM.append(gamma)
            
            # split the folds
            INDS = np.array(range(n_samples))
            idx = np.random.permutation(n_samples)
            INDS = INDS[idx]
            vals = np.array_split(INDS, cv)
            perfs_train = []
            perfs_val = []
            
            j = 1
            for val in vals:
                i += 1 
                sys.stderr.write('\rIteration: %d/%d [ gamma %d -- lambd %.3f -- val %d/%d ]' %(i, n_iters, gamma, lambd, j, cv))
                sys.stderr.flush()
                train = np.setdiff1d(range(n_samples),val)
                
                # define and fit the model
                if method=='svm':
                    clf = KernelSVM(lambd=lambd)
                if method=='krr':
                    clf = KRR(lambd=lambd, threshold=0)
                clf.fit(K_train[train.reshape(-1,1), train], y_train[train])
                
                # get prediction and score on training and validation sets
                score_train = clf.score(K_train[train.reshape(-1,1), train], y_train[train])
                score_val =  clf.score(K_train[val.reshape(-1,1), train], y_train[val])
                
                # store perfomances
                perfs_train.append(score_train)
                perfs_val.append(score_val)
                j+= 1
                
            TRAIN.append(np.mean(np.array(perfs_train)))
            VAL.append(np.mean(np.array(perfs_val)))
            
    df = pd.DataFrame({'gamma':GAM, 'lambd':LAMBD, 'train':TRAIN, 'val':VAL})
    
    tt = time.time() - tt
    if verbose:
        print('Done in %.3f'%(tt), 'seconds.')
    
    return df


def CvSearch_Spectrum(X_train, y_train, lambds, kmer_sizes, kernel,method='svm', cv=5,
                      dataset_idx=None, available=True, scale=True, center=True, verbose=False):
    """
    Perform cross validation on the following hyperparamters:
       - regularization parameter
       - kmer_size of the spectrum kernel
    Parameters:
      X_train: np.array, the training set
      y_train: np.array, the training labels
      lambds: list or np.array, regularization parameters to test
      cv: int, number of folds on which we perform cross validation (5 iterations by default)
    Returns:
      df: pd.DataFrame, dataframe storing the averaged perfomances over the folds for each combination of parameters
    """
    tt = time.time()
    n_iters = cv * len(lambds) * len(kmer_sizes)
    n_samples = y_train.shape[0]
    LAMBD, KMER_S, TRAIN, VAL = [], [], [], []
    i=0
    
    for kmer_size in kmer_sizes:
        
        # compute or load Gram matrix for current kmer_size value
        if available:  # if Gram matrix pre-computed
            sys.stderr.write('\rLoading of the %s Kernel: kmer_size %d...                            ' %(kernel,kmer_size))
            folder_dir = './mykernels/' +str(kernel) + '/'
            K_ = load_kernel(dataset_idx=dataset_idx, kernel = kernel,folder_dir=folder_dir, kmer_size=kmer_size)        
            K_train = K_[:n_samples, :n_samples]
            
        else:   # if Gram matrix not precomputed 
            sys.stderr.write('\rComputation of the %s Kernel: kmer_size %d...                         ' %(kernel,kmer_size))
            K_train = SpectrumKernel(X=X_train, Y=X_train, kmer_size=kmer_size)
        
        # scale and center Gram matrix if needed
        if scale:
            K_train = scale_gram(K_train)
        if center:
            K_train = center_gram(K_train)
            
        
        for lambd in lambds:
            LAMBD.append(lambd)
            KMER_S.append(kmer_size)
            
            # split the folds
            INDS = np.array(range(n_samples))
            idx = np.random.permutation(n_samples)
            INDS = INDS[idx]
            vals = np.array_split(INDS, cv)
            perfs_train = []
            perfs_val = []
            
            j = 1
            for val in vals:
                i += 1 
                sys.stderr.write('\rIteration: %d/%d [ kmer_size %d -- lambd %.3f -- val %d/%d ]' %(i, n_iters, kmer_size, lambd, j, cv))
                #sys.stderr.flush()
                train = np.setdiff1d(range(n_samples),val)
                
                # define and fit the model
                if method=='svm':
                    clf = KernelSVM(lambd=lambd)
                if method=='krr':
                    clf = KRR(lambd=lambd, threshold=0)
                clf.fit(K_train[train.reshape(-1,1), train], y_train[train])
                
                # get prediction and score on training and validation sets
                score_train = clf.score(K_train[train.reshape(-1,1), train], y_train[train])
                score_val =  clf.score(K_train[val.reshape(-1,1), train], y_train[val])
                
                # store perfomances
                perfs_train.append(score_train)
                perfs_val.append(score_val)
                j+= 1
                
            TRAIN.append(np.mean(np.array(perfs_train)))
            VAL.append(np.mean(np.array(perfs_val)))
            
    df = pd.DataFrame({'kmer_size':KMER_S, 'lambd':LAMBD, 'train':TRAIN, 'val':VAL})
    
    tt = time.time() - tt
    if verbose:
        print('Done in %.3f'%(tt), 'seconds.')
    
    return df

def CvSearch_Sum(K_train, y_train, lambds, method='svm', cv=5, verbose=True):
    """
    Perform cross validation on the following hyperparamters:
       - regularization parameter
    Parameters:
      K_train: np.array, the gram matrices we consider in the polynomial combination
      y_train: np.array, the training labels
      lambds: list or np.array, regularization parameters to test
      method: str, 'svm' or 'krr' the methods to fit the model
      cv: int, number of folds on which we perform cross validation (5 iterations by default)
    Returns:
      df: pd.DataFrame, dataframe storing the averaged perfomances over the folds for each combination of parameters
    """
    tt = time.time()
    n_iters = cv * len(lambds)
    n_samples = y_train.shape[0]
    LAMBD, TRAIN, VAL = [], [], []
    i=0
    
    for lambd in lambds:
        LAMBD.append(lambd)
        
        # split the folds
        INDS = np.array(range(n_samples))
        idx = np.random.permutation(n_samples)
        INDS = INDS[idx]
        vals = np.array_split(INDS, cv)
        perfs_train = []
        perfs_val = []
            
        j = 1
        for val in vals:
            i += 1 
            sys.stderr.write('\rIteration: %d/%d [lambd %.3f -- val %d/%d ]' %(i, n_iters, lambd, j, cv))
            sys.stderr.flush()
            train = np.setdiff1d(range(n_samples),val)
                
            # define and fit the model
            if method=='svm':
                clf = KernelSVM(lambd=lambd)
            else: 
                clf = KRR(lambd=lambd, threshold=0)
            clf.fit(K_train[train.reshape(-1,1), train], y_train[train])
            
            # get prediction and score on training and validation sets
            score_train = clf.score(K_train[train.reshape(-1,1), train], y_train[train])
            score_val =  clf.score(K_train[val.reshape(-1,1), train], y_train[val])
            
            # store perfomances
            perfs_train.append(score_train)
            perfs_val.append(score_val)
            j+= 1
                
        TRAIN.append(np.mean(np.array(perfs_train)))
        VAL.append(np.mean(np.array(perfs_val)))
           
    df = pd.DataFrame({'lambd':LAMBD, 'train':TRAIN, 'val':VAL})
    
    tt = time.time() - tt
    if verbose:
        print('Done in %.3f'%(tt), 'seconds.')
    
    return df

def get_best_sum(df):
    """
    Get the best combination of hyperparameters from the dataframe returned by the algorithm CvSearch_Sum
    Parameters:
      df: pd.DataFrame, dataframe storing the averaged perfomances over the folds for each combination of parameters
    Returns:
      best_lambd: best regularization parameter found by cross-validation
      best_score: best validation score 
    """
    idx = np.argmax(df.val.values)
    best_score = np.max(df.val.values)
    
    best_lambd = df.lambd[idx]
    
    return best_lambd, best_score
    
    
    
def CvSearch_Poly(K_train, y_train, degrees, lambds, method='krr', cv=5, n_iter=5, verbose=True):
    """
    Perform cross validation on the following hyperparamters:
       - regularization parameter
       - degree of the polynomial combination
    Parameters:
      K_train: np.array, the gram matrices we consider in the polynomial combination
      y_train: np.array, the training labels
      method: str, 'SVM' or 'KRR' that stand for the classifier we used ('krr' by default)
      degree: list or np.array, degrees of polynomial combination to test
      lambds: list or np.array, regularization parameters to test
      cv: int, number of folds on which we perform cross validation (5 iterations by default)
      n_iter: int, maximum number of iterations to learn the polynomial combination (5-fold CV by default)
    Returns:
      df: pd.DataFrame, dataframe storing the averaged perfomances over the folds for each combination of parameters
    """
    
    tt = time.time()
    n_iters = cv * len(degrees) * len(lambds) 
    n_samples = y_train.shape[0]
    DEG, LAMBD, TRAIN, VAL = [], [], [], []
    i=0

    for degree in degrees:
        for lambd in lambds:
            DEG.append(degree)
            LAMBD.append(lambd)

            # split the folds
            INDS = np.array(range(n_samples))
            idx = np.random.permutation(n_samples)
            INDS = INDS[idx]
            vals = np.array_split(INDS, cv)
            perfs_train = []
            perfs_val = []

            j = 1
            for val in vals:
                i += 1
                sys.stderr.write('\rIteration: %d/%d [degree %d -- lambd %.3f -- val %d/%d]' \
                                 %(i, n_iters, degree, lambd, j, cv))
                #sys.stderr.write(r'Iteration: {i}/{n_iters} [ degree {degree} -- lambd {lambd} -- combin_kmer_sizes {combin} -- val {j}/{cv} ]\n')
                #sys.stderr.flush()
                train = np.setdiff1d(range(n_samples),val)

                # define and fit the model
                clf = PolyKerOpt(lambd=lambd, tol=1e-07, degree=degree, method=method)
                clf.fit(K_train[:, train.reshape(-1,1), train], y_train[train], n_iter=n_iter)

                # get prediction and score on training and validation sets
                score_train = clf.score(K_train[:, train.reshape(-1,1), train], y_train[train])
                score_val =  clf.score(K_train[:, val.reshape(-1,1), train], y_train[val])

                # store perfomances
                perfs_train.append(score_train)
                perfs_val.append(score_val)
                j+= 1

            TRAIN.append(np.mean(np.array(perfs_train)))
            VAL.append(np.mean(np.array(perfs_val)))

    df = pd.DataFrame({'degree':DEG, 'lambd':LAMBD, 'train':TRAIN, 'val':VAL})

    tt = time.time() - tt
    if verbose:
        print('Done in %.3f'%(tt), 'seconds.')

    return df
    
def get_best_poly(df, combin=False):
    """
    Get the best combination of hyperparameters from the dataframe returned by the algorithm CvSearchPolyKernel
    Parameters:
      df: pd.DataFrame, dataframe storing the averaged perfomances over the folds for each combination of parameters
    Returns:
      best_degree: best degree found by cross-validation
      best_lambd: best regularization parameter found by cross-validation
      best_score: best validation score 
    """
    idx = np.argmax(df.val.values)
    best_score = np.max(df.val.values)
    best_degree = df.degree[idx]
    best_lambd = df.lambd[idx]
    if combin:
        best_combin = df.combin_kmer_sizes[idx]
        return best_degree, best_lambd, best_combin, best_score
    else:
        return best_degree, best_lambd, best_score