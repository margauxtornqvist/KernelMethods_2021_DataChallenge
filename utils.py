#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:46:55 2021

@author: margauxtornqvist, theo uscidda, mohammad fahes

"""

import numpy as np 
import pandas as pd
import os
from preprocessing import center_gram, scale_gram

def check_DNA(x):
    """
    Checks that the sequence is of length 100 and that all the nucleotides belongs to characters 'A','C','G','T'
    """
    assert len(x)==101,'Protein sequence not of length 10'
    assert set(x).issubset(set('ACGT'))
    return x 


def load_train_data(data_folder, mode):
    """
    Loads training data
    Parameters:
        - data_folder : (str), path to data folder
        - mode: (str), 'raw' or 'proprocessed'
    Returns:
        - X,Y: (np.array,np.array), array with 3 datasets, targets stacked together
    """
    assert mode in ['raw','preprocessed']
    X,Y = [],[]
    for k in range(3):
        if mode == 'preprocessed':
            Xk = pd.read_csv(data_folder + 'Xtr'+str(k)+'_mat100.csv', sep=' ', header = None).values
        if mode == 'raw':
            Xk = pd.read_csv(data_folder + 'Xtr'+str(k)+'.csv', index_col = 0)\
                .dropna()\
                .applymap(check_DNA)\
                .values
            
        Yk = pd.read_csv(data_folder + 'Ytr'+str(k)+'.csv', index_col = 0)\
            .values\
            .squeeze()
        
        X.append(Xk)
        Y.append(Yk)
        
    return np.array(X), np.array(Y)

def load_test_data(data_folder, mode):
    """
    Loads test data
    Parameters:
        - data_folder : (str), path to data folder
        - mode: (str), 'raw' or 'proprocessed'
    Returns:
        - X: (np.array), array with 3 datasets stacked together
    """
    assert mode in ['raw','preprocessed']
    X = []
    for k in range(3):
        if mode == 'preprocessed':
            Xk = pd.read_csv(data_folder + 'Xte'+str(k)+'_mat100.csv',  sep=' ', header = None).values
        if mode == 'raw':
            Xk = pd.read_csv(data_folder + 'Xte'+str(k)+'.csv', index_col =0)\
                .dropna()\
                .applymap(check_DNA)\
                .values
        X.append(Xk)
    return np.array(X)

def kaggle_submit(y_pred, experiment_path, num_submit):
    """
    Retrieves a dataframe which containes the predictions on the test set to submit to the kaggle competition
    Parameters: 
      - y_pred: np.array, predictions on each dataset with labels in {-1, 1}
      - experiment_path: str, name of the experiment path in which to store the output
      - num_submit: int, number of the submission
    Returns:
      - df: pd.DataFrame, sumission file
    """
    if not os.path.isdir(experiment_path) :
          os.mkdir(experiment_path)

    # convert labels from {-1, 1} to {0, 1}
    y_pred = (y_pred + 1)/2
    
    # prepare submission dataframe
    y_pred = y_pred.astype('int32')
    df = pd.DataFrame(data = y_pred, columns = ['Bound'])
    df.index.name = 'Id'
    
    # save submission dataframe
    df.to_csv(experiment_path + '/Yte' + str(num_submit) + '.csv')  
    print('Model saved with success to: ', experiment_path,'/Yte' + str(num_submit) + '.csv')
    
    return df


def predict_kaggle(models, experiment_path, X_train, X_test):
    """
    Retrieves a dataframe which containes the predictions on the test set to submit to the kaggle competition
    Parameters: 
        - models: (list), list of models
        - experiment_path: (str), name of the experiment path in which to store the output
        - X_train: (np.array)
        - X_test: (np.array)
    """
    y_pred = []

    if not os.path.isdir(experiment_path) :
          os.mkdir(experiment_path)

    for k in range(3):
        # predict on test set
        model_k = models[k]
        y_predk = model_k.predict_test(X_train[k],X_test[k])
        y_pred.append(y_predk)

    data = np.concatenate(y_pred)
    # convert labels from {-1,1} to {0,1}
    data = (data + 1)/2
    data = data.astype('int32')
    df = pd.DataFrame(data = data, columns = ['Bound'])
    df.index.name = 'Id'
    df.to_csv(experiment_path + '/Yte.csv')  
    print('Model saved with success to: ', experiment_path,'/Yte.csv')
    
    return df

def save_kernel(K, folder_dir, dataset_idx, kernel ,kmer_size=None, lambd_exp=None):
    """
    Saves spectrum kernel under .txt format
    Parameters:
        - K: np.array, kernel under array format 
        - folder_dir: str, folder
        - dataset_idx: str, the dataset on which the kernel is computed
        - kmer_size: int, kmer_size if spectrum kernel or exp_spectrum kernel
        - lambd_exp: float, lambd_exp if exp_spectrum kernel 
    """
    if not os.path.isdir(folder_dir) :
        os.mkdir(folder_dir)
    if kmer_size!=None:
        if lambd_exp==None:
            np.savetxt(folder_dir + '%s_kmer%d_set%d.txt'%(kernel, kmer_size, dataset_idx), K)
        if lambd_exp!=None:
            np.savetxt(folder_dir + 'exp_spectrum_kmer%d_lambd-exp%.3f_set%d.txt'%(kmer_size, lambd_exp, dataset_idx), K)
    
def load_kernel(folder_dir, dataset_idx, kernel, kmer_size=None, lambd_exp=None):
    """
    Gets spectrum kernel if exists
    Parameters:
        - folder_dir: str, folder
        - dataset_idx: str, the dataset on which the kernel is computed
        - kmer_size: int, dataset idx
        - lambd_exp: float, lambd_exp if exp_spectrum kernel 
    Returns:
        - K :np.array, loaded kernel 
    """
    if not os.path.isdir(folder_dir) :
        os.mkdir(folder_dir)
    if kmer_size!=None:
        if lambd_exp==None:
            K_ = np.loadtxt(folder_dir + '%s_kmer%d_set%d.txt'%(kernel,kmer_size, dataset_idx))
        if lambd_exp!=None:
            K_ = np.loadtxt(folder_dir + 'exp_spectrum_kmer%d_lambd-exp%.3f_set%d.txt'%(kmer_size, lambd_exp, dataset_idx))
    return K_


def get_gram(kernels, X_train, X_test, dataset_idx, scale=True, center=True, sum_gram=False):
    """
    Build the Gram matrices considered to learn the polynomial combination of kernels.
    Rmk: Since the kernels matrices built on strings are already computed, you can put bag of word training set and 
         testing to compute RBF kernels on vectorial data and then combine string kernel and vector kernels
    Parameters:
      kernels: list of list, list s.t. each element kernels[ii] corresponds to a gram matrix to build
                             (e.g if the first gram matrix we want to build is the Gaussian Gram matrix with parameter gamma=10:
                              kernels[ii][0]=='rbf', kernels[ii][1]=='gamma')
      X_train: np.array, traning set with vectorial data
      X_test: np.array, testing set with vectorial data
      scale: boolean, if True each gram matrix is scaled
      center: boolean, if True each gram matrix is centered in the RKHS
      sum_kernels: boolean, if True returns the sum of the Gram matrices else returns the array with all Gram matrices
    Returns:
      K_train, np.array, the gram matrices on the training set (used to learn the parameters)
      K_test, np.array, the gram matrices on the testing set (used to predict for new data)
    """
    l, n, m = len(kernels), len(X_train), len(X_test)
    K_train, K_test = np.zeros((l, n, n)), np.zeros((l, m, n))
    
    # concatenate training set and testing set 
    X_ = np.concatenate((X_train, X_test), axis=0)
    
    for ii in range(l):
        
        # RBF KERNEL
        if kernels[ii][0]=='rbf':
            
            # build Gram matrix on the whole dataset 
            K_ = GaussianKernel(X_, X_, gamma=kernels[ii][1])
            
        
        # SPECTRUM KERNEL
        if kernels[ii][0]=='spectrum':
            
            # build Gram matrix on the whole dataset
            folder_dir = './mykernels/spectrum/'
            K_ = load_kernel(dataset_idx=dataset_idx, folder_dir=folder_dir,kernel ='spectrum' ,kmer_size=kernels[ii][1])
        
        if kernels[ii][0]=='mismatch':
            
            # build Gram matrix on the whole dataset
            folder_dir = './mykernels/mismatch/'
            K_ = load_kernel(dataset_idx=dataset_idx, folder_dir=folder_dir,kernel ='mismatch' ,kmer_size=kernels[ii][1])
            
        
        # EXPONENTIAL SPECTRUM KERNEL
        if kernels[ii][0]=='exp_spectrum':
            
            # build Gram matrix on the whole dataset
            folder_dir = './mykernels/exp_spectrum/'
            K_ = load_kernel(dataset_idx=dataset_idx, folder_dir=folder_dir, kernel ='exp_spectrum', kmer_size=kernels[ii][1], 
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