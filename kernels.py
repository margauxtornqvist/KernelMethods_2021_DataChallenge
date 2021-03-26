#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:46:57 2021

@author: margauxtornqvist, theo uscidda, mohammad fahes
"""
"""
The main functions in this library are :
    
    - GaussianKernel : Retrieves the Gram matrix for the Gaussian kernel of the input data X
    - SpectrumKernel : Retrieves the Gram matrix for the Spectrum kernel of the input data X
    - MismatchKernel : Retrieves the Gram matrix for the Mismatch kernel of the input data X
    - SpectrumKernel_TFIDF : Retrieves the Gram matrix for the Spectrum kernel of the input data X, with the option to use TF-IDF normaliaztion
    - ExponentialSpectrumKernel: Retrieves the Gram matrix for the Exponential Spectrum kernel of the input data X
"""

# imports
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from itertools import product
from scipy.sparse import csr_matrix
from preprocessing import center_gram, scale_gram



#-------------------------- Gaussian Kernel -------------------------- 

def GaussianKernel(x,y,gamma):
    """
    Get the Gram matrix for Gaussian kernel with parameter gamma
    Parameters:
      x: np.array, data 
      y: np.array, data 
      gamma: float, parameter of the kernel
    Returns:
      K: np.array, Gram matrix of the input data X
    """
    return np.exp(- gamma * cdist(x, y, 'sqeuclidean'))

#-------------------------- Spectrum Kernel -------------------------- 


def sub_string_dict(s,kmer_size):
    """
    Retrieves a dictionary which contains the occurence of each substrings of length k contained in the string S
    Parameters:
      s: string
      kmer_size: int, length of the substring to look ocurrences for
    """
    # substring dictionary
    ss_dict = {}
    for i in range(len(s)-kmer_size):
        sub_string = s[i:i+kmer_size]
        ss_dict[sub_string] = ss_dict.setdefault(sub_string, 0) + 1
    return ss_dict

def InnerProduct_Spectrum(string_i,string_j,kmer_size):
    """
    Computes the inner product between two strings i and j w.r.t the spectrum kernel
    K(x_i,x_j) = SUM_u(Phi_u(x_i),Phi_u(x_j) for u in A^k), A^k all substrings of x_i and x_j
    Parameters:
      string_i: string
      string_j: string
      kmer_size: int, length of the substring to look ocurrences for
    """
    dic_i = sub_string_dict(string_i,kmer_size)
    dic_j = sub_string_dict(string_j,kmer_size)

    # common substrings (intersection)
    inter_ss = set(dic_i.keys()) & set(dic_j.keys())
    K = 0
    # sum over all common substrings
    for ss in inter_ss:
        K += dic_i[ss]*dic_j[ss]
    return K 


def SpectrumKernel(X,Y,kmer_size):
    """
    Retrieves the spectrum kernel of the input data X
    Parameters:
      X: np.array, input data of size number of points x dimension if data
      kmer_size: int, length of the substrings to retrieve occurences for 
    """
    n = len(X)
    m = len(Y)
    K = np.zeros((n,m))

    if X.all() == Y.all():
      for i in range(n):
        K[i,i] = n-kmer_size+1
        for j in range(i):                
            K[i,j] = InnerProduct_Spectrum(X[i][0],X[j][0],kmer_size)
            K[j,i] = K[i,j]

    else:
      for i in range(n):
          for j in range(m):                
              K[i,j] = InnerProduct_Spectrum(X[i][0],Y[j][0],kmer_size)

    return K 


#-------------------------- Mismatch Kernel -------------------------- 

def all_comb(alphabet,k):
    yield from product(*([alphabet] * k)) 

def alter(string):
    l=[]
    string = list(string)
    string_init = deepcopy(string)
    for i,c in enumerate(string):
        if c=='A':
            string[i] = 'G'
            l.append(''.join(string))
            string[i] = 'C'
            l.append(''.join(string))
            string[i] = 'T'
            l.append(''.join(string))
            
        if c=='G':
            string[i] = 'A'
            l.append(''.join(string))
            string[i] = 'C'
            l.append(''.join(string))
            string[i] = 'T'
            l.append(''.join(string))
            
        if c=='C':
            string[i] = 'G'
            l.append(''.join(string))
            string[i] = 'A'
            l.append(''.join(string))
            string[i] = 'T'
            l.append(''.join(string))
            
        if c=='T':
            string[i] = 'G'
            l.append(''.join(string))
            string[i] = 'C'
            l.append(''.join(string))
            string[i] = 'A'
            l.append(''.join(string))
        string = deepcopy(string_init)
    return l

def kmers(X,k):
    d = len(X)
    idx = 0
    kmer_list = []
    for j in range(d - k + 1):
        kmer = X[j: j + k]
        kmer_list.append(kmer)
    return kmer_list

def phi_vect(X,k,dct_0,dict_negbr):
    kmer_list = kmers(X,k)
    for kmer in kmer_list:
        dct_0[kmer] += 1
        values = dict_negbr[kmer]
        for v in values:
            dct_0[v] += 1
    w = np.fromiter(dct_0.values(), dtype=int, count=len(dct_0))
    for key in dct_0.keys():
        dct_0[key] = 0
    return w

def normalize_K(K):
    
    if K[0, 0] == 1:
        print('Kernel already normalized')
    else:
        n = K.shape[0]
        diag = np.sqrt(np.diag(K))
        for i in range(n):
            d = diag[i]
            for j in range(i+1, n):
                K[i, j] /= (d * diag[j])
                K[j, i] = K[i, j]
        np.fill_diagonal(K, np.ones(n))
    return K

def initialize(k):
    all_comb_list = []
    for x in all_comb('GATC',k):
        all_comb_list.append(''.join(x))
    all_comb_possible = np.array(all_comb_list)
    dict_negbr = {}
    for element in all_comb_possible:
        dict_negbr[element] = alter(element)

    dct_0 = {}
    for key in dict_negbr.keys():
        dct_0[key]=0
    return dct_0, dict_negbr

def MismatchKernel(X,k):
    dct_0, dict_negbr = initialize(k)
    n = X.shape[0]
    K = np.zeros((n, n))
    phi_km_x = np.zeros((n,len(dct_0)))
    for i, x in tqdm(enumerate(X)):
        phi_km_x[i] = phi_vect(x,k,dct_0,dict_negbr)
    K = np.inner(phi_km_x,phi_km_x)
    return K

#-------------------------- Spectrum Kernel with TFIDF (second version) -------------------------- 


def get_phi_u(x, k, betas):
    """
    Compute feature vector of sequence x for Spectrum Kernel SP(k)
    :param x: string, DNA sequence
    :param k: int, length of k-mers
    :param betas: list, all combinations of k-mers drawn from 'A', 'C', 'G', 'T'
    :return: np.array, feature vector of x
    """
    phi_u = np.zeros(len(betas))
    for i in range(len(x) - k + 1):
        kmer = x[i:i + k]
        for i, b in enumerate(betas):
            phi_u[i] += (b == kmer)
    return phi_u


def SpectrumKernel_TFIDF(X, k, tf_idf = False):
    """
    Compute K(x, y) for each x, y in DNA sequences for Spectrum Kernel SP(k)
    :param: X: pd.DataFrame, features
    :param k: int, length of k-mers
    :return: np.array, kernel
    """
    n = X.shape[0]
    l =  X.shape[1]
    K = np.zeros((n, n))
    betas = [''.join(c) for c in product('ACGT', repeat=k)]
    phi_u = []
    for i, x in enumerate(X.loc[:, 'seq']):
        phi_u.append(get_phi_u(x, k, betas))
        
    # total number of times the word occurs in a document
    if tf_idf :
        df = np.array(phi_u)
        df_count = df.copy()
        df_count[df_count > 0] = 1
        df_count = df_count.sum(axis = 0)
        # total number of docs / total number of docs each word appears at least once
        idf = np.log(n / (df_count +1))
        tot_doc_lenght = l - k + 1
        df /= tot_doc_lenght
        # computing tf_idf matrix
        phi_u = pd.DataFrame(np.multiply(df, idf)).values
    
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        for j, y in enumerate(X.loc[:, 'seq']):
            if j >= i:
                K[i, j] = np.dot(phi_u[i], phi_u[j])
                K[j, i] = K[i, j]
    K = K
    return K

#-------------------------- Exponential Spectrum Kernel -------------------------- 


def hamming_dist(s1,s2):
    """
    Computes the hamming distance between two strings
    which is defined by the sum of a elements equal to 1 if s1[i] == s2[i], else 0.
    """
    assert len(s1) == len(s2)
    dist = 0
    for k in range(len(s1)):
        if s1[k] == s2[k]:
            dist +=1
    return dist

def InnerProduct_Exponential(string_i,string_j, kmer_size, lambd_exp):
    """
    Computes the inner product between two strings i and j w.r.t the spectrum kernel
    K(x_i,x_j) = SUM_u lambd_exp**hamming(Phi_u(x_i),Phi_u(x_j) for u in A^k), A^k all substrings of x_i and x_j
    Parameters:
      string_i: string
      string_j: string
      k: int, length of the substring to look ocurrences for
      lambd_exp: float, exponential parameter
    """
    dic_i = sub_string_dict(string_i,kmer_size)
    dic_j = sub_string_dict(string_j,kmer_size)
    K = 0
    # sum of the exponential levenstein distance between all substrings in dic_i and dic_j
    for ss_i in dic_i.keys() :
        for ss_j in dic_j.keys():
            K += lambd_exp**hamming_dist(ss_i,ss_j )*dic_i[ss_i]*dic_j[ss_j]
    return K 

def ExponentialSpectrumKernel(X, Y, kmer_size, lambd_exp):
    """
    Retrieves the spectrum kernel of the input data X
    Parameters:
      X: np.array, input data of size number of points x dimension if data
      kmer_size: int, length of the substrings to retrieve occurences for 
      lambd_exp: float, parameter for the exponential spectrum
    """
    n = len(X)
    m = len(Y)
    K = np.zeros((n,m))

    if X.all() == Y.all():
      for i in range(n):
          K[i,i] = n-kmer_size+1
          for j in range(i):                
              K[i,j] = InnerProduct_Exponential(X[i][0],X[j][0], kmer_size, lambd_exp)
              K[j,i] = K[i,j]
    else:
      for i in range(n):
          for j in range(m):                
              K[i,j] = InnerProduct_Exponential(X[i][0],Y[j][0],kmer_size , lambd_exp)


    return K 







