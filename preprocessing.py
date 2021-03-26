#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:54:31 2021

@author: margauxtornqvist
"""
import numpy as np


def center_gram(K):
    """
    Center the Gram matrix in the RKHS
    Parameters:
      K: np.array, the gram matrices associated to each kernel considered in the combination
    Returns:
      np.array, the centered gram matrix
    """
    n = K.shape[0]
    B = np.eye(n) - np.ones((n, n)) / n
    return np.linalg.multi_dot([B, K, B])


def scale_gram(K):
    """
    Scale the Gram matrix in the RKHS
    Parameters:
      K: np.array, the gram matrices associated to each kernel considered in the combination
    Returns:
      np.array, the normalized gram matrix
    """
    diag = np.sqrt(np.diag(K))[:, np.newaxis]
    return (1 / diag) * K * (1 / diag.T)
