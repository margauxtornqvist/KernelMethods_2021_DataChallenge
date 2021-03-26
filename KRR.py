#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:48:41 2021

@author: margauxtornqvist
"""
import numpy as np


class KRR:
    """
    Kernel Ridge Regression (KRR)


    Parameters:
      lambd: float, regularization parameter 
      threshold: float, threshold to discretize prediction
    """

    def __init__(self, lambd, threshold):
        self.lambd = lambd
        self.threshold = threshold
        self.alpha = None         
  
    def fit(self, K_train, y_train):
        """
        Fit the model on the data with Newton Raphson algorithm (with fixed step size)
        Parameters:
          K_train: np.array, Gram matrix on the training set
          y_train: np.array, training labels
        """
        # fit the model
        n = len(y_train)
        self.alpha = np.linalg.solve(K_train + self.lambd * n * np.eye(n), y_train)
                
    def predict(self, K):
        """
        Get the prediction on the training set
        Parameters:
          K: np.array, Gram matrix to do predictions, can be:
                             - K_train --> predictions on the training set
                             - K_test --> predictions on the testing set (not square matrix)
        Returns:
          np.array, predictions (in {-1,1})
        """
        assert self.alpha is not None, "Fit model on data"
        return 2 * (K @ self.alpha >= self.threshold) - 1

    def score(self, K, y):
        """
        Returns the accuracy 
        Parameters:
          K: np.array, Gram matrix to do predictions, can be:
                             - K_train --> predictions on the training set
                             - K_test --> predictions on the testing set
          y: np.array, training or testing labels
        Returns:
          acc: float in [0, 100], accuracy in %
        """
        assert self.alpha is not None, "Fit model on data"
        return 100 * (y == self.predict(K)).mean()