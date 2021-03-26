#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:20:52 2021

@author: margauxtornqvist
"""
import numpy as np 
from cvxopt import matrix, spmatrix, solvers
from scipy.optimize import minimize


class KernelSVM:
    """
    Kernel SVM algorithm

    Implementation of Kernel SVM algorthm by solving the dual of the problem which is a quadratic program.
    The dimensions of the problem and the number of constraints are 2n where n is the number of points

    Parameters:
      lambd : float, regularization parameter 
      alpha: np.array, solution of the QP 
      obj: float, primal objective of the SVM problem
    """
    def __init__(self,lambd):
        self.lambd = lambd
        self.alpha = None   
        self.obj = None      
                
    def fit(self, K_train, y_train):
        """
        Get the coordinates of the prediction in the RKHS for SVM by solving the SVM dual
        Parameters:
          K_train: np.array, Gram matrix on the training set
          y_train: np.array, training labels
        """
        # cvxopt formalization 
        n = len(y_train)
        P = matrix(K_train) 
        q = matrix(-y_train, tc='d')
        G = matrix(np.concatenate((np.diag(y_train), -np.diag(y_train))), tc='d')
        h = matrix(np.concatenate( (((np.ones(n) * (1 / (2 * self.lambd * n))) , np.zeros(n))) ))
        
        solvers.options['show_progress'] = False

        # solve the problem
        sol = solvers.qp(P, q, G, h)
        self.alpha, self.obj = np.array(sol['x']).squeeze(), sol['primal objective']

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
        return np.sign(K @ self.alpha) 

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