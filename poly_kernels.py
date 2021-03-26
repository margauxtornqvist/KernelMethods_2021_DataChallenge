#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:07:03 2021

@author: margauxtornqvist
"""
import numpy as np
from numba import jit

#-------------------------- plynomial Kernel -------------------------- 


class PolyKerOpt():
    
    def __init__(self, lambd=.01, tol=1e-06, degree=3, method='svm', verbose=False):
        self.lambd = lambd
        self.tol = tol
        self.degree = degree
        self.method = method
        self.verbose  = verbose         
                
    def scale(self, coef, norm):
        """
        Used while learning the coefficient to combine the kernels:
        Re-scale the coefficient according to the choosen norm.
        Parameters:
          coef: np.array, coefficient to learn polynomial combination of kernels 
          norm: str, 'l1' or 'l2' to denote norm with which we learn the coefficient to combine the kernels
        Returns:
           np.array, the scaled coefficients
        """
        if norm=='l1':
            return coef / np.linalg.norm(coef, ord=1)
        elif norm=='l2':
            return coef / np.linalg.norm(coef, ord=2)
        else:
            raise Exception('The only available norms are L1 and L2 norms.')
            
    def bound(self, coeff, u_0, gamma, norm):
        """
        Used while learning the coefficient to combine the kernels:
        Re-scale and re-center the coefficient according to the choosen norm, the choosen gamma and the choosen mu_0 (c.f notations of the article).
        Parameters:
          coef: np.array, coefficient to learn polynomial combination of kernels  
          u_0: np.array, reference parameter to center the coeffficent
          gamma: float, reference parameter to scale the coeffficent
          norm: str, 'l1' or 'l2' to denote norm with which we learn the coefficient
        Returns:
          np.array, the scaled coefficients
        """
        # scale and center coef
        u__ = coeff - u_0
        u__ = np.abs(self.scale(u__, norm) * gamma)
        
        return u__ + u_0
    
    def KrrIterate(self, Kernels, y_train, coef):
        """
        Perform a KRR iteration.
        Parameters:
          y_train: np.array, the labels 
          coef: np.array, coefficient to learn polynomial combination of kernels 
        Returns:
          c: np.array, solution of the empirical risk minimization 
        """
        # get the polynomial combination of kernels
        K_w = np.sum((Kernels * coef[:, None, None]), axis=0) ** self.degree
        
        # solve the KRR empirical risk minimization
        N, D = K_w.shape
        c = np.linalg.solve(np.linalg.inv(K_w + self.lambd * np.eye(N, D)), y_train[:, np.newaxis])
        
        return c
    
    def SvmIterate(self, Kernels, y_train, coef):
        """
        Perform a SVM iteration.
        Parameters:
          y_train: np.array, the labels 
          coef: np.array, coefficient to learn polynomial combination of kernels 
        Returns:
          c: np.array, solution of the minimization problem
        """
        nb_samples = y_train.shape[0]
        C = 1 / ( 2 * self.lambd * nb_samples)
        r = np.arange(nb_samples)
        o = np.ones(nb_samples)
        z = np.zeros(nb_samples)
            
        # get the polynomial combination of kernels
        K_w  = np.sum(Kernels * coef[:, None, None], axis=0) ** (self.degree)
        
        # cvxopt fomalization        
        P = matrix(K_w.astype(float), tc='d')
        q = matrix(-y_train, tc='d')
        G = spmatrix(np.r_[y_train, -y_train], np.r_[r, r + nb_samples], np.r_[r, r], tc='d')
        h = matrix(np.r_[o * C, z], tc='d')
        
        solvers.options['show_progress'] = False
        
        # solve the SVM empirical risk minimization
        sol = solvers.qp(P, q, G, h)
        c = np.ravel(sol['x'])[:, np.newaxis]
        
        return c
    
    def gradUpdate(self, Kernels, coef, delta):
        """
        Perform a projection-based gradient descent update (c.f notations of the article).
        Parameters:
          y_train: np.array, the labels 
          coef: np.array, coefficient to learn polynomial combination of kernels 
          delta: np.array, solution of the empirical risk minimization for the last classifier iteration
        Returns:
          c: np.array, solution of the minimization problem
        """
        # get the polynomial combination of kernels
        K_t = np.sum(Kernels * coef[:, None, None], axis=0) ** (self.degree-1)
        
        # compute the gradient w.r.t each coef of the combination
        grad = np.zeros(len(Kernels))
        for m in range(len(Kernels)):
            grad[m] = delta.T.dot((K_t * Kernels[m])).dot(delta)
            
        return - self.degree * grad
    
    def fit(self, Kernels, y_train, u_0=0, gamma=1, norm='l2', n_iter=5, lr=1, verbose=False):
        """
        Fit the model and learn the polynomial combination of Kernels (c.f notations of the article).
        Rmk: thanks to the negativity of the gradient, the thresholding operator to keep non negative coef is not required (c.f article for computations).
        Parameters:
          y_train: np.array, the labels 
          coef: np.array, coefficient to learn polynomial combination of kernels 
          u_0: np.array, reference parameter to center the coeffficent
          gamma: float, reference parameter to scale the coeffficent
          norm: str, 'l1' or 'l2' to denote norm with which we learn the coefficient
          n_iter: int, maximal number of projection-based gradient descent iterations
          lr: float, learning rate
        """
        tt = time.time()
            
        # randomly initialize scaled and centered coef  
        coef = np.random.normal(0, 1, len(Kernels)) / len(Kernels)
        coef = self.bound(coef, u_0, gamma, norm)
        new_coef = 0
        score_prev = np.inf
        
        for ii in range(n_iter):
            
            if verbose:
                sys.stderr.write('\rLearning of the coefficient: iteration %d/%d' %(ii, n_iter))
                sys.stderr.flush()
        
            ### STEP 1: ###
            # solve the empirical risk minimzation
            if self.method=='svm':
                delta = self.SvmIterate(Kernels, y_train, coef)
            else:
                delta = self.KrrIterate(Kernels, y_train, coef)
            
            ### STEP 2: ###
            # projected-based gradient update on the coef
            grad = self.gradUpdate(Kernels, coef, delta)
            new_coef = coef - lr * grad
            
            ### STEP 3: ###
            # scale and center the new coef
            new_coef = self.bound(new_coef, u_0, gamma, norm)
            
            # check stopping criterion
            score = np.linalg.norm(new_coef - coef, np.inf)
            if score>score_prev:
                lr *= 0.9  # learning rate decay if new step required
            if score<self.tol:
                self.coef = coef
                self.delta = delta
            
            coef = new_coef
            score_prev = score.copy()
            
        if verbose:
            sys.stderr.write('\rLearning of the coefficient: Done!             ') # keep spaces
            sys.stderr.flush()
        
        self.coef, self.delta = coef, delta
        
    def predict(self, Kernels):
        """
        Get the prediction on the training set.
        Returns:
          y_pred: np.array, the predicted labels on the training set
        """
        # get the polynomial combination of kernels
        K_w = np.sum(Kernels * self.coef[:, None, None], axis=0) ** (self.degree)
        
        # predict the labels
        y_pred = np.sign(K_w.dot(self.delta)).flatten()
            
        return y_pred
    
    def score(self, Kernels, y):
        """
        Predict the labels with learned polynomial combination of kernels.
        Parameters:
          y_train: np.array, the training labels (training score) or the testing labels (testing score)
        Returns:
          float, accuracy of the prediction in [0, 100]
        """ 
        return 100 * (self.predict(Kernels)==y).mean()