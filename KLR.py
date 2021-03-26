#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:10:08 2021

@author: margauxtornqvist
"""
import numpy as np 

class KLR:
    """
    Kernel Logistic Regression (KLR)

    Parameters:
    K: np.array, Gram matrix which sizes size n_samples x n_samples 
    kernel: string, 'rbf', 'sepctrum', 'mismatch'...
    gamma: float, parameter of the rbf kernel
    lambd : float, regularization parameter 
    """

    def __init__(self, kernel, gamma, lambd):
        self.lambd = lambd
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = None
        self.K_train = None
        self.K_test = None

    def fit_gram_matrix(self, X_train, X_test=None):  
        """
        Get the Gram matrix 
        Parameters:
          X: np.array, training data 
          gamma: float, parameter of the kernel
        Returns:
          K: np.array, Gram matrix of the input data X
        """
        
        # fit Gram matrix on the training set if needed
        if np.all(X_test==None):
            if self.kernel=='rbf':
                self.K_train = GaussianKernel(X_train, X_train, self.gamma)
            #if self.kernel == 'spectrum':
              #
        
        # fit Gram matrix for prediction on unseen data 
        # Warning !! self.K_test is not a square matrix, but a matrix of size n_samples_train x n_samples_test
        else:
            if self.kernel == 'rbf':
                self.K_test = GaussianKernel(X_test, X_train, self.gamma)
            #if self.kernel == 'spectrum':
              #
  
    def sigmoid(self, z):
        """
        Returns the sigmoid function
        Parameters:
          z: np.array
        Returns:
          self.sigmoid: sigmoid function evaluated at point z
          """
        return  1 / (1 + np.exp(-z))

    def loss(self, z):
        """
        Returns the logistic loss function (loss of KLR)
        Parameters:
          z: np.array
        Returns:
          self.loss: logistic loss evaluated at point z
        """
        return np.log(1 + np.exp(-z))

    def d1_loss(self, z):
        """
        Returns the first derivative of the loss function 
        Parameters:
          z: np.array
        Returns:
          self.d1_loss: first derivative of logistic loss evaluated at point z
        """
        return - self.sigmoid(-z)

    def d2_loss(self, z):
        """
        Returns the second derivative of the loss fucntion
        Parameters:
          z: np.array
        Returns:
          self.d2_loss: second derivative of logistic loss evaluated at point z
        """
        return  self.sigmoid(z) * self.sigmoid(-z)

    def objective(self, X, y, alpha):
        """
        Returns the regularized sigmoid loss
        Parameters:
          X: np.array matrix n_samples x dim, training examples
          y: np.array vector n_samples, training labels
          w: np.array vector dim, point in which we evaluate the objective
        Returns:
          self.objective: regularized logistic loss evaluated at point w
        """
        n = len(y)
        return (1/n) * np.sum(self.loss(y * self.K_train @ alpha)) + self.lambd * alpha.T @ self.K_train @ alpha

    def gradient(self, X, y, alpha):
        """
        Returns the gradient of the regularized logistic loss
        Parameters:
          X: np.array n_samples x dim, training examples
          y: np.array n_samples x 1, training labels
          w: np.array vector dim, point in which we evaluate the gradient of the objective
        Returns:
          self.objective: gradient regularized logistic loss evaluated at point w
        """
        n = len(y)
        P = np.diag(self.d1_loss(y * self.K_train @ alpha))
        return (1/n) * self.K_train @ P @ y + self.lambd * self.K_train @ alpha
    
    def hessian(self, X, y, alpha):
        """
        Returns the hessian of the regularized logistic loss
        Parameters:
          X: np.array n_samples x dim, training examples
          y: np.array n_samples x 1, training labels
          w: np.array vector dim, point in which we evaluate the hessian of the objective
        Returns:
          self.objective: gradient regularized logistic loss evaluated at point w
        """
        n = len(y)
        W = np.diag(self.d2_loss(y * self.K_train @ alpha))
        return (1/n) * self.K_train @ W @ self.K_train + self.lambd * self.K_train 


    def fit(self, X, y, optim='hand', eps=1e-3, lr=5*1e-2, max_iter=50, verbose=False):
        """
        Fit the model on the data with Newton Raphson algorithm (with fixed step size)
        Parameters:
          X: np.array n_samples x dim, training examples
          y: np.array n_samples x 1, training labels
          eps: float, stopping criterion
          lr: float, learning rate
          max_iter: int, maximum number of iterations
          verbose: Boolean, wheteher to print of not information during the training
        Returns:
          self.alpha: solution to the minimization problem
        """
        
        # fit the Gram matrix if needed
        if np.all(self.K_train==None):
            self.fit_gram_matrix(X)
            
        ### Optimization with our own Newton's method (with fixed step size)
        if optim == 'hand': 
            # simplify notations
            obj = lambda w: self.objective(X, y, w)
            grad = lambda w: self.gradient(X, y, w)
            hess = lambda w: self.hessian(X, y, w)

            # intialization 
            num_iter = 0
            w = np.zeros(len(X))
            trackL = [obj(w)]

            while num_iter <= max_iter:

                # newton step
                step = np.linalg.pinv(hess(w)) @ grad(w)

                # break before the update if infinite likelihood (in case of separable data)
                if np.isnan(obj(w - lr * step)):
                    break

                # update
                w = w - lr * step
                trackL.append(obj(w))
                num_iter = num_iter + 1

                # check optimization issue
                if obj(w) > trackL[-2]:
                    print("The optimization does't work because the objective is increasing throughout the iterations.")
                    print("Decrease the learning rate or modify the initialization.")
                    break 

                # check the stopping criterion
                if abs(obj(w)- trackL[-2]) <= eps:
                    break 

                if verbose:
                    print(f'###### Iteration {num_iter}: objective = {obj(w)} ######')

            # get the optimal point 
            self.alpha = w
            return self.alpha
        
        ### DOESN'T WORK for the moment !!! See later...
        ### Optimization with Newton-CG algorithm of the library scipy.optimize
        else: 
            
            # simplify notations
            obj = lambda w: self.objective(X, y, w)
            grad = lambda w: self.gradient(X, y, w)
            hess = lambda w: self.hessian(X, y, w)
            
            # use scipy.optimize library to perform optimization 
            res = minimize(obj, method='Newton-CG',
                           jac=grad, hess=hess,
                           x0=np.zeros(len(X)), 
                           options={'xtol': 1e-3, 'disp': True})
            
            # get the optimal point 
            self.alpha = res.x
            return self.alpha
    
    def predict_train(self):
        """
        Get the prediction on the training set
        Returns:
          y_pred: int in {-1,1}, prediction on training set
        """
        assert self.alpha is not None, "Fit model on data"
        return np.sign(self.K_train @ self.alpha) 
    
    def predict_test(self, X_train, X_test):
        """
        Get the prediction on the testing set (new unseen data)
        Parameters:
          X_train: X: np.array n_samples x dim, training examples
          X_test: X: np.array n_samples x dim, training examples
        Returns:
          y_pred: int in {-1,1}, prediction on testing set
        """
        assert self.alpha is not None, "Fit model on data"
        self.fit_gram_matrix(X_train=X_train, X_test=X_test)
        return np.sign(self.K_test @ self.alpha) 

    def score_train(self, y_train):
        """
        Returns the accuracy 
        Parameters:
          y_train: np.array n_samples x 1, training labels
        Returns:
          acc: float in [0, 100], accuracy in %
        """
        assert self.alpha is not None, "Fit model on data"
        return 100 * (y_train == self.predict_train()).mean()
    
    def score_test(self, y_test, X_train, X_test):
        """
        Returns the accuracy 
        Parameters:
          y_test: np.array n_samples x 1, training labels
          X_train: X: np.array n_samples x dim, training examples
          X_test: X: np.array n_samples x dim, training examples
        Returns:
          acc: float in [0, 100], accuracy in %
        """
        assert self.alpha is not None, "Fit model on data"
        return 100 * (y_test == self.predict_test(X_train=X_train, X_test=X_test)).mean()