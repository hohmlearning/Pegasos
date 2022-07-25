# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 22:55:14 2022

@author: Manue
"""
import numpy as np
import sys
path = r'E:\Eigene Dateien\2022\Project_Git_hub'
if path not in sys.path:
    sys.path.append(path) 
    
import numpy as np
from tqdm import tqdm
    
from Evaluation_Metric import Metric_regression, Metric_classification

class Kernel_polynomial ():
    def __init__(self, c=1, p=1):
        '''
        Parameters
        ----------
        c : float - trade off high-order and low-order terms :The default is 1.
        p : float - the degree of the polynomial kernel

        Returns
        -------
        None.

        '''
        self.c = c
        self.p = p
        self.name = 'Polynomial kernel (c={}, p={})'.format(self.c, self.p)
        
    def __call__ (self, x, y):
        '''
        K(x, y) = (<x, y> + c)^p

        Parameters
        ----------
        x : numpy array (n datapoints x features)
        y : numpy array (m datapoints x features)

        Returns
        -------
        K : Kernel_matrix (n datapoints x m datapoints)
        '''
        dot_product = np.dot(x, y.T)
        K = (dot_product + self.c)
        K = K**self.p
        return (K)
    
class Pegasos_kernel_classification (Metric_classification):
    def __init__ (self, kernel, regularization, epoch_max, verbose=True):
        '''
        Pegasos: Primal Estimated sub-GrAdient SOlver for (Support Vector Machines) SVM [1]
        
        Kernel Pegasos:
        The mathematical formulation are justified in the orginal paper [1]. 
        In analogy to neural networks, in the maximum amount of epochs must be given. 
        Thus, in contrast to the orginal paper, on each epoch the dataset is first shuffled
        then the datapoints are selected (batchsize=1). Thus, the randomness lies
        in the order of the selection, whereby the datapoints are equaly likely
        selected.
        

        Parameters
        ----------
        regularization : float - L2 regulation (<theta, theta>)
        epoch_max : float

        Returns
        -------
        None.
        [1] - Shalev-Shwartz, S., Singer, Y., Srebro, N., & Cotter, A. (2011). Pegasos: Primal estimated sub-gradient solver for svm. Mathematical programming, 127(1), 3-30.
        '''
        np.random.seed(42)
        self.kernel = kernel
        self.name = 'Kernel Pegasos: ' + kernel.name
        self.regularization = regularization
        self.learning_rate = np.ones(1, dtype=np.float64)
        self.learning_rate = self.learning_rate / (regularization * 1)
        self.t =  np.ones(1, dtype=np.int64)
        if verbose == False:
            self.fit = self.fit_silence
        
        self.epoch_max = int(epoch_max)
        self.feature_matrix = None
        self.labels = None
        self.batch_order = None
        self.alpha = None
        self.theta_0 = np.zeros(1, dtype=np.float64)
        self.decision = np.zeros(1, dtype=np.float64)
        
    def shuffle(self):
        '''
        For faster and non biased convergence, the dataset is shufled using 
        the indices of X_train, y_train.

        Returns
        -------
        None.

        '''
        n_shufled = np.random.permutation(self.batch_order)
        return(n_shufled)
    
    def decision_boundary (self, X, Z):
        '''
        w_{t+1} = 1/(learning_rate) * <alpha, kernel * labels> + theta_0

        Parameters
        ----------
        feature_matrix : Numpy array: (datapoints x features)
        Returns
        -------
        y_hat: Numpy array: (datapoints x 1)

        '''
        kernel = self.kernel(X,Z)
        decision_boundary = self.learning_rate * np.dot(self.alpha, (kernel * self.labels).T) 
        decision_boundary = decision_boundary + self.theta_0
        return (decision_boundary)
    
    def predict (self, feature_matrix):
        decision_boundary = self.decision_boundary(feature_matrix, self.feature_matrix)
        decision = np.sign(decision_boundary)
        return (decision)
    
    def accurracy(self, feature_matrix, y_true):
        y_hat = self.predict(feature_matrix)
        accuracy__ = self.fun_accurracy(y_true, y_hat)
        return (accuracy__)
    
    def single_epoch (self, epoch):
        '''
        Runs Pegasos for one epoch. First the dataset is shuffled. For each
        update the self.learning_rate is updated (1/(t*regularization)). 
        Consequently, the learning rate is strictly monotonous decreasing. 
        No need for adjusting the learning rate.
        
        The objective function with the hinge loss is minimized using the stochastic 
        sub-gradient descent.

        Parameters
        ----------
        epoch : float

        Returns
        -------
        None.

        '''
        self.batch_order = self.shuffle()
        
        for count, datapoint in enumerate(self.batch_order):
            self.t = (epoch-1) * self.batch_order.shape[0] + count+1
            self.learning_rate = 1 / (self.regularization * self.t) #
            x_datapoint = self.feature_matrix[datapoint,:]
            label = self.labels[datapoint]
            
            self.decision = label * self.decision_boundary(x_datapoint, self.feature_matrix)
            
            if self.decision < 1:
                self.alpha[datapoint] += 1
                self.theta_0 = self.theta_0 + self.learning_rate * label
            else:
                None 
            
    def fit (self, feature_matrix, labels):
        '''
        Initializes the alphas (theta) and batch order according to the
        feature_matrix and the labels. Then, runs the Pegasos for each epoch
        till the maximum epoch.

        Parameters
        ----------
        feature_matrix :  Numpy array: (datapoints x features)
        labels : Numpy array: (datapoints x 1)

        Returns
        -------
        None.

        '''
        self.feature_matrix = feature_matrix
        self.labels = labels
        self.batch_order = np.arange(feature_matrix.shape[0], dtype=int)
        self.alpha = np.zeros(feature_matrix.shape[0], dtype=int)
        
        print('#'*10)
        print('Regularizaion:', self.regularization)
        print('Max epoch:', self.epoch_max)
        print(self.name)
        for epoch in tqdm(range(1, self.epoch_max+1)):
            self.single_epoch(epoch)
        print('#'*10)
        
    def fit_silence (self, feature_matrix, labels):
        '''
        Initializes the alphas (theta) and batch order according to the
        feature_matrix and the labels. Then, runs the Pegasos for each epoch
        till the maximum epoch.

        Parameters
        ----------
        feature_matrix :  Numpy array: (datapoints x features)
        labels : Numpy array: (datapoints x 1)

        Returns
        -------
        None.

        '''
        self.feature_matrix = feature_matrix
        self.labels = labels
        self.batch_order = np.arange(feature_matrix.shape[0], dtype=int)
        self.alpha = np.zeros(feature_matrix.shape[0], dtype=int)
        
        for epoch in range(1, self.epoch_max+1):
            self.single_epoch(epoch)
    
class Pegasos_kernel_regression (Pegasos_kernel_classification, Metric_regression):
    def __init__ (self, kernel, regularization, epoch_max, epsilon, verbose=True):
        '''
        Pegasos: Primal Estimated sub-GrAdient SOlver for (Support Vector Machines) SVM [1]
        
        Kernel Pegasos:
        The mathematical formulation are justified in the orginal paper [1]. 
        In analogy to neural networks, in the maximum amount of epochs must be given. 
        Thus, in contrast to the orginal paper, on each epoch the dataset is first shuffled
        then the datapoints are selected (batchsize=1). Thus, the randomness lies
        in the order of the selection, whereby the datapoints are equaly likely
        selected.
        

        Parameters
        ----------
        regularization : float - L2 regulation (<theta, theta>)
        epoch_max : float
        

        Parameters
        ----------
        kernel : 
        regularization : 
        epoch_max : 
        epsilon : float, defines, when the loss is not fined. Therefore, can act as a regularization parameter

        Returns
        -------
        None.
        [1] - Shalev-Shwartz, S., Singer, Y., Srebro, N., & Cotter, A. (2011). Pegasos: Primal estimated sub-gradient solver for svm. Mathematical programming, 127(1), 3-30.

        '''
        super().__init__(kernel, regularization, epoch_max, verbose)
        self.epsilon = np.ones(1, dtype=np.float64) 
        self.epsilon = self.epsilon * epsilon
        self.y_hat = np.ones(1, dtype=np.float64)
        self.label = np.ones(1, dtype=np.float64)
        
    def decision_boundary (self, X, Z):
        kernel = self.kernel(X,Z)
        decision_boundary = - self.learning_rate * np.dot(self.alpha, kernel.T)
        decision_boundary = decision_boundary + self.theta_0
        return (decision_boundary)
        
    def predict (self, feature_matrix):
       decision_boundary = self.decision_boundary(feature_matrix, self.feature_matrix)
       return (decision_boundary)
   
    def MSE (self, X_feature_matrix, y_true):
        y_hat = self.predict(X_feature_matrix)
        MSE__ = super().fun_MSE(y_true, y_hat)
        return (MSE__)
    
    def single_epoch (self, epoch):
        '''
        Runs Pegasos for one epoch. First the dataset is shuffled. For each
        update the self.learning_rate is updated (1/(t*regularization)). 
        Consequently, the learning rate is strictly monotonous decreasing. 
        No need for adjusting the learning rate.
        
        The objective function with the epsilon-insensitive loss is minimized 
        using the stochastic sub-gradient descent.
        
        Parameters
        ----------
        epoch : float

        Returns
        -------
        None.

        '''
        self.batch_order = self.shuffle()
        
        for count, datapoint in enumerate(self.batch_order):
            self.t = (epoch-1) * self.batch_order.shape[0] + count+1
            self.learning_rate =  1 / (self.regularization * self.t) #1 / np.sqrt(t)
            x_datapoint = self.feature_matrix[datapoint,:]
            label = self.labels[datapoint]
            
            self.y_hat = self.predict(x_datapoint)
            self.decision = self.y_hat - label
            
            if self.decision > 0:
               if  self.decision > self.epsilon:
                   self.alpha[datapoint] += 1
                   self.theta_0 = self.theta_0 - self.learning_rate
               else:
                   None
            else:
                self.decision = label - self.y_hat
                if  self.decision > self.epsilon:
                    self.alpha[datapoint] += -1
                    self.theta_0 = self.theta_0 + self.learning_rate
                else:
                    None
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    