# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 21:55:10 2022

@author: Hohm
"""
import sys
path = r'E:\Eigene Dateien\2022\Project_Git_hub'
if path not in sys.path:
    sys.path.append(path) 
    
import numpy as np
from tqdm import tqdm
    
from Evaluation_Metric import Metric_regression, Metric_classification

class Pegasos_classification(Metric_classification):
    def __init__(self, regularization, epoch_max, verbose=True):
        '''
        Pegasos: Primal Estimated sub-GrAdient SOlver for (Support Vector Machines) SVM [1]
        
        The mathematical formulation are justified in the orginal paper [1]. 
        In analogy to neural networks, in the maximum amount of epochs must be given. 
        Thus, in contrast to the orginal paper, on each epoch the dataset is first shuffled
        then the datapoints are selected (batchsize=1). Thus, the randomness lies
        in the order of the selection, whereby the datapoints are equaly likely
        selected.
        
        L(l, w; (x,y)) = l/2 ||theta||² + 1/m sum(max{0, 1-y y_hat)})
        with: y_hat = <theta, x> + theta_0
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
        self.name = 'Primal Pegasos Linear'
        self.regularization = np.ones(1, dtype=np.float64) * regularization
        self.learning_rate = np.ones(1, dtype=np.float64) 
        self.learning_rate  = self.learning_rate / (regularization * 1)
        self.epoch_max = int(epoch_max)
        self.verbose = verbose
        self.theta_0 = np.zeros(1, dtype=np.float64)
        self.t = np.ones(1, dtype=np.int64) 
        
        self.feature_matrix = None
        self.labels = None
        self.theta = None
        self.batch_order = None
        
        if verbose == False:
            self.fit = self.fit_silence
        
    def decision_boundary(self, feature_matrix):
        '''
        Simple inner product of the weights (theta) and the bias (theta_0).

        Parameters
        ----------
        feature_matrix : Numpy array: (datapoints x features)
        Returns
        -------
        y_hat: Numpy array: (datapoints x 1)

        '''
        y_hat = np.dot(feature_matrix, self.theta.T) + self.theta_0
        return (y_hat)
    
    def predict (self, feature_matrix):
        '''
        Final prediction element [-1, 1].

        Parameters
        ----------
        feature_matrix :  Numpy array: (datapoints x features)

        Returns
        -------
        y_hat : Numpy array: (datapoints x 1)

        '''
        y_hat = self.decision_boundary(feature_matrix)
        y_hat = np.sign(y_hat)
        return (y_hat)
    
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

    def fit (self, feature_matrix, labels):
        '''
        Initializes the weights (theta) and batch order according to the
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
        self.theta = np.zeros(feature_matrix.shape[1]) 
        self.feature_matrix = feature_matrix
        self.labels = labels
        self.batch_order = np.arange(feature_matrix.shape[0], dtype=int)
       
        print('#'*10)
        print('Regularizaion:', float(self.regularization))
        print('Max epoch:', self.epoch_max)
        print(self.name)
        for epoch in tqdm(range(1, self.epoch_max+1)):
            self.single_epoch(epoch)
        print('#'*10)
        
    def fit_silence (self, feature_matrix, labels):
        '''
        Initializes the weights (theta) and batch order according to the
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
        self.theta = np.zeros(feature_matrix.shape[1]) 
        self.feature_matrix = feature_matrix
        self.labels = labels
        self.batch_order = np.arange(feature_matrix.shape[0], dtype=int)
       
        for epoch in range(1, self.epoch_max+1):
             self.single_epoch(epoch)
            
        
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
            self.learning_rate =  1 / (self.regularization * self.t) 
            x_datapoint = self.feature_matrix[datapoint,:]
            label = self.labels[datapoint]
           
            self.decision = label * self.decision_boundary(x_datapoint)
           
            if self.decision < 1:
                self.theta = (1 - self.learning_rate * self.regularization) *  self.theta +  self.learning_rate * x_datapoint * label
                self.theta_0 = self.theta_0 + self.learning_rate * label
            else:
                self.theta = (1 - self.learning_rate * self.regularization) *  self.theta 
                
    def accuracy(self, feature_matrix, y_true):
        y_hat = self.predict(feature_matrix)
        accuracy__ = self.fun_accurracy(y_true, y_hat)
        return (accuracy__)
                
class Pegasos_regression(Pegasos_classification, Metric_regression):
    def __init__(self, regularization, epoch_max, epsilon=1E-3, verbose=True):
        '''
        L(l, w; (x,y)) = l/2 ||theta||² + 1/m sum(max{0, |y_m - y_m_hat| - epsilon})
        with: y_hat = <theta, x> + theta_0
        Parameters
        ----------
        regularization : TYPE
            DESCRIPTION.
        epoch_max : TYPE
            DESCRIPTION.
        epsilon : TYPE, optional
            DESCRIPTION. The default is 1E-3.

        Returns
        -------
        None.

        '''
        super().__init__(regularization, epoch_max, verbose)
        self.epsilon = np.ones(1, dtype=np.float64) * epsilon
        self.y_hat = np.ones(1, dtype=np.float64)
        self.label = np.ones(1, dtype=np.float64)
        
    def MSE (self, X_feature_matrix, y_true):
        y_hat = self.predict(X_feature_matrix)
        MSE__ = super().fun_MSE(y_true, y_hat)
        return (MSE__)
             
    def predict (self, feature_matrix):
        y_hat = super().decision_boundary(feature_matrix)
        return (y_hat)  
    
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
            self.label = self.labels[datapoint]
            
            self.y_hat = self.predict(x_datapoint)
            
            if self.label >= self.y_hat:
                self.decision = self.label - self.y_hat
                if self.decision > self.epsilon:
                    self.theta = (1 - self.learning_rate * self.regularization) *  self.theta + self.learning_rate * x_datapoint 
                    self.theta_0 = self.theta_0 + self.learning_rate 
                else:
                    self.theta = (1 - self.learning_rate * self.regularization) *  self.theta
                   
            else:
                self.decision = self.y_hat - self.label
                if self.decision > self.epsilon:
                    self.theta = (1 - self.learning_rate * self.regularization) *  self.theta - self.learning_rate * x_datapoint 
                    self.theta_0 = self.theta_0 - self.learning_rate 
                else:
                    self.theta = (1 - self.learning_rate * self.regularization) *  self.theta
                
