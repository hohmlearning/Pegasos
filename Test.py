# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:35:13 2022

@author: hohm
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from Primal_Pegasos import Pegasos_regression
from Cross_validation import preparation_cross_validation
golden_section_search = __import__('20220716_Golden_Section_Search').golden_section_search
#%%
#https://archive-beta.ics.uci.edu/ml/datasets/auto+mpg
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
#%%
dataset = raw_dataset.copy()
dataset = dataset.dropna()
#%%
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()
#%%
columns = dataset.columns
dataset_numpy = np.array(dataset)
X = dataset_numpy[:,1:]
y = dataset_numpy[:,0]
#%%
X = X - X.mean(axis=0)
X = X / X.var(axis=0)**0.5
#%%
np.random.seed(42)
n = dataset.shape[0]
n_random = np.random.permutation(n)
n_train = n_random[:int(dataset.shape[0]*0.8)]
n_test = n_random[int(dataset.shape[0]*0.8):]

X_train = X[n_train,:]
y_train = y[n_train]

X_test = X[n_test,:]
y_test =  y[n_test]
#%%
'''
y_train_ = np.expand_dims(y_train, axis=1)
data_set_train = np.hstack((y_train_, X_train))
data_set_train = pd.DataFrame(data_set_train, columns=columns)

columns_1 = columns[:6]
columns_2 = [columns[0]] + list(columns[6:])
sns.pairplot(data_set_train[columns_1], diag_kind='kde')
sns.pairplot(data_set_train[columns_2], diag_kind='kde')
'''
#%%
def MSE_l (l, epoch_max, epsilon, X_train, y_train, X_val, y_val):
    Pegasos = Pegasos_regression(regularization=l,
                                 epoch_max=epoch_max,
                                 epsilon=epsilon)
    Pegasos.fit(X_train, y_train)
    MSE_ = Pegasos.MSE(X_val, y_val)
    return (MSE_) 
#%%
l_opt_list = []
epoch_max = 100
n_bags = 5
np.random.seed(42)
bags_list = preparation_cross_validation(X_train, y_train, n_bags)
#%%
legend_list = ['$n_{bag}$ = ' + str(i+1) for i in range(n_bags)]
for n_val in range (n_bags):
    X_train_ = []
    y_train_ = []
    X_val_ = np.array(bags_list[n_val].x)
    y_val_ =  np.array(bags_list[n_val].y)
    for n in range (n_bags):
        if n_val != n:
            X_train_ += bags_list[n].x
            y_train_ += bags_list[n].y
    X_train_ = np.array(X_train_)
    y_train_ = np.array(y_train_)

    MSE_min = lambda l: MSE_l(l, epoch_max=epoch_max, epsilon=1E-3, X_train=X_train_, y_train=y_train_, X_val=X_val_, y_val=y_val_)
    MSE_min_vec = np.vectorize(MSE_min)
    l_array = np.linspace(0.0001, 0.01, 100)
    plt.plot(l_array, MSE_min_vec(l_array))
plt.legend(legend_list, 'upper left')
plt.plot()
    #l_opt = golden_section_search(MSE_min, 0, 10000, 1E-8)
    #l_opt_list.append(l_opt)
    #MSE_mean += MSE_min(l_opt) / n_bags 
    #print('MSE (on validation set {}/{}) = {:.2f} | alpha = {:.2g}'.format(n_val+1, n_bags, MSE_min(l_opt), l_opt))
#%%
regularization  = 0.1
epoch_max = 1000 
epsilon=1E-3
Pegasos = Pegasos_regression(regularization=0.1,
                             epoch_max=100,
                             epsilon=1E-3)
#%%
Pegasos.fit(X_train, y_train)


























