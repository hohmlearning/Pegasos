# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:35:13 2022

@author: hohm
"""

import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Evaluation_Metric import Metric_regression
from sklearn.linear_model import LinearRegression
from Primal_Pegasos import Pegasos_regression
Pegasos_kernel = __import__('Kernel_Pegasos').Pegasos_kernel_regression
Kernel = __import__('Kernel_Pegasos').Kernel_polynomial   
from Cross_validation import preparation_cross_validation
golden_section_search = __import__('20220716_Golden_Section_Search').golden_section_search
#%%
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
#%%
'''
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
'''
#%%
X_nan = np.isnan(data, dtype=bool)
X_nan = X_nan.sum(axis=1)
X_no_nan = X_nan == 0
X = data[X_no_nan,:]
y = target[X_no_nan]

X = X - X.mean(axis=0)
X = X / X.var(axis=0)**0.5
#%%
np.random.seed(42)
n = X.shape[0]
split_fraction = 0.8
n_random = np.random.permutation(n)
n_train = n_random[:int(n*split_fraction)]
n_test = n_random[int(n*split_fraction):]

X_train = X[n_train,:]
y_train = y[n_train]

X_test = X[n_test,:]
y_test =  y[n_test]
#%%
'''
y_train_ = np.expand_dims(y_train, axis=1)
data_set_train = np.hstack((y_train_, X_train))
data_set_train = pd.DataFrame(data_set_train, columns=column_names)

columns_1 = column_names[-6:]
columns_2 = [column_names[-1]] + list(column_names[:-6])
sns.pairplot(data_set_train[columns_1], diag_kind='kde')
sns.pairplot(data_set_train[columns_2], diag_kind='kde')
'''
#%%
def MSE_l (l, epoch_max, epsilon, X_train, y_train, X_val, y_val):
    Pegasos = Pegasos_regression(regularization=l,
                                 epoch_max=epoch_max,
                                 epsilon=epsilon,
                                 verbose=False)
    Pegasos.fit(X_train, y_train)
    MSE_ = Pegasos.MSE(X_val, y_val)
    return (MSE_) 
#%%
l_opt_list = []
epoch_max_diagram = 100
n_bags = 20
np.random.seed(42)
bags_list = preparation_cross_validation(X_train, y_train, n_bags)
#%%
legend_list = ['$n\mathrm{_{bag}}$ = ' + str(i+1) for i in range(n_bags)]
l_array = np.linspace(0.00001, 25, 100)
for n_val in tqdm(range (n_bags)):
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

    MSE_min = lambda l: MSE_l(l, epoch_max=epoch_max_diagram, epsilon=1E-8, X_train=X_train_, y_train=y_train_, X_val=X_val_, y_val=y_val_)
    MSE_min_vec = np.vectorize(MSE_min)
    plt.plot(l_array, MSE_min_vec(l_array))

plt.title('Epoch$\mathrm{_{max}}$ = ' + '{:.0f}'.format(epoch_max_diagram))
plt.xlabel('$\lambda$ / -')
plt.ylabel('MEDV$\mathrm{_{val}}$')
plt.legend(legend_list, loc='upper right', fontsize='xx-small')
plt.show()
#%%
epoch_max = 100
MSE_mean = 0
l_opt_list = []
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

    MSE_min = lambda l: MSE_l(l, epoch_max=epoch_max, epsilon=1E-8, X_train=X_train_, y_train=y_train_, X_val=X_val_, y_val=y_val_)
    l_opt = golden_section_search(MSE_min, 1E-3, 10, 1E-3)
    l_opt_list.append(l_opt)
    MSE_mean += MSE_min(l_opt) / n_bags 
    print('MSE (on validation set {}/{}) = {:.2f} | lambda = {:.2g}'.format(n_val+1, n_bags, MSE_min(l_opt), l_opt))
best_l_array = np.array(l_opt_list)
print('Mean MSE Cross Validation = {:.2f}'.format(MSE_mean))
print('Best lambda = {:.3g} +- {:.3g}'.format(best_l_array.mean(), best_l_array.var(ddof=1)**0.5))

l_best_linear = best_l_array.mean()
Pegasos_linear_CV = Pegasos_regression(regularization=l_best_linear,
                             epoch_max=epoch_max,
                             epsilon=1E-8)
Pegasos_linear_CV.fit(X_train, y_train)
#%%
def MSE_kernel (l, Pegasos, Kernel, epoch_max, epsilon, X_train, y_train, X_val, y_val):
    Pegasos_ = Pegasos(kernel=Kernel,
                              regularization=l,
                              epoch_max=epoch_max,
                              epsilon=epsilon,
                              verbose=False)
    Pegasos_.fit(X_train, y_train)
    MSE_ = Pegasos_.MSE(X_val, y_val)
    return (MSE_) 
#%%
MSE_mean = 0
Kernel_linear = Kernel(c=0, p=1)
l_opt_list = []
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

    MSE_kernel_lambda = lambda l: MSE_kernel(l,
                                             Pegasos=Pegasos_kernel, 
                                             Kernel=Kernel_linear,  
                                             epoch_max=epoch_max, 
                                             epsilon=1E-8, 
                                             X_train=X_train_, 
                                             y_train=y_train_, 
                                             X_val=X_val_, 
                                             y_val=y_val_)
    
    l_opt = golden_section_search(MSE_kernel_lambda, 1E-5, 100, 1E-3)
    l_opt_list.append(l_opt)

    MSE_mean += MSE_min(l_opt) / n_bags 
    print('MSE (on validation set {}/{}) = {:.2f} | lambda = {:.2g}'.format(n_val+1, n_bags, MSE_min(l_opt), l_opt))
best_l_array = np.array(l_opt_list)
print('Mean MSE Cross Validation = {:.2f}'.format(MSE_mean))
print('Best lambda = {:.3g} +- {:.3g}'.format(best_l_array.mean(), best_l_array.var(ddof=1)**0.5))

l_best = best_l_array.mean()
Pegasos_kernel_linear_CV = Pegasos_kernel(kernel=Kernel_linear,
                                          regularization=l_best,
                                          epoch_max=epoch_max,
                                          epsilon=1E-8,
                                           )
Pegasos_kernel_linear_CV.fit(X_train, y_train)
#%%
MSE_mean = 0
c_best_list = []
l_best_list = []
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


    best_val = 1E8
    best_l = 0
    best_c = 0
    for c in [0, 1, 5, 10, 50, 100]:
        Kernel_2 = Kernel(c=c, p=2)
        for l in [1E-5, 1E-4, 1E-3, 1E-2, 0.5, 0.1, 0.5, 1, 10]:
            Pegasos_reg_kernel_2 = Pegasos_kernel(kernel=Kernel_2, 
                                                regularization=l, 
                                                epoch_max=epoch_max, 
                                                epsilon=1E-8,
                                                verbose=False)
            Pegasos_reg_kernel_2.fit(X_train_, y_train_)
            MSE_Pegasos_train_2 = Pegasos_reg_kernel_2.MSE(X_train, y_train)
            MSE_Pegasos_val_2 = Pegasos_reg_kernel_2.MSE(X_val_, y_val_)
            #print('Train:', MSE_Pegasos_train_2)
            #print('Val:', MSE_Pegasos_val_2)
            if MSE_Pegasos_val_2 < best_val:
                best_l = l
                best_val = MSE_Pegasos_val_2        
                best_c = c
    c_best_list.append(best_c)
    l_best_list.append(best_l)

    MSE_mean += best_val / n_bags 
    print('MSE (on validation set {}/{}) = {:.2f} | c = {} | lambda = {:.2g}'.format(n_val+1, n_bags, best_val, best_c, best_l))
best_l_array = np.array(l_best_list)
c_best_array = np.array(c_best_list)
print('Mean MSE Cross Validation = {:.2f}'.format(MSE_mean))
print('Best lambda = {:.3g} +- {:.3g}'.format(best_l_array.mean(), best_l_array.var(ddof=1)**0.5))

l_best = best_l_array.mean()
c_best = c_best_array.mean()
Kernel_2 = Kernel(c=c_best, p=2)
Pegasos_kernel_quadratic_CV = Pegasos_kernel(kernel=Kernel_2,
                                          regularization=l_best,
                                          epoch_max=epoch_max,
                                          epsilon=1E-8,
                                           )
Pegasos_kernel_quadratic_CV.fit(X_train, y_train)
#%%
MSE_mean = 0
c_best_list = []
l_best_list = []
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


    best_val = 1E8
    best_l = 0
    best_c = 0
    for c in [0, 1, 5, 10, 50, 100]:
        Kernel_3 = Kernel(c=c, p=3)
        for l in [1, 5, 10, 50, 100, 1000]:
            Pegasos_reg_kernel_3 = Pegasos_kernel(kernel=Kernel_3, 
                                                regularization=l, 
                                                epoch_max=epoch_max, 
                                                epsilon=1E-8,
                                                verbose=False)
            Pegasos_reg_kernel_3.fit(X_train_, y_train_)
            MSE_Pegasos_train_3 = Pegasos_reg_kernel_3.MSE(X_train, y_train)
            MSE_Pegasos_val_3 = Pegasos_reg_kernel_3.MSE(X_val_, y_val_)
            #print('Train:', MSE_Pegasos_train_2)
            #print('Val:', MSE_Pegasos_val_2)
            if MSE_Pegasos_val_3 < best_val:
                best_l = l
                best_val = MSE_Pegasos_val_3        
                best_c = c
    c_best_list.append(best_c)
    l_best_list.append(best_l)

    MSE_mean += best_val / n_bags 
    print('MSE (on validation set {}/{}) = {:.2f} | c = {} | lambda = {:.2g}'.format(n_val+1, n_bags, best_val, best_c, best_l))
best_l_array = np.array(l_best_list)
c_best_array = np.array(c_best_list)
print('Mean MSE Cross Validation = {:.2f}'.format(MSE_mean))
print('Best lambda = {:.3g} +- {:.3g}'.format(best_l_array.mean(), best_l_array.var(ddof=1)**0.5))

l_best = best_l_array.mean()
c_best = c_best_array.mean()
Kernel_3 = Kernel(c=c_best, p=3)
Pegasos_kernel_kubic_CV = Pegasos_kernel(kernel=Kernel_3,
                                          regularization=l_best,
                                          epoch_max=epoch_max,
                                          epsilon=1E-8,
                                           )
Pegasos_kernel_kubic_CV.fit(X_train, y_train)
#%%
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_test_sklearn = np.dot(X_test, lin_reg.coef_) + lin_reg.intercept_
#%%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
compare_weights = pd.DataFrame()
compare_weights['Name'] = ['Bias'] + list(column_names[:-1])
compare_weights['Linear Regression'] = np.append(lin_reg.intercept_, lin_reg.coef_)
compare_weights['Pegasos linear'] = np.append(Pegasos_linear_CV.theta_0, Pegasos_linear_CV.theta)
pd.set_option('display.precision', 1)
print(compare_weights.T)   
#%%
MSE_compare = pd.Series(dtype=float)
ARD_compare = pd.Series(dtype=float) 
MSE_compare['Sklearn'] = Metric_regression().fun_MSE(y_test, y_test_sklearn)
MSE_compare['Pegasos (linear)'] = Pegasos_linear_CV.MSE(X_test, y_test)
MSE_compare['Pegasos (Kernel - linear)'] = Pegasos_kernel_linear_CV.MSE(X_test, y_test)
MSE_compare['Pegasos (Kernel - quadratic)'] = Pegasos_kernel_quadratic_CV.MSE(X_test, y_test)
MSE_compare['Pegasos (Kernel - kubic)'] = Pegasos_kernel_kubic_CV.MSE(X_test, y_test)

ARD_compare['Sklearn'] = 0
ARD_compare['Pegasos (linear)'] = -(MSE_compare['Pegasos (linear)'] - MSE_compare['Sklearn'])  / MSE_compare['Sklearn']*100
ARD_compare['Pegasos (Kernel - linear)'] = -(MSE_compare['Pegasos (Kernel - linear)'] - MSE_compare['Sklearn'])  / MSE_compare['Sklearn']*100
ARD_compare['Pegasos (Kernel - quadratic)'] = -(MSE_compare['Pegasos (Kernel - quadratic)'] - MSE_compare['Sklearn'])  / MSE_compare['Sklearn']*100
ARD_compare['Pegasos (Kernel - kubic)'] = -(MSE_compare['Pegasos (Kernel - kubic)'] - MSE_compare['Sklearn'])  / MSE_compare['Sklearn']*100

MSE_DF = MSE_compare.to_frame(name='MSE')
ARD_DF = ARD_compare.to_frame(name='Deviation / %')
performance_DF = pd.concat([ARD_DF, MSE_DF], axis=1)
print(performance_DF)
#%%
MSE_train = []
MSE_val = []
train_samples = []
Pegasos_linear_CV = Pegasos_regression(regularization=l_best_linear,
                             epoch_max=epoch_max,
                             epsilon=1E-8,
                             verbose=False)

n = X_train.shape[0]
split_fraction = 0.8
n_split = int(n * split_fraction)
X_val_ = X_train[n_split:,:]
y_val_ = y_train[n_split:]

lin_reg_sub = LinearRegression()
lin_reg_sub.fit(X_train[:n_split,:], y_train[:n_split])
y_val = np.dot(X_val_, lin_reg.coef_) + lin_reg.intercept_
MSE_epsilon = Metric_regression().fun_MSE(y_val, y_val_)

n_start = 1
for n_sample in tqdm(range(n_start, n_split)):
    train_samples.append(n_sample)
    X_train_sub = X_train[:n_sample,:]
    y_train_sub = y_train[:n_sample]
    
    Pegasos_linear_CV.fit(X_train_sub, y_train_sub)
    MSE_train.append(Pegasos_linear_CV.MSE(X_train_sub, y_train_sub))
    MSE_val.append(Pegasos_linear_CV.MSE(X_val_, y_val_))
   
plt.plot(train_samples, [MSE_epsilon for n in range(n_start, n_split)])
plt.plot(train_samples, MSE_train)
plt.plot(train_samples, MSE_val)
plt.xlim(0, n_split)
plt.xlabel('$n_\mathrm{{train}}$ / -')
plt.ylabel('MSE')
plt.legend(['$\epsilon$','Training', 'Testing'])
plt.yscale('log')
plt.show()
    
    
    






















