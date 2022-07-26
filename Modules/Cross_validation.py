# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 23:44:32 2021

@author: hohm
"""

import random

class ErrorTooFewData(Exception):
    def __init__(self, message=''):
        self.message = message

class Bag():
    def __init__(self):
        self.x = list()
        self.y = list()
        
def preparation_cross_validation (array_x, array_y, k_fold):
    '''
    This function splits the X and y data k-folds. First, the functions checks,
    whether the amount of data is sufficient for the number of k-folds.
    
    Arguments:
        array_x: list or array
        array_y list or array
        k_fold: integer >= 2
    
    return: List of k_fold Bag objects with x and y values.
    '''
    array_x = list(array_x)
    array_y = list(array_y)
    if len(array_x) != len(array_y):
        raise Exception('Len array x: "{}", but len array y: "{}"!'.format(len(array_x), len(array_y)))
    elif k_fold > len(array_x):
        raise ErrorTooFewData('Len data: {}, but k_fold: {}'.format(len(array_x), k_fold))
    
    i_choosen = []
    bags_list = list()
    
    for number_bag in range(k_fold):
        bags_list.append(Bag())
      
    while len(array_x) != len(i_choosen):
        i_reservoir = set([i for i in range(len(array_x))]) - set(i_choosen)
        i_not_choosen = list(i_reservoir)
        i_array = random.choice(i_not_choosen)
        n_bag = len(i_choosen)%k_fold
        bags_list[n_bag].x.append(array_x[i_array])
        bags_list[n_bag].y.append(array_y[i_array])
        i_choosen.append(i_array)
    return(bags_list)