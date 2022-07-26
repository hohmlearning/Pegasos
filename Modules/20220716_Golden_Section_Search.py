"""
Created on Sat Jul 16 12:32:15 2022
@author: Manue
"""
import pdb
import numpy as np

def fun_quadratic (x):
    return (x**4-x**3)
    
def golden_section_search_slow_congergence (function, x_min, x_max, eps):
    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
    lambda_golden = 2 / (1 + 5**0.5) 
    
    delta = x_max - x_min
    c = x_min + (1 - lambda_golden) * delta 
    d = x_min + lambda_golden * delta 
    
    while delta > eps:
        y_c = function(c)
        y_d = function(d)
        
        if y_c < y_d:
            x_max = d 
        else:
            x_min = c
            
        delta = x_max - x_min
        c = x_min + (1 - lambda_golden) * delta 
        d = x_min + lambda_golden * delta 
    return ((c+d)/2)
            
def golden_section_search (function, x_min, x_max, eps):
    '''
    Finds the minimum of a function in the intervall [x_min, x_max]. The function
    must be stricty unimodal and one dimensionally.
    Parameters
    ----------
    function : function one dimension : function
    x_min : lower bound : float
    x_max : upper bound : float
    eps : termination precision : float
    Returns
    -------
    (x_max + x_min)/2 : root : float
    '''
    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
    delta = x_max - x_min
    lambda_golden = 2 / (1 + 5**0.5) 
    n_convergence = int(np.log(eps/delta) / np.log(lambda_golden)) + 1
    
    c = x_min + (1 - lambda_golden) * delta 
    d = x_min + lambda_golden * delta 
    y_c = function(c)
    y_d = function(d)
    
    for n in range(n_convergence):
        delta = delta * lambda_golden
        if y_c < y_d:
            x_max = d 
            d = c
            c = x_min + (1 - lambda_golden) * delta  
            y_d = y_c 
            y_c = function(c)
        else:
            x_min = c
            c = d
            d = x_min + lambda_golden * delta
            y_c = y_d 
            y_d = function(d)       
    return ((x_max + x_min)/2)

if __name__ == '__main__':
    x_min = golden_section_search_slow_congergence(function=fun_quadratic, 
                                   x_min=-10, 
                                   x_max=10, 
                                   eps=1E-3)        
     
    print('{:.7f}'.format(x_min))           
                
       
    x_min = golden_section_search(function=fun_quadratic, 
                                   x_min=-10, 
                                   x_max=10, 
                                   eps=1E-3)        
     
    print('{:.7f}'.format(x_min))   
