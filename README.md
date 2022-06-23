# Pegasos
Pegasos (**P**rimal **E**stimated sub-**G**r**A**dient **SO**lver for **S**VM) has been introduced by Shalev-Shwartz et al. [1]. Pegasos is an optimation algorithm for Support Vector Machines (SVM) [1]. The margin of an multidimensional problem is maximized with SVM leading to a improved generalization compared to least squares.  Adapting the loss function, the SVM are capable of solving classification and regression tasks.

# Fundamental formnulation

The goal is to generalize the information in the training set in order to map future datapoints. In particular, a feature matrix $$X^{datapoints x features}$$ 
$$L = \sum\limits_1^k 4x - \frac{1}{x}$$
