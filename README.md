# Pegasos
Pegasos (**P**rimal **E**stimated sub-**G**r**A**dient **SO**lver for **S**VM) has been introduced by Shalev-Shwartz et al. [1]. Pegasos is an optimation algorithm for Support Vector Machines (SVM) [1]. The margin of an multidimensional problem is maximized with SVM leading to a improved generalization compared to least squares.  Adapting the loss function, the SVM are capable of solving classification and regression tasks.

# Fundamental formulation of the primal problem

The goal is to generalize the information in the training set in order to map future datapoints. In particular, weights $w^d$ and bias $w_0$ are optimized on the feature matrix $X^{n; d}$ to generalizize on the labels $y^d$, with $n$ datapoints and $d$ features [1]:

$$L(w, w_0; X, y) =  \frac{\lambda}{2} ||w||^2 +  \frac{1}{m} \sum\limits_{X,y} loss(w, w_0; X, y)$$

$m$ is the number of randomly selected examples. The loss function $loss$ is choosen according to the problem. In June 2022, the implemented loss functions for respective tasks are shown in Table 1.
