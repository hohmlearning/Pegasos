# Pegasos
Pegasos (**P**rimal **E**stimated sub-**G**r**A**dient **SO**lver for **S**VM) has been introduced by Shalev-Shwartz et al. [1]. Pegasos is an optimation algorithm for Support Vector Machines (SVM) [1]. The margin of an multidimensional problem is maximized with SVM leading to a improved generalization compared to least squares.  Adapting the loss function, the SVM are capable of solving classification and regression tasks.

# Fundamental formulation of the primal problem

The goal is to generalize the information in the training set in order to map future datapoints. In particular, weights $w^d$ and bias $w_0$ are optimized on the feature matrix $X^{n; d}$ to generalizize on the labels $y^d$, with $n$ datapoints and $d$ features [1]:

$$L(w, w_0; X, y) =  \frac{\lambda}{2} ||w||^2 +  \frac{1}{m} \sum\limits_{X,y} loss(w, w_0; X, y)$$

$m$ is the number of randomly selected examples. 
<br>
<br>
The loss function $loss$ is choosen according to the problem. In June 2022, the implemented loss functions for respective tasks are shown in **table 1**.

#### **Table 1**: Implemented loss functions for respective tasks in June 2022.

| Task | Name loss function | $loss(w, w_0; X, y) = $ | Requirements |
|:--------------|:-------------|:----------------|:-------------:|
|Binary Classification       |Hinge loss       | max{0, 1-$y_i$ * (< $w, X_i$ > + $w_0$)}        | $y$ in {-1, 1}       |
|Regression       | $\epsilon$-insensitive loss      | max{0, abs{$ y_i - (< w, X_i > + w_0)$}$  - \epsilon$}  | -   |

<br>
<br>

The SVM problem are solved with stochastic sub-gadient derived in the original paper [1]. The Pegasos algorithm without bias term is given in **figure 1**. 

![Primal Pegasos](https://user-images.githubusercontent.com/107933496/175406566-621d0689-f0e4-4318-9eae-fc7c2aeeb7dc.PNG)
#### **Figure 1**: Primal Pegasos copied from original paper [1].

The learning rate \eta is decreased with increasing iteration $t$. Therefore, there is no need for tuning $\eta$. The optional step is not implemented.
