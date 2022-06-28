
# Pegasos
Pegasos (**P**rimal **E**stimated sub-**G**r**A**dient **SO**lver for **S**VM) has been introduced by Shalev-Shwartz et al. [1]. Pegasos is an optimation algorithm for Support Vector Machines (SVM) [1]. The margin of a multidimensional problem is maximized with SVM leading to an improved generalization compared to least squares.  Adapting the loss function, the SVMs are capable of solving classification and regression tasks.\
In this project the Pegasos algorithm for binary classification and regression is implemented with minor changes.

## Fundamental formulation of the primal problem

The goal is to generalize the information in the training set in order to map future datapoints. In particular, weights $w^d$ and bias $w_0$ are optimized on the feature matrix $X^{n; d}$ to generalize on the labels $y^d$, with $n$ datapoints and $d$ features [1]:

$$L(w, w_0; X, y) =  \frac{\lambda}{2} ||w||^2 +  \frac{1}{m} \sum\limits_{X,y} loss(w, w_0; X, y)$$

$m$ is the number of randomly selected examples. 
<br>
<br>
The loss function $loss$ is chosen according to the problem. In June 2022, the implemented loss functions for respective tasks are shown in **table 1**.

#### **Table 1**: Implemented loss functions for respective tasks in June 2022.

| Task | Name loss function | $loss(w, w_0; X, y) = $ | Requirements |
|:--------------|:-------------|:----------------|:-------------:|
|Binary Classification       |Hinge loss       | max{0, 1-$y_i$ * (< $w, X_i$ > + $w_0$)}        | $y$ in {-1, 1}       |
|Regression       | $\epsilon$-insensitive loss      | max{0, abs{$ y_i - (< w, X_i > + w_0)$}$  - \epsilon$}  | -   |

<br>
<br>

The SVM problem is solved with stochastic sub-gadient derived in the original paper [1]. The Pegasos algorithm without bias term is given in **figure 1**.

![Primal Pegasos](https://user-images.githubusercontent.com/107933496/175406566-621d0689-f0e4-4318-9eae-fc7c2aeeb7dc.PNG)
#### **Figure 1**: Primal Pegasos copied from original paper [1].

The learning rate $\eta$ is decreased with increasing 
iteration $t$. Therefore, there is no need for tuning 
$\eta$. The optional projection step is not implemented. 
In addition to the algorithm in **figure 1**, the bias term $w_0$ is updated, whenever a mistake was made:
$$ w_0 &larr; w_0 + \eta * y_i$$
Importantly, the bias term $w_0$ is not regularized.

The randomization of the sample selecting is modified. In contrary to the original paper [1], the dataset ($X,y$) is shuffled randomly at each epoch. Therefore, in each epoch, each example is used once.
The code snipset for running one epoch is given:

```python

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
                    self.theta = (1 - self.learning_rate * self.regularization) *  self.

```

## Kernalized Pegasos
SVM is not only suited for solving linear relationships between features and labels but also nonlinear such as polynomial or radial basis function kernel. The features are 
implicity transferred to a higher space. Only the feature inner product is needed. Therefore, there is no need for transforming features manually for higher feature space. 

Also, the mapping function is never explicitly calculated, the features are implicity transformed with a mapping function $\phi(X)$:

$$L(w, w_0; X, y) =  \frac{\lambda}{2} ||w||^2 +  \frac{1}{m} \sum\limits_{X,y} loss(w, w_0; \phi(X), y)$$ 

<br>
However, the Pegasos algorithm is implemented with a kernel:
$$ K(x,x') = <\phi(x),\phi(x')> $$
<br>
The kernels implemented are listed in **table 2**.

#### **Table 2**: Kernels implemented in June 2022.
 |Name | Formulation | Details|
 |:--------------|:-------------|:----------------|
 |Polynomial|$K(x,x')=( < x,x' > + c)^{d}$|$c>=0$|
 <br>
 $d$ is the degree and 
 $c$ a parameter trading of the influence of higher-order terms in the polynomial. Therefore,
 $c$ acts as a regularization parameter.
<br>
<br>
The weight vector $w$ 
is calculated implicity. A vector $\alpha_t [j]$ counts the number of the performed corrections for every example 
$j$ at the timestep
$t$. The mathematical reformulations are referred to the original paper [1]. 
The Kernalized Pegasos algorithm without bias term is given in figure 2.
<br>

![grafik](https://user-images.githubusercontent.com/107933496/176115516-804bf9e7-c8bf-43ca-b70a-bcad98781af6.png)

#### **Figure 2**: Kernalized Pegasos copied from original paper [1].

In addition to the algorithm in **figure 2**, a bias term $w_0$ is updated. The code snppet of a single update step is shown:

```python
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
```

## Literatur
[1] - Shalev-Shwartz, S., Singer, Y., Srebro, N., & Cotter, A. (2011). Pegasos: Primal estimated sub-gradient solver for svm. Mathematical programming, 127(1), 3-30.












