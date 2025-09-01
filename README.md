# Simple-MLP-medol

## This is a simple one-hidden-layer MLP model designed to learn the mathematical principles behind it. It is highly simplified with the ability to predict the expected number given $R^n$ -> $R$.

In a mathematical sense, to predict a number given related information, we want to guess a function

$$f(x)$$

such that $f(x)$ returns some value close to the true value.

In statistics, $f(x)$ is always based on some parameters $\theta$ and a chosen structure, which we write

$$f(x;\theta)$$ 

to mean "our prediction of y given input x, under model parameter $\theta$."

We need to pick $\theta$ such that the predicted values $f(x;\theta)$ are as consistent as possible with the observed $y_i$ in the training set.

$$\displaystyle \hat \theta = \arg\max_\theta \prod_{i=1}^{N}p(y_i|x_i;\theta)$$

For example, linear regression, assumed the $y|x \backsim \text{Norm}(f(x;\theta), \sigma^2)$.

And we define it as our likelihood function, $L(\theta)$. Minimizing the negative log-likelihood, which is usually called loss, is equal to maximizing the likelihood function. For example, the likelihood function for Regression is

$$L(\theta) = \frac{1}{N} \sum_{i=1}^{N} r^2 = \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i:\theta))^2$$

For a one-hidden-layer MLP with \(h\) hidden neurons, the structure is:

$$
z_1 = w_1 x + b_1, \quad 
a_1 = f_{\text{act}}(z_1),
$$

$$
\hat{y} = f(x;\theta) = w_2 a_1 + b_2.
$$

- $w_1 \in \mathbb{R}^{h \times d}, b_1 \in \mathbb{R}^h$  
- $w_2 \in \mathbb{R}^{1 \times h}, b_2 \in \mathbb{R}$  
- $f_{\text{act}}$ is a non-linear activation (e.g., ReLU, tanh).

And the loss function is:

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N \big(y_i - f(x_i;\theta)\big)^2,
$$

which is the **Mean Squared Error (MSE)**, also known as the same concept in linear regression. The initiative behind this idea is to minimize the difference between the predicted value $\hat {y} _ i = f(x)$ and the true value of $y_i$.

To minimize the MSE, we use gradient descent on the loss function on the training set. 

- For the output layer:

$$
\frac{\partial \mathcal{L}}{\partial w_2} 
= \frac{1}{N} \sum_{i=1}^N r_i (a_{1,i})^\top,
\qquad
\frac{\partial \mathcal{L}}{\partial b_2} 
= \frac{1}{N} \sum_{i=1}^N r_i
$$

- For the hidden layer :

$$
\delta_1 = (w_2^T R) \odot f'_{\text{act}}(Z)
$$

$$
\frac{\partial \mathcal{L}}{\partial w_1} 
= \frac{1}{N} \sum_{i=1}^N \delta_{1,i} x_i^\top,
\qquad
\frac{\partial \mathcal{L}}{\partial b_1} 
= \frac{1}{N} \sum_{i=1}^N \delta_{1,i}
$$

These derivatives are used in gradient descent updates:

$$
\hat \theta_i = \theta_{i-1} - c \cdot \Delta \theta
$$ 

where c is the learning rate.

After we have the $\hat \theta$, we have the $\hat f(x)$, which can predict a number.
## Reference
Most of the above is from,
Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. Retrieved from http://www.deeplearningbook.org


## Testing part
For a given function, $y_i = \frac{x_1}{10}(x_2 +x_3)$, the result for training set is: 
<img width="1161" height="781" alt="image" src="https://github.com/user-attachments/assets/cf613fec-5be8-4a46-bcad-14a8d93a2a3f" />

For the testing set is:
<img width="1096" height="795" alt="image" src="https://github.com/user-attachments/assets/6c434e94-cd28-4cca-95f2-e192920d4129" />

*midNeuron = 10 is chosen to have the result.
*$x1, x2, x3$ are normalized 











