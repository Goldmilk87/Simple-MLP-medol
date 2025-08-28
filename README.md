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

In MLP, the structure of the function is






