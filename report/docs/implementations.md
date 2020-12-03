---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(implementations)=

# Implementations

I implemented:

1. Steepest gradient descent
2. Accelerated gradient descent ( Nesterov momentum )
3. Adam

for the following 3 problems:

1. Linear regression with one variable
2. Linear regression with multiple variables
3. Logistic regression with multiple variables

The convergence for the three problems looks as follows:

````{div} full-width
```{figure} /_static/convergence/linear_regression_one_variable.png
:scale: 100%
:name: lin_reg_one_variable

Convergence for linear regression in one variable
```
````

````{div} full-width
```{figure} /_static/convergence/linear_regression_multiple_variable.png
:scale: 100%
:name: lin_reg_multiple_variable

Convergence for linear regression in multiple variable
```
````

````{div} full-width
```{figure} /_static/convergence/logistic_regression.png
:scale: 100%
:name: log_reg

Convergence for logistic regression in multiple variables
```
````

One particular anomaly stands out in the logistic regression charts that the loss function actually increases and decreases like ripples. Adam makes no guarantees that its a descent algorithm, hence this kind of behaviour is possible. We do observe that the loss eventually decreases though.

In the 3 experiments run above Adam seems to perform poorly compared to Nesterov accelerated gradient descent. It's difficult to tell at this point if it is due to the algorithm itself or due to the choice of parameters.