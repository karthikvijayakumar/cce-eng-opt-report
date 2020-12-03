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

(momentum)=

# Momentum

One of the aspects of standard gradient descent with fixed step size is that each step is independent of the other.

This can either make it rapidly switch directions close to the local minima when the step size is too high or be excruciatingly slow in converging.

Momentum tries to bake in historical information as the name suggests

The first form of momentum proposed was by Polyak called the heavy ball method

**Heavy ball method**

Do updates as follows

$$
    x_{t+1} = x_{t} − \gamma \nabla f(x_{t}) + \mu (x_{t} - x_{t−1})
$$

This works in practice for most problems. However there are some special functions for which the above algorithm provably does not converge


Nesterov suggested a slightly different form of momentum

**Nesterov momentum**

Nesterov maintain's two seequences

$$
    z^{k+1} = \beta z^{k} + \nabla f(w^{k})
$$
$$
    w^{k+1} = w^{k} - \alpha z^{k+1}
$$

Nesterov showed that the above can provably converge under certain conditions. We will focus on nesterov momentum for the rest of this document

## Nesterov momentum

Lets take a deeper look at nesterov momentum and see how it is different from simple gradient descent

Gradient descent with fixed step size looks as follows

$$
    w_{k+1} = w_{k} - \alpha \nabla f(w_{k})
$$

In nesterov iterates lets first unroll the $z_{k}$ 

$$    
    z_{k} = \beta z_{k-1} + \nabla f(w_{k-1})
$$
$$
    z_{k} = \beta \left( \beta z_{k-2} + \nabla f(w_{k-2} \right) + \nabla f(w_{k-1})
$$
$$
    z_{k} = \beta^{k} z^{0} + \frac{1}{\beta} \left( \sum_{i=1}^{k} \beta^{i} f(w_{k-i}) \right)
$$

If we substitute the unrolled $z_{k}$ into the formula for $w_{k+1}$ then we get

$$
    w_{k+1} = w_{k} - \alpha \nabla f(w_{k}) -\alpha \left( \beta^{k+1} z^{0} + \sum_{i=1}^{k} \beta^{i} f(w_{k-i})   \right)
$$

We notice that the update involves gradients from previous iterates in nesterov which is markedly different from gradient descent

The $\beta$ term controls how much the gradients from previous iterates contribute to the current update.

## Issues with momentum

There are notably a few issues with momentum

* If one has to choose a fixed learning rate and momentum. finding the right $\alpha$ and $\beta$ values can be tricky
* The steps in each directions is equally multiplied with the step size. Some directions need stronger updates compared to the others