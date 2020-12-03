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

(lower_bounds_convergence)=

# Lower bounds on convergence

Before we state the theorem on lower bounds, lets first define broadly what first order methods are

In gradient descent

$$
    x_{k+1} = x_{k} - \alpha_{k} \nabla f(x_{k}); \alpha \epsilon \mathbb{R}^{+}
$$

This means

$$
    x_{k+1} = x_{0} - \sum_{i=1}^{k} \alpha_{i} \nabla f(x_{i})
$$

The k'th iterate is a linear combination of the first iterate and gradients of previous iterates


In standard gradient descent each component of the gradient is multiplied by the same step size. This need not be the case though. We could multiply each component of the gradient with different scaling factors. This leads to the general definiion of first order methods


**First order method**

Iterate using 

$$
    x_{k+1} = x_{k} - \Gamma_{k} \nabla f(x_{k}); \Gamma_{ii} \epsilon \mathbb{R}^{+}
$$

which becomes 

$$
    x_{k+1} = x_{0} - \sum_{i=1}^{k} \Gamma_{i} \nabla f(x_{i})
$$

where $\Gamma$ is a diagonal matrix

**Theorem 1.5 ( Nesterov's lower bound )**

There exist convex functions with L-Lipschitz continuous gradients such that for any first order method we have

$$
    f(x_{k}) - f(x_{*}) >= 3L\frac{\Vert x_{0} - x_{*} \Vert_{2}^{2}}{32(k+1)^2}
$$

The above theorem implies using first order methods we can need $O\left( \frac{1}{\sqrt \epsilon} \right)$ iterations to reach an $\epsilon$ approximate solution

Infact Nesterov's showed that the convergence $O\left( \frac{1}{\sqrt \epsilon} \right)$ is optimal and gave the algorithm for it, which is accelerated gradient descent

The proof of the Nesterov's theorem is quite involved and we will skip it here. Details can be found in [section 2.1.2 of Nesterov's test](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.693.855&rep=rep1&type=pdf)