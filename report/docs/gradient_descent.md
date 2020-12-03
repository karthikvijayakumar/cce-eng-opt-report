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

(gradient_descent)=

# Gradient descent

*Algorithm*

1. Set initial point $x_{0}$ to an arbitrary value in $\mathbb{R}^{n}$
2. Update by setting

$$
    x_{k+1} = x_{k} - \alpha_{k} \nabla f(x_{k});    \alpha_{k} \epsilon \mathbb{R}^{+}
$$

3. Repeat (2) until a stopping criterion is met.

    The stopping condition could be either on the function value or on the gradient or the x values themselves

There are multiple things to think about with the above algorithm:

1. How to choose ${\alpha_{k}}_{k=1}^{\infty}$
2. Does the above algorithm converge? If so for what choices of ${ \alpha_{k} }_{k=1}^{\infty}$ and how fast?


## Choosing step size ${\alpha_{k}}_{k=1}^{\infty}$

Keeping ${\alpha_{k}}_{k=1}^{\infty} = \alpha$ ( fixed ) can be disadavantageous. Ideally we want to adapt $\alpha$ as the iteration progresses to enable faster convergence.

### Line search

One possibility is line search

$$
    \alpha_{k} = argmin_{\alpha} f(x_{k} - \alpha \nabla f(x_{k}))
$$

(i.e find the alpha that gives the most minimization at this step )


Note that $f(x_{k} - \alpha \nabla f(x_{k}))$ is a restriction of $f$. Restrictions of convex functions are convex and hence the above minimization problem for finding $\alpha_{k}$ is in turn a convex problem. However that can be complex in own right and may not offer analytical or closed form solutions

### Backtracking line search ( Armijo rule )

One way to tackle the minimization problem is to not find the minimum but find enough drop in $f(x_{k} - \alpha \nabla f(x_{k}))$ that we can guarantee convergence

Algorithm:

Given $\alpha_{0} >= 0, \beta, \eta \epsilon (0,1)$

1. Pick $x_{0}$ arbitrary point in $\mathbb{R}^{n}$
2. Set $\alpha_{k} = \alpha_{0} \beta^{i}$ for the smallest integer $i$ such that $x_{k+1} = x_{k} - \alpha_{k} \nabla f(x_{k})$ satisfies

$$
    f(x_{k+1}) <= f(x_{k}) - \frac{1}{2} \alpha_{k} \Vert \nabla f(x_{k}) \Vert_{2}^{2}
$$