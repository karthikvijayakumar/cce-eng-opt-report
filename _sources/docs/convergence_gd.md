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

(convergence_gradient_descent)=

# Convergence of gradient descent

A general analysis on the convergence of gradient descent can prove to be very difficult. We will focus on the case where $\nabla f$ shows one or both of the following proerties

1. Lipschitz continuous
2. Strong convexity

## Lipschitz continuity

A function $f : \mathbb{R}^{n} -> \mathbb{R}$ is Lipschitz continuosu with Lipschitz constant $L$ if for any $x,y \epsilon \mathbb{R}^{n}$

$$
    \Vert f(y) - f(x) \Vert_{2} <= L \Vert y-x \Vert_{2}
$$

## Strong convexity

A differentiable function $f : \mathbb{R}^{n} -> \mathbb{R}$ is $\mu$ strongly convex if for any $x,y \epsilon \mathbb{R}^{n}$

$$
    f(y) \geq f(x) + \nabla f(x)^{T} (y-x) + \frac{\mu}{2} \Vert y-x \Vert_{2}^{2}
$$

This is equivalent to 

$$
    \left< \nabla f(x) - \nabla f(y), x-y \right> \geq \alpha \Vert y-x \Vert_{2}^{2}
$$


*Note*

Compare the first condition above to the first order condition for convex functions

$$
    f(y) \geq f(x) + \nabla f(x)^{T}(y-x)
$$

The additional term gives a stronger lower bound ( probably which lead to the name strong convexity )

## Convergence under Lipschitz continuity

In this section we prove that gradient descent converges in $O(1/\epsilon)$ steps to an $\epsilon$ approximate solution when $\nabla f$ is Lipschitz continuous

We need a few results to help us with the proof

### Monotonicity of gradient

(monotonicity_of_gradient)=
**Lemma 1.1** 

A differentiable function $f : \mathbb{R}^{n} -> \mathbb{R}$ is convex iff
$$
    (\nabla f(y) - \nabla f(x))^{T}(y-x) >= 0
$$

**Proof**

Part 1: Convexity => $(\nabla f(y) - \nabla f(x))^{T}(y-x) >= 0$

If f is convex by first order condition

$$
\begin{eqnarray}
f(x) \geq f(y) + \nabla f(y)^{T}(x-y) \\
f(y) \geq f(x) + \nabla f(x)^{T}(y-x)
\end{eqnarray}
$$

Adding the two equations above gives us
$$
    \left( \nabla f(y)^{T} - \nabla f(x)^{T} \right) \left( y-x \right) \geq 0
$$

Part 2: $(\nabla f(y) - \nabla f(x))^{T}(y-x) >= 0$ => Convexity

Consider the restriction of the function f -> $g_{a,b}$

$$
    g_{a,b}(\alpha): [0,1] -> \mathbb{R}
$$

$$
    g_{a,b}(\alpha) := f(\alpha a + (1-\alpha)b)
$$

$$
    g'_{a,b}(\alpha) = \nabla f(\alpha a + (1-\alpha)b)^{T}(a-b)
$$

For any $\alpha \epsilon [0,1]$ we have

$$
\begin{align}
g'_{a,b}(\alpha) - g'_{a,b}(0) &= \left( \nabla f(\alpha a + (1-\alpha)b) \right) (a-b) \\
    &= \frac{1}{\alpha} \left( \nabla f(\alpha a + (1-\alpha)b) \right) \left( \alpha a - (1-\alpha)b - b \right) \\
    &\geq 0 (  since  \nabla f(y) - \nabla f(x))^{T}(y-x) >= 0 ) \\
\end{align}
$$

Writing $f(x)$ in terms of the restriction and using the above inequality gives

$$
\begin{align}    
    f(x) &= g_{xy}(1) \\
    &= g_{xy}(0) + \int_{0}^{1} g'_{xy}(\alpha)d\alpha \\
    &\geq g_{xy}(0) + \int_{0}^{1} g'_{xy}(0)d\alpha \\
    &\geq g_{xy}(0) + g'_{xy}(0) \\
    &=f(y) + \nabla f(y)^{T}(x-y)
\end{align}
$$

This is the first order condition for convexity => f is convex

### First order upper bound
(first_order_upper_bound)=
**Theorem 1.1** 

If the gradient of a function is L-Lipschitz continuous with Lipschitz constant L

$$
    \Vert \nabla f(y) - \nabla f(x) \Vert_{2} <= \Vert y-x \Vert_{2}
$$

then for any $x \epsilon \mathbb{R}^{n}$ the following holds

$$
    f(y) <= f(x) + \nabla f(x)^{T}(y-x) + \frac{L}{2}  \Vert y-x \Vert_{2}^{2}
$$

**Proof**

Consider the function

$$
    g(x) := \frac{L}{2} x^{T}x - f(x)
$$

For g

$$
    \nabla g(x) = Lx - \nabla f(x)
$$

For $x,y \epsilon \mathbb{R}^{n}$

$$
\begin{align}
    \nabla g(y) - \nabla g(x) &= Ly - \nabla f(x) - LX + \nabla f(x) \\
    &= L(y-x) - \left( \nabla f(y) - \nabla f(x) \right) \\
    \left( \nabla g(y) - \nabla g(x) \right )^{T}(y-x) &= L \Vert y-x \Vert_{2}^{2} - \left( \nabla f(y) - \nabla f(x) \right)^{T}(y-x) \\
\end{align}
$$

By Cauchy Schwarz inequality

$$
\begin{align}
    \left< \nabla f(y) - \nabla f(x), y-x \right> & \leq \Vert \nabla f(y) - \nabla f(x) \Vert \Vert y-x \Vert \\
    & \leq L \Vert y-x \Vert^{2}
\end{align}
$$

This implies

$$
    \left( \nabla g(y) - \nabla g(x) \right)^{T} (y-x) \geq 0
$$

The above implies g is convex by the monotonicity of the gradient lemma proved above

By first order condition for convexity

$$
\begin{align}
    \frac{L}{2} y^{T}y - f(y) &= g(y) \\
    &\geq g(x) + \nabla g(x)^{T}(y-x) \\
    &= \frac{L}{2} x^{T}x - f(x) +(Lx - \nabla f(x))^{T}(y-x)
\end{align}
$$

This implies

$$
\begin{align}
    f(y) &\leq f(x) + \nabla f(x)^{T}(y-x) + \frac{L}{2}(y^{T}y - x^{T}x) - Lx^{T}(y-x) \\
    &\leq  f(x) + \nabla f(x)^{T}(y-x) + \frac{L}{2}(y^{T}y - x^{T}x) - 2x^{T}y + 2x^{T}x) \\
    &= f(x) + \nabla f(x)^{T}(y-x) + \frac{L}{2}(y^{T}y + x^{T}x) - 2x^{T}y) \\
    &= f(x) + \nabla f(x)^{T}(y-x) + \frac{L}{2}(y-x)^{T}(y-x) \\
    &= f(x) + \nabla f(x)^{T}(y-x) + \frac{L}{2} \Vert y-x \Vert_{2}^{2}
\end{align}
$$

Hence proved.

### Convergence of successive iterates
(convergence_of_successive_iterates)=
**Theorem 1.2** 

Let $x_{i}$ be the ith iteration of gradient descent and $\alpha_{i} >= 0$. if $\nabla f$ is L-Lipschitz continuous then

$$
    f(x_{k+1}) <= f(x_{k}) - \alpha_{k} ( 1 - \frac{\alpha_{k}L}{2} ) \Vert \nabla f(x_{k}) \Vert_2^{2}
$$

**Proof**

Applying the first order upper bound from theorem 1.1 we have

$$
    f(x_{k+1}) <= f(x_{k}) + \nabla f(x_{k})^{T}(x_{k+1} - x_{k}) + \frac{L}{2} \Vert x_{k+1} - x_{k} \Vert_{2}^{2}
$$

In gradient descent the iterates are computed as
$$
    x_{k+1} = x_{k} - \alpha_{k} \nabla f(x_{k})
$$

This implies
$$
  x_{k+1} - x_{k} = - \alpha_{k} \nabla f(x_{k})
$$


$$
\begin{align}
    f(x_{k+1}) &\leq f(x_{k}) + \nabla f(x_{k})^{T}(-\alpha_{k}\nabla f(x_{k})) + \frac{L}{2}\alpha_{k}^{2} \Vert \nabla f(x_{k}) \Vert_{2}^{2} \\
    &\leq f(x_{k}) + \alpha_{k} \Vert \nabla f(x_{k}) \Vert_{2}^{2} + \frac{L}{2}\alpha_{k}^{2} \Vert \nabla f(x_{k}) \Vert_{2}^{2} \\
    &\leq f(x_{k}) + \alpha_{k} \left( 1- \frac{\alpha_{k} L}{2} \right) \Vert \nabla f(x_{k}) \Vert_{2}^{2} \\
\end{align}
$$

Hence proved

### Gradient descent is a descent method
(gradient_descent_is_a_descent_method)=

**Corollary 1.1** Gradient descent is a descent method if $\alpha_{k} <= \frac{1}{L}$

**Proof**

Substituting $\alpha_{k} \leq \frac{1}{L}$ into the inequality from the above theorem gives us the result

### Convergence rate under Lipschitz continuity
(convergence_rate_under_lipschitz)=

**Theorem 1.3 ( Main result )** : Assume 

- f is convex
- $\nabla f$ is L-Lipschitzz continuous 
- There exists a point $x^{*}$ at which f achieves a finite minimum

If we setup the step size of gradient descent $\alpha_{k} = \alpha = \frac{1}{L}$ for every iteration then

$$
    f(x_{k}) - f(x_{*}) <= \frac{ \Vert x_{0}-x_{*} \Vert^{2}}{2\alpha k}
$$

**Proof**

By first order characterisation of convexity

$$
    f(x_{k-1}) + \nabla f(x_{k-1})^{T}(x_{*} - x_{k-1}) \leq f(x_{*})
$$

Using the previous corollary

$$
\begin{align}
    f(x_{k}) & \leq f(x_{k-1}) - \frac{\alpha_{k-1}}{2} \Vert \nabla f(x_{k-1}) \Vert_{2}^{2} \\
    f(x_{k-1}) & \geq f(x_{k}) + \frac{\alpha_{k-1}}{2} \Vert \nabla f(x_{k-1}) \Vert_{2}^{2}
\end{align}
$$

Combining the two

$$
\begin{align}
f(x_{k}) + \frac{\alpha_{k-1}}{2} \Vert \nabla f(x_{k-1}) \Vert_{2}^{2} + \nabla f(x_{k-1})^{T}(x_{*} - x_{k-1}) \leq f(x_{*}) \\
f(x_{k}) - f(x_{*}) \leq \nabla f(x_{k-1})^{T}(x_{*} - x_{k-1}) - \frac{\alpha_{k-1}}{2} \Vert \nabla f(x_{k-1}) \Vert_{2}^{2} \\
\end{align}
$$

Lets expand the RHS term

$$
\begin{align}
    RHS &= \nabla f(x_{k-1})^{T}(x_{*} - x_{k-1}) - \frac{\alpha_{k-1}}{2} \Vert \nabla f(x_{k-1}) \Vert_{2}^{2} \\
    &= \frac{1}{2\alpha} \left( 2\alpha \nabla f(x_{k-1})^{T}(x_{k-1} - x_{*}) - \alpha^{2} \Vert \nabla f(x_{k-1}) \Vert_{2}^{2}  \right) \\
    &= \frac{1}{2\alpha} \left( \Vert x_{k-1} - x_{*} \Vert_{2}^{2} - \Vert x_{k-1} - x_{*} \Vert_{2}^{2} + 2\alpha \nabla f(x_{k-1})^{T}(x_{k-1} - x_{*}) -\alpha^{2} \Vert \nabla f(x_{k-1})\Vert_{2}^{2} \right) \\
    &= \frac{1}{2\alpha} \left( \Vert x_{k-1} - x_{*} \Vert_{2}^{2} - \Vert x_{k-1} - x_{*} - \alpha \nabla f(x_{k-1}) \Vert_{2}^{2}  \right) \\
    &= \frac{1}{2\alpha} \left( \Vert x_{k-1}-x_{*} \Vert_{2}^{2} - \Vert x_{k}-x_{*} \Vert_{2}^{2} \right)
\end{align}
$$

Hence we have
$$
    f(x_{k}) - f(x_{*}) \leq \frac{1}{2\alpha} \left( \Vert x_{k-1}-x_{*} \Vert_{2}^{2} - \Vert x_{k}-x_{*} \Vert_{2}^{2} \right)
$$

By previous corollary we know that $\left( f(x_{k}) \right)_{k=1}^{\inf}$ is decreasing, hence

$$
\begin{align}
    f(x_{k}) & \leq \sum_{i=1}^{k} \frac{1}{k} f(x_{i}) \\
    f(x_{k}) - f(x_{*}) & \leq \left[  \sum_{i=1}^{k} \frac{1}{k} f(x_{i}) \right]  - f(x_{*}) \\
    &= \frac{1}{k} \left( \sum_{i=1}^{k} f(x_{i}) -f(x_{*}) \right) \\
    &= \frac{1}{2 \alpha_{min} k} \left( \Vert x_{0}-x_{*} \Vert_{2}^{2} - \Vert x_{k}-x_{*} \Vert_{2}^{2} \right) \\
    & \leq \frac{\Vert x_{0} - x_{*} \Vert_{2}^{2}}{2 \alpha_{min} k}
\end{align}
$$

Hence we have

$$
    f(x_{k}) - f(x_{*}) \leq \frac{\Vert x_{0} - x_{*} \Vert_{2}^{2}}{2 \alpha_{min} k}
$$

Theorem 1.3 shows that if we know the Lipschitz constant of the gradient then we can choose a fixed step $\alpha$ to get arbitrarily close to the minimum. In particular to get an $\epsilon$ approximate solution we would need $O(\frac{1}{\epsilon})$ steps

In practice we may not know the Lipschitz constant of a function analytically. In which case there are 2 ways to proceed forward:

1. Find an upper bound for the Lipschitz constant if possible
2. Use backtracking line search ( Armijo rule  )
    We can show that bactracking line search is capable of hitting the $\alpha_{k} <= \frac{1}{L}$ condition

For the latter we have the following results

**Lemma 1.2** Lower bound on step size with backtracking search

If the gradient of a function $f: \mathbb{R}^{n} -> \mathbb{R}$ is Lipschitz continuous with Lipschitz constant L, the step size obtained by backtracking line search with $\eta = 0.5$ satisfies
$$
    \alpha_{k} >= \alpha_{min} := min \left( \alpha^{0}, \frac{\beta}{L} \right) 
$$

**Theorem 1.4** Convergence with backtracking line search

If f is convex and $\nabla f$ is L-Lipschitz continuous, gradient descent with backtracking line search produces a sequence of points that satisfy

$$
    f(x_{k}) - f(x_{*}) <= \frac{\Vert x_{0}-x_{*} \Vert_{2}^{2}}{2\alpha_{min}k}
$$

We skip the proofs of the above theorem and lemma. They are available in [Section 3.2 of CF.Granda's lecture notes](https://cims.nyu.edu/~cfgranda/pages/OBDA_fall17/notes/convex_optimization.pdf)

## Convergence under Lipschitz continuity and strong convexity

**Theorem 1.5**

Let f be L-smooth and $\mu$ strongly convex. From a given $x_{0} \epsilon \mathbb{R}^{n}$ and $0 \lt \alpha \leq \frac{1}{L}$, the iterates

$$
    x_{t+1} = x_{t} - \alpha \nabla f(x_{t})
$$

converge according to 

$$
    \Vert x_{t+1} - x_{*} \Vert_{2}^{2} \leq (1-\alpha \mu)^{t+1} \Vert x^{0} - x^{*} \Vert_{2}^{2}    
$$

In particular when $\alpha = \frac{1}{L}$, the iterates converge linearly with a rate of $\frac{\mu}{L}$

We skip the proof here. It can be found in [Robert Gower's notes](https://gowerrobert.github.io/pdf/M2_statistique_optimisation/grad_conv.pdf)