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

(variations)=

# Variations of accelerated gradient descent

We will focus on 3 variations/improvements of nesterov accelerated gradient descent in this section

1. Adagrad
2. Adadelta
3. Adam

Each of these 3 try to sequentially improve upon on some aspects from the previous one

## Adagrad

Adagrad tries to tackle the issue of needing seperate learning rates for different parameters by using a diagonal matrix for a step size instead of a scalar. Note that we saw the use of diagonal matrices in the definition of first order methods previously.

Understandably just using a diagonal matrix makes it more difficult for the user in that there are more parameters to set. To tackle this adagrad builds the diagonal matrices as the inverse of the root mean square of the sum of gradients seen so far. Directions where the gradients are large will get slowed down and places where the gradient has become tiny could get emphasized.


*Algorithm ( for the i'th component )*

1. Compute $g_{t,i} = \nabla_{x} f(x_{t})_{i}$, the gradient for the i'th component
2. Update the i'th component of the iterate $x_{t+1,i}$ using 
$$
    x_{t+1,i} = x_{t,i} - \frac{\eta}{\sqrt{ \sum_{j=1}^{t} g_{j,i}^{2} + \epsilon}} g_{t,i}
$$

Above we observe that each component has a differently scaled step size at each iteration. The scaling is based on how the gradient for that parameter have been in the past.

The $\epsilon$ above is a very small number ( typically $10^{-8}$ ) introduced to prevent division by zero errors

*Algorithm ( vectorized for all components )*

1. Compute $g_{t} = \nabla_{x} f(x_{t})$, the gradient. 
2. Set $G_{t}$ to be an nxn diagonal matrix where

$$
    \left( G_{t} \right)_{i,i} = \sum_{j=1}^{t} g_{j,i}^{2}
$$

2. Compute the iterate $x_{t+1,i}$ using 
$$
    x_{t+1} = x_{t} - \eta \left( \sqrt{ G_{t} + \epsilon} \right)^{-1} g_{t}
$$

$\eta$ above is the learning rate as in vanilla gradient descent

## Adadelta

One of the drawbacks of Adagrad was that the $G_{t}$ was accumalating gradients and could get large after severla iterations making the learning rate very small after which the iterates wont change much.

Adadelta improved upon this by using an exponential moving average of the gradients instead of a simple sum.

The exponential average is defined as follows:

$$
    E[g^{2}]_{t} = \gamma E[g^{2}]_{t-1} + (1-\gamma)E[g^{2}]_{t-1}
$$

when rolled out this becomes

$$
    E[g^{2}]_{t} = \gamma^{t} E[g^{2}]_{0} + \sum_{j=1}^{t-1} (1-\gamma)^{j}E[g^{2}]_{t-j}
$$

*Algorithm*

1. Compute $g_{t} = \nabla_{x} f(x_{t})$, the gradient. 
2. Compute $E[g^{2}]_{t}$ using

$$
    \left( E[g^{2}]_{t} \right)_{i,i} = \gamma E[g^{2}]_{t-1} + (1-\gamma)E[g^{2}]_{t-1}
$$

2. Compute the iterate $x_{t+1}$ using 

$$
    x_{t+1} = x_{t} - \eta \left( \sqrt{ E[g^{2}]_{t} + \epsilon} \right)^{-1} g_{t}
$$


Some points to note:

* $\eta$ above is the learning rate as in vanilla gradient descent
* $\epsilon$ is a very small number introduced to prevent division by zero errors
* $\gamma$ is parameter that has been introduced to control the impact of past gradient on the current step size. This is akin to the momentum term in nesterov accelerated gradient descent.

## Adam

Adam stands for Adaptive Moment Estimation.

Adagrad and Adadelta used the accumalated sum of gradients to adjust the step size at each step. Adam uses the accumalated gradient itself in addition to the accumalated squared gradients to adjust the step size.

The high level idea is to do the following:

1. Compute $g_{t} = \nabla_{x} f(x_{t})$, the gradient. 
2. Compute $m_{t}$ and $v_{t}$ using

$$
\begin{eqnarray}
m_{t} = \beta_{1}m_{t-1} + (1-\beta_{1})g_{t} \\
v_{t} = \beta_{2}v_{t-1} + (1-\beta_{2})g_{t}^2
\end{eqnarray}
$$

3. Compute the iterate $x_{t+1,i}$ using 
$$
    x_{t+1} = x_{t} - \eta \left( \sqrt{ m_{t} + \epsilon} \right)^{-1} m_{t}
$$

Points to note from the above:

* The $m_{t}$ term above is the same as the $E[g^{2}]_{t}$ in Adadelta
* Instead of multiplying the step size with gradient at the current step, in Adam its multiplied with the exponential running average of the gradients
* Two parameters $\beta_{1}$ and $\beta_{2}$ were introduced to control the impact of recent gradients ( and corresponding squares ) on the updates. They are typically set to 0.9 and 0.999 respectively.
* $m_{0}$ and $v_{0}$ are set to zero


The last 2 points above mean that new gradients will take a long time to relfect in the iterates and $m_{t}$ and $v_{t}$ would be biased towards zero ( particularly in the initial iterations )

To correct the zero bias issue, one multiplies $m_{t}$ and $v_{t}$ with a factor.

Define 

$$
\begin{eqnarray}
\hat m_{t} = \frac{m_{t}}{1-\beta_1^{t}} \\
\hat v_{t} = \frac{v_{t}}{1-\beta_2^{t}}
\end{eqnarray}
$$

The final algorithm looks as follows:

*Algorithm*

1. Compute $g_{t} = \nabla_{x} f(x_{t})$, the gradient. 
2. Compute $m_{t}$ and $v_{t}$ using

$$
\begin{align}
    m_{t} &= \beta_{1} m_{t-1} + (1-\beta_{1})g_{t} \\
    v_{t} &= \beta_{2} v_{t-1} + (1-\beta_{2})g_{t}^{2}
\end{align}
$$

3. Compute $\hat m_{t}$ and $\hat v_{t}$ using

$$
\begin{align}
\hat m_{t} &= \frac{m_{t}}{1-\beta_1^{t}} \\
\hat v_{t} &= \frac{v_{t}}{1-\beta_2^{t}}
\end{align}
$$

4. Compute the iterate $x_{t+1,i}$ using the following

$$
    x_{t+1} = x_{t} - \frac{\eta}{\left( \sqrt{ \hat m_{t} + \epsilon} \right)} \hat m_{t}
$$

## Convergence guarantees

In previous sections we showed that gradient descent is guaranteed to converge for certain choice of step sizes. There exist similar guarantees for Nesterov accelerated gradient descent.

One could ask for similar convergence guarantees for variations of momentum like Adam, Adagrad etc.

It turns out that Adam is not guaranteed to converge on all convex problems. A counter example convex problem was shown by [Reddi et. all in 2018](https://arxiv.org/pdf/1904.09237.pdf)

Adam is known to converge under the following conditions stochastically

1. There exists $x_{*}$ such that $f(x_{*}) \leq f(x) \forall x$
2. The $l_{\inf}$ of the gradient is bounded almost surely
3. The gradient of the function $\nabla f$ is L-smooth ( Lipschitz continuous with constant L )

A proof of the convergence for both Adam and Adagrad can be found in [Alexandre et. all](https://arxiv.org/pdf/2003.02395.pdf)