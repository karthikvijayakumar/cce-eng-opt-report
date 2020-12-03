# Accelerated gradient descent and related algorithms

Project report for the CCE course "Engineering Optimization"

The project focusses on accelerated gradient descent and related algorithms. We first take a look at how gradient descent works, its convergence guarantees, the need for momentum and variations of momentum

The outline of the report is as follows:

1. Gradient descent
    1. Algorithm
    2. Choice of step sizes
    3. Line search
    4. Backtracking line search
2. Convergence of gradient descent
    1. Lipschitz continuity
    2. Strong convexity
    3. Convergence under lipschitz continuity
    4. Convergence under lipschitz continuity and strong convexity
3. Lower bounds on first order convergence
4. Momentum
5. Nesterov accelerated gradient descent
6. Observed issues with GD
7. Variations of accelerated gradient descent


References used while building this report:

1. Lecture notes from CFGranda, NYU on convex optimization ( [link to particular lecture](https://cims.nyu.edu/~cfgranda/pages/OBDA_fall17/notes/convex_optimization.pdf), [course website](https://cims.nyu.edu/~cfgranda/pages/OBDA_fall17/schedule.html) )
2. "Why momentum works" from distil.pub ( https://distill.pub/2017/momentum/ )
3. "An overview of gradient descent optimization algorithms" ruder.io ( [website](https://ruder.io/optimizing-gradient-descent/index.html), [arxiv paper](https://arxiv.org/abs/1609.04747) )
4. Convergence theorems for gradient descent, Robert.M.Grower ( [link](https://gowerrobert.github.io/pdf/M2_statistique_optimisation/grad_conv.pdf) )

```{note}
This report was built using the new Sphinx-based [Jupyter Book
2.0](https://beta.jupyterbook.org/intro.html) tool set.
```