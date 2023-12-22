Splotch
===================

Overview
-------------
Splotch is a hierarchical generative probabilistic model for analyzing spatial transcriptomics data [1].

Features
-------------
- Supports complex hierarchical experimental designs and model-based analysis of replicates
- Full Bayesian inference with Hamiltonian Monte Carlo (HMC) using the adaptive HMC sampler as implemented in **Stan** [2]
- Analysis of expression differences between anatomical regions and conditions using posterior samples
- Different anatomical annotated regions are modelled using a linear model
- Zero-inflated Poisson likelihood for counts
- Conditional autoregressive (CAR) prior for spatial random effect

Usage
-------------
Implementing a single Stan program with the features listed above that would simultaneously support arbitratory hierarchical experiment designs without making the use of the code extremely cumbersome and complicated is difficult.
Therefore, we have made available two different implementations (see the directories `human` and `mouse`) supporting the common two- and three-level hierarchical experimental designs

![Two-level hierarchical model](https://raw.githubusercontent.com/tare/Splotch/master/images/two-level_hierarchical_model.png)
![Three-level hierarchical model](https://raw.githubusercontent.com/tare/Splotch/master/images/three-level_hierarchical_model.png)

These two implementations can be easily modified to support other experimental designs.

Alternatively, we could write a Stan code generator, in the same way as in RStanArm [3], that would generate the approriate Stan code automatically based on the experimental design description.


### References
[1] Ståhl, Patrik L., et al. "Visualization and analysis of gene expression in tissue sections by spatial transcriptomics." Science 353.6294 (2016): 78-82.

[2] Carpenter, Bob, et al. "Stan: A probabilistic programming language." Journal of statistical software 76.1 (2017).

[3] Stan Development Team (2016). rstanarm: Bayesian applied regression modeling via Stan. R package version 2.13.1. <http://mc-stan.org/>.
