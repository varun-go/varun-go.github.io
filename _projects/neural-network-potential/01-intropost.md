---
layout: single
title: "Part 1 - Behler-Parinello Neural Network Potentials"
permalink: /projects/neural-network-potential/part-1/
# classes: wide
use_math: true
toc: true
toc_sticky: true
---
## Motivation

> [Disclaimer]\
> In the following, there may be a slight abuse of math notation for the sake of clarity.


## What is a Neural Network Potential?

<!-- 
Notes:
1. Add that we are looking at the flavor of NNPs which estimate that the potential energy of a system can be decomposed into atomic (or site) contributions.

-->

## Local descriptors for atomic environments

<!-- 
Notes:
1. Add information about other descriptors such as SOAP 
2. 

-->

### Atom-centered symmetry functions (ACSFs)

These functions are used to describe the local atomic environment around a central atom. These functions combine the pairwise distances between the central atom and its neighbors. 

**Cutoff Function ($G_1$)**

First, we consider the cutoff function. The cutoff function is used to weigh the contribution of atoms that are farther away from the central atom in the symmetry functions. These function smoothly decay to 0 at the cutoff distance. Two commonly used cutoff functions are:

$$
r_{ij} = | \mathbf{r}_i - \mathbf{r}_j |
$$

$$
f_c(r_{ij}) =
\begin{cases}
\frac{1}{2} \left[ \cos\left(\frac{\pi r_{ij}}{r_c}\right) + 1 \right], & r_{ij} \leq r_c \\
0, & r_{ij} > r_c
\end{cases}
$$

$$
f_c(r_{ij}) =
\begin{cases}
\tanh^3[1-\left(\frac{r_{ij}}{r_c}\right)], & r_{ij} \leq r_c \\
0, & r_{ij} > r_c
\end{cases}
$$

In accordance to the reference \[1\], the first ACSF is a sum of the cutoff functions for the neighbors of the central atom \(i\):

$$
G_1^i = \sum_{j \neq i} f_c(r_{ij})
$$

**Radial Symmetry Function ($G_2$)**

The radial symmetry function captures the local atomic density:

$$
G_2^i = \sum_{j \neq i} e^{-\eta (r_{ij} - r_s)^2} f_c(r_{ij})
$$

where $ \eta $ and $ r_s $ are parameters that alter the width and position of the Gaussian, respectively. These parameters can be selected to focus on certain distances from the central atom; for example, one may interested in capturing the presence of atoms at the distance of the first coordination shell around the central atom. Generally, a range of values for $ \eta $ and $ r_s $ are used to capture the local atomic environment at different scales. Since the Gaussian terms are multiplied by the cutoff function, the contribution of atoms that are farther away from the central atom is reduced.

**Angular Symmetry Function ($G_3$)**

The angular symmetry function captures the local atomic environment's angular distribution:

## Constructing the AEVs 

```python
import torch

```

## Designing Neural Networks for Potential Energy Prediction 

## References
1. Atom-centered symmetry functions for constructing high-dimensional
neural network potentials (2011) THE JOURNAL OF CHEMICAL PHYSICS 134, 074106 (2011)