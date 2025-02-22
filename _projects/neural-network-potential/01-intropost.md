---
layout: single
title: "Neural-Network Potentials - Part 1" 
permalink: /projects/neural-network-potential/part-1/
# classes: wide
use_math: true
toc: true
toc_sticky: true
---

# Introduction

This post covers the main ideas behind **one** type of neural-network potential (NNP) in molecular simulations called the **Behler-Parrinello NNP**. I cover the basic premise behind the approach, the model architecture, how feature vectors are created, and also, apply the workflow to train a NNP for water. 

While there are existing workflows that can automate the construction of NNPs, those approaches abstract away a lot from the user. Constructing our own model should help us understand what exactly is happening under the hood and could be useful for debugging and troubleshooting when using automated workflows.

<!-- {: .notice--info}
**Note:** Some points will expanded in future posts.  -->

# Why are Neural-Network Potentials Needed?

Molecular simulations are a powerful technique for studying atomistic- and molecular-scale phenomena at a spatial and temporal resolutions that can be difficult for experiments. One technique, called molecular dynamics, allows us to simulate the dynamic behavior of atoms and molecules by numerically integrating Newton's equations of motions. 

Here, the forces between atoms are calculated using a potential energy function that describes the interactions between atoms.

$$
\mathbf{F} = -\nabla V
$$

Molecular dynamics simulations have been used to study a plethora of interesting phenomena such as phase transitions, conformational changes of biomolecules (e.g., protein folding), and even portions of viruses (e.g., SARS-CoV-2 spike protein).

While molecular dynamics simulations are powerful, a big challenge is the trade-off between the accuracy of the simulations and the computational cost. Generally, to enable the simulation of large systems or of systems for long timescales, approximations are made in the representation of the potential energy surface. These empirical potentials are based on simple functional forms and are parameterized to reproduce experimental data or *ab initio* calculations. Furthermore, many variants of empirical potentials exist that are tailored to specific systems or phenomena. On the other hand, more accurate *ab initio* simulations are expensive and are limited to small systems. 

A large focus has been on developing potentials that can bridge the gap between empirical potentials and *ab initio* simulations. 

## Potential Energy Surfaces

Before we jump to where neural-network potentials fall into this picture, we first take a step back. We can view the potential energy of the system as a function which maps atomic coordinates to a scalar value.

$$
V: \mathbb{R}^{3N} \rightarrow \mathbb{R}
$$ 

where $N$ is the number of atoms in the system.

Empirical potentials approximate this mapping using a intuitive functional form with a set of parameters. Specifically, they view the potential energy of the system as a sum of bonded and non-bonded interactions. 

$$
V = V_{\text{bonded}} + V_{\text{non-bonded}}
$$ 

The bonded interactions consist of harmonic potentials for bonds and angles with torsional potentials for dihedrals. The non-bonded interactions are modeled using Lennard-Jones and Coulombic potentials.

$$
V = V_{\text{bond}} + V_{\text{angle}} + V_{\text{dihedral}} + V_{\text{LJ}} + V_{\text{Coulomb}}
$$

$$
V_{\text{bond}} = \sum_{\text{bonds}} k_b (r - r_{eq})^2
\\
\\
V_{\text{angle}} = \sum_{\text{angles}} k_{\theta} (\theta - \theta_{eq})^2
\\
V_{\text{dihedral}} = \sum_{\text{dihedrals}} k_{\phi} (1 + \cos(n \phi - \delta))
\\
V_{\text{LJ}} = \sum_{i < j} 4 \epsilon \left[ \left( \frac{\sigma}{r_{ij}} \right)^{12} - \left( \frac{\sigma}{r_{ij}} \right)^6 \right]
\\ 
V_{\text{Coulomb}} = \sum_{i < j} \frac{q_i q_j}{4 \pi \epsilon_0 r_{ij}}
$$

The parameters ($k_b$, $r_{eq}$, $\theta_{eq}$, $k_{\theta}$, $k_{\phi}$, $\epsilon$, $\sigma$, $q_i$, $q_j$) are determined by fitting the potential energy function to experimental data or *ab initio* calculations.

We see that the main objective is to find a mapping between the atomic coordinates and the potential energy of the system, if we could learn this mapping with parametric or non-parametric models, we could potentially have a more accurate representation of the **true** potential energy surface. we could avoid the approximations made in empirical potentials. 
NNPs tackle this problem by learning the potential energy function using neural networks. 

we could learn this mapping with machine learning techniques if we have sufficient number of atomic configurations and their corresponding potential energies. Neural-network potentials fit into this picture by learning the potential energy function using neural networks.

# Neural-Network Potentials (NNPs)

 
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

First, we consider the cutoff function which is for 2-body interactions. The cutoff function is used to weigh pair contributions based on the Euclidean distance between the particles. The function smoothly decay to 0 at the cutoff distance. Two commonly used cutoff functions are:

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

where $r_{ij}$ is the pairwise distance between particle $i$ and particle $j$, and $r_c$ is the cutoff distance. 

{: .notice--info}
**Note:**
Other cutoff functions can be used as well. For example, the [n2p2](https://compphysvienna.github.io/n2p2/) package which helps automate the training of neural network potentials using the BP symmetry functions has multiple [cutoff functions](https://compphysvienna.github.io/n2p2/api/cutoff_functions.html) to choose from.


In accordance to reference \[1\](), the first ACSF is a sum of the cutoff functions for the neighbors of the central atom \(i\):

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

The angular symmetry function captures the local atomic environment's 3-body interactions. The function is defined as:

$$
G_3^i = 2^{1 - \zeta} \sum_{j \neq i} \sum_{k \neq i,j} (1 + \lambda \cos \theta_{ijk})^\zeta e^{-\eta (r_{ij}^2 + r_{ik}^2 + r_{jk}^2)} f_c(r_{ij}) f_c(r_{ik}) f_c(r_{jk})
$$

This function has a new variable, $ \theta_{ijk} $, which is the angle between the three atoms with the central atom $ i $ at the vertex.

The remaining new parameters are $ \zeta $ and $ \lambda $, and $ \eta $. The values of $ \zeta $ are limited to {-1, 1} and controls the importance placed on acute and obtuse angles, respectively. Similar to $G_1$, the $ \eta $ parameter controls the width of the Gaussian term.

The double sum ensures that all possible combinations of neighbors are considered. The inclusion of the cutoff functions ($f_c$) ensure that if any of the pairwise distances are greater than the cutoff distance, the contribution of the 3-body term is zero.

## Constructing the AEVs 

```python
import torch

```

## Designing Neural Networks for Potential Energy Prediction 

## References
1. Atom-centered symmetry functions for constructing high-dimensional
neural network potentials (2011) THE JOURNAL OF CHEMICAL PHYSICS 134, 074106 (2011)