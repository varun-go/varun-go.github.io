---
layout: single
title: "Neural-Network Potentials - Part 1" 
permalink: /projects/neural-network-potential/part-1/
classes: wide
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

## Main Idea

As stated in the previous section, NNPs aim to approximate the potential energy surface using neural networks. The main idea is to learn the mapping between the atomic coordinates and the potential energy of the system. How can we do this? 

The idea introduced by Behler and Parrinello in their seminal [paper](http://dx.doi.org/10.1103/PhysRevLett.98.146401) from 2007 is to decompose the potential energy of a configuration into atomic/site contributions:

$$
V(\mathbf{R}) = \sum_{i=1}^{N} E_i
$$ 
where $E_i$ is the energy contribution of atom/site $i$ in the system and $N$ is the number of atoms/sites in the system. 

{: .notice--info}
> **Note:** From now on, I will only refer to atomic contributions, but the same concepts can be applied to site contributions as well.

A model is used to estimate the atomic energy. In the Behler-Parrinello NNP, the model is a feedforward neural network that takes as input a feature vector that describes the local atomic environment around a specific atom. We will refer to this feature vector as the atomic environment vector (AEV), such that $AEV_{i}$ is the feature vector for atom $i$ in the system. Thus, the atomic energy is predicted as:

$$
E_i = f_{\theta_i}(AEV_i)
$$
    
where $f_{\theta_i}$ is the neural network with parameters $\theta$. We can initially think of there being a separate neural network for each atom in the system. This network will predict the atomic energy for the atom based on its local atomic environment. However, to ensure that the system is invariant to the permutation of atoms, the atomic energies are predicted using the same neural network. To understand this point, consider a single configuration of a single-component system. If two particle labels are permuted, the potential energy of the system should remain the same. To ensure this invariance, a single network is used to predict the atomic energies. Thus, the workflow would look like this:

<!--
insert figure showing the neural network architecture
-->

The learning process involves training the neural network to minimize the difference between the sum of the predicted atomic energies and the *true* configuration energy. The *true* atomic energies can obtained from any source (e.g., *ab initio* calculations, semi-empirical methods, etc.). The key point is that the neural network is trained to predict the atomic energies that sum up to the configuration energy (i.e., the label value). The loss function can be constructed as sum of squared differences between the predicted atomic energies and the *true* configuration energy:

$$
\mathcal{L} = \sum_{l=1}^{L}[E_{l,pred}- E_{l, true}]^2 \\
            = \sum_{l=1}^{L}[\sum_{i=1}^{N} (E_i)) - E_{l,true}]^2 \\
$$

where $L$ is the number of training configurations and $E_{l,true}$ is the *reference* configuration energy for the $l$-th configuration. 

A trained model should be able to predict the atomic energy contribution based on its local atomic environment and the sum of the atomic energies should be close to the configuration energy.

If differentiable activation functions are used in the neural network, the forces acting on the atoms can be calculated by taking the gradient of the potential energy with respect to the atomic coordinates. Thus, the NNPs can be used to perform molecular dynamics simulations.

## Single-component vs Multi-component Systems

For a single-component system, a singular neural network can be used to predict the atomic energies. However, for multi-component systems, the atomic energies are predicted using multiple neural networks. For example, consider a system such as water, where there are two types of atoms (hydrogen and oxygen). In this case, two neural networks are used to predict the atomic energies. The oxygen network is used to predict the atomic energy of oxygen atoms, and the hydrogen network is used to predict the atomic energy of hydrogen atoms. The predicted atomic energies are summed up to obtain the configuration energy.

$$ 
E_{\text{config}} = \sum_{i=1}^{N_{\text{O}}} E_{\text{O}_i} + \sum_{j=1}^{N_{\text{H}}} E_{\text{H}_i} 

$$

where $N_{\text{O}}$ and $N_{\text{H}}$ are the number of oxygen and hydrogen atoms in the system, respectively.

Pictorially, this workflow would look like this:


Now, let's dive into the details of the feature vectors used to describe the local atomic environments.

<!-- 
Notes:
1. Add that we are looking at the flavor of NNPs which estimate that the potential energy of a system can be decomposed into atomic (or site) contributions.

-->

# Local descriptors for atomic environments

<!-- 
Notes:
1. Add information about other descriptors such as SOAP 
2. 
-->

The atomic environment vector (AEV) is a feature vector that describes the local atomic environment around a specific atom. There are different methods for constructing the AEV, but we will focus on the method used in Behler-Parrinello NNP. These networks 
use atom-centered symmetry functions (ACSFs) to construct the AEV. The ACSFs have a set of parameters that capture different angular and radial information about the local atomic environment. The AEVs are constructed by combining the values of many of these functions values. 

## Atom-centered symmetry functions (ACSFs)

These functions are used to describe the local atomic environment around a central atom. These functions use internal coordinate representations of the system by using pairwise distances and angular information. As a result, they are translationally and rotationally invariant. This aspect is important as it helps reduce the complexity of the neural network.

**Cutoff Function ($G_1$)**

First, we consider the cutoff function which is for 2-body interactions. The cutoff function is used to weigh pair contributions based on the Euclidean distance between a central particle and its neigbhors. The function smoothly decays to 0 at the cutoff distance. Two commonly used cutoff functions are:

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

{% include figure 
  image_path="/assets/images/nnp_post_1/cutoff_functions.svg" 
  alt="A graph showing the two classes of cutoff functions mentioned in the text."
  caption="Visualization of two classes of cutoff function for a cutoff distance of 12 Bohr (~6.3 Angstroms)."
  width="25%" 
  height="25%"
  title="Tooltip text when hovering"
  %}

The figure shows that the functions monotically decrease to zero at the cutoff distance. Both functions are continuous and differentiable, which is important for training the neural network.

The first ACSF is a sum of the cutoff functions for the neighbors of the central atom \(i\):

$$
G_1^i = \sum_{j \neq i} f_c(r_{ij})
$$

**Radial Symmetry Function ($G_2$)**

The radial symmetry function captures the local atomic density:

$$
G_2^i = \sum_{j \neq i} e^{-\eta (r_{ij} - r_s)^2} f_c(r_{ij})
$$

where $ \eta $ and $ r_s $ are parameters that alter the width and position of the Gaussian, respectively. These parameters can be selected to focus on certain distances from the central atom; for example, one may interested in capturing the presence of atoms at the distance of the first coordination shell around the central atom. Generally, a range of values for $ \eta $ and $ r_s $ are used to capture the local atomic environment at different scales and each of these parameter values would contribute another row to the AEV vector. Since the Gaussian terms are multiplied by the cutoff function, the contribution of atoms that are farther away from the central atom is reduced.

{% include figure 
image_path="/assets/images/nnp_post_1/radial_symmetry_functions_fig1.svg" 
alt=" "
caption="$G_2$ symmetry function values for different $\eta$ values (units of Bohr$^{-2}$) when $R_s = 0$ considering only a single neighbor particle. The cutoff distance is 12 Bohr (~6.3 Angstroms)." 
width="25%" 
height="25%"
title="Tooltip text when hovering"
%}

The figure above shows that smaller values of $ \eta $ result in a wider Gaussian term, allowing neighbors at larger pairwise distances to be captured. On the other hand, larger values of $ \eta $ result in a narrower Gaussian term, localizing the contribution of the neighbor particles to a smaller range. The cutoff function ensures that the contributions decay to zero if the pairwise distance is greater than the cutoff distance.

The figure below shows the effect of $R_s$ on the $G_2$ symmetry function. The values of $R_s$ are selected to capture the local atomic environment at different scales. As $R_s$ increases, the emphasis is on larger pairwise distnaces thought the effect is diminished by the cutoff function.

{% include figure 
image_path="/assets/images/nnp_post_1/radial_symmetry_functions_fig2.svg" 
alt=" "
caption="$G_2$ symmetry function values for different $R_s$ values (units of Bohr) for a fixed $\eta$ value of 0.5 Bohr$^-2$ when considering only a single neighbor particle. The cutoff distance is 12 Bohr (~6.3 Angstroms)." 
width="25%" 
height="25%"
title="Tooltip text when hovering"
%}

**Angular Symmetry Function ($G_4$)**

{: .notice--info}
**Note:**
I have intentionally skipped the $G_3$ symmetry function as it is not used in the Behler-Parrinello NNP that we will be constructing later in the post. There is also a $G_5$ symmetry function which we will also not cover. More details about these functions can be found in the follwing [paper](https://doi.org/10.1063/1.3553717).

The angular symmetry function captures the local atomic environment's 3-body interactions. The function is defined as:

$$
G_4^i = 2^{1 - \zeta} \sum_{j \neq i} \sum_{k \neq i,j} (1 + \lambda \cos \theta_{ijk})^\zeta e^{-\eta (r_{ij}^2 + r_{ik}^2 + r_{jk}^2)} f_c(r_{ij}) f_c(r_{ik}) f_c(r_{jk})
$$

This function has a new variable, $ \theta_{ijk} $, which is the angle between the three atoms with the central atom $ i $ at the vertex.

The remaining new parameters are $ \zeta $ and $ \lambda $, and $ \eta $. The values of $ \zeta $ are limited to {-1, 1} and controls the importance placed on acute and obtuse angles, respectively. Similar to $G_1$, the $ \eta $ parameter controls the width of the Gaussian term.

The double sum ensures that all possible combinations of neighbors are considered. The inclusion of the cutoff functions ($f_c$) ensure that if any of the pairwise distances are greater than the cutoff distance, the contribution of the 3-body term is zero.

The two figures below reflect the effect of the $\lambda$ and $\zeta$ parameters on the $G_4$ symmetry function.

{% include figure 
image_path="/assets/images/nnp_post_1/angular_symmetry_functions_fig1.svg" 
alt=" "
caption="$G_4$ symmetry function values for different $\theta_{ijk}$ values. The values are shown for $\lambda = 1$ which emphasizes the importance of acute angles. $\eta = 1$ and all pair distances ($r_{ij}$, $r_{ik}$, $r_{jk}$) are set equal to each other at 0.5 Bohr to isolate the effect of the $\theta_{ijk}$ term. The cutoff distance is 12 Bohr (~6.3 Angstroms)."
width="25%" 
height="25%"
title=""
%}

{% include figure 
image_path="/assets/images/nnp_post_1/angular_symmetry_functions_fig2.svg" 
alt=" "
caption="$G_4$ symmetry function values for different $R_s$ values (units of Bohr) for a fixed $\eta$ value of 0.5 $Bohr^-2$ when considering only a single neighbor particle. The cutoff distance is 12 Bohr (~6.3 Angstroms)." 
width="25%" 
height="25%"
title=""
%}

Since different parameters values of the ACSFs can highlight specific areas of the local atomic environment, multiple parameters values for the same symmetry functions can be used. The AEV vector consists of the values of the ACSFs with different parameter values. For exmaple, the AEV vector for a single atom could look like this:

$$
AEV_i =
\begin{bmatrix}
G_1^i \\
G_2^i(\eta_1, R_{s1}) \\
G_2^i(\eta_2, R_{s2}) \\
G_4^i(\zeta_1, \lambda_1, \eta_1) \\
G_4^i(\zeta_2, \lambda_2, \eta_2) \\
\end{bmatrix}
$$


## Designing Neural Networks for Potential Energy Prediction 

Now, with the basics of the Behler-Parrinello NNP covered, we will not move on to constructing a NNP for water. We will use the the following paper as a reference. The dataset for this paper is available [here]().




## References
1. Atom-centered symmetry functions for constructing high-dimensional
neural network potentials (2011) THE JOURNAL OF CHEMICAL PHYSICS 134, 074106 (2011)