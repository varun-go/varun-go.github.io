---
layout: single
title: "Neural-Network Potentials - Part 2" 
permalink: /projects/neural-network-potential/part-2/
classes: wide
use_math: true
toc: true
toc_sticky: true
---

# Introduction

In the previous post, we covered the main idea behind the **Behler-Parrinello Neural-Network Potential (NNP)**. We saw the how the NNP can be used to approximate the potential energy surface, one scheme for generating feature vectors (e.g., the **Behler-Parinello symmetry functions**), and the parameters associated with these symmetry functions which allows one to *capture* the chemical environment of an atom/particle.

In this post, we will cover how one may go about building an NNP with an emphasis on understanding the implementation details. Our goal is to build a NNP that can be used in a molecular dynamics simulation.

Our procedure for buidling a NNP will involve the following steps:
1. Generate training data
2. Generate feature vectors
3. Build the neural network
4. Train the model
5. Use the neural network in a molecular dynamics simulation

I will provide code snippets in Python to illustrate the implementation details. I will make a note of all the libraries that are used in the code snippets as they come up.

# Generating training data

Remembering that a NNP is used to approximate a potential energy surface, we need to sample points on this energy surface; our training data for the model will consist of a set of atomic configurations and their corresponding potential energy values. Generally, these energies would be calculated using a high-fidelity method such as density functional theory (DFT), allowing us to use the speed in inference of the NNP to calculate the potential energy of unseen data. To get benefits of generating a NNP, the accuracy of the training should be greater than that of a molecular mechanics force field. However, since this post is for illustrative purposes, we will simplify the problem. Specifically, training data will be generated from molecular mechanics force field and we will train a NNP to reproduce the force field.

To generate the training data, we run a molecular dynamics (MD) simulation of water molecules using a molecular mechanics force field. We will use the `OpenMM` library to run the simulation. `OpenMM` is an open-source MD simulation library that is designed for high performance simulations on CPUs and GPUs. It has a Python API that allows for easy integration with other Python libraries such as [Plumed](https://www.plumed.org) ([OpenMM-Plumed](https://github.com/openmm/openmm-plumed)) for doing enhanced-sampling simulations and [PyTorch](https://pytorch.org) ([OpenMM-Torch](https://github.com/openmm/openmm-torch)) to interface with neural networks.

Below is the Python script we will use to generate the training data. The script will run a MD simulation of water molecules using the TIP3P water model. To create the system, we will use the [OpenMMTools](https://openmmtools.readthedocs.io/en/stable/) library which provides a set of easy-to-use tools for setting up MD simulations. 

```python
import os
from openmm.app import *
from openmm import *
from openmm.unit import *
import openmmtools
import numpy as np

# generate the water box system w/ default parameters
water = openmmtools.testsystems.WaterBox(model='tip3p', 
box_edge=25*angstroms) 
# export the structure to a pdb file
system_file_name = 'output_mm_waters_nvt_run_01'
PDBFile.writeFile(water.topology, water.positions, open(f'{system_file_name}.pdb', 'w'))

# use the pdb file to create a modeller object
water_box = PDBFile(f'{system_file_name}.pdb')
modeller = Modeller(water_box.topology, water_box.positions)
system = water.system
num_particles = system.getNumParticles()

# add a CMMotionRemover to remove the center of mass motion every 20 steps
system.addForce(CMMotionRemover(20)) 
# Create an integrator with a time step of 2 fs
integrator = NoseHooverIntegrator(
                            300*kelvin,       # Temperature of heat bath
                            1.0/picoseconds,  # Friction coefficient
                            2*femtoseconds,  # Time step
                            )
# add a barostat to control the pressure
barostat = MonteCarloBarostat(1*bar, 300*kelvin, 10)
system.addForce(barostat)

# identify the platform to run the simulation 
platform = Platform.getPlatformByName('CUDA')

# Create a simulation context
simulation = Simulation(
  modeller.topology,
  system,
  integrator,
  platform)
simulation.context.setPeriodicBoxVectors(*modeller.topology.getPeriodicBoxVectors())

# set initial positions
simulation.context.setPositions(modeller.positions)

# Minimize energy
simulation.minimizeEnergy()

# set initial velocities
simulation.context.setVelocitiesToTemperature(300*kelvin)

# Run NPT equilibration
print('Running NPT equilibration!')
simulation.step(500000) # 1 ns

# Switch to NVT, remove the barostat
system.removeForce(system.getNumForces()-1)

nsteps = 2500000 # 5 ns
recording_frequency = 100

outfile = os.path.join(f'{system_file_name}.txt')
dcdfile = os.path.join(f'{system_file_name}.dcd')
checkpointfile = os.path.join(f'{system_file_name}.chk')

simulation.reporters.append(
        DCDReporter(
            dcdfile, 
            100))
simulation.reporters.append(
        StateDataReporter(
            outfile,
            100,
            time=True,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True, 
            temperature=True,
            volume=True,
            density=True))
simulation.reporters.append(
        CheckpointReporter(
            checkpointfile,
            10000))

# Create numpy arrays to store the important quantites every 50 steps
forces = np.zeros((nsteps // recording_frequency, num_particles, 3))
positions = np.zeros((nsteps // recording_frequency, num_particles, 3))
energy = np.zeros((nsteps // recording_frequency, 1))
box_vectors = np.zeros((nsteps // recording_frequency, 3, 3))
elements = [atom.element.symbol for atom in modeller.topology.atoms()]

print('Running NVT production!')
simulation.step(nsteps)

# record the state of the system periodically
for i in range(nsteps // recording_frequency):
    simulation.step(recording_frequency)
    state = simulation.context.getState(
        getForces=True, 
        getPositions=True, 
        enforcePeriodicBox=True, 
        getEnergy=True)

    # Store items
    forces[i] = state.getForces(asNumpy=True)
    positions[i] = state.getPositions(asNumpy=True)
    energy[i] = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    box_vectors[i] = state.getPeriodicBoxVectors(asNumpy=True)

# Save the forces to a file
np.save(f'forces_{system_file_name}.npy', forces)
np.save(f'positions_{system_file_name}.npy', positions)
np.save(f'energies_{system_file_name}.npy', energy)
np.save(f'box_vectors_{system_file_name}.npy', box_vectors)
np.save(f'elements_{system_file_name}.npy', elements)

# write out final frame
state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
# update topology with new correct vectors
simulation.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
PDBFile.writeFile(simulation.topology, state.getPositions(), open(f'{system_file_name}_final_frame.pdb', 'w'))

print('Finished production!')
```
The key points to note from this script are:
1. The water box system is generated using the `openmmtools.testsystems.WaterBox` class with default parameters. This will create a cubic box with a box edge of 25 Ångstroms (~500 water molecules).
1. We store the forces, positions, energies, box vectors, and elements of the system. The forces and positions are at each time step are of shape `(num_particles, 3)`.
2. We run an initial NPT equilibration run followed by an NVT production run. The configurations are stored from the NVT run. 

Now that we have sampled the PES associated with the TIP3P water model, we can use this data to train a NNP.

# Building the NNP potential

To build the NNP, we need to create both the set of feature vectors and the neural network model. We will use the [ANI](http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract) methodology to do both steps. ANI uses a variation of the Behler-Parinello symmetry functions to generate the feature vectors and a neural network model to approximate the potential energy surface (see the References at the bottom of the page for more details). We will use the `torchani` [library](https://aiqm.github.io/torchani/index.html) to build the NNP which is a PyTorch implementation of the ANI potential. 

For this section, I will break down the code into smaller snippets to explain the implementation details.

First, we load all the data from the MD simulation. We convert the units of the data to be compatible with `torchani` (e.g., Angstroms for distances and Hartree for energies). We also load the species of the atoms in the system.

```python
import numpy as np
import tqdm 
import math
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchani
from torchani.units import hartree2kcalmol

# load data from MM simulations
position_file = 'positions_output_mm_waters_nvt_run_01.npy'
forces_file = 'forces_output_mm_waters_nvt_run_01.npy'
energy_file = 'energies_output_mm_waters_nvt_run_01.npy'
box_vector_file = 'box_vectors_output_mm_waters_nvt_run_01.npy'
species_file = 'elements_output_mm_waters_nvt_run_01.npy'

# load the data
positions = np.load(position_file)
positions = torch.tensor(positions) * 10 # convert from nm to Angstrom, since torchani uses Angstrom

energies = np.load(energy_file)
energies = torch.tensor(energies)
energies = torchani.units.kjoulemol2hartree(energies) # convert to Hartree, since torchani uses Hartree

forces = np.load(forces_file)
forces = torch.tensor(forces) / 10 # convert from 1/nm to 1/Angstrom
forces = torchani.units.kjoulemol2hartree(forces) # convert to Hartree, H/Å

box_vectors = np.load(box_vector_file)
box_vectors = torch.tensor(box_vectors) * 10 # convert from nm to Angstrom
species = np.load(species_file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Next, we use the `torchani` library to generate the feature vectors using the 
AEVComputer class. The AEVComputer class is a `torch.nn.Module` that computes the atomic environment vectors (AEVs). An instance of the AEVComputer class is created with the parameters of the symmetry functions. Since the AEVComputer is a `torch.nn.Module`, it can be backpropagated through during training. This detail is important as it allows us to compute derivatives with respect to the inputs of this transformation which are the atomic positions!

Below is the code snippet that creates the AEVComputer instance. Here, we use the default parameters for the symmetry functions from the TorchANI [documentation](https://aiqm.github.io/torchani/examples/nnp_training.html).

```python
# first, we set up the parameters of the symmetry functions
Rcr = 5.2000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([5.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)

species_order = ['H', 'O']
num_species = len(species_order)
aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
energy_shifter = torchani.utils.EnergyShifter(None)
```
In the code above, we also creates specify the species of the atoms in the system. This order is important as it dictates the order of the atomic neural networks which are used to compute the atomic energies. 

Next, 


# Using the ANI potential in a molecular dynamics simulation

# Conclusion

# References
**ANI potential papers**
