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
system_file_name = 'output_mm_waters_run_01'
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

# Create a numpy array to store the important quantites every 50 steps
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
For clarity, I have excluded code that would write out the trajectory to file and the creation of periodic checkpoint files. The key points to note are:
1. The water box system is generated using the `openmmtools.testsystems.WaterBox` class with default parameters. This will create a cubic box with a box edge of 25 Ångstroms (~500 water molecules).
1. We store the forces, positions, energies, box vectors, and elements of the system. The forces and positions are at each time step are of shape `(num_particles, 3)`.
2. We run an initial NPT equilibration run followed by an NVT production run. The configurations are stored from the NVT run. 

# 



# Building an ANI potential

# Using the ANI potential in a molecular dynamics simulation

# Conclusion