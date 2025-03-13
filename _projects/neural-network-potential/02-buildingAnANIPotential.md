---
layout: single
title: "Building a Neural-Network Potential using PyTorch" 
permalink: /projects/neural-network-potential/part-2/
classes: wide
use_math: true
toc: true
toc_sticky: true
---

# Introduction

In the previous post, we covered the main idea behind the **Behler-Parrinello Neural-Network Potential (NNP)**. We saw the how the NNP can be used to approximate the potential energy surface, one scheme for generating feature vectors (e.g., the **Behler-Parrinello symmetry functions**), and the parameters associated with these symmetry functions which allows one to *capture* the chemical environment of an atom/particle.

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

To build the NNP, we need to create both the set of feature vectors and the neural network model. We will use the [ANI](http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract) methodology to do both steps. ANI uses a variation of the Behler-Parrinello symmetry functions to generate the feature vectors and a neural network model to approximate the potential energy surface (see equation 4 in the ANI [paper](http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract) for more details). We will use the `torchani` [library](https://aiqm.github.io/torchani/index.html) to build the NNP which is a PyTorch implementation of the ANI potential. 

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

Next, we create training and validation datasets using the `torch.utils.data.Dataset` class. This class can store the data in a format that can be used by the `torch.utils.data.DataLoader` class, which can load the data in batches during training. 

```python
class ANI_Dataset(Dataset):
    def __init__(self,
                 positions: torch.Tensor,
                 energies: torch.Tensor,
                 forces: torch.Tensor,
                 box_vectors: torch.Tensor,
                 species: torch.Tensor,
                 device='cpu'):
        ''' Dataset class for the ANI model '''
        self.device = device
        self.coordinates = positions
        self.energies = energies
        self.forces = forces
        self.box_vectors = box_vectors
        self.species = species

    def __len__(self):
        return len(self.species)  # Number of molecules

    def __getitem__(self, idx):
        return {
            'species': self.species[idx].to(self.device),
            'coordinates': self.coordinates[idx].to(self.device),
            'energies': self.energies[idx].to(self.device),
            'box_vectors': self.box_vectors[idx].to(self.device),
            'forces': self.forces[idx].to(self.device),  
        }
# update species vector based on species_order
species = [species_order.index(i) for i in species]
# now copy species vector along the last dimension, applying to all frames
species = np.broadcast_to(species, positions.shape[:-1])
species = torch.tensor(species, device=device)
# create the dataset
dataset = ANI_Dataset(positions, energies, forces, box_vectors, species, device=device)
# save the dataset to a file, if needed for later use
torch.save(dataset, 'mm_water_dataset.pt')

# split into training and validation sets
# 80-20 train-validation split
train_data, val_data = random_split(
    dataset, 
    [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

# Create DataLoader objects for batching
training = DataLoader(train_data, batch_size=10, collate_fn=torchani.data.collate_fn, shuffle=True)
validation = DataLoader(val_data, batch_size=10, collate_fn=torchani.data.collate_fn)
```

Next, we create the atomic neural networks using the `torch.nn.Sequential` class. The atomic neural networks are feedforward neural networks that takes as input the AEVs and outputs the atomic energies. 

```python
# define atomic neural networks.
aev_dim = aev_computer.aev_length

# ANI default architecture
H_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)
O_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)
atomic_neural_nets = torchani.ANIModel([H_network, O_network])

# use the KaiMing initialization for the weights and zero initialization for the biases
def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)

atomic_neural_nets.apply(init_params)
```
The order of the atomic neural networks should match the order of the species in the system. The architecture of the atomic neural networks can be optimized, but the default ANI architecture is a good starting point.

Now, we can combine the AEVComputer with the atomic neural networks to create the neural network model. The model will take as input the atomic positions and species and output the total energy of the system.

```python 
# Let's now create a pipeline of AEV Computer --> Neural Networks.
model = torchani.nn.Sequential(aev_computer, atomic_neural_nets).to(device)
```

Since AEVComputer is a `torch.nn.Module`, it can be used in the forward pass of the model. This idea of making featurization a part of the model is very powerful as it allows for end-to-end training of the model; this idea could be used in many different applications in chemistry and materials science.

Next, we initialize the optimizer. The default ANI approach uses separate weight decay for the weights and the biases of the model. An Adam optimizer is used for the weights and a SGD optimizer for the biases. More details are in the [TorchANI documentation](https://aiqm.github.io/torchani/examples/nnp_training.html).

```python

, we create the neural network model that combines the atomic neural networks with the AEVComputer and the EnergyShifter. The model is then trained using the training and validation datasets.

```python

AdamW = torch.optim.AdamW([
    # H networks
    {'params': [H_network[0].weight]},
    {'params': [H_network[2].weight], 'weight_decay': 0.00001},
    {'params': [H_network[4].weight], 'weight_decay': 0.000001},
    {'params': [H_network[6].weight]},
    # O networks
    {'params': [O_network[0].weight]},
    {'params': [O_network[2].weight], 'weight_decay': 0.00001},
    {'params': [O_network[4].weight], 'weight_decay': 0.000001},
    {'params': [O_network[6].weight]},
])
SGD = torch.optim.SGD([
    # H networks
    {'params': [H_network[0].bias]},
    {'params': [H_network[2].bias]},
    {'params': [H_network[4].bias]},
    {'params': [H_network[6].bias]},
    # O networks
    {'params': [O_network[0].bias]},
    {'params': [O_network[2].bias]},
    {'params': [O_network[4].bias]},
    {'params': [O_network[6].bias]},
], lr=1e-3)
###############################################################################
# Setting up a learning rate scheduler to do learning rate decay
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)
```

Finally, we train the model using the training and validation datasets. The training loop is shown below.

```python
# Train the model by minimizing the MSE loss, until validation RMSE no longer
# improves during a certain number of steps, decay the learning rate and repeat
# the same process, stop until the learning rate is smaller than a threshold.
#
# We first read the checkpoint files to restart training. We use `latest.pt`
# to store current training state.
latest_checkpoint = 'latest.pt'

def validate():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    model.train(False)
    with torch.no_grad():
        for properties in validation:
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device).float()
            box_vectors = properties['box_vectors'][0].to(device).float()
            pbc = torch.tensor([True, True, True], device=device)
            _, predicted_energies = model((species, coordinates), pbc = pbc, cell = box_vectors)
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]
    model.train(True)
    
    return hartree2kcalmol(math.sqrt(total_mse / count))

###############################################################################
# In this tutorial, we are setting the maximum epoch to a very small number,
# only to make this demo terminate fast. For serious training, this should be
# set to a much larger value
mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
max_epochs = 10
early_stopping_learning_rate = 1.0E-5
best_model_checkpoint = 'best.pt'
force_coefficient = 0.1  # controls the importance of energy loss vs force loss

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    rmse = validate()
    print('RMSE (kcal/mol) at epoch', AdamW_scheduler.last_epoch + 1, ':', rmse)

    learning_rate = AdamW.param_groups[0]['lr']

    if learning_rate < early_stopping_learning_rate:
        break

    # checkpoint
    if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(atomic_neural_nets.state_dict(), 
                   best_model_checkpoint)

    AdamW_scheduler.step(rmse)
    SGD_scheduler.step(rmse)
    for i, properties in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
        true_energies = properties['energies_kcalmol'].to(device).float()
        box_vectors = properties['box_vectors'][0].to(device).float()
        true_forces = properties['forces_kcalmol'].to(device).float()
        
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        pbc = torch.tensor([True, True, True], device=device)
#         _, predicted_energies = model((species, coordinates))
        _, predicted_energies = model((species, coordinates), pbc = pbc, cell = box_vectors)

        # We can use torch.autograd.grad to compute force. Remember to
        # create graph so that the loss of the force can contribute to
        # the gradient of parameters, and also to retain graph so that
        # we can backward through it a second time when computing gradient
        # w.r.t. parameters.
        forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]

        #######################################################################
        # # LOSS FUNCTION: energy loss + force loss
        # # Now the total loss has two parts, energy loss and force loss
        # energy_loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
        # force_loss = (mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms).mean()
        # loss = energy_loss + force_coefficient * force_loss
        #######################################################################
        # LOSS FUNCTION: energy loss only
        loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
        #######################################################################

        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()
        AdamW.step()
        SGD.step()

    torch.save({
        'nn': atomic_neural_nets.state_dict(),
        'AdamW': AdamW.state_dict(),
        'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)
```
Two key points to note from the training loop are:
1. The forward pass of the model uses two keyword arguments `pbc` and `cell` to account for periodic boundary conditions. 
2. Two loss functions are shown in the code snippet. The first loss function includes both the energy loss and the force loss. The second loss function only includes the energy loss. The force loss is computed using the `torch.autograd.grad` function which computes the gradient of the predicted energies with respect to the atomic positions -- which is easily done since the AEVComputer is a `torch.nn.Module`. In theory, one could also incorporate the virial tensor in the loss function; this approach is taken in the Deep Potential NNP [approach](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.143001).

Great! We have setup a NNP using the ANI methodology. The trained model can be used to predict the potential energy of unseen data and can be used in a molecular dynamics simulation.

# Using the ANI potential in a molecular dynamics simulation
As stated previously, OpenMM is a powerful library and one aspect that adds this power is the availability of plugins on top of the core library. One such plugin is the [OpenMM-Torch](https://github.com/openmm/openmm-torch) plugin which allows for the integration of PyTorch neural networks with OpenMM. We show how this plugin can be used to integrate the ANI potential with OpenMM. This workflow is based on an [example](https://github.com/openmm/openmm-torch/blob/master/tutorials/openmm-torch-nnpops.ipynb) for using the pre-trained ANI-2x model in OpenMM.

```python
import os
from openmm.app import *
from openmm import *
from openmm.unit import *
import openmmtools
import numpy as np
import torch as torch
import torchani
from openmmtorch import TorchForce

# generate the water box system
water = openmmtools.testsystems.WaterBox(model='tip3p')
system_file_name = 'output_ml_waters_nvt_run_01'

# use the pdb file to create a modeller object
water_box = PDBFile(f'output_ml_waters_nvt_run_01.pdb')
modeller = Modeller(water_box.topology, water_box.positions)

system = water.system

# Remove MM constraints
while system.getNumConstraints() > 0:
  system.removeConstraint(0)

# Remove MM forces
while system.getNumForces() > 0:
  system.removeForce(0)

# The system should not contain any additional force and constrains
assert system.getNumConstraints() == 0
assert system.getNumForces() == 0

# Get the list of atomic numbers
atomic_symbols = [atom.element.symbol for atom in water.topology.atoms()]
atomic_numbers =[atom.element.atomic_number for atom in water.topology.atoms()]
num_particles = system.getNumParticles()

##############################################
# DEFINING THE ANI MODEL
##############################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# order of species in the model, i.e., order of atomic NNPs
nnp_species_order = ['H', 'O']
num_species = len(nnp_species_order)

# setting up the AEV computer
Rcr = 5.2000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([5.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)

aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
energy_shifter = torchani.utils.EnergyShifter(None)

# define atomic neural networks.
aev_dim = aev_computer.aev_length
H_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)
O_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)
atomic_neural_nets = torchani.ANIModel([H_network, O_network])
atomic_neural_nets.load_state_dict(torch.load('tip3p_nnp_model.pt')) # load parameters from a pre-trained model

# creating a pipeline of species AEV Computer --> Neural Networks.
custom_ani_model = torchani.nn.Sequential(aev_computer, atomic_neural_nets)
####################################################

# now, create a tensor for the input species based on the species order in the atomic neural networks
# this tensor is fixed, the species order should not change
species = torch.tensor([nnp_species_order.index(s) for s in atomic_symbols], device=device)

# now create a wrapper class for the ANI model to integrate with OpenMM
class NNP(torch.nn.Module):
  '''
    A torch module wrapper for the ANI model
  '''
  def __init__(self, model, species):
    super().__init__()

    # store custom ANI model
    self.model = model
    self.species = species.unsqueeze(0) # add a batch dimension

  def forward(self, positions, boxvectors):

    # tensor of booleans for periodic boundary conditions
    pbc = torch.tensor(([True,True,True]))

    # Prepare the positions
    positions = positions.unsqueeze(0).float() * 10 # nm --> Å
    boxvectors = boxvectors * 10 # nm --> Å
    # inferential forward pass
    result = self.model((self.species, positions),
                        cell=boxvectors, 
                        pbc=pbc.to(device='cuda'))
    
    # Get the potential energy
    energy = result.energies[0] * 2625.5 # Hartree --> kJ/mol
    return energy

# Create an instance of the model
nnp = NNP(custom_ani_model, species)

# Save the NNP to a file and load it with OpenMM-Torch
torch.jit.script(nnp).save('tip3p_nnp_model_openmm.pt')
force = TorchForce('tip3p_nnp_model_openmm.pt')
force.setUsesPeriodicBoundaryConditions(True)  # by default, PBC is False

# Add the NNP to the system
system.addForce(force)
assert system.getNumForces() == 1
# add a CMMotionRemover to remove the center of mass motion
system.addForce(CMMotionRemover(20))

# Create an integrator with a time step of 0.25 fs
integrator = NoseHooverIntegrator(
                            300*kelvin,       # Temperature of heat bath
                            1.0/picoseconds,  # Friction coefficient
                            0.25*femtoseconds, # Time step
                            )
platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(
  modeller.topology,
  system,
  integrator,
  platform)
simulation.context.setPeriodicBoxVectors(*modeller.topology.getPeriodicBoxVectors())
simulation.context.setPositions(modeller.positions)

# run a short equilibration
print('Running equilibration!')
simulation.step(250000) # 50 ps

# remaining code for creating reporters and running the simulation
# ...
```

In the code snippet above, we first create the `torch.nn.Module` class for the ANI model. This requires us to specify the AEVConverter and the atomic NNPs. We load the pre-trained parameters of the atomic NNPs from a file. We then create a wrapper class for the ANI model that takes as input the atomic positions and box vectors and outputs the potential energy of the system. This new model is saved to a file using the `torch.jit.script` function. This function serializes the model to a file which can be loaded with OpenMM-Torch. The NNP is added to the OpenMM system using the `TorchForce` class. The rest of the code is similar to the previous OpenMM simulation script.

It should be noted that the the ANI model can be optimized using the `NNPOps` [package](https://github.com/openmm/NNPOps) which provides a set of tools for optimizing the neural network potentials.  

# Conclusion

That's it! In this post, we went through the worfklow for how to train a Behler-Parrinello style NNP for use in MD simulations. We used PyTorch and the `torchani` library to build the NNP, using the awesome functionality of making the featurization part of the model using the `torch.nn.Module` class. We created an example script for training the model and lastly, we showed how one can use a PyTorch-based NNP in an OpenMM simulation using the `OpenMM-Torch` plugin.

It should be noted that in practice, the training of the NNP would be done using high-quality data and there would likely be many steps to optimize both the architecture of the atomic NNPs and the hyperparameters of the model. 

Creating this post was a great learning experience for me since utilizing a pre-trained NNP is different from training a NNP from scratch. I hope this post can help others who are interested in using NNPs in their MD simulations. If you have any questions or would like to chat about NNPs, feel free to reach out to me! 