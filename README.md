# A Model for Place Field Reorganization During Reward Maximization
This repository contains the code to train place field parameters using the reinforcement learning framework. 

Additional code is needed to analyze and generate the figures. These will be available at a later stage. 

## Installation
The code was develped on a MacOS and Linux.

To get started, install python==3.8, JAX, numpy, scipy and matplotlib; or use the following

### Using PIP 
```
pip install -r requirements.txt
```
### Using Conda
```
conda env create -f environment.yml
conda activate example-project
```

Depending on your machine, you may need additional or alternative packages for GPU-enabled Jax. Please consult the Jax installation instructions for details.

## Running the Project

### JAX
The initial code was developed using JAX to use its autograd function. This code compiles the objective and optimizes the network parameters using gradients. Hence, to experiment with other place field descriptions or objective functions, use the code in jax folder.

### Numpy
The place field reorganization was developed in numpy to speed up run time and to implement an online learning version. Use the code in numpy folder to run this code. This folder also includes the Successor Representation agent described in the paper. 


### 1D or 2D environments
The main executable code for this project is contained within the 1D and 2D directories, where each of these directories includes a main.py file that serves as the entry point.

To run the 1D agent:
```
python 1D/main.py
```

To run the 2D agent:
```
python 2D/main.py
```

