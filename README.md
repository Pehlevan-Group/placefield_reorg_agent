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
conda activate pfagent
```

Depending on your machine, you may need additional or alternative packages for GPU-enabled Jax. Please consult the Jax installation instructions for details.

## Running the Project

### JAX
The code was initially developed using JAX to use its autograd function. This code compiles the objective and optimizes the network parameters using gradients. Hence, to experiment with other place field descriptions or objective functions, use the code in the jax folder.

### Numpy
The place field reorganization model was re-written in numpy to speed up run time and reduce memory issues with JAX. The model is also implemented as an online learning version so as to add Gaussian noise to place field parameters at each time step to model neural drift. Use the code in the numpy folder to run this code. This folder also includes the Successor Representation agent described in the paper. 


### 1D or 2D environments
The main executable code for this project is contained within the 1D and 2D directories, where each of these directories includes a main.py file that serves as the entry point.


## References

If the publication or code was helpful in anyway, please consider citing using the following BibTeX entry:

```bibtex
@article{kumar2024pfreorg,
  title={A Model for Place Field Reorganization During Reward Maximization},
  author={Kumar, M Ganesh and Bordelon, Blake and Zavatone-Veth, Jacob and Pehlevan, Cengiz},
  year={2024},
}