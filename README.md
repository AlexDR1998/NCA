# NCA
Neural Cellular Automata (NCA). Used primarily for modelling spatio-temporal patterning phenomena in my PhD research. See this pre-print: https://arxiv.org/abs/2310.14809 . Inspired by and based on the work of Mortvinstev et al: https://distill.pub/2020/growing-ca/ , this work extends their NCA framework to learn local rules that yield a time series of images, rather than growing one image from one pixel. This extension allows for modelling dynamical emergent behaviour, and can reproduce the behaviour of classic PDEs (heat, reaction diffusion).

# How to use
See the demo jupyter notebook. The code here was used for running experiments in https://arxiv.org/abs/2310.14809 , but moving forward a version based on JAX is included in https://github.com/AlexDR1998/Differentiable-Patterning

## Requirements 
 - tensorflow 2.13.0
 - tensorboard 2.13.0
 - numpy 1.24.4
 - scipy 1.9.0
 - scikit-image 0.19.1
 - tqdm 4.64.0
 - matplotlib 3.7.2


## Code structure
The code is structured as followed.
- NCA code (Tensorflow version)
  - NCA/NCA_class.py contains the actual NCA model wrapped in a class for convenience, with saving/loading functionality.
  - NCA/trainer/ contains all the code for fitting NCA to data:
    - NCA/trainer/NCA_trainer.py includes the basic NCA_Trainer class for fitting an NCA to data
    - NCA/trainer/NCA_PDE_trainer.py includes a subclass of NCA_Trainer that is tweaked to train NCA to PDE trajectories
    - NCA/trainer/NCA_IC_trainer.py includes a subclass of NCA_Trainer that optimises initial conditions (while keeping NCA model parameters fixed) to explore stability
    - NCA/trainer/NCA_trainer_stem_cells.py includes a subclass of NCA_Trainer that is tweaked to train on specific biological data
    - NCA/trainer/NCA_train_utils.py includes functions for sweeping through training hyperparameters
    - NCA/trainer/NCA_loss_functions.py includes definitions of all the different loss functions explored
  - NCA/NCA_utils.py contains a lot of helper functions, mainly file i.o. and stuff that should have been in tensorflow i.e. periodic padding.
- Other non NCA stuff
  - NCA/PDE_solver.py contains the PDE_Solver class that performs finite difference / euler numerical solutions of PDEs for the NCA to be trained on
  - NCA/GOL_solver.py contains the GOL_Solver class that simulates conway's game of life, as a test to see if the NCA can recover that behaviour.

