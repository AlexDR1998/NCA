# NCA
Neural Cellular Automata (NCA). Used primarily for modelling human embryonic stem cells for use in my PhD research.  Inspired by and based on the work of Mortvinstev et al: https://distill.pub/2020/growing-ca/ , this work extends their NCA framework to learn local rules that yield a time series of images, rather than growing one image from one pixel. This extension allows for modelling dynamical emergent behaviour, and can reproduce the behaviour of classic PDEs (heat, reaction diffusion) 

# How to use
See the demo jupyter notebook, it includes details of how to actually use this code on a range of problems

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
- Fundamental NCA code
  - NCA/NCA_class.py contains the actual NCA model wrapped in a class for convenience, with saving/loading functionality.
  - NCA/trainer/ contains all the code for fitting NCA to data:
    - NCA/trainer/NCA_trainer.py includes the basic NCA_Trainer class for fitting an NCA to data
    - NCA/trainer/NCA_PDE_trainer.py includes a subclass of NCA_Trainer that is tweaked to train NCA to PDE trajectories
    - NCA/trainer/NCA_IC_trainer.py includes a subclass of NCA_Trainer that optimises initial conditions (while keeping NCA model parameters fixed) to explore stability
    - NCA/trainer/NCA_trainer_stem_cells.py includes a subclass of NCA_Trainer that is tweaked to train on specific biological data
    - NCA/trainer/NCA_train_utils.py includes functions for sweeping through training hyperparameters
    - NCA/trainer/NCA_loss_functions.py includes definitions of all the different loss functions explored
  - NCA/NCA_utils.py contains a lot of helper functions, mainly file i.o. and stuff that should have been in tensorflow i.e. periodic padding.
- Fundamental non NCA stuff
  - NCA/PDE_solver.py contains the PDE_Solver class that performs finite difference / euler numerical solutions of PDEs for the NCA to be trained on
  - NCA/GOL_solver.py contains the GOL_Solver class that simulates conway's game of life, as a test to see if the NCA can recover that behaviour.
- Work in progress extra stuff
  - NCA/NCA_visualise.py contains methods for visualising and interpretting trained NCA models, including saving videos of trajectories.
  - NCA/NCA_analyse.py contains methods for exploring perturbations of NCA models, still very much work in progress
