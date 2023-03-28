# NCA
Neural Cellular Automata (NCA). Used primarily for modelling human embryonic stem cells for use in my PhD research.  Inspired by and based on the work of Mortvinstev et al: https://distill.pub/2020/growing-ca/ , this work extends their NCA framework to learn local rules that yield a time series of images, rather than growing one image from one pixel. This extension allows for modelling dynamical emergent behaviour, and can reproduce the behaviour of classic PDEs (heat, reaction diffusion) 

## Code structure
The code in the NCA/ subdirectory is structured as followed.
- Fundamental NCA code
  - NCA_class.py contains the actual NCA model wrapped in a class for convenience, with saving/loading functionality.
  - NCA_train.py contains the NCA_Trainer class for fitting an NCA to data, as well as subclasses that tweak the functionality for PDEs or exploring the stability of NCA - initial condition pairs
  - NCA_utils.py contains a lot of helper functions, mainly file i.o. and stuff that should have been in tensorflow i.e. periodic padding.
  - NCA_train_utils.py contains helper functions for NCA_train.py, mainly different loss functions
- Fundamental non NCA stuff
  - PDE_solver.py contains the PDE_Solver class that performs finite difference / euler numerical solutions of PDEs for the NCA to be trained on
  - GOL_solver.py contains the GOL_Solver class that simulates conway's game of life, as a test to see if the NCA can recover that behaviour.
- Work in progress extra stuff
  - NCA_visualise.py and NCA_analyse.py contain some methods for exploring, visualising and interpretting trained NCA models. Still in development
  - 
