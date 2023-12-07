import jax
#from jax import config
jax.config.update("jax_enable_x64", True)
import equinox as eqx
import jax.numpy as jnp
import time

import matplotlib.pyplot as plt

from PDE.solver.semidiscrete_solver import PDE_solver
from PDE.trainer.PDE_trainer import PDE_Trainer
from NCA_JAX.NCA_visualiser import my_animate
from NCA_JAX.utils import *
from NCA_JAX.model.NCA_model import NCA
from NCA_JAX.model.boundary import NCA_boundary
from NCA_JAX.trainer.data_augmenter_tree import DataAugmenter as DataAugmenterNCA
from tqdm import tqdm

model_name = "models/micropattern_radii_experiments/micropattern_radii_sized_c32_b1_r1e-2_v1_"
#data_name = "data/micropattern_size_c32_b1_v1_predictions.pickle"
N_MODELS = 135
CHANNELS=32
steps=64
MODEL_NUMBER=5
PDE_TIMESTEPS=8

#model_name = "models/micropattern_radii_experiments/micropattern_radii_sized_c32_b0_r1e-2_v1_"
#data_name = "data/micropattern_size_c32_b0_v1_predictions.pickle"
#N_MODELS = 8
#CHANNELS=32
#steps=64




data = load_pickle("data/micropattern_data_size_sorted.pickle")
masks = load_pickle("data/micropattern_masks_size_sorted.pickle")

DA = DataAugmenterNCA(data,hidden_channels=CHANNELS-4)
x0,y_true = DA.split_x_y(1)

#print(DA.return_true_data())
boundary_callbacks = []
for m in masks:
    boundary_callbacks.append(NCA_boundary(m))

model_names = [model_name +str(x)+".eqx" for x in range(N_MODELS)]
models = []
models_raw = []
for n in tqdm(model_names):
    nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PERIODIC=False)
    nca_loaded = nca.load(n)
    models_raw.append(nca_loaded)
	


X = [models_raw[MODEL_NUMBER].run(steps,x0[MODEL_NUMBER][0],boundary_callbacks[MODEL_NUMBER])]
pde = PDE_solver(CHANNELS,False,dx=1,dt=0.01)
trainer = PDE_Trainer(pde,X)
trainer.train(t=PDE_TIMESTEPS,iters=1000,SAMPLING=2,WARMUP=10)
