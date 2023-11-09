from NCA_JAX.model.NCA_model import NCA
from NCA_JAX.trainer.NCA_trainer import *
from NCA_JAX.utils import *
from NCA_JAX.NCA_visualiser import *
from NCA_JAX.trainer.data_augmenter_tree import DataAugmenter
import optax
import numpy as np
import jax.numpy as jnp
import jax
import random
import sys
from tqdm import tqdm

CHANNELS=16
t = 64
B=int(sys.argv[1])
L = int(sys.argv[2])
### Load pre-processed data

# For use on laptop - for testing
#data = load_pickle("data/micropattern_data_size_sorted.pickle")
#masks = load_pickle("data/micropattern_masks_size_sorted.pickle")

# For use on eddie
data = load_pickle("../Data/micropattern_radii/micropattern_data_size_sorted.pickle")
masks = load_pickle("../Data/micropattern_radii/micropattern_masks_size_sorted.pickle")


# Format and pad with hidden channel zeros
DA = DataAugmenter(data,hidden_channels=12)
x0,y_true = DA.split_x_y(1)

#print(DA.return_true_data())
# Construct list of boundary mask callback functions
boundary_callbacks = []
for m in masks:
    boundary_callbacks.append(NCA_boundary(m))
	
model_name = "models/micropattern_radii_sized_b"+str(B)+"_r1e-2_v2_"
#model_name = "models/micropattern_radii_experiments/micropattern_radii_sized_b"+str(B)+"_r1e-2_v2_"
model_names = [model_name +str(x)+".eqx" for x in range(L)]
models = []
models_raw = []
for n in model_names:
    nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PERIODIC=False)
    nca_loaded = nca.load(n)
    models.append(nca_loaded)
    #v_nca = jax.vmap(nca_loaded,in_axes=(0,None,0),out_axes=0,axis_name="N") # boundary is independant of time N
    #vv_nca = lambda x,callback,key_array:jax.tree_util.tree_map(v_nca,x,callback,key_array)
    #models.append(vv_nca)
	

# def run(v_nca):
#     x = x0
#     key = jax.random.PRNGKey(int(time.time()))
    
#     for i in range(t):
#         key = jax.random.fold_in(key,i)
#         key_array = key_pytree_gen(key,(len(x),x[0].shape[0]))
#         x = v_nca(x,boundary_callbacks,key_array)
#     return x


# only_obs = lambda x:x[:,:4]

# x_pred = []
# for nca in tqdm(models):
#     output = run(nca)

#     x_pred.append(list(map(only_obs,output)))


x_pred = []
for nca in tqdm(models_raw):
    x_pred_model = []
    for i,x0_i in tqdm(enumerate(x0)): 
        x = nca.run(t,x0_i[0],boundary_callbacks[i])
        x_pred_model.append(x[-1,:4])
        #print(x.shape)
    #print(x)
    x_pred.append(x_pred_model)

save_pickle(x_pred,"../Data/micropattern_radii/micropattern_size_b"+str(B)+"_v2_predictions.pickle",overwrite=True)