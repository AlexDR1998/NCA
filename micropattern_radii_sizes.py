from NCA_JAX.model.NCA_model import NCA
from NCA_JAX.trainer.NCA_trainer import *
from NCA_JAX.utils import *
from NCA_JAX.NCA_visualiser import *
import optax
import numpy as np
import jax.numpy as jnp
import jax
import random
import sys




index=int(sys.argv[1])-1


CHANNELS = 16
t = 64
iters=2000
#BATCHES = 2

# Select which subset of data to train on
data,masks,shapes = load_micropattern_radii("../Data/micropattern_radii/Chir_Fgf_*")
#data,masks,shapes = load_micropattern_radii("../Data/micropattern_radii/Chir_Fgf_*/processed/*")
#print(shapes)
combined = list(zip(data,masks,range(len(data)),shapes))
combined_sorted = sorted(combined,key=lambda pair:pair[3])

#random.shuffle(combined)
data,masks,inds,shapes = zip(*combined_sorted)
data = list(data)
masks = list(masks)
inds = list(inds)
shapes = list(shapes)
print(shapes)
print(inds)


data = data[index*5:(index+1)*5]
masks= masks[index*5:(index+1)*5]
inds = inds[index*5:(index+1)*5]
shapes = shapes[index*5:(index+1)*5]
print("Selected batches: "+str(inds))
print("Size of selected images: "+str(shapes))
schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
optimiser= optax.adamw(schedule)

# Remove most of the data augmentation - don't need shifting or extra batches or intermediate propagation
class data_augmenter_subclass(DataAugmenter):
	 #Redefine how data is pre-processed before training
	 def data_init(self,batches):
		  data = self.return_saved_data()
		  self.save_data(data)
		  return None  
	 def data_callback(self, x, y, i):
		 x_true,_ =self.split_x_y(1)
		 reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
		 x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
		 return x,y

nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PERIODIC=False)
opt = NCA_Trainer(nca,
				  data,
				  model_filename="micropattern_radii_sized_b5_r1e-2_"+str(index),
				  BOUNDARY_MASK=masks,
				  DATA_AUGMENTER = data_augmenter_subclass)

opt.train(t,iters,optimiser=optimiser)