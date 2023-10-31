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
BATCHES = 2

# Select which subset of data to train on
#data,masks = load_micropattern_radii("../Data/micropattern_radii/Chir_Fgf_*")
data,masks = load_micropattern_radii("../Data/micropattern_radii/Chir_Fgf_*/processed/*")
combined = list(zip(data,masks,range(len(data))))
random.shuffle(combined)
data,masks,inds = zip(*combined)
data = list(data)
masks = list(masks)
inds = list(inds)
data = data[:BATCHES]
masks= masks[:BATCHES]
inds = inds[:BATCHES]
print("Selected batches: "+str(inds))

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
				  model_filename="micropattern_radii_random_b2_r1e-2_"+str(index),
				  BOUNDARY_MASK=masks,
				  DATA_AUGMENTER = data_augmenter_subclass)

opt.train(t,iters,optimiser=optimiser)