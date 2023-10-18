from NCA_JAX.model.NCA_model import NCA
from NCA_JAX.trainer.NCA_trainer import *
from NCA_JAX.utils import *
from NCA_JAX.NCA_visualiser import *
import optax
import numpy as np
import jax.numpy as jnp
#import numpy as np
#import equinox as eqx
#import jax
#import jax.numpy as jnp
#from tqdm import tqdm

#N = 5
#BATCHES = 7
CHANNELS = 16
t = 64
iters=100
data_1 = load_emoji_sequence(["crab.png","alien_monster.png","butterfly.png"],downsample=4)
data_2 = load_emoji_sequence(["crab.png","alien_monster.png","butterfly.png"],downsample=2)
data = [data_1[0],data_2[0]]

#load_micropattern_radii("../Data/micropattern_radii/Chir_Fgf_*/processed/*")
# class data_augmenter_subclass(DataAugmenter):
# 	 #Redefine how data is pre-processed before training
# 	 def data_init(self):
# 		  data = self.return_saved_data()
# 		  data = self.duplicate_batches(data, 4)
# 		  data = self.pad(data, 10) 		
# 		  self.save_data(data)
# 		  return None
# 	
# 	# Redefine how data is processed during NCA training
# 	def data_callback(self, x, y, i):
# 		x = x.at[1:].set(x[:-1]) 
# 		x_true,_ =self.split_x_y(1)
# 		x = x.at[0].set(x_true[0])
# 		return x,y
	

# mask = np.zeros((2,data.shape[-2]+10,data.shape[-1]+10))
# mask[0,::2]=1 
# mask[1,:,::2]=1
# mask = jnp.array(mask)

schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
optimiser= optax.adamw(schedule)
#optimiser= optax.chain(optax.clip_by_block_rms(1.0),optimiser)


nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PERIODIC=False)


opt = NCA_Trainer(nca,
				  data,
				  model_filename="jax_pytree_multisize_batch_gpu")#,
				  #SHARDING=4)
				  #BOUNDARY_MASK=mask)


opt.train(t,iters,optimiser=optimiser)





#print(nca.layers[3].weight[0,0])

#nca = opt.NCA_model
#print(nca.layers[3].weight[0,0])

#nca2 = NCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PERIODIC=True)
#nca2 = nca2.load("models/jax_eddie_test.eqx")
#trajectory = nca2.run(200,data[0,0])[:,:3]
#my_animate(trajectory)

#print(nca2.layers[3].weight[0,0])


