import jax
import time
from NCA_JAX.trainer.data_augmenter_tree import DataAugmenter

class DataAugmenterPDE(DataAugmenter):
	"""
		Inherits the methods of DataAugmenter, but overwrites the batch cloning in the init
	"""
	def data_init(self,SHARDING=None):
		return None
	
	
	def sub_trajectory_split(self,L,key=jax.random.PRNGKey(int(time.time()))):
		"""
		Splits data into x (initial conditions) and y (following sub trajectory of length L).
		So that x[:,i]->y[:,i+1:i+1+L] is learned

		Parameters
		----------
		L : Int
			Length of subdirectory

		Returns
		-------
		x : float32[BATCHES,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : float32[BATCHES,L,CHANNELS,WIDTH,HEIGHT]
			Following sub trajectories

		"""
		pos = jax.random.randint(key,shape=(1,),minval=0,maxval=self.data_true[0].shape[0]-L-1)[0]
		x = jax.tree_util.tree_map(lambda data:data[pos],self.data_saved)
		y = jax.tree_util.tree_map(lambda data:data[pos+1:pos+1+L],self.data_saved)
		
		return x,y
		
		