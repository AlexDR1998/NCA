import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import datetime
import time
from NCA_JAX.trainer.data_augmenter_tree import DataAugmenter
import NCA_JAX.trainer.loss as loss
from NCA_JAX.model.boundary import NCA_boundary
from PDE.trainer.tensorboard_log import PDE_Train_log
import diffrax
class PDE_Trainer(object):
	
	
	def __init__(self,
			     PDE_model,
			     NCA_model,
				 data,
				 model_filename=None,
				 DATA_AUGMENTER = DataAugmenter,
				 BOUNDARY_MASK = None, 
				 SHARDING = None, 
				 directory="models/"):
		"""
		

		Parameters
		----------
		
		PDE_model : object callable - (float32 array [N_CHANNELS,_,_]) -> (float32 array [N_CHANNELS,_,_])
			PDE model to train to fit NCA trajectory
		
		NCA_model : object callable - (float32 array [N_CHANNELS,_,_],PRNGKey) -> (float32 array [N_CHANNELS,_,_])
			trained NCA object
			
		data : float32 array [BATCHES,N,OBS_CHANNELS,_,_]
			set of trajectories (initial conditions) to train PDE to NCA on
		
		model_filename : str, optional
			name of directories to save tensorboard log and model parameters to.
			log at :	'logs/gradient_tape/model_filename/train'
			model at : 	'models/model_filename'
			if None, sets model_filename to current time
		
		DATA_AUGMENTER : object, optional
			DataAugmenter object. Has data_init and data_callback methods that can be re-written as needed. The default is DataAugmenter.
		BOUNDARY_MASK : float32 [N_BOUNDARY_CHANNELS,WIDTH,HEIGHT], optional
			Set of channels to keep fixed, encoding boundary conditions. The default is None.
		SHARDING : int, optional
			How many parallel GPUs to shard data across?. The default is None.
		
		directory : str
			Name of directory where all models get stored, defaults to 'models/'

		Returns
		-------
		None.

		"""
		self.NCA_model = NCA_model
		
		# Set up variables 
		self.CHANNELS = self.NCA_model.N_CHANNELS
		self.OBS_CHANNELS = data[0].shape[1]
		self.SHARDING = SHARDING
		
		
		# Set up data and data augmenter class
		self.DATA_AUGMENTER = DATA_AUGMENTER(data,self.CHANNELS-self.OBS_CHANNELS)
		self.DATA_AUGMENTER.data_init(self.SHARDING)
		self.data = self.DATA_AUGMENTER.return_saved_data()
		self.BATCHES = len(self.data)
		print("Batches = "+str(self.BATCHES))
		# Set up boundary augmenter class
		# length of BOUNDARY_MASK PyTree should be same as number of batches
		
		self.BOUNDARY_CALLBACK = []
		for b in range(self.BATCHES):
			if BOUNDARY_MASK is not None:
			
				self.BOUNDARY_CALLBACK.append(NCA_boundary(BOUNDARY_MASK[b]))
			else:
				self.BOUNDARY_CALLBACK.append(NCA_boundary(None))
		
		#print(jax.tree_util.tree_structure(self.BOUNDARY_CALLBACK))
		# Set logging behvaiour based on provided filename
		if model_filename is None:
			self.model_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
			self.IS_LOGGING = False
		else:
			self.model_filename = model_filename
			self.IS_LOGGING = True
			self.LOG_DIR = "logs/"+self.model_filename+"/train"
			self.LOGGER = PDE_Train_log(self.LOG_DIR, data)
			print("Logging training to: "+self.LOG_DIR)
		self.directory = directory
		self.MODEL_PATH = directory+self.model_filename
		print("Saving model to: "+self.MODEL_PATH)
	
	
	
	@eqx.filter_jit	
	def loss_func(self,x,y):
		"""
		NOTE: VMAP THIS OVER BATCHES TO HANDLE DIFFERENT SIZES OF GRID IN EACH BATCH
	
		Parameters
		----------
		x : float32 array [N,CHANNELS,_,_]
			NCA state
		y : float32 array [N,OBS_CHANNELS,_,_]
			data
		Returns
		-------
		loss : float32 array [N]
			loss for each timestep of trajectory
		"""
		x_obs = x[:,:self.OBS_CHANNELS]
		y_obs = y[:,:self.OBS_CHANNELS]
		return loss.euclidean(x_obs,y_obs)
	
	def train(self,
		      t,
			  iters,
			  optimiser=None,
			  STATE_REGULARISER=1.0,
			  WARMUP=64,
			  LOSS_SAMPLING = 64,			        
			  key=jax.random.PRNGKey(int(time.time()))):
		
		
		@eqx.filter_jit
		def make_step(pde,x,y,t,opt_state,key):	
			@eqx.filter_value_and_grad(has_aux=True)
			def compute_loss(pde_diff,pde_static,x,y,t,key):
				_pde = eqx.combine(pde_diff,pde_static)
				v_pde = jax.vmap(_pde,in_axes=0,out_axes=0,axis_name="N")
				vv_pde= lambda x: jax.tree_util.tree_map(v_pde,x) # different data batches can have different sizes
				v_loss_func = lambda x,y: jnp.array(jax.tree_util.tree_map(self.loss_func,x,y))
				term = diffrax.ODETerm(vv_pde)