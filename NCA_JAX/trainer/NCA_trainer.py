import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import datetime
import NCA_JAX.trainer.loss as loss
from NCA_JAX.trainer.tensorboard_log import NCA_Train_log
from NCA_JAX.trainer.data_augmenter_tree import DataAugmenter
from NCA_JAX.utils import key_pytree_gen
from NCA_JAX.model.boundary import NCA_boundary
from tqdm import tqdm
import time

class NCA_Trainer(object):
	"""
	General class for training NCA model to data trajectories
	"""
	
	def __init__(self,
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
		
		NCA_model : object callable - (float32 array [N_CHANNELS,_,_],PRNGKey) -> (float32 array [N_CHANNELS,_,_])
			the NCA object to train
			
		data : float32 array [BATCHES,N,OBS_CHANNELS,_,_]
			set of trajectories to train NCA on
		
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
			self.LOGGER = NCA_Train_log(self.LOG_DIR, data)
			print("Logging training to: "+self.LOG_DIR)
		self.directory = directory
		self.MODEL_PATH = directory+self.model_filename
		print("Saving model to: "+self.MODEL_PATH)
		
	@eqx.filter_jit	
	def loss_func(self,x,y,key,SAMPLES):
		"""
		NOTE: VMAP THIS OVER BATCHES TO HANDLE DIFFERENT SIZES OF GRID IN EACH BATCH

		Parameters
		----------
		x : float32 array [N,CHANNELS,_,_]
			NCA state
		y : float32 array [N,OBS_CHANNELS,_,_]
			data
		key : jax.random.PRNGKey
			Jax random number key. Only useful for loss functions that are stochastic (i.e. subsampled).
		Returns
		-------
		loss : float32 array [N]
			loss for each timestep of trajectory
		"""
		x_obs = x[:,:self.OBS_CHANNELS]
		y_obs = y[:,:self.OBS_CHANNELS]
		return loss.euclidean(x_obs,y_obs)
		
		#return loss.random_sampled_euclidean(x_obs, y_obs, key, SAMPLES=SAMPLES)
	@eqx.filter_jit
	def intermediate_reg(self,x,full=True):
		"""
		Intermediate state regulariser - tracks how much of x is outwith [0,1]
		
		NOTE: VMAP THIS OVER BATCHES TO HANDLE DIFFERENT SIZES OF GRID IN EACH BATCH

		Parameters
		----------
		x : float32 array [N,CHANNELS,_,_]
			NCA state
		full : boolean
			Flag for whether to only regularise observable channel (true) or all channels (false)
		Returns
		-------
		reg : float
			float tracking how much of x is outwith range [0,1]

		"""
		if not full:
			x = x[:,:self.OBS_CHANNELS]
		return jnp.mean(jnp.abs(x)+jnp.abs(x-1)-1)
	
	def train(self,
		      t,
			  iters,
			  optimiser=None,
			  STATE_REGULARISER=1.0,
			  WARMUP=64,
			  LOSS_SAMPLING = 64,			        
			  key=jax.random.PRNGKey(int(time.time()))):
		"""
		Perform t steps of NCA on x, compare output to y, compute loss and gradients of loss wrt model parameters, and update parameters.

		Parameters
		----------
		t : int
			number of NCA timesteps between x[N] and x[N+1]
		iters : int
			number of training iterations
		optimiser : optax.GradientTransformation
			the optax optimiser to use when applying gradient updates to model parameters.
			if None, constructs adamw with exponential learning rate schedule
		STATE_REGULARISER : int optional
			Strength of intermediate state regulariser. Defaults to 1.0
		WARMUP : int optional
			Number of iterations to wait for until starting model checkpointing
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).
		Returns
		-------
		None
		"""
		
		@eqx.filter_jit
		def make_step(nca,x,y,t,opt_state,key):
			"""
			

			Parameters
			----------
			nca : object callable - (float32 [N_CHANNELS,_,_],PRNGKey) -> (float32 [N_CHANNELS,_,_])
				the NCA object to train
			x : float32 array [BATCHES,N,CHANNELS,_,_]
				NCA state
			y : float32 array [BATCHES,N,OBS_CHANNELS,_,_]
				true data
			t : int
				number of NCA timesteps between x[N] and x[N+1]
			opt_state : optax.OptState
				internal state of self.OPTIMISER
			key : jax.random.PRNGKey, optional
				Jax random number key. 
				
			Returns
			-------
			nca : object callable - (float32 array [N_CHANNELS,_,_],PRNGKey) -> (float32 array [N_CHANNELS,_,_])
				the NCA object with updated parameters
			opt_state : optax.OptState
				internal state of self.OPTIMISER, updated in line with having done one update step
			loss_x : (float32, (float32 array [BATCHES,N,CHANNELS,_,_], float32 array [BATCHES,N]))
				tuple of (mean_loss, (x,losses)), where mean_loss and losses are returned for logging purposes,
				and x is the updated NCA state after t iterations

			"""
			
			@eqx.filter_value_and_grad(has_aux=True)
			def compute_loss(nca_diff,nca_static,x,y,t,key):
				# Gradient and values of loss function computed here
				_nca = eqx.combine(nca_diff,nca_static)
				v_nca = jax.vmap(_nca,in_axes=(0,None,0),out_axes=0,axis_name="N") # boundary is independant of time N
				vv_nca = lambda x,callback,key_array:jax.tree_util.tree_map(v_nca,x,callback,key_array)
				reg_log = jnp.zeros(len(x))
				v_intermediate_reg = lambda x:jax.numpy.array(jax.tree_util.tree_map(self.intermediate_reg,x))
				_loss_func = lambda x,y,key:self.loss_func(x,y,key,SAMPLES=LOSS_SAMPLING)
				v_loss_func = lambda x,y,key_array:jax.numpy.array(jax.tree_util.tree_map(_loss_func,x,y,key_array))				
				
				# Structuring this as function and lax.scan speeds up jit compile a lot

				def nca_step(carry,j): # function of type a,b -> a
					key,x,reg_log = carry
					key = jax.random.fold_in(key,j)
					key_array = key_pytree_gen(key,(len(x),x[0].shape[0]))
					x = vv_nca(x,self.BOUNDARY_CALLBACK,key_array)				
					reg_log+=v_intermediate_reg(x)
					return (key,x,reg_log),None

				(key,x,reg_log),_ = jax.lax.scan(nca_step,(key,x,reg_log),xs=jnp.arange(t))
				#(key,x),_ = eqx.internal.scan(nca_step,(key,x),xs=jnp.arange(t),kind="checkpointed",checkpoints="all")
				#(key,x,_) = eqx.internal.while_loop(cond_func,nca_step,(key,x,0),kind="checkpointed",max_steps=t)
				loss_key = key_pytree_gen(key, (len(x),))
				losses = v_loss_func(x, y, loss_key)
				mean_loss = jnp.mean(losses)+STATE_REGULARISER*(jnp.mean(reg_log)/t)
				return mean_loss,(x,losses)
			
			nca_diff,nca_static = nca.partition()
			loss_x,grads = compute_loss(nca_diff,nca_static,x,y,t,key)
			updates,opt_state = self.OPTIMISER.update(grads, opt_state, nca_diff)
			nca = eqx.apply_updates(nca,updates)
			return nca,opt_state,loss_x#loss_x[0],loss_x[1]
		
		nca = self.NCA_model
		nca_diff,nca_static = nca.partition()
		
		# Set up optimiser
		if optimiser is None:
			schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
			self.OPTIMISER = optax.adamw(schedule)
			#self.OPTIMISER = optax.chain(optax.clip_by_block_rms(1.0),self.OPTIMISER)
		else:
			self.OPTIMISER = optimiser
		opt_state = self.OPTIMISER.init(nca_diff)
		
		
		# Split data into x and y
		x,y = self.DATA_AUGMENTER.split_x_y(1)
		
		#print(x[0].shape)
		best_loss = 100000000
		loss_thresh = 1e16
		model_saved = False
		for i in tqdm(range(iters)):
			key = jax.random.fold_in(key,i)
			nca,opt_state,(mean_loss,(x,losses)) = make_step(nca, x, y, t, opt_state,key)
			
			if self.IS_LOGGING:
				self.LOGGER.tb_training_loop_log_sequence(losses, x, i, nca)
			
			# Check if NaN
			assert not jnp.isnan(mean_loss), "|-|-|-|-|-|-  Loss reached NaN at step "+str(i)+" -|-|-|-|-|-|"
			assert not any(list(map(lambda x: jnp.any(jnp.isnan(x)), x))), "|-|-|-|-|-|-  X reached NaN at step "+str(i)+" -|-|-|-|-|-|"
			assert mean_loss<loss_thresh, "|-|-|-|-|-|-  Loss exceded "+str(loss_thresh)+" at step "+str(i)+", optimisation probably diverging  -|-|-|-|-|-|"
			
			# Do data augmentation update
			x,y = self.DATA_AUGMENTER.data_callback(x, y, i)
			
			# Save model whenever mean_loss beats the previous best loss
			if i>WARMUP:
				if mean_loss < best_loss:
					model_saved=True
					self.NCA_model = nca
					self.NCA_model.save(self.MODEL_PATH,overwrite=True)
					best_loss = mean_loss
					tqdm.write("--- Model saved at "+str(i)+" epochs with loss "+str(mean_loss)+" ---")
		
		assert model_saved, "|-|-|-|-|-|-  Training did not converge, model was not saved  -|-|-|-|-|-|"
		#self.NCA_model = nca
		#self.NCA_model.save(self.MODEL_PATH,overwrite=True)