import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import datetime
import NCA_JAX.trainer.loss as loss
from NCA_JAX.trainer.tensorboard_log import NCA_Train_log
from NCA_JAX.trainer.data_augmenter import DataAugmenter
from NCA_JAX.utils import key_array_gen
from NCA_JAX.model.boundary import NCA_boundary
from tqdm import tqdm
import time
from functools import partial

class NCA_Trainer(object):
	"""
	General class for training NCA model to data trajectories
	"""
	
	def __init__(self,NCA_model,data,model_filename=None,DATA_AUGMENTER = DataAugmenter, BOUNDARY_MASK = None, directory="models/"):
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
		directory : str
			Name of directory where all models get stored, defaults to 'models/'

		Returns
		-------
		None.

		"""
		self.NCA_model = NCA_model
		
		# Set up variables 
		self.CHANNELS = self.NCA_model.N_CHANNELS
		self.OBS_CHANNELS = data.shape[2]
		
		# Set up data and data augmenter class
		self.DATA_AUGMENTER = DATA_AUGMENTER(data,self.CHANNELS-self.OBS_CHANNELS)
		self.data = self.DATA_AUGMENTER.return_true_data()
		
		# Set up boundary augmenter class
		self.BOUNDARY_CALLBACK = NCA_boundary(BOUNDARY_MASK)
		
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
	def loss_func(self,x,y):
		"""
		

		Parameters
		----------
		x : float32 array [CHANNELS,_,_]
			NCA state
		y : float32 array [OBS_CHANNELS,_,_]
			data

		Returns
		-------
		loss : float32 array [BATCHES]
			loss for each batch of trajectories
		"""
		x_obs = x[:,:,:self.OBS_CHANNELS]
		y_obs = y[:,:,:self.OBS_CHANNELS]
		#x_obs = x[:self.OBS_CHANNELS]
		#y_obs = y[:self.OBS_CHANNELS]
		return loss.euclidean(x_obs,y_obs)
	
	@eqx.filter_jit
	def intermediate_reg(self,x,full=True):
		"""
		Intermediate state regulariser - tracks how much of x is outwith [0,1]

		Parameters
		----------
		x : float32 array [CHANNELS,_,_]
			NCA state
		full : boolean
			Flag for whether to only regularise observable channel (true) or all channels (false)
		Returns
		-------
		reg : float
			float tracking how much of x is outwith range [0,1]

		"""
		if not full:
			x = x[:,:,:self.OBS_CHANNELS]
			#x = x[:self.OBS_CHANNELS]
		return jnp.mean(jnp.abs(x)+jnp.abs(x-1)-1)
	
	
		
		
	def train(self,t,iters,optimiser=None,STATE_REGULARISER=1.0,key=jax.random.PRNGKey(int(time.time()))):
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
			if None, constructs adam with gradient clipping and cosine learning rate schedule
		STATE_REGULARISER : int optional
			Strength of intermediate state regulariser. Defaults to 1.0
			
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
			x : float32 array [N,CHANNELS,_,_]
				NCA state
			y : float32 array [N,OBS_CHANNELS,_,_]
				true data
			t : int
				number of NCA timesteps between x[N] and x[N+1]
			opt_state : optax.OptState
				internal state of self.OPTIMISER
			key : jax.random.PRNGKey, optional
				Jax random number key. 
				
			Returns
			-------
			grad : pytree
			opt_state : optax.OptState
				internal state of self.OPTIMISER, updated in line with having done one update step
			loss_x : (float32, (float32 array [N,CHANNELS,_,_], float32 array [N]))
				tuple of (mean_loss, (x,losses)), where mean_loss and losses are returned for logging purposes,
				and x is the updated NCA state after t iterations

			"""
			
			@eqx.filter_value_and_grad(has_aux=True)
			def compute_loss(nca_diff,nca_static,x,y,t,key):
				# Gradient and values of loss function computed here
				_nca = eqx.combine(nca_diff,nca_static)
				nca = jax.vmap(lambda x,key:_nca(x,boundary_callback=self.BOUNDARY_CALLBACK,key=key),axis_name="N")
					  
				
				reg_log = 0
				
				# Structuring this as function and lax.scan speeds up jit compile a lot

				def nca_step(carry,j): # function of type a,b -> a
					key,x,reg_log = carry
					key = jax.random.fold_in(key,j)
					key_array = key_array_gen(key,(x.shape[0]))
					x = nca(x,key_array)
					#x = nca(x,self.BOUNDARY_CALLBACK,key)
					
					reg_log+=self.intermediate_reg(x)
					return (key,x,reg_log),None

				(key,x,reg_log),_ = jax.lax.scan(nca_step,(key,x,reg_log),xs=jnp.arange(t))
				
				losses = self.loss_func(x, y)
				reg_term = STATE_REGULARISER*(reg_log/t)
				#mean_loss = jnp.mean(losses)+STATE_REGULARISER*(reg_log/t)
				return reg_term,(x,losses)

			
			nca_diff,nca_static = nca.partition()
			(reg_term,(x,losses)),grads = compute_loss(nca_diff,nca_static,x,y,t,key)
			grads = jax.lax.pmean(grads,axis_name="batch")
			mean_loss = jnp.mean(jax.lax.pmean(losses,axis_name="batch"))*reg_term
			
			updates,opt_state = self.OPTIMISER.update(grads, opt_state, nca_diff)
			nca = eqx.apply_updates(nca,updates)
			return nca,opt_state,(mean_loss,(x,losses))#loss_x[0],loss_x[1]
			
			#return loss_x,grads
		
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
		
		# Initialise data and split into x and y
		self.DATA_AUGMENTER.data_init()
		x,y = self.DATA_AUGMENTER.split_x_y(1)
		
		pmake_step = jax.pmap(make_step,in_axes=(None,0,0,None,0,0),out_axes=(None,0,0),axis_name="batch")

		
		for i in tqdm(range(iters)):
			key = jax.random.fold_in(key,i)
			key_array = key_array_gen(key,(x.shape[0]))
			nca,opt_state,(mean_loss,(x,losses)) = pmake_step(nca, x, y, t, opt_state, key_array)
			
			if self.IS_LOGGING:
				self.LOGGER.tb_training_loop_log_sequence(losses, x, i, nca)
			# Do data augmentation update
			x,y = self.DATA_AUGMENTER.data_callback(x, y, i)
				
		self.NCA_model = nca
		self.NCA_model.save(self.MODEL_PATH,overwrite=True)