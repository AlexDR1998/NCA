import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import datetime
import time
from PDE.trainer.data_augmenter_pde import DataAugmenterPDE
import NCA_JAX.trainer.loss as loss
from NCA_JAX.model.boundary import NCA_boundary
from PDE.trainer.tensorboard_log import PDE_Train_log
from PDE.trainer.optimiser import non_negative_diffusion
from PDE.solver.semidiscrete_solver import PDE_solver
import diffrax
from tqdm import tqdm
class PDE_Trainer(object):
	
	
	def __init__(self,
			     PDE_solver,
				 data,
				 model_filename=None,
				 DATA_AUGMENTER = DataAugmenterPDE,
				 BOUNDARY_MASK = None, 
				 SHARDING = None, 
				 directory="models/"):
		"""
		

		Parameters
		----------
		
		PDE_solver : object callable - (float32 array [T], float32 array [N_CHANNELS,_,_]) -> (float32 array [N_CHANNELS,_,_])
			PDE solver that returns T timesteps of integrated parameterised PDE model. Parameters are to be trained
		
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
		#self.NCA_model = NCA_model
		self.PDE_solver = PDE_solver
		
		# Set up variables 

		self.OBS_CHANNELS = self.PDE_solver.func.N_CHANNELS#data[0].shape[1]
		
		# Set up data and data augmenter class
		self.DATA_AUGMENTER = DATA_AUGMENTER(data)
		self.DATA_AUGMENTER.data_init()
		self.BATCHES = len(data)
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
			  WARMUP=64,
			  SAMPLING = 8,			        
			  key=jax.random.PRNGKey(int(time.time()))):
		"""
		

		Parameters
		----------
		t : Int
			Number of timesteps for the PDE to predict at once. 
		iters : Int
			Number of training iterations.
		optimiser : optax.GradientTransformation
			the optax optimiser to use when applying gradient updates to model parameters.
			if None, constructs adamw with exponential learning rate schedule
		
		WARMUP : int optional
			Number of iterations to wait for until starting model checkpointing. Default is 64
		SAMPLING : TYPE, optional
			DESCRIPTION. The default is 8.
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).

		Returns
		-------
		TYPE
			DESCRIPTION.
		TYPE
			DESCRIPTION.
		TYPE
			DESCRIPTION.

		"""
		
		
		@eqx.filter_jit
		def make_step(pde,x,y,t,opt_state,key):	
			"""
			

			Parameters
			----------
			pde : object callable - (float32 [T], float32 [N_CHANNELS,_,_]) -> (float32 [T], float32 [T,N_CHANNELS,_,_])
				the PDE solver to train
			x : float32 array [BATCHES,N,CHANNELS,_,_]
				input state
			y : float32 array [BATCHES,N,OBS_CHANNELS,_,_]
				true predictions (time offset with respect to input axes)
			t : int
				number of PDE timesteps to predict - mapping X[:,i]->X[:,i+1:i+t]
			opt_state : optax.OptState
				internal state of self.OPTIMISER
			key : jax.random.PRNGKey, optional
				Jax random number key. 
				
			Returns
			-------
			pde : object callable - (float32 [T], float32 [N_CHANNELS,_,_]) -> (float32 [T], float32 [T,N_CHANNELS,_,_])
				the PDE solver with updated parameters
			opt_state : optax.OptState
				internal state of self.OPTIMISER, updated in line with having done one update step
			loss_x : (float32, (float32 array [BATCHES,N,CHANNELS,_,_], float32 array [BATCHES,N]))
				tuple of (mean_loss, (x,losses)), where mean_loss and losses are returned for logging purposes,
				and x is the updated PDE state after t iterations

			"""
			@eqx.filter_value_and_grad(has_aux=True)
			def compute_loss(pde_diff,pde_static,x,y,t,key):
				_pde = eqx.combine(pde_diff,pde_static)
				#v_pde = jax.vmap(lambda x:_pde(jnp.linspace([0,t,t+1]),x)[1][1:],in_axes=0,out_axes=0,axis_name="N")
				v_pde = lambda x:_pde(jnp.linspace(0,t,t+1),x)[1][1:] # Don't need to vmap over N
				vv_pde= lambda x: jax.tree_util.tree_map(v_pde,x) # different data batches can have different sizes
				v_loss_func = lambda x,y: jnp.array(jax.tree_util.tree_map(self.loss_func,x,y))
				y_pred=vv_pde(x)
				losses = v_loss_func(y_pred,y)
				mean_loss = jnp.mean(losses)
				return mean_loss,(y_pred,losses)
			
			pde_diff,pde_static=pde.partition()
			loss_x,grads = compute_loss(pde_diff, pde_static, x, y, t, key)
			updates,opt_state = self.OPTIMISER.update(grads, opt_state, pde_diff)
			pde = eqx.apply_updates(pde,updates)
			return pde,opt_state,loss_x
		
		
		# Initialise training
		pde = self.PDE_solver
		pde_diff,pde_static = pde.partition()
		if optimiser is None:
			#schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
			#self.OPTIMISER = optax.adam(schedule)
			self.OPTIMISER = non_negative_diffusion(learn_rate=1e-2,iters=iters)
		else:
			self.OPTIMISER = optimiser
		opt_state = self.OPTIMISER.init(pde_diff)
		
		#x_full,y_full = self.DATA_AUGMENTER.split_x_y(1)
		#x,y=self.DATA_AUGMENTER.random_N_select(x_full,y_full,SAMPLING)
		data_steps = self.DATA_AUGMENTER.data_saved[0].shape[0]
		
		best_loss = 100000000
		loss_thresh = 1e16
		model_saved = False
		error = 0
		error_at = 0
		for i in tqdm(range(iters)):
			key = jax.random.fold_in(key,i)
			x,y = self.DATA_AUGMENTER.sub_trajectory_split(L=t,key=key)
			pde,opt_state,(mean_loss,(x,losses)) = make_step(pde, x, y, t, opt_state,key)
			
			if self.IS_LOGGING:
				self.LOGGER.tb_training_loop_log_sequence(losses, x, i, pde)
			
			
			if jnp.isnan(mean_loss):
				error = 1
				error_at=i
				break
			elif any(list(map(lambda x: jnp.any(jnp.isnan(x)), x))):
				error = 2
				error_at=i
				break
			elif mean_loss>loss_thresh:
				error = 3
				error_at=i
				break
			# Check if training has crashed or diverged yet
			if error==0:
				# Do data augmentation update
				#x,y = self.DATA_AUGMENTER.data_callback(x, y, i)
				#x_full,y_full = self.DATA_AUGMENTER.split_x_y(1)
				# Save model whenever mean_loss beats the previous best loss
				if i>WARMUP:
					if mean_loss < best_loss:
						model_saved=True
						self.PDE_solver = pde
						self.PDE_solver.save(self.MODEL_PATH,overwrite=True)
						best_loss = mean_loss
						tqdm.write("--- Model saved at "+str(i)+" epochs with loss "+str(mean_loss)+" ---")
		if error==0:
			print("Training completed successfully")
		elif error==1:
			print("|-|-|-|-|-|-  Loss reached NaN at step "+str(error_at)+" -|-|-|-|-|-|")
		elif error==2:
			print("|-|-|-|-|-|-  X reached NaN at step "+str(error_at)+" -|-|-|-|-|-|")
		elif error==3:
			print( "|-|-|-|-|-|-  Loss exceded "+str(loss_thresh)+" at step "+str(error_at)+", optimisation probably diverging  -|-|-|-|-|-|")
		#assert model_saved, "|-|-|-|-|-|-  Training did not converge, model was not saved  -|-|-|-|-|-|"
		if error!=0 and model_saved==False:
			print("|-|-|-|-|-|-  Training did not converge, model was not saved  -|-|-|-|-|-|")
		elif self.IS_LOGGING and model_saved:
			x,y = self.DATA_AUGMENTER.split_x_y(1)
			self.LOGGER.tb_training_end_log(self.PDE_solver,x,data_steps,self.BOUNDARY_CALLBACK)