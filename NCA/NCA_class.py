import numpy as np
import tensorflow as tf
import datetime
from NCA.NCA_utils import periodic_padding

class NCA(tf.keras.Model):
	""" 
		Neural Cellular Automata class. Is a subclass of keras.model,
		so that model loading and saving are inherited. 
		Heavily inspired by work at https://distill.pub/2020/growing-ca/ 
		modified and extended for purpose of modelling stem cell differentiation experiments

	
	"""


	def __init__(self,N_CHANNELS,
			     FIRE_RATE=0.5,
			     ENV_CHANNEL_DATA=None,
			     ACTIVATION="relu",
			     LAYERS=2,
			     OBS_CHANNELS=4,
			     REGULARIZER=0.001,
			     PADDING="periodic",
			     KERNEL_TYPE="ID_LAP_AV",
			     ORDER=1):
		"""
			Initialiser for neural cellular automata object

			Parameters
			----------
			N_CHANNELS : int
				Number of channels to include - first 4 are visible as RGBA, others are hidden
			FIRE_RATE : float in [0,1]
				Controls stochasticity of cell updates, at 1 all cells update every step, at 0 no cells update
			ENV_CHANNEL_DATA : float32 array [size,size,E] optional
				Array encoding information about the environment e.g. presence of mircropatterns or unusual boundary condition shapes 
                E is the number of channels used to encode envirnonment
			ACTIVATION : string optional
				String corresponding to tensorflow activation functions
			LAYERS : int optional
				How many hidden layers in the neural network
			OBS_CHANNELS : int optional
				How many of the channels are 'observable', i.e. fitted to target data
			REGULARIZER : float optional
				Regularizer coefficient for layer weights
			PADDING : string optional
				zero, flat or periodic boundary conditions
			KERNEL_TYPE : string optional
				What type of kernels to use. Valid options are:
				 "ID_LAP", "ID_AV", "ID_DIFF", 						- Identity and 1 other
				 "ID_LAP_AV","ID_DIFF_AV","ID_DIFF_LAP",			- Identity and 2 others
				 "ID_DIFF_LAP_AV",									- Identity and all 3 others
				 "DIFF_LAP_AV","DIFF_AV","LAP_AV"					- Average and other non-identity
				 "GOL"                                              - Identity and Moore neighbourhood
			ORDER : int optional
				Highest order polynomial terms of channels.
				1 - only linear channels, no cross terms
				2 - up to squared cross terms
				3 - not yet implemented
		"""

		 
		super(NCA,self).__init__()
		self.N_CHANNELS=N_CHANNELS # RGBA +hidden layers
		self.OBS_CHANNELS=OBS_CHANNELS
		self.FIRE_RATE=FIRE_RATE # controls stochastic updates - i.e. grid isn't globaly synchronised
		self.N_layers = LAYERS
		self.ACTIVATION = ACTIVATION
		self.PADDING = PADDING
		self.KERNEL_TYPE=KERNEL_TYPE
		self.ORDER = ORDER
		
		if OBS_CHANNELS>N_CHANNELS:
			print("Too many observable channels, setting OBS_CHANNELS = "+str(N_CHANNELS))
			OBS_CHANNELS = N_CHANNELS
		# Set up environment channels
		self._ENV_CHANNEL_DATA_raw = np.copy(ENV_CHANNEL_DATA)
		if ENV_CHANNEL_DATA is not None:
			#print(ENV_CHANNEL_DATA)
			ENV_CHANNEL_DATA = np.array(ENV_CHANNEL_DATA)
			self.set_env_channels(ENV_CHANNEL_DATA)
			assert self.OBS_CHANNELS + self.ENV_CHANNELS <= self.N_CHANNELS, "Not enough channels"
		else:
			self.ENV_CHANNEL_DATA = None   

		
		#-------------------------------------------------------------------------------
		#--- Set up dense nn for perception vector based on LAYERS parameter
		
		
		if LAYERS==3:
			self.dense_model = tf.keras.Sequential([
				tf.keras.layers.Conv2D(4*self.N_CHANNELS,1,
									   activation=ACTIVATION,
									   kernel_regularizer=tf.keras.regularizers.L1(REGULARIZER),
									   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.1),
									   use_bias=False,
									   trainable=True),
				tf.keras.layers.Conv2D(2*self.N_CHANNELS,1,
									   activation=ACTIVATION,
									   kernel_regularizer=tf.keras.regularizers.L1(REGULARIZER),
									   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.1),
									   use_bias=False,
									   trainable=True),
				tf.keras.layers.Conv2D(self.N_CHANNELS,1,
									   activation=None,
									   kernel_initializer=tf.keras.initializers.Zeros(),
									   trainable=True)])
		
		elif LAYERS==2:
			self.dense_model = tf.keras.Sequential([
				tf.keras.layers.Conv2D(4*self.N_CHANNELS,1,
									   activation=ACTIVATION,
									   kernel_regularizer=tf.keras.regularizers.L1(REGULARIZER),
									   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.1),
									   use_bias=False,
									   trainable=True),
				tf.keras.layers.Conv2D(self.N_CHANNELS,1,
									   activation=None,
									   kernel_initializer=tf.keras.initializers.Zeros(),
									   trainable=True)])
		

		#--- Prepare convolution kernels
		_i = np.array([0,1,0],dtype=np.float32)
		I  = np.outer(_i,_i)
		dx = (np.outer([1,2,1],[-1,0,1])/8.0).astype(np.float32)
		dy = dx.T
		lap = np.array([[0.25,0.5,0.25],
						[0.5,-3,0.5],
						[0.25,0.5,0.25]]).astype(np.float32)
		av = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.float32)/9.0
		moore=np.array([[1,1,1],[1,0,1],[1,1,1]]).astype(np.float32)
		

		#------------------------------------------------------------------------------
		#--- Combine kernels together based on KERNEL_TYPE parameter
		if self.KERNEL_TYPE=="ID_LAP":
			kernel = tf.stack([I,lap],-1)[:,:,None,:]
		if self.KERNEL_TYPE=="ID_LAP_AV":
			kernel = tf.stack([I,lap,av],-1)[:,:,None,:]
		if self.KERNEL_TYPE=="ID_AV":
			kernel = tf.stack([I,av],-1)[:,:,None,:]
		if self.KERNEL_TYPE=="ID_DIFF":
			kernel = tf.stack([I,dx,dy],-1)[:,:,None,:]
		if self.KERNEL_TYPE=="ID_DIFF_AV":
			kernel = tf.stack([I,dx,dy,av],-1)[:,:,None,:]
		if self.KERNEL_TYPE=="ID_DIFF_LAP":
			kernel = tf.stack([I,dx,dy,lap],-1)[:,:,None,:]
		if self.KERNEL_TYPE=="ID_DIFF_LAP_AV":
			kernel = tf.stack([I,dx,dy,lap,av],-1)[:,:,None,:]
		if self.KERNEL_TYPE=="LAP_AV":
			kernel = tf.stack([lap,av],-1)[:,:,None,:]
		if self.KERNEL_TYPE=="DIFF_LAP_AV":
			kernel = tf.stack([dx,dy,lap,av],-1)[:,:,None,:]
		if self.KERNEL_TYPE=="DIFF_AV":
			kernel = tf.stack([dx,dy,av],-1)[:,:,None,:]
		if self.KERNEL_TYPE=="GOL":
			kernel = tf.stack([I,moore],-1)[:,:,None,:]
		#--- If including 2nd order channels (squares and cross multiplication), 
		#    expand KERNEL appropriately

		if self.ORDER==1:
			self.KERNEL = tf.repeat(kernel,self.N_CHANNELS,2)
		elif self.ORDER==2:
			self.tri_ind = tf.convert_to_tensor(np.array(np.triu_indices(self.N_CHANNELS)).T)
			self.KERNEL = tf.repeat(kernel,self.N_CHANNELS + (self.N_CHANNELS*(self.N_CHANNELS+1))//2,2)
		#if ENV_CHANNEL_DATA is not None:
		#	self(tf.zeros_like(self.ENV_CHANNEL_MASK))
		#else:
		
		self(tf.zeros([1,3,3,self.N_CHANNELS])) # Dummy call to build the model
		
		
		print(self.dense_model.summary())
	
		
	
	def set_env_channels(self,ENV_CHANNEL_DATA):
		"""
		Sets the hard coded NCA channels that encode environmental information
		
		If ENV_CHANNEL_DATA is none, instead passes identity values to ENV_CHANNEL_MASK, so that no masking occurs

		Parameters
		----------
		ENV_CHANNEL_DATA : float32 array [N_BATCHES,size,size,E] optional
			Array encoding information about the environment e.g. presence of mircropatterns or unusual boundary condition shapes 
			E is the number of channels used to encode envirnonment

		Returns
		-------
		None.
		
		Saves
		-----
		
		self.ENV_CHANNEL_DATA : float32

		"""
		ENV_CHANNEL_DATA = np.array(ENV_CHANNEL_DATA) # Ensure that lists get converted to numpy arrays
		N_BATCHES = ENV_CHANNEL_DATA.shape[0]
		X = ENV_CHANNEL_DATA.shape[1]
		Y = ENV_CHANNEL_DATA.shape[2]
		self.ENV_CHANNELS = ENV_CHANNEL_DATA.shape[3]
		env_mask = np.zeros((N_BATCHES,X,Y,self.N_CHANNELS),dtype="float32")
		env_mask[...,self.OBS_CHANNELS:self.OBS_CHANNELS+self.ENV_CHANNELS] = 1.0
		self.ENV_CHANNEL_MASK = tf.convert_to_tensor(env_mask,dtype=tf.float32)
		
		
		_env = tf.convert_to_tensor(ENV_CHANNEL_DATA,dtype=tf.float32)
		self.ENV_CHANNEL_DATA = tf.concat((tf.zeros((N_BATCHES,X,Y,self.OBS_CHANNELS),dtype=tf.float32),
								       _env,
									   tf.zeros((N_BATCHES,X,Y,self.N_CHANNELS - self.ENV_CHANNELS - self.OBS_CHANNELS),dtype=tf.float32)),
								      axis=-1)
		
		
		#self.ENV_INDS = [[x] for x in range(self.OBS_CHANNELS,self.OBS_CHANNELS+ENV_CHANNELS.shape[-1])]
		
		#print("ENV_CHANNEL_PATTERN shape: "+str(self.ENV_CHANNEL_DATA.shape))
		#print("ENV_CHANNEL_MASK shape: "+str(self.ENV_CHANNEL_MASK.shape))
	def env_channel_callback(self,X):
		"""
		

		Parameters
		----------
		X : float32 tensor [batches,size,size,N_CHANNELS]
			

		Returns
		-------
		x_masked : float32 tensor [batches,size,size,N_CHANNELS]
			channels OBS_CHANNELS - OBS_CHANNELS+E are set to environment mask

		"""
		
		#--- Optional batch masking here for when during training N_BATCHES is overwritten to account for multiple timesteps
		B_x = X.shape[0]
		B_env=self.ENV_CHANNEL_DATA.shape[0]
		B = min(B_x,B_env)
		
		return (1-self.ENV_CHANNEL_MASK[:B])*X[:B] + self.ENV_CHANNEL_MASK[:B] * self.ENV_CHANNEL_DATA[:B]
		
		
	def upscale_kernel(self):
		"""
			Replaces kernels with higher resolution versions - DOESN'T WORK WELL
		"""
		#_i = np.array([0,0,1,0,0],dtype=np.float32)
		#I  = np.outer(_i,_i)
		I = np.array([[0,0,0,0,0],
					  [0,0,1,0,0],
					  [0,1,1,1,0],
					  [0,0,1,0,0],
					  [0,0,0,0,0]]).astype(np.float32)/5.0
		#dx = (np.outer([1,2,1],[-1,0,1])/8.0).astype(np.float32)
		#dy = dx.T
		lap = np.array([[0,0,1,0,0],
						[0,1,2,1,0],
						[1,2,-16,2,1],
						[0,1,2,1,0],
						[0,0,1,0,0]]).astype(np.float32)*3.0/16.0
		av = np.array([[1,1,1,1,1],
					   [1,1,1,1,1],
					   [1,1,1,1,1],
					   [1,1,1,1,1],
					   [1,1,1,1,1]]).astype(np.float32)/25.0
		kernel = tf.stack([I,lap,av],-1)[:,:,None,:]
		#kernel = tf.stack([I,av,dx,dy],-1)[:,:,None,:]
		#kernel = tf.stack([I,av],-1)[:,:,None,:]
		self.KERNEL = tf.repeat(kernel,self.N_CHANNELS,2)

	
	def __str__(self):
		"""
			Print method
		"""
		
		print("Neural Cellular Automata model")
		print("_________________________________________________________________________________")
		print("Stochastic firing rate:         {fr}".format(fr=self.FIRE_RATE))
		print("Number of channels:             {hc}".format(hc=self.N_CHANNELS))
		print("Activation function: 		   {ac}".format(ac=self.ACTIVATION))
		print("_________________________________________________________________________________")
		print("		Neural Network update function:")


		self.dense_model.summary()		
		print("_________________________________________________________________________________")		
		print("		Convolution kernels:")
		for i in range(self.KERNEL.shape[-1]):
			print(self.KERNEL[:,:,0,i])
		print("_________________________________________________________________________________")

		return " "
		
	@tf.function
	def perceive(self,x):
		"""
			Constructs a field y where each "pixel" represents the perception of local environments.
			Choice of kernels here is important.

			Parameters
			----------
			x : float32 tensor [batches,size,size,N_CHANNELS]
				state space of NCA, first 4 channels are typically visualised as RGBA,
				rest are hidden channels

			Returns
			-------
			y : float32 tensor [batches,size,size,depth]
				perception field, where each coordinate [batch,size,size] is a vector encoding
				local structure of x at [batch,size,size]. Used as input to self.dense_model
		"""
		if self.ORDER==1:
			y = tf.nn.depthwise_conv2d(x,self.KERNEL,[1,1,1,1],"SAME")
		elif self.ORDER==2:
			x_sq = tf.einsum('bxyi,bxyj->ijbxy',x,x)
			x_sq_flat = tf.transpose(tf.gather_nd(params=x_sq,indices=self.tri_ind),[1,2,3,0])
			x_2 = tf.concat([x,x_sq_flat],axis=-1)
			y = tf.nn.depthwise_conv2d(x_2,self.KERNEL,[1,1,1,1],"SAME")
		
		return y
	
	@tf.function
	def call(self,x,step_size=1.0,fire_rate=None):
		"""
			Applies a neural network (the trainable part of the model) to the perception field.
			
			Parameters
			----------
			x : float32 tensor [batches,size,size,N_CHANNELS]
				state space of NCA, first 4 channels are typically visualised as RGBA,
				rest are hidden channels
			step_size : float=1.0
				scale size of updates
			fire_rate : float=None
				controls probability of each pixel updating

			Returns
			-------
			x_new : float32 tensor
				new state space of NCA, with (stochastically masked) update applied across all channels and batches
		"""
		#print(x.shape)
		if fire_rate is not None:
			self.FIRE_RATE = fire_rate
		#if self.ENV_CHANNELS is not None:
		#	self.env_channel_callback(x)
		#tf.print(x[0,...,0])
		
		#--- If non-zero padding option given, pad x
		if self.PADDING=="periodic":
			x = periodic_padding(x,2)
		if self.PADDING=="flat":
			x = tf.pad(x,tf.constant([[0,0],[2,2],[2,2],[0,0]]),"SYMMETRIC")
		


		y = self.perceive(x)
		dx = (self.dense_model(y)*step_size)



		update_mask = tf.random.uniform(tf.shape(x[:,:,:,:1]),minval=0.0,maxval=1.0) <= self.FIRE_RATE

		x_new = x + dx*tf.cast(update_mask,tf.float32)
		
		if self.PADDING!="zero":
			x_new = x_new[:,2:-2,2:-2]
		

			
		return x_new


	def run_init(self,x0,T,N_BATCHES=1,ENV_CHANNELS=None):
		"""
			Helper function that initialises some stuff needed for running NCA trajectories.
			
			Parameters
			----------
			x0 : float32 array [batches,size,size,channels]
				Initial condition for NCA simulation
			T : int 
				number of timesteps to run for		
			N_BATCHES : int=1
				number of batches of simulations to run in parallel
			
			Returns	
			-------
			trajectory : float32 array [T,batches,size,size,channels]
				array of zeros where trajectory will be written to
		"""
		#print("X0 shape: "+str(x0.shape))
		#--- Initialise stuff
		TARGET_SIZE = x0.shape[1]
		x0 = x0[0:N_BATCHES] # If initial condition is too wide in batches dimension, reduce it
		trajectory = np.zeros((T,N_BATCHES,TARGET_SIZE,TARGET_SIZE,self.N_CHANNELS),dtype="float32")

		#--- Setup initial conditions
		if x0.shape[-1]<self.N_CHANNELS: # If x0 has less channels than the NCA, pad zeros to x0
			z0 = tf.zeros((N_BATCHES,TARGET_SIZE,TARGET_SIZE,self.N_CHANNELS-x0.shape[-1]),dtype="float32")
			x0 = tf.concat((x0,z0),axis=-1)

		
		assert trajectory[0].shape == x0.shape
		
		if ENV_CHANNELS is not None:
			self.setup_env_channels(ENV_CHANNELS)
		if self.ENV_CHANNEL_DATA is not None:
			x0 = self.env_channel_callback(x0)
			
		trajectory[0] = x0
		return trajectory


	def run(self,x0,T,N_BATCHES=1,ENV_CHANNELS=None,PADDING=None):
		"""
			Iterates self.call several times to perform a NCA simulation.
			
			Parameters
			----------
			x0 : float32 array [batches,size,size,channels]
				Initial condition for NCA simulation
			T : int 
				number of timesteps to run for		
			N_BATCHES : int=1
				number of batches of simulations to run in parallel
			ENV_CHANNELS : optional boolean array [size,size]
				Mask representing presence of environmental factors (i.e. adhesive surfaces for cells)
			PADDING : optional string
				Overwrites self.PADDING if provided

			Returns	
			-------
			trajectory : float32 array [T,batches,size,size,channels]
				time series resulting from running NCA for T steps starting at x0
		"""

		#--- Optionally overwrite padding and environment variables at runtime
		if PADDING is not None:
			self.PADDING=PADDING



		trajectory=self.run_init(x0,T,N_BATCHES=N_BATCHES,ENV_CHANNELS=ENV_CHANNELS)
		
		#--- Run T iterations of NCA
		for t in range(1,T):
			trajectory[t] = self.call(trajectory[t-1])
			if self.ENV_CHANNEL_DATA is not None:
				trajectory[t] = self.env_channel_callback(trajectory[t])
		trajectory = tf.convert_to_tensor(trajectory, dtype=tf.float32)

		return trajectory


	def run_detailed(self,x0,T,N_BATCHES=1,ENV_CHANNELS=None):
		"""
			Iterates self.call several times to perform a NCA simulation.
			Also tracks and returns trajectories of increments for each kernel
			
			Parameters
			----------
			x0 : float32 array [batches,size,size,channels]
				Initial condition for NCA simulation
			T : int 
				number of timesteps to run for		
			N_BATCHES : int=1
				number of batches of simulations to run in parallel
			
			Returns	
			-------
			trajectory : float32 array [T,batches,size,size,channels]
				time series resulting from running NCA for T steps starting at x0
			k_trajectory : float32 array [T,batches,kernels,size,size,channels]
				time series of increments
		"""

		#--- Initialise trajectory variables
		K = self.KERNEL.shape[3]
		trajectory=self.run_init(x0,T,N_BATCHES=N_BATCHES,ENV_CHANNELS=ENV_CHANNELS)
		TARGET_SIZE = x0.shape[1]
		k_trajectory = np.zeros((T,N_BATCHES,K,TARGET_SIZE,TARGET_SIZE,self.N_CHANNELS))
		k_trajectory_dx = np.zeros((T,N_BATCHES,K,TARGET_SIZE,TARGET_SIZE,self.N_CHANNELS))
		SUBKERNELS = tf.repeat(self.KERNEL[...,None],K,-1).numpy()
		print(SUBKERNELS.shape)
		for i in range(K):
			mask = np.zeros(self.KERNEL.shape).astype(int)
			mask[:,:,:,i]=1
			SUBKERNELS[...,i]*=mask
		SUBKERNELS = tf.convert_to_tensor(SUBKERNELS)

		@tf.function
		def partial_perception(x,i):
			y = tf.nn.depthwise_conv2d(x,SUBKERNELS[...,i],[1,1,1,1],"SAME")
			return y


		@tf.function
		def partial_call(x,i,step_size=1.0,fire_rate=None):
			"""
				Applies a neural network (the trainable part of the model) to the partial perception field.
				
				Parameters
				----------
				x : float32 tensor [batches,size,size,N_CHANNELS]
					state space of NCA, first 4 channels are typically visualised as RGBA,
					rest are hidden channels
				step_size : float=1.0
					scale size of updates
				fire_rate : float=None
					controls probability of each pixel updating
				i : int
					Which kernel to select

				Returns
				-------
				x_new : float32 tensor
					new state space of NCA, with (stochastically masked) update applied across all channels and batches
			"""
			#print(x.shape)
			if self.ENV_CHANNELS is not None:
				x = self.env_channel_callback(x)
			y = partial_perception(x,i)
			#print(y.shape)
			dx = self.dense_model(y)*step_size
			if fire_rate is None:
				fire_rate = self.FIRE_RATE
			update_mask = tf.random.normal(tf.shape(x[:,:,:,:1])) <= fire_rate

			x_new = x + dx*tf.cast(update_mask,tf.float32)
			return x_new,dx

		
		#--- Run T iterations of NCA
		for t in range(1,T):
			trajectory[t] = self.call(trajectory[t-1])
			for k in range(K):
				k_trajectory[t,:,k],k_trajectory_dx[t,:,k] = partial_call(trajectory[t-1],k)
			
		trajectory = tf.convert_to_tensor(trajectory, dtype=tf.float32)
		return trajectory,k_trajectory,k_trajectory_dx

	def get_config(self):
		return {"N_CHANNELS":self.N_CHANNELS,
				"FIRE_RATE": self.FIRE_RATE,
				"LAYERS":self.N_layers,
				"KERNEL_TYPE":self.KERNEL_TYPE,
				"ORDER":self.ORDER,
				"ACTIVATION":self.ACTIVATION,
				"ENV_CHANNEL_DATA":self._ENV_CHANNEL_DATA_raw}
				#"dense_model":self.dense_model}
	
	@classmethod
	def from_config(cls,config):
		return cls(**config)

	
	def save_wrapper(self,filename=None,directory="models/"):
		"""
			Saves the trainable part of the model - the dense nn trained on the perception field.
			Wrapper for keras.models.save function, puts things in the right directory etc.

			Parameters
			----------
			filename : str optional
				Name of subdirectory where keras SavedModel files are contained
			directory : str optional
				Name of directory where all models get stored, defaults to 'models/'
		"""

		if filename is None:
			filename=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		
		self.save(directory+filename)



def load_wrapper(filename,directory="models/"):
	"""
		Loads the trainable part of the model - the dense nn trained on the perception field

		Parameters
		----------
		filename : str
			Name of subdirectory where keras SavedModel files are contained
		directory : str optional
			Name of directory where all models get stored, defaults to 'models/'
			
	"""
	return tf.keras.models.load_model(directory+filename,custom_objects={"NCA":NCA})
	

	
	
