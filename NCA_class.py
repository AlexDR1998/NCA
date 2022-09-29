import numpy as np
import tensorflow as tf
import scipy as sp
import datetime


class NCA(tf.keras.Model):
	""" 
		Neural Cellular Automata class. Is a subclass of keras.model,
		so that model loading and saving are inherited. 
		Heavily inspired by work at https://distill.pub/2020/growing-ca/ 
		modified and extended for purpose of modelling stem cell differentiation experiments

		Don't use this directly, instead use one of the subclasses with specified neural net architecture
	
	"""


	def __init__(self,N_CHANNELS,FIRE_RATE=0.5,ADHESION_MASK=None,ACTIVATION="swish",LAYERS=2,REGULARIZER=0.01):
		"""
			Initialiser for neural cellular automata object

			Parameters
			----------
			N_CHANNELS : int
				Number of channels to include - first 4 are visible as RGBA, others are hidden
			FIRE_RATE : float in [0,1]
				Controls stochasticity of cell updates, at 1 all cells update every step, at 0 no cells update
			ADHESION_MASK : boolean array [size,size] optional
				A binary mask indicating the presence of the adesive micropattern surface
			ACTIVATION : string optional
				String corresponding to tensorflow activation functions
			REGULARIZER : float optional
				Regularizer coefficient for layer weights
		"""



		super(NCA,self).__init__()
		self.N_CHANNELS=N_CHANNELS # RGBA +hidden layers
		self.FIRE_RATE=FIRE_RATE # controls stochastic updates - i.e. grid isn't globaly synchronised
		self.N_layers = LAYERS
		self.ACTIVATION = ACTIVATION



		if ADHESION_MASK is not None:
			self.ADHESION_MASK=np.repeat(ADHESION_MASK[...,np.newaxis],N_CHANNELS,axis=-1) # Signals where cells can adhere to the micropattern
			
		else:
			self.ADHESION_MASK=None
			ones = tf.ones(4)

		
		#--- Set up dense nn for perception vector - removed and added to subclasses
		
		
		if LAYERS==2:
			self.dense_model = tf.keras.Sequential([
				tf.keras.layers.Conv2D(4*self.N_CHANNELS,1,
									   activation=ACTIVATION,
									   kernel_regularizer=tf.keras.regularizers.L1(0.01),
									   use_bias=False),
				tf.keras.layers.Conv2D(2*self.N_CHANNELS,1,
									   activation=ACTIVATION,
									   kernel_regularizer=tf.keras.regularizers.L1(0.01),
									   use_bias=False),
				tf.keras.layers.Conv2D(self.N_CHANNELS,1,activation=None,kernel_initializer=tf.keras.initializers.Zeros())])
		
		elif LAYERS==1:
			self.dense_model = tf.keras.Sequential([
				tf.keras.layers.Conv2D(4*self.N_CHANNELS,1,
									   activation=ACTIVATION,
									   kernel_regularizer=tf.keras.regularizers.L1(0.01),
									   use_bias=False),
				tf.keras.layers.Conv2D(self.N_CHANNELS,1,activation=None,kernel_initializer=tf.keras.initializers.Zeros())])
		


		#--- Set up convolution kernels
		_i = np.array([0,1,0],dtype=np.float32)
		I  = np.outer(_i,_i)
		#dx = (np.outer([1,2,1],[-1,0,1])/8.0).astype(np.float32)
		#dy = dx.T
		lap = np.array([[0.25,0.5,0.25],
						[0.5,-3,0.5],
						[0.25,0.5,0.25]]).astype(np.float32)
		av = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.float32)/9.0
		kernel = tf.stack([I,lap,av],-1)[:,:,None,:]
		#kernel = tf.stack([I,av,dx,dy],-1)[:,:,None,:]
		#kernel = tf.stack([I,av],-1)[:,:,None,:]
		self.KERNEL = tf.repeat(kernel,self.N_CHANNELS,2)
		self(tf.zeros([1,3,3,self.N_CHANNELS])) # Dummy call to build the model
		print(self.dense_model.summary())
	
	
	def __str__(self):
		"""
			Print method
		"""
		
		print("Neural Cellular Automata model")
		print("_________________________________________________________________________________")
		print("Stochastic firing rate:         {fr}".format(fr=self.FIRE_RATE))
		print("Number of hidden channels:      {hc}".format(hc=self.N_CHANNELS))
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

		y = tf.nn.depthwise_conv2d(x,self.KERNEL,[1,1,1,1],"SAME")
		
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
		y = self.perceive(x)
		#print(y.shape)
		dx = self.dense_model(y)*step_size
		if fire_rate is None:
			fire_rate = self.FIRE_RATE
		update_mask = tf.random.normal(tf.shape(x[:,:,:,:1])) <= fire_rate

		x_new = x + dx*tf.cast(update_mask,tf.float32)
		return x_new


	def run_init(self,x0,T,N_BATCHES=1,ADHESION_MASK=None):
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
			_mask : float32 tensor [batvhes,size,size,channels]
				mask to indicate which channel contains information about adhesive surfaces or other environmental constants
		"""
		_mask = None
		#--- Initialise stuff
		TARGET_SIZE = x0.shape[1]
		x0 = x0[0:N_BATCHES] # If initial condition is too wide in batches dimension, reduce it
		trajectory = np.zeros((T,N_BATCHES,TARGET_SIZE,TARGET_SIZE,self.N_CHANNELS),dtype="float32")
	
		#--- Setup initial conditions
		if x0.shape[-1]<self.N_CHANNELS: # If x0 has less channels than the NCA, pad zeros to x0
			z0 = tf.zeros((N_BATCHES,TARGET_SIZE,TARGET_SIZE,self.N_CHANNELS-x0.shape[-1]),dtype="float32")
			x0 = tf.concat((x0,z0),axis=-1)
		assert trajectory[0].shape == x0.shape
		
		

		#--- Setup the adhesion and decay mask, if provided
		if ADHESION_MASK is not None:	
			self.ADHESION_MASK=np.repeat(ADHESION_MASK[...,np.newaxis],self.N_CHANNELS,axis=-1)
			if (self.ADHESION_MASK.shape[0]==1) and (N_BATCHES>1):
				self.ADHESION_MASK=np.repeat(self.ADHESION_MASK,N_BATCHES,axis=0)
			#ones = np.ones(5)
			
		if self.ADHESION_MASK is not None:
			if (self.ADHESION_MASK.shape[0]==1) and (N_BATCHES>1):
				self.ADHESION_MASK=np.repeat(self.ADHESION_MASK,N_BATCHES,axis=0)
			_mask = tf.zeros((x0.shape),dtype="float32")
			_mask[...,4]=1
			x0 = _mask*self.ADHESION_MASK[:N_BATCHES] + (1-_mask)*x0
			#ones = np.ones(5)
			self.ADHESION_MASK = tf.convert_to_tensor(self.ADHESION_MASK)
		#if self.ADHESION_MASK is None:
			#ones = np.ones(4)
			
		trajectory[0] = x0
		return trajectory,_mask


	def run(self,x0,T,N_BATCHES=1,ADHESION_MASK=None):
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
			
			Returns	
			-------
			trajectory : float32 array [T,batches,size,size,channels]
				time series resulting from running NCA for T steps starting at x0
		"""

		#--- Set up variables
		trajectory,_mask=self.run_init(x0,T,N_BATCHES=N_BATCHES,ADHESION_MASK=ADHESION_MASK)
		
		#--- Run T iterations of NCA
		for t in range(1,T):
			trajectory[t] = self.call(trajectory[t-1])
			if self.ADHESION_MASK is not None:
				trajectory[t] = _mask*self.ADHESION_MASK[:N_BATCHES] + (1-_mask)*trajectory[t]
		trajectory = tf.convert_to_tensor(trajectory, dtype=tf.float32)

		return trajectory


	def run_detailed(self,x0,T,N_BATCHES=1,ADHESION_MASK=None):
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

	def get_config(self):
		return {"N_CHANNELS":self.N_CHANNELS,
				"FIRE_RATE": self.FIRE_RATE}
				#"ADHESION_MASK":self.ADHESION_MASK,
				#"dense_model":self.dense_model}
	
	@classmethod
	def from_config(cls,config):
		return cls(**config)

	
	def save_wrapper(self,filename=None):
		"""
			Saves the trainable part of the model - the dense nn trained on the perception field.
			Wrapper for keras.models.save function, puts things in the right directory etc.

			Parameters
			----------
			filename : str
				Name of directory where keras SavedModel files are contained
		"""

		if filename is None:
			filename=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		
		self.save("models/"+filename)



"""===========================================================================================================================

	Below are subclasses of the NCA class with specific activation functions and network architectures

"""
"""

class NCA_sigmoid_2layer(NCA):
	
		#Sub-class of NCA with sigmoidal activation functions and 2 hidden layers
	

	def __init__(self,N_CHANNELS,FIRE_RATE=0.5,ADHESION_MASK=None):
		super(NCA_sigmoid_2layer,self).__init__(N_CHANNELS,FIRE_RATE=0.5,ADHESION_MASK=None)

		#--- Set up dense nn for perception vector
		self.dense_model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(4*self.N_CHANNELS,1,
								   activation=tf.nn.sigmoid,
								   kernel_regularizer=tf.keras.regularizers.L1(0.01),
								   use_bias=False),
			tf.keras.layers.Conv2D(2*self.N_CHANNELS,1,
								   activation=tf.nn.sigmoid,
								   kernel_regularizer=tf.keras.regularizers.L1(0.01),
								   use_bias=False),

			tf.keras.layers.Conv2D(self.N_CHANNELS,1,activation=None,kernel_initializer=tf.keras.initializers.Zeros())])

		self.N_layers = 2
		self(tf.zeros([1,3,3,self.N_CHANNELS])) # Dummy call to build the model
		print(self.dense_model.summary())

	def run(self,*args,**kwargs):
		return super().run(*args,**kwargs)

	#def call(self,*args,**kwargs):
	#	return super().call(*args,**kwargs)

	#def call(self,*args,**kwargs):
	#	return super().call(*args,**kwargs)


class NCA_sigmoid_1layer(NCA):
	
		Sub-class of NCA with sigmoidal activation functions and 1 hidden layers
	

	def __init__(self,N_CHANNELS,FIRE_RATE=0.5,ADHESION_MASK=None):
		super(NCA_sigmoid_1layer,self).__init__(N_CHANNELS,FIRE_RATE=0.5,ADHESION_MASK=None)

		#--- Set up dense nn for perception vector
		self.dense_model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(4*self.N_CHANNELS,1,
								   activation=tf.nn.sigmoid,
								   kernel_regularizer=tf.keras.regularizers.L1(0.01),
								   use_bias=False),

			tf.keras.layers.Conv2D(self.N_CHANNELS,1,activation=None,kernel_initializer=tf.keras.initializers.Zeros())])

		self.N_layers = 1
		self(tf.zeros([1,3,3,self.N_CHANNELS])) # Dummy call to build the model
		print(self.dense_model.summary())

class NCA_swish_2layer(NCA):
	
		Sub-class of NCA with sigmoidal activation functions and 2 hidden layers
	

	def __init__(self,N_CHANNELS,FIRE_RATE=0.5,ADHESION_MASK=None):
		super(NCA_swish_2layer,self).__init__(N_CHANNELS,FIRE_RATE=0.5,ADHESION_MASK=None)

		#--- Set up dense nn for perception vector
		self.dense_model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(4*self.N_CHANNELS,1,
								   activation=tf.nn.swish,
								   kernel_regularizer=tf.keras.regularizers.L1(0.01),
								   use_bias=False),
			tf.keras.layers.Conv2D(2*self.N_CHANNELS,1,
								   activation=tf.nn.swish,
								   kernel_regularizer=tf.keras.regularizers.L1(0.01),
								   use_bias=False),

			tf.keras.layers.Conv2D(self.N_CHANNELS,1,activation=None,kernel_initializer=tf.keras.initializers.Zeros())])

		self.N_layers = 2
		self(tf.zeros([1,3,3,self.N_CHANNELS])) # Dummy call to build the model
		print(self.dense_model.summary())

class NCA_swish_1layer(NCA):
	
		#Sub-class of NCA with sigmoidal activation functions and 1 hidden layers
	

	def __init__(self,N_CHANNELS,FIRE_RATE=0.5,ADHESION_MASK=None):
		super(NCA_swish_1layer,self).__init__(N_CHANNELS,FIRE_RATE=0.5,ADHESION_MASK=None)

		#--- Set up dense nn for perception vector
		self.dense_model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(4*self.N_CHANNELS,1,
								   activation=tf.nn.swish,
								   kernel_regularizer=tf.keras.regularizers.L1(0.01),
								   use_bias=False),

			tf.keras.layers.Conv2D(self.N_CHANNELS,1,activation=None,kernel_initializer=tf.keras.initializers.Zeros())])

		self.N_layers = 1
		self(tf.zeros([1,3,3,self.N_CHANNELS])) # Dummy call to build the model
		print(self.dense_model.summary())

class NCA_linear_1layer(NCA):
	
		#Sub-class of NCA with no activation functions and 1 hidden layers. Basically just a matrix multiplication
	

	def __init__(self,N_CHANNELS,FIRE_RATE=0.5,ADHESION_MASK=None):
		super(NCA_linear_1layer,self).__init__(N_CHANNELS,FIRE_RATE=0.5,ADHESION_MASK=None)

		#--- Set up dense nn for perception vector
		self.dense_model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(4*self.N_CHANNELS,1,
								   activation=None,
								   kernel_regularizer=tf.keras.regularizers.L1(0.01),
								   use_bias=False),

			tf.keras.layers.Conv2D(self.N_CHANNELS,1,activation=None,kernel_initializer=tf.keras.initializers.Zeros())])

		self.N_layers = 1
		self(tf.zeros([1,3,3,self.N_CHANNELS])) # Dummy call to build the model
		print(self.dense_model.summary())

"""




def load_wrapper(filename):
	"""
		Loads the trainable part of the model - the dense nn trained on the perception field

		Parameters
		----------
		filename : str
			Name of directory where keras SavedModel files are contained
	"""
	return tf.keras.models.load_model("models/"+filename,custom_objects={"NCA":NCA})
	

	
	
