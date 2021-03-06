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
	
	"""


	def __init__(self,N_CHANNELS,FIRE_RATE=0.5,DECAY_FACTOR=1.0,ADHESION_MASK=None):
		"""
			Initialiser for neural cellular automata object

			Parameters
			----------
			N_CHANNELS : int
				Number of channels to include - first 4 are visible as RGBA, others are hidden
			FIRE_RATE : float in [0,1]
				Controls stochasticity of cell updates, at 1 all cells update every step, at 0 no cells update
			DECAY_FACTOR : float in [0,1]
				Controls how quickly hidden channel values degrade over time. 1 = no decay, 0 = instant decay
			ADHESION_MASK : boolean array [size,size] optional
				A binary mask indicating the presence of the adesive micropattern surface
		"""



		super(NCA,self).__init__()
		self.N_CHANNELS=N_CHANNELS # RGBA +hidden layers
		self.FIRE_RATE=FIRE_RATE # controls stochastic updates - i.e. grid isn't globaly synchronised
		self.DECAY_FACTOR=DECAY_FACTOR



		if ADHESION_MASK is not None:
			self.ADHESION_MASK=np.repeat(ADHESION_MASK[...,np.newaxis],N_CHANNELS,axis=-1) # Signals where cells can adhere to the micropattern
			
		else:
			self.ADHESION_MASK=None
			ones = tf.ones(4)
			decay = tf.ones(N_CHANNELS-4)*self.DECAY_FACTOR

		
		#--- Set up dense nn for perception vector
		self.dense_model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(4*self.N_CHANNELS,1,activation=tf.nn.swish,kernel_regularizer=tf.keras.regularizers.L1(0.001)),
			tf.keras.layers.Conv2D(2*self.N_CHANNELS,1,activation=tf.nn.swish,kernel_regularizer=tf.keras.regularizers.L1(0.001)),
			tf.keras.layers.Conv2D(self.N_CHANNELS,1,activation=None,kernel_initializer=tf.keras.initializers.Zeros())])

		self.N_layers = 2
		#--- Set up convolution kernels
		_i = np.array([0,1,0],dtype=np.float32)
		I  = np.outer(_i,_i)
		dx = (np.outer([1,2,1],[-1,0,1])/8.0).astype(np.float32)
		dy = dx.T
		lap = np.array([[0.25,0.5,0.25],
						[0.5,-3,0.5],
						[0.25,0.5,0.25]]).astype(np.float32)
		av = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.float32)/9.0
		kernel = tf.stack([I,dx,dy,lap,av],-1)[:,:,None,:]
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
		print("Hidden channel decay factor:    {df}".format(df=self.DECAY_FACTOR))
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

		#--- Initialise stuff
		TARGET_SIZE = x0.shape[1]
		x0 = x0[0:N_BATCHES] # If initial condition is too wide in batches dimension, reduce it
		trajectory = np.zeros((T,N_BATCHES,TARGET_SIZE,TARGET_SIZE,self.N_CHANNELS),dtype="float32")
	
		#--- Setup initial conditions
		if x0.shape[-1]<self.N_CHANNELS: # If x0 has less channels than the NCA, pad zeros to x0
			z0 = np.zeros((N_BATCHES,TARGET_SIZE,TARGET_SIZE,self.N_CHANNELS-x0.shape[-1]),dtype="float32")
			x0 = np.concatenate((x0,z0),axis=-1)
		assert trajectory[0].shape == x0.shape
		
		

		#--- Setup the adhesion and decay mask, if provided
		if ADHESION_MASK is not None:	
			self.ADHESION_MASK=np.repeat(ADHESION_MASK[...,np.newaxis],self.N_CHANNELS,axis=-1)
			if (self.ADHESION_MASK.shape[0]==1) and (N_BATCHES>1):
				self.ADHESION_MASK=np.repeat(self.ADHESION_MASK,N_BATCHES,axis=0)
			ones = np.ones(5)
			decay = np.ones(self.N_CHANNELS-5)*self.DECAY_FACTOR
		if self.ADHESION_MASK is not None:
			if (self.ADHESION_MASK.shape[0]==1) and (N_BATCHES>1):
				self.ADHESION_MASK=np.repeat(self.ADHESION_MASK,N_BATCHES,axis=0)
			_mask = np.zeros((x0.shape),dtype="float32")
			_mask[...,4]=1
			x0 = _mask*self.ADHESION_MASK[:N_BATCHES] + (1-_mask)*x0
			ones = np.ones(5)
			decay = np.ones(self.N_CHANNELS-5)*self.DECAY_FACTOR
		if self.ADHESION_MASK is None:
			ones = np.ones(4)
			decay = np.ones(self.N_CHANNELS-4)*self.DECAY_FACTOR
		trajectory[0] = x0
		

		decay_mask_single = np.concatenate((ones,decay),axis=0)
		_decay_mask = decay_mask_single[np.newaxis,np.newaxis,np.newaxis]
		self.DECAY_MASK = tf.cast(np.tile(_decay_mask,(N_BATCHES,TARGET_SIZE,TARGET_SIZE,1)),dtype=tf.float32)
		"""
		print("Decay mask shape:")
		print(self.decay_mask.shape)
		print("Hidden decay rate:")
		print(self.decay_mask[...,6])
		print("Observable decay rate:")
		print(self.decay_mask[...,0])
		"""
		#--- Run T iterations of NCA
		for t in range(1,T):
			trajectory[t] = self.call(trajectory[t-1])
			if self.ADHESION_MASK is not None:
				trajectory[t] = _mask*self.ADHESION_MASK[:N_BATCHES] + (1-_mask)*self.DECAY_MASK*trajectory[t]
			else:
				trajectory[t] = self.DECAY_MASK*trajectory[t]
		
		return trajectory


	
	def get_config(self):
		return {"N_CHANNELS":self.N_CHANNELS,
				"FIRE_RATE": self.FIRE_RATE,
				"DECAY_FACTOR":self.DECAY_FACTOR}
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

def load_wrapper(filename):
	"""
		Loads the trainable part of the model - the dense nn trained on the perception field

		Parameters
		----------
		filename : str
			Name of directory where keras SavedModel files are contained
	"""
	return tf.keras.models.load_model("models/"+filename,custom_objects={"NCA":NCA})
	

	
	
