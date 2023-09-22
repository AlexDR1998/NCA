import jax.numpy as jnp
import jax
import time
import equinox as eqx
from jax.experimental import mesh_utils

class DataAugmenter(object):
	
	def __init__(self,data_true,hidden_channels=0):
		"""
		Class for handling data augmentation for NCA training. 
		data_init is called before training,
		data_callback is called during training
		
		Also handles JAX array sharding, so all methods of NCA_trainer work
		on multi-gpu setups. Currently splits data onto different GPUs by batches

		Parameters
		----------
		data_true : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			true un-augmented data
		hidden_channels : int optional
			number of hidden channels to zero-pad to data. Defaults to zero
		"""
		self.OBS_CHANNELS = data_true.shape[2]
		
		data_true = jnp.pad(data_true,((0,0),(0,0),(0,hidden_channels),(0,0),(0,0))) # Pad zeros onto hidden channels
		
		self.data_true = data_true
		self.data_saved = data_true

	def data_init(self,SHARDING = None):
		"""
		Chain together various data augmentations to perform at intialisation of NCA training

		"""
		data = self.return_saved_data()
		if SHARDING is not None:
			data = self.duplicate_batches(data, SHARDING)
			data = self.pad(data,10)
			shard = jax.sharding.PositionalSharding(mesh_utils.create_device_mesh((SHARDING,1,1,1,1)))
			data = jax.device_put(data,shard)
			jax.debug.visualize_array_sharding(data[:,0,0,0])
		else:	
			data = self.duplicate_batches(data, 4)
			data = self.pad(data, 10)
		
		self.save_data(data)
		return None
		
	#@eqx.filter_jit
	def data_callback(self,x,y,i):
		"""
		Called after every training iteration to perform data augmentation and processing		


		Parameters
		----------
		x : float32[N-N_steps,BATCHES,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : float32[N-N_steps,BATCHES,CHANNELS,WIDTH,HEIGHT]
			Final states
		i : int
			Current training iteration - useful for scheduling mid-training data augmentation

		Returns
		-------
		x : float32[N-N_steps,BATCHES,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : float32[N-N_steps,BATCHES,CHANNELS,WIDTH,HEIGHT]
			Final states

		"""
		am=10
		
		if hasattr(self,"PREVIOUS_KEY"):
			x = self.unshift(x, am, self.PREVIOUS_KEY)
			y = self.unshift(y, am, self.PREVIOUS_KEY)

		
		x = x.at[:,1:].set(x[:,:-1]) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
		x_true,_ =self.split_x_y(1)
		x = x.at[:,0].set(x_true[:,0]) # Keep first initial x correct
		if i < 500:
			x = x.at[::2,:,:self.OBS_CHANNELS].set(x_true[::2,:,:self.OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
		key=jax.random.PRNGKey(int(time.time()))

		x = self.shift(x,am,key=key)
		y = self.shift(y,am,key=key)
		self.PREVIOUS_KEY = key
		return x,y
		
	def split_x_y(self,N_steps=1):
		"""
		Splits data into x (initial conditions) and y (final states). 
		Offset by N_steps in N, so x[:,N]->y[:,N+N_steps] is learned

		Parameters
		----------
		N_steps : int, optional
			How many steps along data trajectory to learn update rule for. The default is 1.

		Returns
		-------
		x : float32[BATCHES,N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : float32[BATCHES,N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Final states

		"""
		return self.data_saved[:,:-N_steps],self.data_saved[:,N_steps:]
	@eqx.filter_jit
	def concat_x_y(self,x,y):
		"""
		Joins x and y together as one data tensor along time axis - useful for 
		data processing that affects each batch differently, but where
		identitcal transformations are needed for x and y. i.e. random shifting

		Parameters
		----------
		x : float32[:,N,...]
			Initial conditions
		y : float32[:,N,...]
			Final states

		Returns
		-------
		data : float32[:,2N,...]
			x and y concatenated along axis 1
		"""
		return jnp.concatenate((x,y),axis=1)
	@eqx.filter_jit
	def unconcat_x_y(self,data):
		"""
		Inverse of concat_x_y

		Parameters
		----------
		data : float32[:,2N,...]
			x and y concatenated along axis 0
		Returns
		-------

		x : float32[:,N,...]
			Initial conditions
		y : float32[:,N,...]
			Final states

		"""
		midpoint = data.shape[1]//2
		return data[:,:midpoint],data[:,midpoint:]
	
	@eqx.filter_jit
	def pad(self,data,am):
		"""
		
		Pads spatial dimensions with zeros

		Parameters
		----------
		data : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		am : int
			width to pad with zeros in spatial dimension

		Returns
		-------
		data : float32[BATCHES,N,CHANNELS,WIDTH+2*am,HEIGHT+2*am]
			data padded with zeros

		"""
		return jnp.pad(data,((0,0),(0,0),(0,0),(am,am),(am,am)))
	
	#@eqx.filter_jit
	def shift(self,data,am,key=jax.random.PRNGKey(int(time.time()))):
		"""
		Randomly shifts each trajectory. 

		Parameters
		----------
		data : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		am : int
			possible width to shift by in spatial dimension
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).
			
		Returns
		-------
		data : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data randomly shifted in spatial dimensions

		"""

			
		shifts = jax.random.randint(key,minval=-am,maxval=am,shape=(data.shape[1],2))
		
		for b in range(data.shape[1]):
			data = data.at[b].set(jnp.roll(data[b],shifts[b],axis=(-1,-2)))
		return data
	
	def unshift(self,data,am,key):
		"""
		Randomly shifts each trajectory. If useing same key as shift(), it undoes that shift

		Parameters
		----------
		data : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		am : int
			possible width to shift by in spatial dimension
		key : jax.random.PRNGKey
			Jax random number key.
			
		Returns
		-------
		data : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data randomly shifted in spatial dimensions

		"""

			
		shifts = jax.random.randint(key,minval=-am,maxval=am,shape=(data.shape[1],2))
		
		for b in range(data.shape[1]):
			data = data.at[b].set(jnp.roll(data[b],-shifts[b],axis=(-1,-2)))
		return data
	
	
	
	
	def noise(self,data,am,full=True,key=jax.random.PRNGKey(int(time.time()))):
		"""
		Adds uniform noise to the data
		
		Parameters
		----------
		data : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		am : float in (0,1)
			amount of noise, with 0 being none and 1 being pure noise
		full : boolean optional
			apply noise to observable channels, or all channels?. Defaults to True (all channels)
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).
		Returns
		-------
		noisy : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			noisy data

		"""
		noisy = am*jax.random.uniform(key,shape=data.shape) + (1-am)*data
		if not full:
			#noisy[:,:,self.OBS_CHANNELS:] = data[:,:,self.OBS_CHANNELS:]
			noisy = noisy.at[:,:,self.OBS_CHANNELS:].set(data[:,:,self.OBS_CHANNELS:])
		return noisy
		
	@eqx.filter_jit
	def duplicate_batches(self,data,B):
		"""
		Repeats data along batches axis by B

		Parameters
		----------
		data : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		B : int
			number of repetitions

		Returns
		-------
		data : float32[N,B*BATCHES,CHANNELS,WIDTH,HEIGHT]
			data augmented along batch axis

		"""
		
		return jnp.repeat(data,B,axis=0)
	
	
	def save_data(self,data):
		self.data_saved = data

	def return_saved_data(self):		
		return self.data_saved
	
	def return_true_data(self):
		return self.data_true
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		