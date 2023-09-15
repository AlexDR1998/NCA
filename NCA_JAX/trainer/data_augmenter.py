import jax.numpy as jnp
import jax
import time

class DataAugmenter(object):
	def __init__(self,data_true,hidden_channels=0):
		"""
		

		Parameters
		----------
		data_true : float32[N,BATCHES,CHANNELS,WIDTH,HEIGHT]
			true un-augmented data
		hidden_channels : int optional
			number of hidden channels to zero-pad to data. Defaults to zero
		"""
		self.OBS_CHANNELS = data_true.shape[2]
		
		data_true = jnp.pad(data_true,((0,0),(0,0),(0,hidden_channels),(0,0),(0,0)))
		
		self.data_true = data_true
		self.data_saved = data_true

		
	def data_init(self,B,am):
		"""
		Chain together various data augmentations to perform at intialisation of NCA training

		Parameters
		----------
		B : int
			NUmber of batches to duplicate.
		am : int
			Width to pad and shift with

		"""
		data = self.return_saved_data()
		data = self.duplicate_batches(data, B)
		data = self.pad(data, am)
		data = self.shift(data, am)
		self.save_data(data)
		
		
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
		
		
		x = x.at[1:].set(x[:-1]) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
		x_true,_ =self.split_x_y(1)
		x = x.at[0].set(x_true[0]) # Keep first initial x correct
		x = x.at[:,::2].set(x_true[:,::2]) # Set every other intermediate initial condition to correct initial condition
		return x,y
		
	def split_x_y(self,N_steps=1):
		"""
		Splits data into x (initial conditions) and y (final states). 
		Offset by N_steps in N, so x[N]->y[N+N_steps] is learned

		Parameters
		----------
		N_steps : int, optional
			How many steps along data trajectory to learn update rule for. The default is 1.

		Returns
		-------
		x : float32[N-N_steps,BATCHES,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : float32[N-N_steps,BATCHES,CHANNELS,WIDTH,HEIGHT]
			Final states

		"""
		return self.data_saved[:-N_steps],self.data_saved[N_steps:]
	
	
	def pad(self,data,am):
		"""
		
		Pads spatial dimensions with zeros

		Parameters
		----------
		data : float32[N,BATCHES,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		am : int
			width to pad with zeros in spatial dimension

		Returns
		-------
		data : float32[N,BATCHES,CHANNELS,WIDTH+2*am,HEIGHT+2*am]
			data padded with zeros

		"""
		return jnp.pad(data,((0,0),(0,0),(0,0),(am,am),(am,am)))
	
	
	def shift(self,data,am,key=jax.random.PRNGKey(int(time.time()))):
		"""
		Randomly shifts each trajectory. 

		Parameters
		----------
		data : float32[N,BATCHES,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		am : int
			possible width to shift by in spatial dimension
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).
			
		Returns
		-------
		data : float32[N,BATCHES,CHANNELS,WIDTH,HEIGHT]
			data randomly shifted in spatial dimensions

		"""
		
			
		shifts = jax.random.randint(key,minval=-am,maxval=am,shape=(data.shape[1],2))
		for b in range(data.shape[1]):
			#data[:,b] = jnp.roll(data[:,b],shifts[b],axis=(-1,-2))
			data = data.at[:,b].set(jnp.roll(data[:,b],shifts[b],axis=(-1,-2)))
		return data
	
	def noise(self,data,am,full=True,key=jax.random.PRNGKey(int(time.time()))):
		"""
		Adds uniform noise to the data
		
		Parameters
		----------
		data : float32[N,BATCHES,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		am : float in (0,1)
			amount of noise, with 0 being none and 1 being pure noise
		full : boolean optional
			apply noise to observable channels, or all channels?. Defaults to True (all channels)
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).
		Returns
		-------
		noisy : float32[N,BATCHES,CHANNELS,WIDTH,HEIGHT]
			noisy data

		"""
		noisy = am*jax.random.uniform(key,shape=data.shape) + (1-am)*data
		if not full:
			#noisy[:,:,self.OBS_CHANNELS:] = data[:,:,self.OBS_CHANNELS:]
			noisy = noisy.at[:,:,self.OBS_CHANNELS:].set(data[:,:,self.OBS_CHANNELS:])
		return noisy
		
	
	def duplicate_batches(self,data,B):
		"""
		Repeats data along batches axis by B

		Parameters
		----------
		data : float32[N,BATCHES,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		B : int
			number of repetitions

		Returns
		-------
		data : float32[N,B*BATCHES,CHANNELS,WIDTH,HEIGHT]
			data augmented along batch axis

		"""
		
		return jnp.repeat(data,B,axis=1)
	
	
	def save_data(self,data):
		self.data_saved = data

	def return_saved_data(self):		
		return self.data_saved
	
	def return_true_data(self):
		return self.data_true
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		