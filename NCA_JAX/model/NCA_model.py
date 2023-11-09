import jax
import jax.numpy as jnp
import equinox as eqx
import time
from pathlib import Path
from typing import Union

class NCA(eqx.Module):
	layers: list
	KERNEL_STR: list
	N_CHANNELS: int
	N_FEATURES: int
	PERIODIC: bool
	FIRE_RATE: float
	N_WIDTH: int
	def __init__(self,
			     N_CHANNELS,
				 KERNEL_STR=["ID","LAP"],
				 ACTIVATION_STR="relu",
				 PERIODIC=True,
				 FIRE_RATE=1.0,
				 key=jax.random.PRNGKey(int(time.time()))):
		"""
		

		Parameters
		----------
		N_CHANNELS : int
			Number of channels for NCA.
		KERNEL_STR : [STR], optional
			List of strings corresponding to convolution kernels. Can include "ID","DIFF","LAP","AV", corresponding to
			identity, derivatives, laplacian and average respectively. The default is ["ID","LAP"].
		ACTIVATION_STR : str, optional
			Decide which activation function to use. The default is "relu".
		PERIODIC : Boolean, optional
			Decide whether to have periodic or fixed boundaries. The default is True.
		FIRE_RATE : float, optional
			Probability that each pixel updates at each timestep. Defuaults to 1, i.e. deterministic update
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).

		Returns
		-------
		None.

		"""
		
		
		key1,key2 = jax.random.split(key,2)
		self.PERIODIC=PERIODIC
		self.N_CHANNELS = N_CHANNELS
		self.FIRE_RATE = FIRE_RATE
		self.KERNEL_STR = KERNEL_STR
		self.N_WIDTH = 1
		
		# Define which convolution kernels to use
		KERNELS = []
		if "ID" in KERNEL_STR:
			KERNELS.append(jnp.array([[0,0,0],[0,1,0],[0,0,0]]))
		if "AV" in KERNEL_STR:
			KERNELS.append(jnp.array([[1,1,1],[1,1,1],[1,1,1]])/9.0)
		if "DIFF" in KERNEL_STR:
			dx = jnp.outer(jnp.array([1.0,2.0,1.0]),jnp.array([-1.0,0.0,1.0]))/8.0
			dy = dx.T
			KERNELS.append(dx)
			KERNELS.append(dy)
		if "LAP" in KERNEL_STR:
			KERNELS.append(jnp.array([[0.25,0.5,0.25],[0.5,-3,0.5],[0.25,0.5,0.25]]))
		
		self.N_FEATURES = N_CHANNELS*len(KERNELS)
		KERNELS = jnp.array(KERNELS) # OHW layout
		KERNELS = jnp.zeros((self.N_CHANNELS,self.N_FEATURES//self.N_CHANNELS,3,3)) + KERNELS[jnp.newaxis]# OIHW layout
		KERNELS = jnp.reshape(KERNELS,(-1,3,3))
		KERNELS = jnp.expand_dims(KERNELS,1)
		
		# Define which activation function to use
		if ACTIVATION_STR == "relu":
			ACTIVATION = jax.nn.relu
		elif ACTIVATION_STR == "tanh":
			ACTIVATION = jax.nn.tanh
		else:
			ACTIVATION = None
		
		@jax.jit
		def periodic_pad(x):
			if self.PERIODIC:
				return jnp.pad(x, ((0,0),(1,1),(1,1)), mode='wrap')
			else:
				return x
		
		@jax.jit
		def periodic_unpad(x):
			if self.PERIODIC:
				return x[:,1:-1,1:-1]
			else:
				return x
		

		self.layers = [
			periodic_pad,
			eqx.nn.Conv2d(in_channels=self.N_CHANNELS,
						  out_channels=self.N_FEATURES,
						  kernel_size=3,
						  use_bias=False,
						  key=key1,
						  padding=1,
						  groups=self.N_CHANNELS),
			periodic_unpad,
			eqx.nn.Conv2d(in_channels=self.N_FEATURES,
						  out_channels=self.N_WIDTH*self.N_FEATURES,
						  kernel_size=1,
						  use_bias=False,
						  key=key1),
			ACTIVATION,
			eqx.nn.Conv2d(in_channels=self.N_WIDTH*self.N_FEATURES, 
						  out_channels=self.N_CHANNELS,
						  kernel_size=1,
						  use_bias=True,
						  key=key2)
			]
		
		
		
		
		# Initialise final layer to zero
		w_zeros = jnp.zeros((self.N_CHANNELS,self.N_WIDTH*self.N_FEATURES,1,1))
		b_zeros = jnp.zeros((self.N_CHANNELS,1,1))
		w_where = lambda l: l.weight
		b_where = lambda l: l.bias
		self.layers[-1] = eqx.tree_at(w_where,self.layers[-1],w_zeros)
		self.layers[-1] = eqx.tree_at(b_where,self.layers[-1],b_zeros)
		
		# Initialise first layer weights as perception kernels
		self.layers[1] = eqx.tree_at(w_where,self.layers[1],KERNELS)
		

		
	def __call__(self,x,boundary_callback=lambda x:x,key=jax.random.PRNGKey(int(time.time()))):
		"""
		

		Parameters
		----------
		x : float32 [N_CHANNELS,_,_]
			input NCA lattice state.
		boundary_callback : callable (float32 [N_CHANNELS,_,_]) -> (float32 [N_CHANNELS,_,_]), optional
			function to augment intermediate NCA states i.e. imposing complex boundary conditions or external structure. Defaults to None
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).

		Returns
		-------
		x : float32 [N_CHANNELS,_,_]
			output NCA lattice state.

		"""
		

		dx = x

		for layer in self.layers:
			dx = layer(dx)
		sigma = jax.random.bernoulli(key,p=self.FIRE_RATE,shape=dx.shape)
		x_new = x + sigma*dx

		#x_new = augment_callback(x_new)
		#print(x_new.shape)
		#print(boundary_callback)
		return boundary_callback(x_new)
		#return x + dx
	
	def partition(self):
		"""
		Behaves like eqx.partition, but moves the hard coded kernels (a jax array) from the "trainable" pytree to the "static" pytree

		Returns
		-------
		diff : PyTree
			PyTree of same structure as NCA, with all non trainable parameters set to None
		static : PyTree
			PyTree of same structure as NCA, with all trainable parameters set to None

		"""
		
		where = lambda nca: nca.layers[1].weight
		kernel = self.layers[1].weight
		diff,static = eqx.partition(self,eqx.is_array)
		diff = eqx.tree_at(where,diff,None)
		static = eqx.tree_at(where,static,kernel,is_leaf=lambda x: x is None)
		return diff, static
	
	def combine(self,diff,static):
		"""
		Wrapper for eqx.combine

		Parameters
		----------
		diff : PyTree
			PyTree of same structure as NCA, with all non trainable parameters set to None
		static : PyTree
			PyTree of same structure as NCA, with all trainable parameters set to None

		"""
		self = eqx.combine(diff,static)
		

	def save(self, path: Union[str, Path], overwrite: bool = False):
		"""
		Wrapper for saving NCA via pickle. Taken from https://github.com/google/jax/issues/2116

		Parameters
		----------
		path : Union[str, Path]
			path to filename.
		overwrite : bool, optional
			Overwrite existing filename. The default is False.

		Raises
		------
		RuntimeError
			file already exists.

		Returns
		-------
		None.

		"""
		suffix = ".eqx"
		path = Path(path)
		if path.suffix != suffix:
			path = path.with_suffix(suffix)
			path.parent.mkdir(parents=True, exist_ok=True)
		if path.exists():
			if overwrite:
				path.unlink()
			else:
				raise RuntimeError(f'File {path} already exists.')
		eqx.tree_serialise_leaves(path,self)
		#with open(path, 'wb') as file:	
			#pickle.dump(self, file)
	
	def load(self, path: Union[str, Path]):
		"""
		

		Parameters
		----------
		path : Union[str, Path]
			path to filename.

		Raises
		------
		ValueError
			Not a file or incorrect file type.

		Returns
		-------
		NCA
			NCA loaded from pickle.

		"""
		suffix = ".eqx"
		path = Path(path)
		if not path.is_file():
			raise ValueError(f'Not a file: {path}')
		if path.suffix != suffix:
			raise ValueError(f'Not a {suffix} file: {path}')
		#with open(path, 'rb') as file:
		#	data = pickle.load(file)
		return eqx.tree_deserialise_leaves(path,self)
		
	def run(self,iters,x,callback,key=jax.random.PRNGKey(int(time.time()))):
		trajectory = []
		trajectory.append(x)
		for i in range(iters):
			key = jax.random.fold_in(key,i)
			x = self(x,callback,key=key)
			trajectory.append(x)
		return jnp.array(trajectory)
		
