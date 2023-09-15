import jax
import jax.numpy as jnp
import equinox as eqx
import time
from jax.tree_util import pytree
import pickle
from pathlib import Path
from typing import Union

class NCA(eqx.Module):
	layers: list
	KERNELS: jax.Array
	N_CHANNELS: int
	N_FEATURES: int
	PERIODIC: bool
	FIRE_RATE: float
	
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
		
		self.KERNELS = jnp.zeros((self.N_CHANNELS,self.N_FEATURES//self.N_CHANNELS,3,3)) # OIHW layout
		self.KERNELS+=KERNELS[jnp.newaxis]
		self.KERNELS = jnp.reshape(self.KERNELS,(-1,3,3))
		self.KERNELS = jnp.expand_dims(self.KERNELS,1)
		
		# Define which activation function to use
		if ACTIVATION_STR == "relu":
			ACTIVATION = jax.nn.relu
		elif ACTIVATION_STR == "tanh":
			ACTIVATION = jax.nn.tanh
		else:
			ACTIVATION = None
		
		
		
		# Define network model
		self.layers = [
			eqx.nn.Conv2d(in_channels=self.N_FEATURES,
						  out_channels=self.N_FEATURES,
						  kernel_size=1,
						  use_bias=False,
						  key=key1),
			ACTIVATION,
			eqx.nn.Conv2d(in_channels=self.N_FEATURES, 
						  out_channels=self.N_CHANNELS,
						  kernel_size=1,
						  use_bias=True,
						  key=key2)
			]
		
		# Initialise final layer to zero
		w_zeros = jnp.zeros((self.N_CHANNELS,self.N_FEATURES,1,1))
		b_zeros = jnp.zeros((self.N_CHANNELS,1,1))
		w_where = lambda l: l.weight
		b_where = lambda l: l.bias
		self.layers[-1] = eqx.tree_at(w_where,self.layers[-1],w_zeros)
		self.layers[-1] = eqx.tree_at(b_where,self.layers[-1],b_zeros)
		
	def periodic_padding(self,x):
		return jnp.pad(x, ((0,0),(1,1),(1,1)), mode='wrap')
	
	def periodic_unpadding(self,x):
		return x[:,1:-1,1:-1]
	
	def percieve(self,x):
		"""
		Performs convolutions of x with each kernel in self.KERNELS. Adds dummy index to x such that lax.conv_general_dilated has 4D inputs and output
		Performs periodic or zero padding depending on self.PERIODIC parameter
		
		Parameters
		----------
		x : jax array [N_CHANNELS,_,_]
			3D array of NCA state, with 1st axis being channel number, and other axes being spatial dimensions.

		Returns
		-------
		z : jax array [N_FEATURES,_,_]
			3D array of perception vectors, with 1st axis being feature number - each channel convolved with each kernel

		"""
		
		
		if self.PERIODIC:
			x = self.periodic_padding(x)
			
		x_ = x[jnp.newaxis]
		z = jax.lax.conv_general_dilated(x_,
								         self.KERNELS,
										 (1,1),
										 "SAME",
										 feature_group_count=self.N_CHANNELS)
		if self.PERIODIC:
			z = self.periodic_unpadding(z[0])
		else:
			 z = z[0]
			 
		return z
		
	def __call__(self,x,key):
		#print(x.shape)
		dx = self.percieve(x)
		for layer in self.layers:
			dx = layer(dx)
		sigma = jax.random.bernoulli(key,p=self.FIRE_RATE,shape=dx.shape)
		return x + sigma*dx
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
		
		where = lambda nca: nca.KERNELS
		kernel = self.KERNELS
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
		suffix = ".pickle"
		path = Path(path)
		if path.suffix != suffix:
			path = path.with_suffix(suffix)
			path.parent.mkdir(parents=True, exist_ok=True)
		if path.exists():
			if overwrite:
				path.unlink()
			else:
				raise RuntimeError(f'File {path} already exists.')
		with open(path, 'wb') as file:
			pickle.dump(self, file)
	
	def load(path: Union[str, Path]) -> pytree:
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
		suffix = ".pickle"
		path = Path(path)
		if not path.is_file():
			raise ValueError(f'Not a file: {path}')
		if path.suffix != suffix:
			raise ValueError(f'Not a {suffix} file: {path}')
		with open(path, 'rb') as file:
			data = pickle.load(file)
		return data

