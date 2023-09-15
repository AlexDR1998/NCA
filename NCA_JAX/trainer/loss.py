import jax.numpy as jnp
import jax
"""
	General format of loss functions here:

	Parameters
	----------
	x : float32 [...,CHANNELS,WIDTH,HEIGHT]
		predictions
	y : float32 [...,CHANNELS,WIDTH,HEIGHT]
		true data

	Returns
	-------
	loss : float32 array [...]
		loss reduced over channel and spatial axes

"""
@jax.jit
def l2(x,y):

	return jnp.sum(((x-y)**2),axis=[-1,-2,-3])
@jax.jit
def euclidean(x,y):
	return jnp.sqrt(jnp.sum(((x-y)**2),axis=[-1,-2,-3]))