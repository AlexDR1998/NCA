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
	return jnp.sqrt(jnp.mean(((x-y)**2),axis=[-1,-2,-3]))

#@jax.jit
def random_sampled_euclidean(x,y,key,SAMPLES=64):
	x_r = jnp.einsum("ncxy->cxyn",x)
	y_r = jnp.einsum("ncxy->cxyn",y)
	x_sub = jax.random.choice(key,x_r.reshape((-1,x_r.shape[-1])),(SAMPLES,),False)
	y_sub = jax.random.choice(key,y_r.reshape((-1,y_r.shape[-1])),(SAMPLES,),False)
	return jnp.sqrt(jnp.mean((x_sub-y_sub)**2,axis=0))