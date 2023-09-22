import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import skimage
# Some convenient helper functions
#@jax.jit
def key_array_gen(key,shape):
	"""
	

	Parameters
	----------
	key : jax.random.PRNGKey, 
		Jax random number key.
	shape : tuple of ints
		Shape to broadcast to

	Returns
	-------
	key_array : uint32[shape,2]
		array of random keys
	"""
	shape = list(shape)
	shape.append(2)
	key_array = jax.random.randint(key,shape=shape,minval=0,maxval=2_147_483_647,dtype="uint32")
	return key_array


def grad_norm(grad):
	"""
	Normalises each vector/matrix in grad 

	Parameters
	----------
	grad : NCA/pytree

	Returns
	-------
	grad : NCA/pytree

	"""
	w_where = lambda l: l.weight
	b_where = lambda l: l.bias
	w1 = grad.layers[3].weight/(jnp.linalg.norm(grad.layers[3].weight)+1e-8)
	w2 = grad.layers[5].weight/(jnp.linalg.norm(grad.layers[5].weight)+1e-8)
	b2 = grad.layers[5].bias/(jnp.linalg.norm(grad.layers[5].bias)+1e-8)
	grad.layers[3] = eqx.tree_at(w_where,grad.layers[3],w1)
	grad.layers[5] = eqx.tree_at(w_where,grad.layers[5],w2)
	grad.layers[5] = eqx.tree_at(b_where,grad.layers[5],b2)
	return grad

def load_emoji_sequence(filename_sequence,impath_emojis="../Data/Emojis/",downsample=2,crop_square=False):
	"""
		Loads a sequence of images in impath_emojis
		Parameters
		----------
		filename_sequence : list of strings
			List of names of files to load
		downsample : int
			How much to downsample the resolution - highres takes ages
	
		Returns
		-------
		images : float32 array [1,T,C,size,size]
			Timesteps of T RGB/RGBA images. Dummy index of 1 for number of batches
	"""
	images = []
	for filename in filename_sequence:
		im = skimage.io.imread(impath_emojis+filename)[::downsample,::downsample]
		if crop_square:
			s= min(im.shape[0],im.shape[1])
			im = im[:s,:s]
		#im = im[np.newaxis] / 255.0
		im = im/255.0
		images.append(im)
	data = np.array(images)
	data = data[np.newaxis]
	data = np.einsum("btxyc->btcxy",data)
	return data

