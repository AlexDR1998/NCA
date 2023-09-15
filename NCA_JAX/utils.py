import jax
import jax.numpy as jnp
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
		images : float32 array [T,1,C,size,size]
			Timesteps of T RGB/RGBA images. Dummy index of 1 for number of batches
	"""
	images = []
	for filename in filename_sequence:
		im = skimage.io.imread(impath_emojis+filename)[::downsample,::downsample]
		if crop_square:
			s= min(im.shape[0],im.shape[1])
			im = im[:s,:s]
		im = im[np.newaxis] / 255.0
		images.append(im)
	data = np.array(images)
	data = np.einsum("tbxyc->tbcxy",data)
	return data