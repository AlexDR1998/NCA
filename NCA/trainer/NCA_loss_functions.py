import numpy as np 
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
#import tensorflow_addons as tfa

vgg_style = VGG16(weights="imagenet", include_top=False, input_shape=[150,150,3])
vgg_style.trainable = False ## Not trainable weights
#------ Loss functions -------
@tf.function
def loss_sliced_wasserstein_channels(X,Y,num=64):
	"""
		Implementation of the sliced wasserstein loss described by Heitz et al 2021

		Projects over channels
		
		Parameters
		----------
		X,Y : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		num : int optional
			Number of random projections to average over

		Returns
		-------
		loss : float32 tensor [N_BATCHES]

	"""
	C = X.shape[-1]
	B = X.shape[0]
	V = tf.random.uniform(shape=(C,num))
	
	V = V/tf.linalg.norm(V,axis=0) # shape (C,num)

	X_proj = tf.einsum("Cn,bxyC->nbxy",V,X)
	Y_proj = tf.einsum("Cn,bxyC->nbxy",V,Y)

	X_proj = tf.reshape(X_proj,(num,B,-1))
	Y_proj = tf.reshape(Y_proj,(num,B,-1))
	
	X_sort = tf.sort(X_proj,axis=2)
	Y_sort = tf.sort(Y_proj,axis=2)

	#print(X_proj.shape)
	#print(Y_sort.shape)

	return tf.math.reduce_mean((X_sort-Y_sort)**2,axis=[0,2])
	#print(tf.math.reduce_sum(X_proj,axis=1))



@tf.function
def loss_sliced_wasserstein_grid(X,Y,num=256):
	"""
		Implementation of the sliced wasserstein loss described by Heitz et al 2021

		Projects over spatial directions
		
		Parameters
		----------
		X,Y : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		num : int optional
			Number of random projections to average over

		Returns
		-------
		loss : float32 tensor [N_BATCHES]

	"""
	#C = X.shape[-1]
	#B = X.shape[0]
	H = X.shape[1]
	W = X.shape[2]
	V = tf.random.uniform(shape=(H,W,num))
	
	V = V/tf.linalg.norm(V,axis=[0,1]) # shape (C,num)
	#print(tf.linalg.norm(V,axis=[0,1]))
	X_proj = tf.einsum("xyn,bxyC->nbC",V,X)
	Y_proj = tf.einsum("xyn,bxyC->nbC",V,Y)

	
	X_sort = tf.sort(X_proj,axis=2)
	Y_sort = tf.sort(Y_proj,axis=2)

	#print(X_proj.shape)
	#print(Y_sort.shape)

	return tf.math.reduce_mean((X_sort-Y_sort)**2,axis=[0,2])
	#print(tf.math.reduce_sum(X_proj,axis=1))




def my_preprocess(x):
	x = tf.clip_by_value(x, 0, 1)
	return preprocess_input(x*255)
@tf.function
def loss_vgg(X,Y):
	"""
	Transform each channel through VGG16, compare euclidean distance of second last layer activations

	Parameters
	----------
	X,Y : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
		Data to compute loss on

	Returns
	-------
	loss : float32 tensor [N_BATCHES]
		

	"""
	_X = tf.image.resize(X,(150,150))[...,:3]
	_Y = tf.image.resize(Y,(150,150))[...,:3]
	## Loading VGG16 model
	
	
	X_vgg = vgg_style(my_preprocess(_X))
	Y_vgg = vgg_style(my_preprocess(_Y))
	return tf.math.reduce_mean(tf.abs(X_vgg-Y_vgg),axis=[1,2,3])
# @tf.function
# def loss_sliced_wasserstein_rotate(X,Y,num=24):
# 	"""
# 		Implementation of the sliced wasserstein loss described by Heitz et al 2021

# 		Projects data along random angle through image
# 		
# 		Parameters
# 		----------
# 		X,Y : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
# 			Data to compute loss on
# 		num : int optional
# 			Number of random projections to average over

# 		Returns
# 		-------
# 		loss : float32 tensor [N_BATCHES]

# 	"""

# 	#C = X.shape[-1]
# 	B = X.shape[0]
# 	#H = X.shape[1]
# 	#W = X.shape[2]
# 	
# 	#X_stack = tf.repeat(X,num,axis=0)
# 	#Y_stack = tf.repeat(Y,num,axis=0)

# 	#--- Randomly rotate each batch
# 	angles = tf.random.uniform(shape=(1,B),minval=0,maxval=359)[0]
# 	X_rot = tfa.image.rotate(X,angles)
# 	Y_rot = tfa.image.rotate(Y,angles)
# 	
# 	#--- Project each rotated batch in orthogonal directions
# 	X_proj_1 = tf.math.reduce_mean(X_rot,axis=1)
# 	Y_proj_1 = tf.math.reduce_mean(Y_rot,axis=1)
# 	X_proj_2 = tf.math.reduce_mean(X_rot,axis=2)
# 	Y_proj_2 = tf.math.reduce_mean(Y_rot,axis=2)


# 	#--- Sort histograms
# 	X_sort_1 = tf.sort(X_proj_1,axis=2)
# 	Y_sort_1 = tf.sort(Y_proj_1,axis=2)
# 	X_sort_2 = tf.sort(X_proj_2,axis=2)
# 	Y_sort_2 = tf.sort(Y_proj_2,axis=2)


# 	diff_1 = tf.math.reduce_mean((X_sort_1-Y_sort_1)**2,axis=[1,2])
# 	diff_2 = tf.math.reduce_mean((X_sort_2-Y_sort_2)**2,axis=[1,2])

# 	return diff_1+diff_2

@tf.function
def loss_spectral(X,Y):
	"""
		Implementation of euclidean distance of FFTs of X and Y

		
		Parameters
		----------
		X,Y : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		

		Returns
		-------
		loss : float32 tensor [N_BATCHES]

	"""

	X_ = tf.einsum("bxyc->bcxy",X)
	Y_ = tf.einsum("bxyc->bcxy",Y)

	X_fft = tf.math.abs(tf.signal.rfft2d(X_))
	Y_fft = tf.math.abs(tf.signal.rfft2d(Y_))

	return tf.math.reduce_euclidean_norm((X_fft-Y_fft),axis=[1,2,3])


@tf.function
def loss_euclidean(X,Y):
	"""
		Implementation of euclidean distance 
	
		Parameters
		----------
		X,Y : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		
	
		Returns
		-------
		loss : float32 tensor [N_BATCHES]
	
	"""
	return tf.math.reduce_euclidean_norm((X-Y),axis=[1,2,3])

	





@tf.function
def loss_spectral_euclidean(X,Y):
	"""
		Implementation of euclidean distance in both real and FFT space

		Parameters
		----------
		X,Y : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		

		Returns
		-------
		loss : float32 tensor [N_BATCHES]

	"""
	loss_real = tf.math.reduce_euclidean_norm((X-Y),axis=[1,2,3])
	loss_fft = loss_spectral(X,Y) 

	combined_loss = loss_real + loss_fft
	return combined_loss
@tf.function
def loss_bhattacharyya(X,Y):
	"""
		Implementation of bhattarcharyya distance of X and Y
		
		As described in: https://en.wikipedia.org/wiki/Bhattacharyya_distance
		
		Note that this assumes X and Y are probability distributions -> normalise them
		
		Parameters
		----------
		X,Y : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		

		Returns
		-------
		loss : float32 tensor [N_BATCHES]
	"""
	eps = 1e-9
	X = tf.math.abs(X)
	Y = tf.math.abs(Y)
	X_norm = tf.math.divide(X+eps,tf.math.reduce_sum(X+eps,axis=[1,2],keepdims=True))
	Y_norm = tf.math.divide(Y+eps,tf.math.reduce_sum(Y+eps,axis=[1,2],keepdims=True))
	BC = tf.math.reduce_sum(tf.math.sqrt(X_norm*Y_norm),axis=[1,2])
	B_loss = -tf.math.log(BC)

	return tf.math.reduce_mean(B_loss,axis=-1) 

@tf.function
def loss_bhattacharyya_modified(X,Y):
	"""
		Implementation of bhattarcharyya distance of X and Y. Modified to account for 
		difference in average amplitude of X and Y
		
		As described in: https://en.wikipedia.org/wiki/Bhattacharyya_distance
		
		Note that this assumes X and Y are probability distributions -> normalise them
		Also compares average amplitude of X and Y to account for information lost in normalisation
		
		Parameters
		----------
		X,Y : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		

		Returns
		-------
		loss : float32 tensor [N_BATCHES]
	"""
	eps = 1e-9
	X_amp = tf.math.reduce_mean(X,axis=[1,2])
	Y_amp = tf.math.reduce_mean(Y,axis=[1,2])
	X = tf.math.abs(X)
	Y = tf.math.abs(Y)
	X_norm = tf.math.divide(X+eps,tf.math.reduce_sum(X+eps,axis=[1,2],keepdims=True))
	Y_norm = tf.math.divide(Y+eps,tf.math.reduce_sum(Y+eps,axis=[1,2],keepdims=True))
	BC = tf.math.reduce_sum(tf.math.sqrt(X_norm*Y_norm),axis=[1,2])
	B_loss = -tf.math.log(BC)*(1+tf.math.abs(X_amp-Y_amp))

	return tf.math.reduce_mean(B_loss,axis=-1) 

@tf.function
def loss_bhattacharyya_euclidean(X,Y):
    """
        Combined loss of bhattachryya and euclidean
		
		Parameters
		----------
		X,Y : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		

		Returns
		-------
		loss : float32 tensor [N_BATCHES]
    """
    b_loss = loss_bhattacharyya_modified(X, Y)
    e_loss = tf.math.reduce_euclidean_norm((X-Y),axis=[1,2,3])
    return b_loss+e_loss

@tf.function
def loss_hellinger(X,Y):
	"""
		Implementation of hellinger distance of X and Y
		
		As described in: https://en.wikipedia.org/wiki/Hellinger_distance
		
		Note that this assumes X and Y are probability distributions -> normalise them
		
		Parameters
		----------
		X,Y : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		

		Returns
		-------
		loss : float32 tensor [N_BATCHES]
	"""
	eps=1e-9
	X = tf.math.abs(X)
	Y = tf.math.abs(Y)
	X_norm = tf.math.divide(X+eps,tf.math.reduce_sum(X+eps,axis=[1,2],keepdims=True))
	Y_norm = tf.math.divide(Y+eps,tf.math.reduce_sum(Y+eps,axis=[1,2],keepdims=True))
	sqrt_diff = tf.math.sqrt(X_norm) - tf.math.sqrt(Y_norm)
	
	H_bc = 0.70710678118*tf.math.reduce_euclidean_norm(sqrt_diff,axis=[1,2])
	return tf.math.reduce_mean(H_bc,axis=-1)

@tf.function
def loss_hellinger_modified(X,Y):
	"""
		Implementation of hellinger distance of X and Y
		
		As described in: https://en.wikipedia.org/wiki/Hellinger_distance
		
		Note that this assumes X and Y are probability distributions -> normalise them
		Also compares average amplitude of X and Y to account for information lost in normalisation
		Parameters
		----------
		X,Y : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		

		Returns
		-------
		loss : float32 tensor [N_BATCHES]
	"""
	eps=1e-9
	X_amp = tf.math.reduce_mean(X,axis=[1,2])
	Y_amp = tf.math.reduce_mean(Y,axis=[1,2])
	X = tf.math.abs(X)
	Y = tf.math.abs(Y)
	X_norm = tf.math.divide(X+eps,tf.math.reduce_sum(X+eps,axis=[1,2],keepdims=True))
	Y_norm = tf.math.divide(Y+eps,tf.math.reduce_sum(Y+eps,axis=[1,2],keepdims=True))
	sqrt_diff = tf.math.sqrt(X_norm) - tf.math.sqrt(Y_norm)
	
	H_bc = 0.70710678118*tf.math.reduce_euclidean_norm(sqrt_diff,axis=[1,2])*(1+tf.math.abs(X_amp-Y_amp))
	return tf.math.reduce_mean(H_bc,axis=-1)




@tf.function
def loss_kl_divergence(X,Y):
	"""
		Wrapper for the tensorflow kullback-leibler divergence, mostly just reshaping data for it.
	

		Parameters
		----------
		X,Y : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		

		Returns
		-------
		loss : float32 tensor [N_BATCHES]

	"""
	kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
	size = tf.cast(X.shape[1]*X.shape[2]*X.shape[3],tf.float32)
	k_comb = lambda xy: kl(xy[0],xy[1])
	loss = tf.vectorized_map(k_comb, (X,Y))
	return loss / size
	

#----- Helper functions for sinkhorn loss

@tf.function
def pdist(x, y):
	print(x.shape)
	dx = x[:, None, :] - y[None, :, :]
	return tf.reduce_sum(tf.square(dx), -1)

@tf.function
def Sinkhorn_step(C, f):
	g = tf.reduce_logsumexp(-f-tf.transpose(C), -1)
	f = tf.reduce_logsumexp(-g-C, -1)
	return f, g

def Sinkhorn(C, f=None, niter=1000):
	n = tf.shape(C)[0]
	if f is None:
		f = tf.zeros(n, np.float32)
	for i in range(niter):
		f, g = Sinkhorn_step(C, f)
	P = tf.exp(-f[:,None]-g[None,:]-C)/tf.cast(n, tf.float32)
	return tf.reduce_sum(P*C)


def loss_sinkhorn(X,Y):
	"""
	
		Wrapper for computing OT loss with sinkhorn algorithm. Code taken from
		https://colab.research.google.com/github/znah/notebooks/blob/master/mini_sinkhorn.ipynb#scrollTo=mP68HY4Bric5
		
		Parameters
		----------
		X,Y : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
	
		Returns
		-------
		loss : float32 tensor [N_BATCHES]

	"""
	B = X.shape[0]
	Ch= X.shape[-1]
	X_flat = tf.reshape(tf.einsum("bxyc->bcxy",X),(B,Ch,-1))
	Y_flat = tf.reshape(tf.einsum("bxyc->bcxy",Y),(B,Ch,-1))
	#print(X_flat.shape)
	b_sink = lambda xy: Sinkhorn(pdist(xy[0],xy[1]))
	#loss = tf.vectorized_map(b_sink, tf.stack((X_flat,Y_flat),-1))
	#@tf.function
	#def _func(elems):
	#	return tf.map_fn(b_sink,elems,fn_output_signature=tf.float32)
		
	loss = tf.map_fn(b_sink,(X_flat,Y_flat),fn_output_signature=tf.float32)
	"""
	loss = []
	for b in range(B):
		loss.append(b_sink(X_flat[b],Y_flat[b]))
	"""
	return tf.cast(loss,tf.float32)
