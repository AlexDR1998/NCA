import numpy as np 
import tensorflow as tf
import tensorflow_addons as tfa
"""
	Some utilities and helper functions used by NCA_train.py

"""


#------ Loss functions -------

def loss_sliced_wasserstein_channels(X,Y,num=64):
	"""
		Implementation of the sliced wasserstein loss described by Heitz et al 2021

		Projects over channels
		
		Parameters
		----------
		X,Y : float23 tensor [N_BATCHES,X,Y,N_CHANNELS]
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




def loss_sliced_wasserstein_grid(X,Y,num=256):
	"""
		Implementation of the sliced wasserstein loss described by Heitz et al 2021

		Projects over spatial directions
		
		Parameters
		----------
		X,Y : float23 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		num : int optional
			Number of random projections to average over

		Returns
		-------
		loss : float32 tensor [N_BATCHES]

	"""
	C = X.shape[-1]
	B = X.shape[0]
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



def loss_sliced_wasserstein_rotate(X,Y,num=24):
	"""
		Implementation of the sliced wasserstein loss described by Heitz et al 2021

		Projects data along random angle through image
		
		Parameters
		----------
		X,Y : float23 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		num : int optional
			Number of random projections to average over

		Returns
		-------
		loss : float32 tensor [N_BATCHES]

	"""

	C = X.shape[-1]
	B = X.shape[0]
	H = X.shape[1]
	W = X.shape[2]
	
	#X_stack = tf.repeat(X,num,axis=0)
	#Y_stack = tf.repeat(Y,num,axis=0)

	#--- Randomly rotate each batch
	angles = tf.random.uniform(shape=(1,B),minval=0,maxval=359)[0]
	X_rot = tfa.image.rotate(X,angles)
	Y_rot = tfa.image.rotate(Y,angles)
	
	#--- Project each rotated batch in orthogonal directions
	X_proj_1 = tf.math.reduce_mean(X_rot,axis=1)
	Y_proj_1 = tf.math.reduce_mean(Y_rot,axis=1)
	X_proj_2 = tf.math.reduce_mean(X_rot,axis=2)
	Y_proj_2 = tf.math.reduce_mean(Y_rot,axis=2)


	#--- Sort histograms
	X_sort_1 = tf.sort(X_proj_1,axis=2)
	Y_sort_1 = tf.sort(Y_proj_1,axis=2)
	X_sort_2 = tf.sort(X_proj_2,axis=2)
	Y_sort_2 = tf.sort(Y_proj_2,axis=2)


	diff_1 = tf.math.reduce_mean((X_sort_1-Y_sort_1)**2,axis=[1,2])
	diff_2 = tf.math.reduce_mean((X_sort_2-Y_sort_2)**2,axis=[1,2])

	return diff_1+diff_2

def loss_spectral(X,Y):
	"""
		Implementation of euclidean distance of FFTs of X and Y

		
		Parameters
		----------
		X,Y : float23 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		

		Returns
		-------
		loss : float32 tensor [N_BATCHES]

	"""
	X = tf.einsum("bxyc->xybc",X)
	Y = tf.einsum("bxyc->xybc",Y)

	X_fft = tf.signal.rfft2d(X)
	Y_fft = tf.signal.rfft2d(Y)

	return tf.math.reduce_mean(tf.math.abs(X_fft-Y_fft),axis=[0,1,3])


X = np.random.uniform(size=(5,128,128,7))
#Y = np.random.uniform(size=(5,128,128,7))
Y = np.zeros((5,128,128,7))
#X[2] = 1

print(loss_sliced_wasserstein_rotate(X,Y))