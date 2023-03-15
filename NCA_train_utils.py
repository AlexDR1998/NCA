import numpy as np 
import tensorflow as tf
import tensorflow_addons as tfa
import io
import matplotlib.pyplot as plt
"""
	Some utilities and helper functions used by NCA_train.py

"""


#------ Loss functions -------
@tf.function
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



@tf.function
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


@tf.function
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
@tf.function
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

	return tf.math.abs(tf.math.reduce_euclidean_norm((X_fft-Y_fft),axis=[0,1,3]))

@tf.function
def loss_spectral_euclidean(X,Y):
	"""
		Implementation of euclidean distance in both real and FFT space

		Parameters
		----------
		X,Y : float23 tensor [N_BATCHES,X,Y,N_CHANNELS]
			Data to compute loss on
		

		Returns
		-------
		loss : float32 tensor [N_BATCHES]

	"""
	loss_real = tf.math.reduce_euclidean_norm((X-Y),axis=[1,2])
	X_ = tf.einsum("bxyc->xybc",X)
	Y_ = tf.einsum("bxyc->xybc",Y)

	X_fft = tf.signal.rfft2d(X_)
	Y_fft = tf.signal.rfft2d(Y_)

	loss_fft = tf.math.abs(tf.math.reduce_euclidean_norm((X_fft-Y_fft),axis=[0,1])) # note that X_fft has different axes as X

	combined_loss = loss_real + loss_fft
	return tf.math.reduce_mean(combined_loss,axis=-1)
@tf.function
def loss_bhattacharyya(X,Y):
	"""
		Implementation of bhattarcharyya distance of X and Y
		
		As described in: https://en.wikipedia.org/wiki/Bhattacharyya_distance
		
		Note that this assumes X and Y are probability distributions -> normalise them
		
		Parameters
		----------
		X,Y : float23 tensor [N_BATCHES,X,Y,N_CHANNELS]
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
		
		Parameters
		----------
		X,Y : float23 tensor [N_BATCHES,X,Y,N_CHANNELS]
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
		X,Y : float23 tensor [N_BATCHES,X,Y,N_CHANNELS]
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
		X,Y : float23 tensor [N_BATCHES,X,Y,N_CHANNELS]
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
		
		Parameters
		----------
		X,Y : float23 tensor [N_BATCHES,X,Y,N_CHANNELS]
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
	return H_bc

def plot_to_image(figure):
	"""Converts the matplotlib plot specified by 'figure' to a PNG image and
	returns it. The supplied figure is closed and inaccessible after this call."""
	# Save the plot to a PNG in memory.
	buf = io.BytesIO()
	plt.savefig(buf, format='png')
	# Closing the figure prevents it from being displayed directly inside
	# the notebook.
	plt.close(figure)
	buf.seek(0)
	# Convert PNG buffer to TF image
	image = tf.image.decode_png(buf.getvalue(), channels=4)
	# Add the batch dimension
	image = tf.expand_dims(image, 0)
	return image



def index_to_trainer_parameters(index):
	"""
		Takes job array index from 1-35 and constructs pair of loss function and optimiser
	
	loss_funcs = [loss_sliced_wasserstein_channels,
				  loss_sliced_wasserstein_grid,
				  loss_sliced_wasserstein_rotate,
				  loss_spectral,
				  loss_bhattacharyya_modified,
				  loss_hellinger_modified,
				  None]
	loss_func_name =["sliced_wasserstein_channels",
					 "sliced_wasserstein_grid",
					 "sliced_wasserstein_rotate",
					 "spectral",
					 "bhattachryya",
					 "hellinger",
					 "euclidean"]

	optimisers = ["Adagrad",
				  "Adam",
				  "Adadelta",
				  "Nadam",
				  "RMSprop"]
	"""
	loss_funcs = [loss_sliced_wasserstein_channels,
				  loss_sliced_wasserstein_grid,
				  loss_sliced_wasserstein_rotate,
				  loss_spectral,
				  loss_bhattacharyya_modified,
				  loss_spectral_euclidean,
				  None]
	loss_func_name =["sliced_wasserstein_channels",
					 "sliced_wasserstein_grid",
					 "sliced_wasserstein_rotate",
					 "spectral",
					 "bhattachryya",
					 "spectral_euclidean",
					 "euclidean"]

	optimisers = ["Adagrad",
				  "Adam",
				  "Adadelta",
				  "Nadam"]
	L = len(loss_funcs)
	
	opt = optimisers[index//L]
	loss= loss_funcs[index%L]
	loss_name = loss_func_name[index%L]
	return loss,opt,loss_name

def index_to_learnrate_parameters(index):
	"""
		Takes job array index from 1-84 and constructs 4-tuple of learn rate, optimiser, training mode and gradient normalisation

	"""
	learn_rates = np.logspace(1,-5,7)
	learn_rates_name = ["1e1","1e0","1e-1","1e-2","1e-3","1e-4","1e-5"]
	optimisers = ["Adagrad",
				  "Adam",
				  "Nadam"]
	training_modes = ["differential","full"]
	grad_norm = [True,False]
	L1 = len(learn_rates)
	L2 = len(optimisers)
	L3 = len(training_modes)
	L4 = len(grad_norm)

	indices = np.unravel_index(index,(L1,L2,L3,L4))
	lr= learn_rates[indices[0]]
	lr_name = learn_rates_name[indices[0]]
	opt = optimisers[indices[1]]
	mode = training_modes[indices[2]]
	grad = grad_norm[indices[3]]
	return lr,lr_name,opt,mode,grad


def index_to_Nadam_parameters(index):
	"""
		Takes index from 1-N and constructs 3-tuple of learn rate, NCA to PDE step ratio and grad_norm

	"""
	learn_rates = np.logspace(1,-5,7)
	learn_rates_name = ["1e1","1e0","1e-1","1e-2","1e-3","1e-4","1e-5"]
	ratios = [1,2,4,8]
	grad_norm = [True,False]
	L1 = len(learn_rates)
	L2 = len(ratios)
	L3 = len(grad_norm)
	indices = np.unravel_index(index,(L1,L2,L3))
	lr = learn_rates[indices[0]]
	lr_name = learn_rates_name[indices[0]]
	ratio = ratios[indices[1]]
	grad = grad_norm[indices[2]]
	return lr,lr_name,ratio,grad



def index_to_mitosis_parameters(index):
	"""
		Takes index from 1-N and constructs 3-tuple of loss function, time sampling rate and gradient normalisation
	"""
	loss_funcs = [loss_spectral,
				  loss_bhattacharyya_modified,
				  loss_spectral_euclidean,
				  None]
	loss_func_name =["spectral",
					 "bhattacharyya",
					 "spectral_euclidean",
					 "euclidean"]
	sampling_rates = [1,2,4,8,16]
	grad_norm = [True,False]
	L1 = len(loss_funcs)
	L2 = len(sampling_rates)
	L3 = len(grad_norm)
	indices = np.unravel_index(index,(L1,L2,L3))
	loss = loss_funcs[indices[0]]
	loss_name = loss_func_name[indices[0]]
	sampling = sampling_rates[indices[1]]
	grad = grad_norm[indices[2]]
	return loss,loss_name,sampling,grad

def index_to_generalise_test(index):
    loss_funcs = [None,
                  loss_bhattacharyya_modified,
                  loss_bhattacharyya_euclidean,
				  loss_spectral,
				  loss_hellinger_modified]
    loss_func_names = ["euclidean",
                       "bhattacharyya",
                       "bhattacharyya_euclidean",
					   "spectral",
					   "hellinger"]
    sampling_rates = [1,2,4,8,16]
    tasks = ["heat",
            "mitosis",
            "coral",
			"gol"]
    L1 = len(loss_funcs)
    L2 = len(sampling_rates)
    L3 = len(tasks)
    indices = np.unravel_index(index,(L1,L2,L3))
    loss = loss_funcs[indices[0]]
    loss_name = loss_func_names[indices[0]]
    sampling = sampling_rates[indices[1]]
    task = tasks[indices[2]]
    return loss,loss_name,sampling,task