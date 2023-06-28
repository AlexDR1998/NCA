import numpy as np 
import tensorflow as tf
import io
import matplotlib.pyplot as plt
from NCA.trainer.NCA_loss_functions import *
"""
	Some utilities and helper functions used by NCA_train.py. Mostly functions for mapping array job indices to training/model parameters

"""


	
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
		Takes index from 1-N and constructs 2-tuple of loss function and time sampling rate 
	"""
	loss_funcs = [loss_spectral,
				  loss_bhattacharyya_modified,
				  loss_hellinger_modified,
				  None]
	loss_func_name = ["spectral","bhattachryya","hellinger","euclidean"]
	sampling_rates = [1,2,4,6,8,12,16,24,32]
	instance = [0,1,2,3]
	L1 = len(loss_funcs)
	L2 = len(sampling_rates)
	L3 = len(instance)
	indices = np.unravel_index(index,(L1,L2,L3))
	loss = loss_funcs[indices[0]]
	loss_name = loss_func_name[indices[0]]
	sampling = sampling_rates[indices[1]]
	i = instance[indices[2]]
	return loss,loss_name,sampling,i

def index_to_generalise_test(index):
    loss_funcs = [None,
                  loss_bhattacharyya_modified,
                  loss_bhattacharyya_euclidean,
				  loss_spectral,
				  loss_hellinger_modified,
				  loss_kl_divergence]
    loss_func_names = ["euclidean",
                       "bhattacharyya",
                       "bhattacharyya_euclidean",
					   "spectral",
					   "hellinger",
					   "kl_divergence"]
    sampling_rates = [1,2,4,8,16]
    tasks = ["heat",
            "mitosis",
            "coral",
			"gol"]
    #sampling_rates = [16,32,48,64]
    #tasks=["emoji"]
    L1 = len(loss_funcs)
    L2 = len(sampling_rates)
    L3 = len(tasks)
    indices = np.unravel_index(index,(L1,L2,L3))
    loss = loss_funcs[indices[0]]
    loss_name = loss_func_names[indices[0]]
    sampling = sampling_rates[indices[1]]
    task = tasks[indices[2]]
    return loss,loss_name,sampling,task

def index_to_generalise_test_2():
	#Rerunning some of index_tp_generalise_test, as fft loss was bugged
	loss = loss_spectral
	loss_name = "spectral"
	sampling = 8
	task = "coral"
	return loss,loss_name,sampling,task
def index_to_model_exploration_parameters(index):
    loss_funcs = [None,loss_spectral]
    loss_func_names = ["euclidean","spectral"]
    tasks = ["heat","mitosis","coral","emoji"]
    layers = [2,3]
    kernels = ["ID_LAP","ID_LAP_AV","ID_DIFF_LAP","ID_DIFF"]
    activations=["linear","relu","swish","tanh"]
    L1 = len(loss_funcs)
    L2 = len(tasks)
    L3 = len(layers)
    L4 = len(kernels)
    L5 = len(activations)
    indices = np.unravel_index(index, (L1,L2,L3,L4,L5))
    LOSS = loss_funcs[indices[0]]
    LOSS_STR = loss_func_names[indices[0]]
    TASK = tasks[indices[1]]
    LAYER = layers[indices[2]]
    KERNEL= kernels[indices[3]]
    ACTIVATION= activations[indices[4]]
    return LOSS,LOSS_STR,TASK,LAYER,KERNEL,ACTIVATION
    

def index_to_model_exploration_parameters_spectral(index):
	#same as above function, but only spectral 2 layer emojis, as fft loss was bugged
	LOSS = loss_spectral
	LOSS_STR = "spectral"
	TASK = "emoji"
	LAYER = 2
	kernels = ["ID_LAP","ID_LAP_AV","ID_DIFF_LAP","ID_DIFF"]
	activations=["linear","relu","swish","tanh"]
	#L1 = len(loss_funcs)
	#L2 = len(tasks)
	#L3 = len(layers)
	L1 = len(kernels)
	L2 = len(activations)
	indices = np.unravel_index(index, (L1,L2))
	#LOSS = loss_funcs[indices[0]]
	#LOSS_STR = loss_func_names[indices[0]]
	#TASK = tasks[indices[1]]
	#LAYER = layers[indices[2]]
	KERNEL= kernels[indices[0]]
	ACTIVATION= activations[indices[1]]
	return LOSS,LOSS_STR,TASK,LAYER,KERNEL,ACTIVATION

def index_to_emoji_symmetry_parameters(index):
	kernels = ["ID_LAP_AV","ID_DIFF_LAP"]
	tasks=["normal","rotated_data","rotation_task","translation_task"]
	L1 = len(kernels)
	L2 = len(tasks)
	indices = np.unravel_index(index,(L1,L2))
	KERNEL = kernels[indices[0]]
	TASK = tasks[indices[1]]
	return KERNEL,TASK
	
def index_to_channel_sample(index):
	channels = [4,6,8,10,12,14,16,32]
	samplings = [1,2,4,8,16,32,64,128]
	L1 = len(channels)
	L2 = len(samplings)
	indices = np.unravel_index(index,(L1,L2))
	N_CHANNELS = channels[indices[0]]
	SAMPLING = samplings[indices[1]]
	return N_CHANNELS,SAMPLING