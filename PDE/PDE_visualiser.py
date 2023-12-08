import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import io



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


def plot_weight_matrices(pde):
	"""
	Plots heatmaps of NCA layer weights

	Parameters
	----------
	pde : object callable - (float32 array [T], float32 array [N_CHANNELS,_,_]) -> (float32 array [T,N_CHANNELS,_,_])
		the PDE solver object to plot parameters of

	Returns
	-------
	figs : list of images
		a list of images

	"""
	w1_v = pde.func.f_v.layers[0].weight[:,:,0,0]
	w2_v = pde.func.f_v.layers[2].weight[:,:,0,0]
	
	w1_d = pde.func.f_d.layers[-1].weight[:,:,0,0]
	
	w1_r = pde.func.f_r.layers[0].weight[:,:,0,0]
	w2_r = pde.func.f_r.layers[2].weight[:,:,0,0]
	figs = []
	
	figure = plt.figure(figsize=(5,5))
	col_range = max(np.max(w1_v),-np.min(w1_v))
	plt.imshow(w1_v,cmap="seismic",vmax=col_range,vmin=-col_range)
	plt.ylabel("Output")
	plt.xlabel("Input")
	plt.title("Advection layer 1")
	figs.append(plot_to_image(figure))
	
	figure = plt.figure(figsize=(5,5))
	col_range = max(np.max(w2_v),-np.min(w2_v))
	plt.imshow(w2_v,cmap="seismic",vmax=col_range,vmin=-col_range)
	plt.ylabel("Output")
	plt.xlabel("Input")
	plt.title("Advection layer 2")
	figs.append(plot_to_image(figure))
	
	figure = plt.figure(figsize=(5,5))
	#col_range = max(np.max(w1_d),-np.min(w1_d))
	#plt.imshow(w1_d,cmap="seismic",vmax=col_range,vmin=-col_range)
	plt.plot(w1_d)
	plt.ylabel("Weight")
	plt.xlabel("Channel")
	plt.title("Diffusion weights")
	figs.append(plot_to_image(figure))

	figure = plt.figure(figsize=(5,5))
	col_range = max(np.max(w1_r),-np.min(w1_r))
	plt.imshow(w1_r,cmap="seismic",vmax=col_range,vmin=-col_range)
	plt.ylabel("Output")
	plt.xlabel("Input")
	plt.title("Reaction layer 1")
	figs.append(plot_to_image(figure))
	
	figure = plt.figure(figsize=(5,5))
	col_range = max(np.max(w2_r),-np.min(w2_r))
	plt.imshow(w2_r,cmap="seismic",vmax=col_range,vmin=-col_range)
	plt.ylabel("Output")
	plt.xlabel("Input")
	plt.title("Reaction layer 2")
	figs.append(plot_to_image(figure))
	

	return figs

def plot_weight_kernel_boxplot(pde):
	"""
	Plots boxplots of PDE 1st layer weights sorted by which channel they correspond to

	Parameters
	----------
	pde : object callable - (float32 array [T], float32 array [N_CHANNELS,_,_]) -> (float32 array [T,N_CHANNELS,_,_])
		the PDE solver object to plot parameters of

	Returns
	-------
	figs : list of images
		a list of images

	"""
	w1_v = pde.func.f_v.layers[0].weight[:,:,0,0]
	#w2_v = pde.func.f_v.layers[2].weight[:,:,0,0]
	
	#w1_d = pde.func.f_d.layers[-1].weight[:,:,0,0]
	
	w1_r = pde.func.f_r.layers[0].weight[:,:,0,0]
	#w2_r = pde.func.f_r.layers[2].weight[:,:,0,0]
	
	figs = []
	
	figure = plt.figure(figsize=(5,5))
	plt.boxplot(w1_v.T)
	plt.xlabel("Channels")
	plt.ylabel("Weights")
	plt.title("Advection 1st layer")
	figs.append(plot_to_image(figure))
	
	#figure = plt.figure(figsize=(5,5))
	#plt.boxplot(w1_d.T)
	#plt.xlabel("Channels")
	#plt.ylabel("Weights")
	#plt.title("Diffusion 1st layer")
	#figs.append(plot_to_image(figure))
	
	figure = plt.figure(figsize=(5,5))
	plt.boxplot(w1_r.T)
	plt.xlabel("Channels")
	plt.ylabel("Weights")
	plt.title("Reaction 1st layer")
	figs.append(plot_to_image(figure))
	return figs


def my_animate(img):
	"""
	Boilerplate code to produce matplotlib animation
	Parameters
	----------
	img : float32 or int array [N,rgb,_,_]
		img must be float in range [0,1] 
	"""
	img = np.clip(img,0,1)
	img = np.einsum("ncxy->nxyc",img)
	frames = [] # for storing the generated images
	fig = plt.figure()
	for i in range(img.shape[0]):
		frames.append([plt.imshow(img[i],vmin=0,vmax=1,animated=True)])
	ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,repeat_delay=0)
	plt.show()