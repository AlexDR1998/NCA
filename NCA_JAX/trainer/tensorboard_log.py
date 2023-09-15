import tensorflow as tf
import numpy as np
import io
import matplotlib.pyplot as plt

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













class NCA_Train_log(object):
	"""
		Class for logging training behaviour of NCA_Trainer classes, using tensorboard
	"""


	def __init__(self,log_dir,data,RGB_mode="RGB"):
		"""
			Initialises the tensorboard logging of training.
			Writes some initial information. Very similar to setup_tb_log_single, but designed for sequence modelling

		"""

		self.LOG_DIR = log_dir
		self.RGB_mode = RGB_mode
		
		
		train_summary_writer = tf.summary.create_file_writer(self.LOG_DIR)
		
		#--- Log the graph structure of the NCA
		#tf.summary.trace_on(graph=True,profiler=True)
		#y = self.NCA_model.perceive(self.x0)
		#with train_summary_writer.as_default():
		#	tf.summary.trace_export(name="NCA Perception",step=0,profiler_outdir=self.LOG_DIR)
		
		#tf.summary.trace_on(graph=True,profiler=True)
		#x = self.NCA_model(self.x0)
		#with train_summary_writer.as_default():
		#	tf.summary.trace_export(name="NCA full step",step=0,profiler_outdir=self.LOG_DIR)
		
		#--- Log the target image and initial condtions
		with train_summary_writer.as_default():
			if self.RGB_mode=="RGB":
				tf.summary.image('True sequence RGB',np.einsum("ncxy->nxyc",data[:,0,:3,...]),step=0,max_outputs=data.shape[0])
			elif self.RGB_mode=="RGBA":
				tf.summary.image('True sequence RGBA',np.einsum("ncxy->nxyc",data[:,0,:4,...]),step=0,max_outputs=data.shape[0])
			
		self.train_summary_writer = train_summary_writer

	def tb_training_loop_log_sequence(self,losses,x,i,nca,write_images=True):
		"""
			Helper function to format some data logging during the training loop

			Parameters
			----------
			
			losses : float32 array [N,BATCHES]
				loss for each timestep of each trajectory

			x : float32 array [N,BATCHES,CHANNELS,_,_]
				NCA state

			i : int
				current step in training loop - useful for logging something every n steps
				
			nca : object callable - (float32 array [N_CHANNELS,_,_],PRNGKey) -> (float32 array [N_CHANNELS,_,_])
				the NCA object being trained
			write_images : boolean optional
				flag whether to save images of intermediate x states. Useful for debugging if a model is learning, but can use up a lot of storage if training many models

		"""
		BATCHES = losses.shape[1]
		N = losses.shape[0]
		with self.train_summary_writer.as_default():
			tf.summary.histogram("Loss",losses,step=i)
			tf.summary.scalar("Mean Loss",np.mean(losses),step=i)
			for n in range(N):
				tf.summary.histogram("Loss of each batch, timestep "+str(n),losses[n],step=i)
				tf.summary.scalar("Loss of averaged over each batch, timestep "+str(n),np.mean(losses[n]),step=i)
			for b in range(BATCHES):
				tf.summary.histogram("Loss of each timestep, batch "+str(b),losses[:,b],step=i)
				tf.summary.scalar("Loss of averaged over each timestep,  batch "+str(b),np.mean(losses[:,b]),step=i)
			if i%10==0:

				# Log weights and biasses of model every 10 training epochs
				weight_matrix_image = []
				w1 = nca.layers[0].weight[:,:,0,0]
				w2 = nca.layers[2].weight[:,:,0,0]
				b2 = nca.layers[2].bias[:,0,0]
				
				tf.summary.histogram('Input layer weights',w1,step=i)
				tf.summary.histogram('Output layer weights',w2,step=i)
				tf.summary.histogram('Output layer bias',b2,step=i)
					
				
				figure = plt.figure(figsize=(5,5))
				col_range = max(np.max(w1),-np.min(w1))
				plt.imshow(w1,cmap="seismic",vmax=col_range,vmin=-col_range)
				plt.ylabel("Output")
				plt.xlabel(r"N_CHANNELS$\star$ KERNELS")
				weight_matrix_image.append(plot_to_image(figure))
				
				figure = plt.figure(figsize=(5,5))
				col_range = max(np.max(w2),-np.min(w2))
				plt.imshow(w2,cmap="seismic",vmax=col_range,vmin=-col_range)
				plt.xlabel("Input from previous layer")
				plt.ylabel("NCA state increments")
				weight_matrix_image.append(plot_to_image(figure))
				
				tf.summary.image("Weight matrices",np.array(weight_matrix_image)[:,0],step=i)
				
				if write_images:
					for b in range(BATCHES):
						if self.RGB_mode=="RGB":
							tf.summary.image('Trajectory batch '+str(b),np.einsum("ncxy->nxyc",x[:,b,:3,...]),step=i,max_outputs=x.shape[0])
						elif self.RGB_mode=="RGBA":
							tf.summary.image('Trajectory batch '+str(b),np.einsum("ncxy->nxyc",x[:,b,:4,...]),step=i,max_outputs=x.shape[0])
				
				
				
				
				
				
				
				
				
# 			for j in range(self.T):
# 				if j==0:
# 					tf.summary.scalar('Mean Loss',loss[0],step=i)
# 				else:
# 					tf.summary.scalar('Loss '+str(j),loss[j],step=i)
# 					if i%10==0:
# 						if self.RGB_mode=="RGB":
# 							tf.summary.image('Model sequence step '+str(j)+' RGB',
# 										 	x[(j-1)*N_BATCHES:(j)*N_BATCHES,...,:3],
# 										 	step=i)
# 						elif self.RGB_mode=="RGBA":
# 							tf.summary.image('Model sequence step '+str(j)+' RGBA',
# 										 	x[(j-1)*N_BATCHES:(j)*N_BATCHES,...,:4],
# 										 	step=i)
# 			#tf.summary.scalar('Loss 1',loss[1],step=i)
# 			#tf.summary.scalar('Loss 2',loss[2],step=i)
# 			#tf.summary.scalar('Loss 3',loss[3],step=i)
# 			tf.summary.histogram('Loss ',loss,step=i)

			
					
					#tf.summary.image('Layer '+str(n)+' weight matrix',tf.einsum("...ijk->...kji",model_params[0]),step=i)
					#try:
					#	tf.summary.histogram('Layer '+str(n)+' biases',model_params[1],step=i)
					#except Exception as e:
					#	pass
	