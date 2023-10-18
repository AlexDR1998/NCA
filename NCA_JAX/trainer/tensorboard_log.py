import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU") # Force tensorflow not to use GPU, as it's only logging data
import numpy as np
from NCA_JAX.NCA_visualiser import *





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
		
		#--- Log the target image and initial condtions
		with train_summary_writer.as_default():
			for b in range(len(data)):
				if self.RGB_mode=="RGB":
					#tf.summary.image('True sequence RGB',np.einsum("ncxy->nxyc",data[0,:,:3,...]),step=0,max_outputs=data.shape[0])
					tf.summary.image('True sequence RGB',np.einsum("ncxy->nxyc",data[b][:,:3,...]),step=b,max_outputs=data[b].shape[0])
				elif self.RGB_mode=="RGBA":
					#tf.summary.image('True sequence RGBA',np.einsum("ncxy->nxyc",data[0,:,:4,...]),step=0,max_outputs=data.shape[0])
					tf.summary.image('True sequence RGBA',np.einsum("ncxy->nxyc",data[b][:,:4,...]),step=b,max_outputs=data[b].shape[0])
			
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
		#print(losses.shape)
		BATCHES = losses.shape[0]
		N = losses.shape[1]
		with self.train_summary_writer.as_default():
			tf.summary.histogram("Loss",losses,step=i)
			tf.summary.scalar("Mean Loss",np.mean(losses),step=i)
			for n in range(N):
				tf.summary.histogram("Loss of each batch, timestep "+str(n),losses[:,n],step=i)
				tf.summary.scalar("Loss of averaged over each batch, timestep "+str(n),np.mean(losses[:,n]),step=i)
			for b in range(BATCHES):
				tf.summary.histogram("Loss of each timestep, batch "+str(b),losses[b],step=i)
				tf.summary.scalar("Loss of averaged over each timestep,  batch "+str(b),np.mean(losses[b]),step=i)
			if i%10==0:

				# Log weights and biasses of model every 10 training epochs
				weight_matrix_image = []
				w1 = nca.layers[3].weight[:,:,0,0]
				w2 = nca.layers[5].weight[:,:,0,0]
				b2 = nca.layers[5].bias[:,0,0]
				
				tf.summary.histogram('Input layer weights',w1,step=i)
				tf.summary.histogram('Output layer weights',w2,step=i)
				tf.summary.histogram('Output layer bias',b2,step=i)				
				#diff,static=nca.partition()
				weight_matrix_figs = plot_weight_matrices(nca)
				tf.summary.image("Weight matrices",np.array(weight_matrix_figs)[:,0],step=i)
				
				kernel_weight_figs = plot_weight_kernel_boxplot(nca)
				tf.summary.image("Input weights per kernel",np.array(kernel_weight_figs)[:,0],step=i)
				
				
				if write_images:
					for b in range(BATCHES):
						if self.RGB_mode=="RGB":
							#tf.summary.image('Trajectory batch '+str(b),np.einsum("ncxy->nxyc",x[b,:,:3,...]),step=i,max_outputs=x.shape[0])
							tf.summary.image('Trajectory batch '+str(b),np.einsum("ncxy->nxyc",x[b][:,:3,...]),step=i,max_outputs=N)
						elif self.RGB_mode=="RGBA":
							#tf.summary.image('Trajectory batch '+str(b),np.einsum("ncxy->nxyc",x[b,:,:4,...]),step=i,max_outputs=x.shape[0])
							tf.summary.image('Trajectory batch '+str(b),np.einsum("ncxy->nxyc",x[b][:,:4,...]),step=i,max_outputs=N)
					if nca.N_CHANNELS > 4:
						b=0
						if self.RGB_mode=="RGB":
							hidden_channels = x[b][:,3:]
						elif self.RGB_mode=="RGBA":
							hidden_channels = x[b][:,4:]
						extra_zeros = (-hidden_channels.shape[1])%3
						hidden_channels = np.pad(hidden_channels,((0,0),(0,extra_zeros),(0,0),(0,0)))
						#print(hidden_channels.shape)
						w = hidden_channels.shape[-2]
						h = hidden_channels.shape[-1]
						hidden_channels_r = np.reshape(hidden_channels,(hidden_channels.shape[0],3,w*(hidden_channels.shape[1]//3),h))
						#tf.summary.image('Trajectory batch 0, hidden channels',np.einsum("ncxy->nxyc",hidden_channels_r),step=i,max_outputs=x.shape[0])
						tf.summary.image('Trajectory batch 0, hidden channels',np.einsum("ncxy->nxyc",hidden_channels_r),step=i,max_outputs=N)

	