import numpy as np
#import silence_tensorflow.auto # shuts up tensorflow's spammy warning messages
import tensorflow as tf
import tensorflow_model_optimization.sparsity.keras as tfmot
#import scipy as sp
from tqdm import tqdm
import datetime
from NCA.trainer.NCA_train_utils import *
from NCA.trainer.NCA_loss_functions import *
from time import time
from scipy.ndimage import rotate as sp_rotate


class NCA_Trainer(object):
	"""
		Class to train NCA to data, as well as logging the process via tensorboard.
		Very general, should work on any sequence of images (of same size)
	
	"""

	def __init__(self,NCA_model,data,N_BATCHES,model_filename=None,RGB_mode="RGBA",CYCLIC=False,directory="models/"):
		"""
			Initialiser method

			Parameters
			----------
			NCA_model : object callable - float32 tensor [batches,size,size,N_CHANNELS],float32,float32 -> float32 tensor [batches,size,size,N_CHANNELS]
				the NCA object to train
			data : float32 tensor [T,batches,size,size,4]
				The image sequence being modelled. data[0] is treated as the initial condition
			N_BATCHES : int
				size of training batch
			model_filename : str
				name of directories to save tensorboard log and model parameters to.
				log at :	'logs/gradient_tape/model_filename/train'
				model at : 	'models/model_filename'
				if None, sets model_filename to current time
			CYCLIC : boolean optional
				Flag for if data is cyclic, i.e. include X_N -> X_0 in training
			RGB_mode : string
				Expects "RGBA" "RGB" or "RGB-A"
				Defines how to log image channels to tensorboard
				RGBA : 4 channel RGBA image (i.e. PNG with transparancy)
				RGB : 3 channel RGB image
				RGB-A : 3+1 channel - a 3 channel RGB image alongside a black and white image representing the alpha channel
			directory : str
				Name of directory where all models get stored, defaults to 'models/'
		"""
		
		


		data = data.astype("float32") # Cast data to float32 for tensorflow
		self.NCA_model = NCA_model
		self.N_BATCHES = N_BATCHES
		self.CYCLIC = CYCLIC
		self.N_CHANNELS = self.NCA_model.N_CHANNELS
		self.OBS_CHANNELS=self.NCA_model.OBS_CHANNELS
		if self.OBS_CHANNELS==3:
			RGB_mode = "RGB"

		self.RGB_mode = RGB_mode
		

		if model_filename is None:
			self.model_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		else:
			self.model_filename = model_filename
		self.directory = directory
		self.LOG_DIR = "logs/gradient_tape/"+self.model_filename+"/train"
		print("Saving to: "+directory+model_filename)
		
		#--- Setup initial condition
		if data.shape[1]==1:
			#If there is only 1 batch of data, repeat it along batch axis N_BATCHES times
			data = np.repeat(data,N_BATCHES,axis=1).astype("float32")
		
		x0 = np.copy(data[:-1])
		target = data[1:]
		
		
		self.T = data.shape[0]
		z0 = np.zeros((x0.shape[0],x0.shape[1],x0.shape[2],x0.shape[3],self.N_CHANNELS-x0.shape[4]))
		x0 = np.concatenate((x0,z0),axis=-1).astype("float32")
		
		
		
		self.x0 = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
		if self.NCA_model.ENV_CHANNEL_DATA is not None:
			self.x0 = self.NCA_model.env_channel_callback(self.x0)
		self.target = target.reshape((-1,target.shape[2],target.shape[3],target.shape[4]))
		self.data = data
		
		self.x0_true = np.copy(self.x0)
		self.x0 = tf.convert_to_tensor(self.x0)
		self.PRUNE = False
		# data augmentation flags
		self.FLIP = False
		self.SHIFT = False
		self.NOISE = False
		self.ROTATE= False
		self.PAD = False
		self.setup_tb_log_sequence()
		print("---Shapes of NCA_Trainer variables---")
		print("data: "+str(self.data.shape))
		print("X0: "+str(self.x0.shape))
		print("Target: "+str(self.target.shape))
		
		

	
	def loss_func(self,x,x_true=None):
		"""
			Loss function for training to minimise. Averages over batches, returns error per time slice (sequence)

			Parameters
			----------
			x : float32 tensor [(T-1)*N_BATCHES,size,size,N_CHANNELS]
				Current state of NCA grids, in sequence training mode
			x_true : float32 tensor optional [(T-1)*N_BATCHES,size,size,N_CHANNELS]
				Target state for NCA grids. Defaults to self.target
			Returns
			-------
			loss : float32 tensor [T]
				Array of errors at each timestep
		"""
		if x_true is None:
			x_true = self.target
		eu = tf.math.reduce_euclidean_norm((x[...,:self.OBS_CHANNELS]-x_true),[-2,-3,-1])
		# self.N_BATCHES: removes the 'hidden' 12h state
		return tf.reduce_mean(tf.reshape(eu,(-1,self.N_BATCHES)),-1)


	def setup_tb_log_sequence(self):
		"""
			Initialises the tensorboard logging of training.
			Writes some initial information. Very similar to setup_tb_log_single, but designed for sequence modelling

		"""


		#if model_filename is None:
		#	current_time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		#	train_log_dir = "logs/gradient_tape/"+current_time+"/train"
		#else:
		
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
				tf.summary.image('True sequence RGB',self.data[:,0,...,:3],step=0,max_outputs=self.data.shape[0])
			elif self.RGB_mode=="RGBA":
				tf.summary.image('True sequence RGBA',self.data[:,0,...,:4],step=0,max_outputs=self.data.shape[0])
			
		self.train_summary_writer = train_summary_writer

	def tb_training_loop_log_sequence(self,loss,x,i):
		"""
			Helper function to format some data logging during the training loop

			Parameters
			----------
			
			loss : float32 tensor [T]
				Array of errors at each timestep (24h, 36h, 48h)

			x : float32 tensor [N_BATCHES,size,size,N_CHANNELS]
				final output of NCA

			i : int
				current step in training loop - useful for logging something every n steps

		"""
		N_BATCHES = self.N_BATCHES
		with self.train_summary_writer.as_default():
			for j in range(self.T):
				if j==0:
					tf.summary.scalar('Mean Loss',loss[0],step=i)
				else:
					tf.summary.scalar('Loss '+str(j),loss[j],step=i)
					if i%10==0:
						if self.RGB_mode=="RGB":
							tf.summary.image('Model sequence step '+str(j)+' RGB',
										 	x[(j-1)*N_BATCHES:(j)*N_BATCHES,...,:3],
										 	step=i)
						elif self.RGB_mode=="RGBA":
							tf.summary.image('Model sequence step '+str(j)+' RGBA',
										 	x[(j-1)*N_BATCHES:(j)*N_BATCHES,...,:4],
										 	step=i)
			#tf.summary.scalar('Loss 1',loss[1],step=i)
			#tf.summary.scalar('Loss 2',loss[2],step=i)
			#tf.summary.scalar('Loss 3',loss[3],step=i)
			tf.summary.histogram('Loss ',loss,step=i)
			tf.summary.scalar("Layer 1 weight sparsity",np.count_nonzero(self.NCA_model.dense_model.layers[0].get_weights()[0]==0),step=i)
			tf.summary.scalar("Layer 2 weight sparsity",np.count_nonzero(self.NCA_model.dense_model.layers[1].get_weights()[0]==0),step=i)
			if i%10==0:

				#model_params=[]
				#for n in range(self.NCA_model.N_layers):
				#	model_params.append(self.NCA_model.dense_model.layers[n].get_weights())



				#print(np.array(model_params[0]).shape)
				#print(np.array(model_params[1]).shape)
				#print(ca.dense_model.layers[0].get_weights()[0])
				#tf.summary.image('Weight matrices',model_params,step=i)
				weight_matrix_image = []
				for n in range(self.NCA_model.N_layers):
					model_params = self.NCA_model.dense_model.layers[n].get_weights()
					tf.summary.histogram('Layer '+str(n)+' weights',model_params[0],step=i)
					
					figure = plt.figure(figsize=(5,5))
					col_range = max(np.max(model_params[0][0,0]),-np.min(model_params[0][0,0]))
					plt.imshow(model_params[0][0,0],cmap="seismic",vmax=col_range,vmin=-col_range)
					if n==0:
						plt.ylabel(r"N_CHANNELS$\star$ KERNELS")
					else:
						plt.ylabel("Input from previous layer")
					if n==self.NCA_model.N_layers-1:
						plt.xlabel("NCA state increments")
					else:
						plt.xlabel("Output")
					weight_matrix_image.append(plot_to_image(figure))
				tf.summary.image("Weight matrices",np.array(weight_matrix_image)[:,0],step=i)
					
					#tf.summary.image('Layer '+str(n)+' weight matrix',tf.einsum("...ijk->...kji",model_params[0]),step=i)
					#try:
					#	tf.summary.histogram('Layer '+str(n)+' biases',model_params[1],step=i)
					#except Exception as e:
					#	pass
				
				"""
				model_params_1 = self.NCA_model.dense_model.layers[1].get_weights()
				model_params_2 = self.NCA_model.dense_model.layers[2].get_weights()
				tf.summary.histogram('Layer 1 weights',model_params_1[0],step=i)
				tf.summary.histogram('Layer 2 weights',model_params_2[0],step=i)
				tf.summary.histogram('Layer 1 biases',model_params_1[1],step=i)
				tf.summary.histogram('Layer 2 biases',model_params_2[1],step=i)
				"""
	
	def tb_write_result(self,iter_n):
		"""
			Log trained behaviour of NCA model to tensorboard

			Parameters
			----------
			iter_n : int
				How many steps between each sequence image
		"""

		with self.train_summary_writer.as_default():
			try:
				grids = self.BEST_TRAJECTORY
			except:
				self.BEST_TRAJECTORY = self.NCA_model.run(self.x0,iter_n*self.T*2,N_BATCHES=self.N_BATCHES).numpy()
				grids = self.BEST_TRAJECTORY
				print("-------Warning - Loss function was not reduced, displaying last model instead----------")
			grids[...,self.OBS_CHANNELS:] = (1+np.tanh(grids[...,self.OBS_CHANNELS:]))/2.0
			for i in range(iter_n*self.T*2):
				if self.RGB_mode=="RGB":
					tf.summary.image('Trained NCA dynamics RGB at step '+str(self.time_of_best_model),
									 grids[i,...,:self.OBS_CHANNELS],
									 step=i)
				elif self.RGB_mode=="RGBA":
					tf.summary.image('Trained NCA dynamics RGBA at step '+str(self.time_of_best_model),
									 grids[i,...,:self.OBS_CHANNELS],
									 step=i)
				elif self.RGB_mode=="RGB-A":
					tf.summary.image('Trained NCA dynamics RGB --- Alpha at step '+str(self.time_of_best_model),
									 np.concatenate((grids[i,...,:self.OBS_CHANNELS-1],
									 				 np.repeat(grids[i,...,self.OBS_CHANNELS-1:self.OBS_CHANNELS],3,axis=-1)),
									 				axis=1),
									 step=i)
				if self.N_CHANNELS>self.OBS_CHANNELS:	
					
					hidden_channels = grids[i,...,self.OBS_CHANNELS:]
					if hidden_channels.shape[-1]%3!=0:
						# Hidden channels is not divisible by 3 (RGB), append zeros to make visualisation easy
						
						new_channels = np.zeros((hidden_channels.shape[0],hidden_channels.shape[1],hidden_channels.shape[2],3 - hidden_channels.shape[-1]%3))
						hidden_channels = np.concatenate((hidden_channels,new_channels),axis=-1)
					
					hidden_channels_r = hidden_channels[...,:3]
					for c in range(1,hidden_channels.shape[-1]//3):
						hidden_channels_r = np.concatenate((hidden_channels_r,hidden_channels[...,3*c:3*(c+1)]),axis=1)
						
					#hidden_channels_r = np.einsum("bxyc->cxyb",hidden_channels)
					tf.summary.image("Trained NCA hidden channel dynamics RGB at step "+str(self.time_of_best_model),
									  hidden_channels_r,
									  step=i)
					
					
# 					try:
# 						hidden_channels = grids[i,...,self.OBS_CHANNELS:]
# 						hidden_channels_reshaped = hidden_channels[...,0]
# 						for k in range(1,hidden_channels.shape[-1]):
# 							hidden_channels_reshaped = tf.concat([hidden_channels_reshaped,hidden_channels[...,k]],axis=1)
# 						tf.summary.image('Trained NCA hidden dynamics (tanh limited) of batch 0 at step '+str(self.time_of_best_model),
# 										 hidden_channels_reshaped,step=i)
# 					except:
# 						pass
					"""
					for j in range((self.N_CHANNELS-1)//3-1):
						try:
							if j==0:
								hidden_channels = grids[i,...,self.OBS_CHANNELS:self.OBS_CHANNELS+3]
							else:
								hidden_channels = np.concatenate((hidden_channels,
																  grids[i,...,((j+1)*3)+1:((j+2)*3)+1]))
						except:
							pass
					
					"""
	
	def prune_setup(self,sparsity=0.5):
		"""		
		Wraps the NCA_model.dense_model network in the tfmot.sparsity.keras.prune_low_magnitude() 
		framework to prune model weights during training, forcing a sparser model.


		Parameters
		----------
		sparsity : float in [0,1]
			target sparsity

		Returns
		-------
		None.

		"""
		
		pruning_params = {'pruning_schedule': tfmot.ConstantSparsity(sparsity, 0),
					      'block_size': (1, 1),
						  'block_pooling_type': 'AVG'}
		
		self.NCA_model.dense_model = tfmot.prune_low_magnitude(self.NCA_model.dense_model,**pruning_params)
		self.PRUNE = True
		self.NCA_model.dense_model.optimizer = self.trainer
		self.PRUNE_CALLBACK = tfmot.UpdatePruningStep()
		self.PRUNE_CALLBACK.set_model(self.NCA_model.dense_model)
		self.PRUNE_LOG = tfmot.PruningSummaries(log_dir=self.LOG_DIR)
		self.PRUNE_LOG.set_model(self.NCA_model.dense_model)
		
	
	def prune_remove(self):
		"""
			Removes the pruning wrappers and writes self.NCA_model.dense_model with the parameters of self.NCA_model.pruned_dense_model
		"""
		self.NCA_model.dense_model = tfmot.strip_pruning(self.NCA_model.dense_model)
		self.PRUNE = False
	def train_step(self,x,iter_n,REG_COEFF=0,update_gradients=True,LOSS_FUNC=None,NORM_GRADS=True):
		"""
			Training step. Runs NCA model once, calculates loss gradients and tweaks model
			
			Parameters
			----------
			x : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
			
			iter_n : int
				number of NCA update s
				teps to run ca

			REG_COEFF : float32 optional
				Strength of intermediate state regulariser - penalises any pixels outwith [0,1]
			
			update_gradients : bool optional
				Controls whether model weights are updated at this step

			LOSS_FUNC : (float32 tensor [N_BATCHES,X,Y,N_CHANNELS])**2 -> float32 tensor [N_BATCHES] optional
				Alternative loss function
			
			NORM_GRADS : bool optional
				Choose whether to normalise gradient updates


		"""
		
		#--- If alternative loss function is provided, use that, otherwise use default
		if LOSS_FUNC is None:
			loss_func =  self.loss_func
		else:
			loss_func = lambda x: LOSS_FUNC(x[...,:self.OBS_CHANNELS],self.target)
		
		
		
		

		with tf.GradientTape() as g:
			reg_log = []
			for i in range(iter_n):
				x = self.NCA_model(x)
				if self.NCA_model.ENV_CHANNEL_DATA is not None:
					#print("X shape: "+str(x.shape))
					#print("ENV_CHANNEL_DATA shape: "+str(self.NCA_model.ENV_CHANNEL_DATA.shape))
					x = self.NCA_model.env_channel_callback(x)
					
				#--- Intermediate state regulariser, to penalise any pixels being outwith [0,1]
				reg_log.append(tf.reduce_sum(tf.nn.relu(-x)+tf.nn.relu(x-1)))
				
			#print(x.shape)
			losses = loss_func(x) 
			reg_loss = tf.cast(tf.reduce_mean(reg_log),tf.float32)
			mean_loss = tf.reduce_mean(losses) + REG_COEFF*reg_loss
					
		if update_gradients:
			
			if self.PRUNE:
				grads = g.gradient(mean_loss,self.NCA_model.dense_model.trainable_variables)
			else:
				grads = g.gradient(mean_loss,self.NCA_model.dense_model.weights)
			
			
			if NORM_GRADS:
				grads = [g/(tf.norm(g)+1e-8) for g in grads]
			
			if self.PRUNE:
				self.trainer.apply_gradients(zip(grads, self.NCA_model.dense_model.trainable_variables))
				
			else:
				self.trainer.apply_gradients(zip(grads, self.NCA_model.dense_model.weights))
		x = np.array(x)
		return x, mean_loss,losses

	def train_sequence(self,TRAIN_ITERS,iter_n,UPDATE_RATE=1,REG_COEFF=0,LOSS_FUNC=None,LEARN_RATE=2e-3,OPTIMIZER="Adagrad",NORM_GRADS=True,INJECT_TRUE=True,LOSS_WEIGHTS=None,PRUNE_MODEL=False,SPARSITY=0.5):
		"""
			Trains the ca to recreate the given image sequence. Error is calculated by comparing ca grid to each image after iter_n/T steps 
			
			Parameters
			----------
			
			TRAIN_ITERS : int
				how many iterations of training
			iter_n : int
				number of NCA update steps to run ca

			UPDATE_RATE: float32 optional
				Controls stochasticity of applying gradient updates, potentially useful for breaking dominance of short time effects

			REG_COEFF : float32 optional
				Strength of intermediate state regulariser - penalises any pixels outwith [0,1]
			
			LOSS_FUNC : (float32 tensor [N_BATCHES,X,Y,N_CHANNELS])**2 -> float32 tensor [N_BATCHES] optional
				Alternative loss function

			LEARN_RATE : float32 optional
				Learning rate for optimisation algorithm

			OPTIMIZER : string optional {"Adagrad","Adam","Adadelta","Nadam","RMSprop"}
				Select which tensorflow.keras.optimizers method to use
			LOSS_WEIGHTS : float32 tensor [T] optional
				If provided, weights how much each timestep in a trajectory contributes to loss calculations
			INJECT_TRUE : bool optional
				If true, after each iter_n iterations of NCA, one sample from the true trajectory is injected back into the current NCA trajectory before the next iteration.
			PRUNE_MODEL : bool optional
				If true, enforce sparsity of network with pruning during training
			SPARSITY : float [0,1] optional
				Target sparsity parameter for network pruning
			Returns
			-------
			None
		"""
		self.INJECT_TRUE = INJECT_TRUE
		
		#--- Setup training algorithm

		lr = LEARN_RATE
		#lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([TRAIN_ITERS//2], [lr, lr*0.1])
		lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(lr, TRAIN_ITERS, 0.99)
		if OPTIMIZER=="Adagrad":
			self.trainer = tf.keras.optimizers.Adagrad(lr_sched)
		elif OPTIMIZER=="Adam":
			self.trainer = tf.keras.optimizers.Adam(lr_sched)
		elif OPTIMIZER=="Adadelta":
			self.trainer = tf.keras.optimizers.Adadelta()#(lr_sched)
		elif OPTIMIZER=="Nadam":
			self.trainer = tf.keras.optimizers.Nadam(LEARN_RATE)#lr_sched)
		elif OPTIMIZER=="RMSprop":
			self.trainer = tf.keras.optimizers.RMSprop(lr_sched)
		else:
			print('No optimizer selected. Please select one of: {"Adagrad","Adam","Adadelta","Nadam","RMSprop"}')
			return None
		
		
	

		


		if LOSS_WEIGHTS is None:
			self.LOSS_WEIGHTS = tf.ones(shape=self.x0.shape[0])
		else:
			self.LOSS_WEIGHTS = tf.repeat(LOSS_WEIGHTS,self.N_BATCHES)
		
		
		
		
		#--- Pruning setup
		if PRUNE_MODEL:
			self.prune_setup(SPARSITY)
			self.PRUNE_CALLBACK.on_train_begin()
		
		#--- Do training loop
		best_mean_loss = 100000
		self.time_of_best_model = 0
		N_BATCHES = self.N_BATCHES
		#print(N_BATCHES)
		#print(self.x0.shape)
		#print(self.x0_true.shape)
		

		start_time = time()
		for i in tqdm(range(TRAIN_ITERS)):
			R = np.random.uniform()<UPDATE_RATE
			if PRUNE_MODEL:
				self.PRUNE_LOG.on_epoch_begin(epoch=i)
				self.PRUNE_CALLBACK.on_train_batch_begin(batch=i)
			x,mean_loss,losses = self.train_step(self.x0,iter_n,REG_COEFF,update_gradients=R,LOSS_FUNC=LOSS_FUNC,NORM_GRADS=NORM_GRADS)#,i%4==0)
			assert not tf.math.reduce_any(tf.math.is_nan(x)), "|-|-|-|-|-|-  X reached NaN  -|-|-|-|-|-|"

		
			self.x0 = self.x0.numpy()
			

			if self.CYCLIC:
				#self.x0 = np.roll(x,N_BATCHES,axis=0)
				self.x0[N_BATCHES:] = x[:-N_BATCHES]
				self.x0[:N_BATCHES] = x[-N_BATCHES:]
			else:
				self.x0[N_BATCHES:] = x[:-N_BATCHES] # updates each initial condition to be final condition of previous chunk of timesteps
			if N_BATCHES>1 and self.INJECT_TRUE:
				#self.x0[::N_BATCHES][1:] = self.x0_true[::N_BATCHES][1:] # update one batch to contain the true initial conditions
				self.x0[::N_BATCHES] = self.x0_true[::N_BATCHES] # update one batch to contain the true initial conditions
			self.x0 = tf.convert_to_tensor(self.x0)
			if self.NOISE:
				self.data_noise_augment()

			if PRUNE_MODEL:
				self.PRUNE_CALLBACK.on_epoch_end(batch=i)

			#--- Save model each time it is better than previous best model (and after 10% of training iterations are done)
			if (mean_loss<best_mean_loss) and (i>TRAIN_ITERS//10):
				if self.model_filename is not None:
					self.NCA_model.save_wrapper(self.model_filename,self.directory)
					tqdm.write("--- Model saved at "+str(i)+" epochs ---")
				
				self.BEST_TRAJECTORY = self.NCA_model.run(self.x0,iter_n*self.T*2,N_BATCHES=self.N_BATCHES).numpy()
				self.time_of_best_model = i
				best_mean_loss = mean_loss

			loss = np.hstack((mean_loss,losses))
			
			
			if i%(self.T*2)==0 and i > 1:
				need_to_reset = True
				if self.FLIP:
					self.data_flip_augment(reset=need_to_reset)
					need_to_reset = False
				if self.ROTATE:
					self.data_rotate_augment(reset=need_to_reset)
					need_to_reset = False
				if self.SHIFT:
					self.data_shift_augment(reset=need_to_reset)
					need_to_reset = False
				if self.NOISE:
					self.data_noise_augment()
			#--- Write to log
			self.tb_training_loop_log_sequence(loss,x,i)
		print("-------- Training complete ---------")
		print("Time taken (seconds): "+str(time()-start_time))
		#--- Write resulting best animation to tensorboard			
		self.tb_write_result(iter_n)

		#--- If a filename is provided, save the trained NCA model.
		#if model_filename is not None:
		#	ca.save_wrapper(model_filename)

	def data_pad_augment(self,WIDTH):
		"""
			Augments training data by padding with extra zeros

			Parameters
			----------
			
			WIDTH : int
				How wide to pad data with zeros

		"""
		
		#--- Reshape x0 and target to be [T-1,batch,size,size,channels]
		self.PAD = True
		self.x0 = self.x0.numpy()
		self.WIDTH = WIDTH
		x0 = self.x0.reshape((self.T-1,self.N_BATCHES,self.x0.shape[1],self.x0.shape[2],-1))
		target = self.target.reshape((self.T-1,self.N_BATCHES,self.target.shape[1],self.target.shape[2],-1))
		
		#--- Pad along [:,:,size,size,:] dimensions
		padwidth = ((0,0),(0,0),(WIDTH,WIDTH),(WIDTH,WIDTH),(0,0))
		x0 = np.pad(x0,padwidth)
		target = np.pad(target,padwidth)
		self.x0 = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
		self.target = target.reshape((-1,target.shape[2],target.shape[3],target.shape[4]))
		self.x0_true = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
		self.x0 = tf.convert_to_tensor(self.x0)
		
	def data_shift_augment(self,reset=True):
		"""
		
		Randomly translates I.C - target pairs
		Should result in NCA that ignores simulation boundary
		
		Doesn't augment first 1/2 batches, but uses them as reference when redoing augmentation
		
		"""
		self.SHIFT = True
		self.x0 = self.x0.numpy()
		#--- Reshape input
		x0 = self.x0.reshape((self.T-1,self.N_BATCHES,self.x0.shape[1],self.x0.shape[2],-1))
		target = self.target.reshape((self.T-1,self.N_BATCHES,self.target.shape[1],self.target.shape[2],-1))
		
		
		#--- If data has already been shifted, undo that operation first
# 		if hasattr(self,"shifts"):
# 			for t in range(x0.shape[0]):
# 				for b in range(x0.shape[1]):
# 					x0[t,b] = np.roll(x0[t,b],shift= -self.shifts[b],axis=(0,1))
# 					target[t,b] = np.roll(target[t,b],shift= -self.shifts[b],axis=(0,1))
			
				
		
		#--- Reset augmented batches to unaugmented values - use 1 instead of 0 as batch 0 has true state injection
		
		if reset:
			x0[:,2:] = x0[:,1:2]
			target[:,2:]=target[:,1:2]
		
		#--- Randomly offset each batch of images
		self.shifts = np.random.randint(low=-self.WIDTH,high=self.WIDTH,size=(x0.shape[1],2))
		for t in range(x0.shape[0]):
			for b in range(2,x0.shape[1]):
				x0[t,b] = np.roll(x0[t,b],shift=self.shifts[b],axis=(0,1))
				target[t,b] = np.roll(target[t,b],shift=self.shifts[b],axis=(0,1))

		#--- Reshape back to [(T-1)*batch,size,size,channels]
		self.x0 = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
		self.target = target.reshape((-1,target.shape[2],target.shape[3],target.shape[4]))
		self.x0_true = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
		self.x0 = tf.convert_to_tensor(self.x0)

	def data_noise_augment(self,AMOUNT=0.0001,mode="full"):
		"""
			Augments training data by adding noise to the I.C. scaled by AMOUNT

			Parameters
			----------
			AMOUNT : float [0,1]
				Interpolates between initial data at 0 and pure noise at 1


		"""
		self.NOISE = True
		self.x0 = self.x0.numpy()
		#--- Reshape input
		x0 = self.x0.reshape((self.T-1,self.N_BATCHES,self.x0.shape[1],self.x0.shape[2],-1))
		#target = self.target.reshape((self.T-1,self.N_BATCHES,self.target.shape[1],self.target.shape[2],-1))
		
		


		#if hasattr(self, "x0_denoise"):
		#	x0[:,0] = self.x0_denoise
			#target[:,0] = self.target_denoise
		#else:
		#	self.x0_denoise = x0[:,0]
			#self.target_denoise= target[:,0]
		#--- add noise to each augmented batch
		
		x0 = AMOUNT*np.random.uniform(size=x0.shape) + (1-AMOUNT)*x0
		#target = AMOUNT*np.random.uniform(size=target.shape) + (1-AMOUNT)*target
		

		#--- Reshape back to [(T-1)*batch,size,size,channels]
		self.x0 = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
		#self.target = target.reshape((-1,target.shape[2],target.shape[3],target.shape[4]))
		self.x0_true = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
		self.x0 = tf.convert_to_tensor(self.x0)
	
	
	def data_rotate_augment(self,reset=True):
		"""
			Augments training data by randomly rotating



		"""
		self.ROTATE = True
		self.x0 = self.x0.numpy()
		#--- Reshape input
		x0 = self.x0.reshape((self.T-1,self.N_BATCHES,self.x0.shape[1],self.x0.shape[2],-1))
		target = self.target.reshape((self.T-1,self.N_BATCHES,self.target.shape[1],self.target.shape[2],-1))
		
		
		#--- If data has already been rotated, undo that operation first - DON'T NEED TO
# 		if self.angles is not None:
# 			for t in range(x0.shape[0]):
# 				for b in range(x0.shape[1]):
# 					x0[t,b] = sp_rotate(x0[t,b],angle= -self.angles[b],axes=(0,1),reshape = False, mode = "nearest",order=1)
# 					target[t,b] = sp_rotate(target[t,b],angle= -self.angles[b],axes=(0,1),reshape = False, mode = "nearest",order=1)
			
		
		#--- Reset augmented batches to unaugmented values - use 1 instead of 0 as batch 0 has true state injection
		if reset:
			x0[:,2:] = x0[:,1:2]
			target[:,2:]=target[:,1:2]
		
		#--- Randomly rotate each batch of images
		self.angles = np.random.randint(low=0,high=360,size=(x0.shape[1]))
		for t in range(x0.shape[0]):
			for b in range(2,x0.shape[1]):
				x0[t,b] = sp_rotate(x0[t,b],angle=self.angles[b],axes=(0,1),reshape = False, mode = "nearest",order=1)
				target[t,b] = sp_rotate(target[t,b],angle=self.angles[b],axes=(0,1),reshape = False, mode = "nearest",order=1)

				
		#--- Reshape back to [(T-1)*batch,size,size,channels]
		self.x0 = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
		self.target = target.reshape((-1,target.shape[2],target.shape[3],target.shape[4]))
		self.x0_true = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
		self.x0 = tf.convert_to_tensor(self.x0)
	def data_flip_augment(self,reset=True):
		"""
		

		Parameters
		----------
		reset : TYPE, optional
			DESCRIPTION. The default is True.

		Returns
		-------
		None.

		"""
		self.FLIP = True
		self.x0 = self.x0.numpy()
		#--- Reshape input
		x0 = self.x0.reshape((self.T-1,self.N_BATCHES,self.x0.shape[1],self.x0.shape[2],-1))
		target = self.target.reshape((self.T-1,self.N_BATCHES,self.target.shape[1],self.target.shape[2],-1))
		
		
		
		#--- Reset augmented batches to unaugmented values - use 1 instead of 0 as batch 0 has true state injection
		if reset:
			x0[:,2:] = x0[:,1:2]
			target[:,2:]=target[:,1:2]
		
		#--- Randomly flip each batch of images
		flip_axes = np.random.randint(2,size=x0.shape[1])
		do_flip = np.random.randint(2,size=x0.shape[1])
		
		for b in range(2,x0.shape[1]):
			if do_flip[b]==1:
				for t in range(x0.shape[0]):
					x0[t,b] = np.flip(x0[t,b],axis=flip_axes[b])
					target[t,b] = np.flip(target[t,b],axis=flip_axes[b])

				
		#--- Reshape back to [(T-1)*batch,size,size,channels]
		self.x0 = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
		self.target = target.reshape((-1,target.shape[2],target.shape[3],target.shape[4]))
		self.x0_true = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
		self.x0 = tf.convert_to_tensor(self.x0)