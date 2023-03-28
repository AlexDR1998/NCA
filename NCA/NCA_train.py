import numpy as np
#import silence_tensorflow.auto # shuts up tensorflow's spammy warning messages
import tensorflow as tf
#import scipy as sp
from tqdm import tqdm
import datetime
from PDE_solver import PDE_solver
from NCA_train_utils import *
from time import time



class NCA_Trainer(object):
	"""
		Class to train NCA to data, as well as logging the process via tensorboard.
		Very general, should work on any sequence of images (of same size)
	
	"""

	def __init__(self,NCA_model,data,N_BATCHES,model_filename=None,RGB_mode="RGBA",directory="models/"):
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
		self.target = target.reshape((-1,target.shape[2],target.shape[3],target.shape[4]))
		self.data = data
		self.x0_true = np.copy(self.x0)
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
		train_log_dir = "logs/gradient_tape/"+self.model_filename+"/train"
		train_summary_writer = tf.summary.create_file_writer(train_log_dir)
		
		#--- Log the graph structure of the NCA
		tf.summary.trace_on(graph=True,profiler=True)
		y = self.NCA_model.perceive(self.x0)
		with train_summary_writer.as_default():
			tf.summary.trace_export(name="NCA Perception",step=0,profiler_outdir=train_log_dir)
		
		tf.summary.trace_on(graph=True,profiler=True)
		x = self.NCA_model(self.x0)
		with train_summary_writer.as_default():
			tf.summary.trace_export(name="NCA full step",step=0,profiler_outdir=train_log_dir)
		
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
			if i%10==0:
				"""
				tf.summary.image('12h GSC - Brachyury T - SOX2 --- Lamina B',
								 np.concatenate((x[:N_BATCHES,...,:3],np.repeat(x[:N_BATCHES,...,3:4],3,axis=-1)),axis=0),
								 step=i)
				tf.summary.image('24h GSC - Brachyury T - SOX2 --- Lamina B',
								 np.concatenate((x[N_BATCHES:2*N_BATCHES,...,:3],np.repeat(x[N_BATCHES:2*N_BATCHES,...,3:4],3,axis=-1)),axis=0),
								 step=i)
				tf.summary.image('36h GSC - Brachyury T - SOX2 --- Lamina B',
								 np.concatenate((x[2*N_BATCHES:3*N_BATCHES,...,:3],np.repeat(x[2*N_BATCHES:3*N_BATCHES,...,3:4],3,axis=-1)),axis=0),
								 step=i)
				tf.summary.image('48h GSC - Brachyury T - SOX2 --- Lamina B',
								 np.concatenate((x[3*N_BATCHES:4*N_BATCHES,...,:3],np.repeat(x[3*N_BATCHES:4*N_BATCHES,...,3:4],3,axis=-1)),axis=0),
								 step=i)
				"""
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
					plt.imshow(model_params[0][0,0])
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
					
					try:
						hidden_channels = grids[i,...,self.OBS_CHANNELS:]
						hidden_channels_reshaped = hidden_channels[...,0]
						for k in range(1,hidden_channels.shape[-1]):
							hidden_channels_reshaped = tf.concat([hidden_channels_reshaped,hidden_channels[...,k]],axis=1)
						tf.summary.image('Trained NCA hidden dynamics (tanh limited) at step '+str(self.time_of_best_model),
										 hidden_channels_reshaped,step=i)
					except:
						pass
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
				if self.NCA_model.ADHESION_MASK is not None:
					x = _mask*self.NCA_model.ADHESION_MASK + (1-_mask)*x
				
				#--- Intermediate state regulariser, to penalise any pixels being outwith [0,1]
				#above_1 = tf.math.maximum(tf.reduce_max(x),1) - 1
				#below_0 = tf.math.maximum(tf.reduce_max(-x),0)
				#reg_log.append(tf.math.maximum(above_1,below_0))
				reg_log.append(tf.reduce_sum(tf.nn.relu(-x)+tf.nn.relu(x-1)))
				
			#print(x.shape)
			losses = self.LOSS_WEIGHTS*loss_func(x) 
			reg_loss = tf.cast(tf.reduce_mean(reg_log),tf.float32)
			mean_loss = tf.reduce_mean(losses) + REG_COEFF*reg_loss
					
		if update_gradients:
			grads = g.gradient(mean_loss,self.NCA_model.dense_model.weights)
			if NORM_GRADS:
				grads = [g/(tf.norm(g)+1e-8) for g in grads]
			self.trainer.apply_gradients(zip(grads, self.NCA_model.dense_model.weights))
		return x, mean_loss,losses

	def train_sequence(self,TRAIN_ITERS,iter_n,UPDATE_RATE=1,REG_COEFF=0,LOSS_FUNC=None,LEARN_RATE=2e-3,OPTIMIZER="Adagrad",NORM_GRADS=True):
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
			
			Returns
			-------
			None
		"""
		
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
		
		
		#--- Setup adhesion and decay masks

		if self.NCA_model.ADHESION_MASK is not None:
			_mask = np.zeros((self.x0.shape),dtype="float32")
			_mask[...,self.OBS_CHANNELS]=1
			print(_mask.shape)
			print(self.NCA_model.ADHESION_MASK.shape)

		

		if LOSS_WEIGHTS is None:
			self.LOSS_WEIGHTS = tf.ones(shape=self.x0.shape[0])
		else:
			self.LOSS_WEIGHTS = tf.repeat(LOSS_WEIGHTS,self.N_BATCHES)
		
		#--- Do training loop
		
		best_mean_loss = 100000
		self.time_of_best_model = 0
		N_BATCHES = self.N_BATCHES
		print(N_BATCHES)
		print(self.x0.shape)
		print(self.x0_true.shape)
		for i in tqdm(range(TRAIN_ITERS)):
			R = np.random.uniform()<UPDATE_RATE
			x,mean_loss,losses = self.train_step(self.x0,iter_n,REG_COEFF,update_gradients=R,LOSS_FUNC=LOSS_FUNC,NORM_GRADS=NORM_GRADS)#,i%4==0)
			

			self.x0[N_BATCHES:] = x[:-N_BATCHES] # updates each initial condition to be final condition of previous chunk of timesteps
			

			if N_BATCHES>1:
				self.x0[::N_BATCHES][1:] = self.x0_true[::N_BATCHES][1:] # update one batch to contain the true initial conditions
			


			#--- Save model each time it is better than previous best model (and after 10% of training iterations are done)
			if (mean_loss<best_mean_loss) and (i>TRAIN_ITERS//10):
				if self.model_filename is not None:
					self.NCA_model.save_wrapper(self.model_filename,self.directory)
					tqdm.write("--- Model saved at "+str(i)+" epochs ---")
				
				self.BEST_TRAJECTORY = self.NCA_model.run(self.x0,iter_n*self.T*2,N_BATCHES=self.N_BATCHES).numpy()
				self.time_of_best_model = i
				best_mean_loss = mean_loss

			loss = np.hstack((mean_loss,losses))
			
			
			#--- Write to log
			self.tb_training_loop_log_sequence(loss,x,i)
		print("-------- Training complete ---------")
		#--- Write resulting best animation to tensorboard			
		self.tb_write_result(iter_n)

		#--- If a filename is provided, save the trained NCA model.
		#if model_filename is not None:
		#	ca.save_wrapper(model_filename)

	def data_pad_augment(self,AUGMENTATION,WIDTH):
		"""
			Augments training data by padding with extra zeros and randomly translating I.C - target pairs
			Should result in NCA that ignores simulation boundary

			Parameters
			----------
			AUGMENTATION : int
				Number of copies of data to augment - multiplies number of batches
			WIDTH : int
				How wide to pad data with zeros

		"""
		
		#--- Reshape x0 and target to be [T-1,batch,size,size,channels]
		x0 = self.x0.reshape((self.T-1,self.N_BATCHES,self.x0.shape[1],self.x0.shape[2],-1))
		target = self.target.reshape((self.T-1,self.N_BATCHES,self.target.shape[1],self.target.shape[2],-1))
		
		#--- Pad along [:,:,size,size,:] dimensions
		padwidth = ((0,0),(0,0),(WIDTH,WIDTH),(WIDTH,WIDTH),(0,0))
		x0 = np.pad(x0,padwidth)
		target = np.pad(target,padwidth)

		#--- Duplicate by AUGMENTATION along batches axis
		x0 = np.repeat(x0,AUGMENTATION,axis=1)
		target = np.repeat(target,AUGMENTATION,axis=1)

		#--- Randomly offset each image
		shifts = np.random.randint(low=-WIDTH,high=WIDTH,size=(x0.shape[1],2))
		for t in range(x0.shape[0]):
			for b in range(x0.shape[1]):
				x0[t,b] = np.roll(x0[t,b],shift=shifts[b],axis=(0,1))
				target[t,b] = np.roll(target[t,b],shift=shifts[b],axis=(0,1))

		#--- Reshape back to [(T-1)*batch,size,size,channels]
		self.x0 = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
		self.target = target.reshape((-1,target.shape[2],target.shape[3],target.shape[4]))
		self.x0_true = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
		self.N_BATCHES*=AUGMENTATION

	def data_noise_augment(self,AMOUNT=0.001):
		"""
			Augments training data by adding noise to the I.C. scaled by AMOUNT

			Parameters
			----------
			AMOUNT : float [0,1]
				Interpolates between initial data at 0 and pure noise at 1


		"""
		noise = np.random.uniform(size=self.x0.shape)
		x0_noisy = AMOUNT*noise + (1-AMOUNT)*self.x0
		self.x0 = x0_noisy




class NCA_Trainer_stem_cells(NCA_Trainer):
	"""
		Subclass of NCA_Trainer that specifically handles quirks of stem cell data.
		Handles the discrepancy of timesteps 0h, 24h, 36h etc...,
		Correctly labels proteins in data for tensorboard logging
	"""
	def __init__(self,NCA_model,data,N_BATCHES,model_filename=None,directory="models/"):

		super().__init__(NCA_model,data,N_BATCHES,model_filename,"RGB-A",directory)
		
		x0 = np.copy(self.data[:])
		x0[1:] = self.data[:-1] # Including 1 extra time slice to account for hidden 12h time
		target = self.data[1:]
		
		z0 = np.zeros((x0.shape[0],x0.shape[1],x0.shape[2],x0.shape[3],self.N_CHANNELS-x0.shape[4]))
		x0 = np.concatenate((x0,z0),axis=-1).astype("float32")
		
		self.x0 = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
		
		self.target = target.reshape((-1,target.shape[2],target.shape[3],target.shape[4]))
		self.x0_true = np.copy(self.x0)
		self.setup_tb_log_sequence()

	'''
	def loss_func(self,x):
		"""
			Loss function for training to minimise. Averages over batches, returns error per time slice (sequence)
			Handles the unobserved 12h timestep in the data!

			Parameters
			----------
			x : float32 tensor [T*N_BATCHES,size,size,N_CHANNELS]
				Current state of NCA grids, in sequence training mode
			
			Returns
			-------
			loss : float32 tensor [T]
				Array of errors at each timestep (24h, 36h, 48h)
		"""

		eu = tf.math.reduce_euclidean_norm((x[self.N_BATCHES:,...,:4]-self.target),[-2,-3,-1])
		# self.N_BATCHES: removes the 'hidden' 12h state
		return tf.reduce_mean(tf.reshape(eu,(-1,self.N_BATCHES)),-1)
	'''
	
	def setup_tb_log_sequence(self):
		"""
			Initialises the tensorboard logging of training.
			Writes some initial information. Very similar to setup_tb_log_single, but designed for sequence modelling

		"""


		#if model_filename is None:
		#	current_time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		#	train_log_dir = "logs/gradient_tape/"+current_time+"/train"
		#else:
		train_log_dir = "logs/gradient_tape/"+self.model_filename+"/train"
		train_summary_writer = tf.summary.create_file_writer(train_log_dir)
		
		#--- Log the graph structure of the NCA
		tf.summary.trace_on(graph=True,profiler=True)
		y = self.NCA_model.perceive(self.x0)
		with train_summary_writer.as_default():
			tf.summary.trace_export(name="NCA Perception",step=0,profiler_outdir=train_log_dir)
		
		tf.summary.trace_on(graph=True,profiler=True)
		x = self.NCA_model(self.x0)
		with train_summary_writer.as_default():
			tf.summary.trace_export(name="NCA full step",step=0,profiler_outdir=train_log_dir)
		
		#--- Log the target image and initial condtions
		with train_summary_writer.as_default():
			tf.summary.image('Sequence GSC - Brachyury T - SOX2',self.data[:,0,...,:3],step=0,max_outputs=self.data.shape[0])
			
			
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
			#for j in range(len(loss)):
			tf.summary.scalar('Mean Loss',loss[0],step=i)
			tf.summary.scalar('Loss 1',loss[1],step=i)
			tf.summary.scalar('Loss 2',loss[2],step=i)
			tf.summary.scalar('Loss 3',loss[3],step=i)
			tf.summary.histogram('Loss ',loss,step=i)
			if i%10==0:
				tf.summary.image('12h GSC - Brachyury T - SOX2 --- Lamina B',
								 np.concatenate((x[:N_BATCHES,...,:3],np.repeat(x[:N_BATCHES,...,3:4],3,axis=-1)),axis=0),
								 step=i)
				tf.summary.image('24h GSC - Brachyury T - SOX2 --- Lamina B',
								 np.concatenate((x[N_BATCHES:2*N_BATCHES,...,:3],np.repeat(x[N_BATCHES:2*N_BATCHES,...,3:4],3,axis=-1)),axis=0),
								 step=i)
				tf.summary.image('36h GSC - Brachyury T - SOX2 --- Lamina B',
								 np.concatenate((x[2*N_BATCHES:3*N_BATCHES,...,:3],np.repeat(x[2*N_BATCHES:3*N_BATCHES,...,3:4],3,axis=-1)),axis=0),
								 step=i)
				tf.summary.image('48h GSC - Brachyury T - SOX2 --- Lamina B',
								 np.concatenate((x[3*N_BATCHES:4*N_BATCHES,...,:3],np.repeat(x[3*N_BATCHES:4*N_BATCHES,...,3:4],3,axis=-1)),axis=0),
								 step=i)
				#print(ca.dense_model.layers[0].get_weights()[0])
				model_params_0 = self.NCA_model.dense_model.layers[0].get_weights()
				model_params_1 = self.NCA_model.dense_model.layers[1].get_weights()
				#model_params_2 = ca.dense_model.layers[2].get_weights()
				tf.summary.histogram('Layer 0 weights',model_params_0[0],step=i)
				tf.summary.histogram('Layer 1 weights',model_params_1[0],step=i)
				#tf.summary.histogram('Layer 2 weights',model_params_2[0],step=i)
				#tf.summary.histogram('Layer 0 biases',model_params_0[1],step=i)
				tf.summary.histogram('Layer 1 biases',model_params_1[1],step=i)
				#tf.summary.histogram('Layer 2 biases',model_params_2[1],step=i)

	def tb_write_result(self,iter_n=24):
		"""
			Log final trained behaviour of NCA model to tensorboard
		"""

		with self.train_summary_writer.as_default():
			grids = self.NCA_model.run(self.x0,8*iter_n,1)
			grids = np.array(grids)
			grids[...,4:] = (1+np.tanh(grids[...,4:]))/2.0
			for i in range(8*iter_n):
				tf.summary.image('Trained NCA dynamics GSC - Brachyury T - SOX2 --- Lamina B',
								 np.concatenate((grids[i,:1,...,:3],
								 				 np.repeat(grids[i,:1,...,3:4],3,axis=-1)),
								 				axis=1),
								 step=i)
				if (self.N_CHANNELS>=7) and (self.N_CHANNELS <10):
					tf.summary.image('Trained NCA hidden dynamics (tanh limited)',
									 grids[i,:1,...,4:7],step=i,
								 	max_outputs=4)
				elif ((self.N_CHANNELS>=10) and (self.N_CHANNELS <13)):
					tf.summary.image('Trained NCA hidden dynamics (tanh limited)',
									 np.concatenate((grids[i,:1,...,4:7],
													 grids[i,:1,...,7:10]),axis=1),
									 step=i,
								 	 max_outputs=4)
				elif ((self.N_CHANNELS>=13) and (self.N_CHANNELS <16)):
					tf.summary.image('Trained NCA hidden dynamics (tanh limited)',
									 np.concatenate((grids[i,:1,...,4:7],
													 grids[i,:1,...,7:10],
									 				 grids[i,:1,...,10:13]),axis=1),
									 step=i,
								 	 max_outputs=4)
				elif (self.N_CHANNELS>=16):
					tf.summary.image('Trained NCA hidden dynamics (tanh limited)',
									 np.concatenate((grids[i,:1,...,4:7],
													 grids[i,:1,...,7:10],
									 				 grids[i,:1,...,10:13],
									 				 grids[i,:1,...,13:16]),axis=1),
									 step=i,
								 	 max_outputs=4)				
								
	def train_sequence(self,TRAIN_ITERS,iter_n,UPDATE_RATE=1,REG_COEFF=0,LOSS_FUNC=None,LEARN_RATE=2e-3,OPTIMIZER="Adagrad",NORM_GRADS=True,LOSS_WEIGHTS=None):
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
			
			Returns
			-------
			None
		"""
		
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
		
		
		#--- Setup adhesion and decay masks

		if self.NCA_model.ADHESION_MASK is not None:
			_mask = np.zeros((self.x0.shape),dtype="float32")
			_mask[...,self.OBS_CHANNELS]=1
			print(_mask.shape)
			print(self.NCA_model.ADHESION_MASK.shape)

		

		if LOSS_WEIGHTS is None:
			self.LOSS_WEIGHTS = tf.ones(shape=self.x0.shape[0])
		else:
			self.LOSS_WEIGHTS = tf.repeat(LOSS_WEIGHTS,self.N_BATCHES)
		
		#--- Do training loop
		
		best_mean_loss = 100000
		self.time_of_best_model = 0
		N_BATCHES = self.N_BATCHES
		print(N_BATCHES)
		print(self.x0.shape)
		print(self.x0_true.shape)
		for i in tqdm(range(TRAIN_ITERS)):
			R = np.random.uniform()<UPDATE_RATE
			x,mean_loss,losses = self.train_step(self.x0,iter_n,REG_COEFF,update_gradients=R,LOSS_FUNC=LOSS_FUNC,NORM_GRADS=NORM_GRADS)#,i%4==0)
			

			self.x0[N_BATCHES:] = x[:-N_BATCHES] # updates each initial condition to be final condition of previous chunk of timesteps
			

			if N_BATCHES>1:
				self.x0[::N_BATCHES][2:] = self.x0_true[::N_BATCHES][2:] # update one batch to contain the true initial conditions
			


			#--- Save model each time it is better than previous best model (and after 10% of training iterations are done)
			if (mean_loss<best_mean_loss) and (i>TRAIN_ITERS//10):
				if self.model_filename is not None:
					self.NCA_model.save_wrapper(self.model_filename,self.directory)
					tqdm.write("--- Model saved at "+str(i)+" epochs ---")
				
				self.BEST_TRAJECTORY = self.NCA_model.run(self.x0,iter_n*self.T*2,N_BATCHES=self.N_BATCHES).numpy()
				self.time_of_best_model = i
				best_mean_loss = mean_loss

			loss = np.hstack((mean_loss,losses))
			
			
			#--- Write to log
			self.tb_training_loop_log_sequence(loss,x,i)
		print("-------- Training complete ---------")
		#--- Write resulting best animation to tensorboard			
		self.tb_write_result(iter_n)
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
				if self.NCA_model.ADHESION_MASK is not None:
					x = _mask*self.NCA_model.ADHESION_MASK + (1-_mask)*x
				
				#--- Intermediate state regulariser, to penalise any pixels being outwith [0,1]
				#above_1 = tf.math.maximum(tf.reduce_max(x),1) - 1
				#below_0 = tf.math.maximum(tf.reduce_max(-x),0)
				#reg_log.append(tf.math.maximum(above_1,below_0))
				reg_log.append(tf.reduce_sum(tf.nn.relu(-x)+tf.nn.relu(x-1)))
				
			#print(x.shape)
			losses = loss_func(x[self.N_BATCHES:]) 
			reg_loss = tf.cast(tf.reduce_mean(reg_log),tf.float32)
			mean_loss = tf.reduce_mean(losses) + REG_COEFF*reg_loss
					
		if update_gradients:
			grads = g.gradient(mean_loss,self.NCA_model.dense_model.weights)
			if NORM_GRADS:
				grads = [g/(tf.norm(g)+1e-8) for g in grads]
			self.trainer.apply_gradients(zip(grads, self.NCA_model.dense_model.weights))
		return x, mean_loss,losses




class NCA_IC_Trainer(NCA_Trainer):
	"""
		Class to "train" initial conditions of a pre-trained NCA to optimise
		some function of trajectory
	"""

	def __init__(self,NCA_model,data,N_BATCHES,log_filename,mode="max"):
		super().__init__(NCA_model,data,N_BATCHES,log_filename)

		#--- Add a tiny bit of noise to x0
		self.x0 = np.clip(self.x0 +0.001*np.random.normal(size=self.x0.shape),0,1)
		

		#--- Remove hidden channels, only optimise the observable channels
		self.x0 = tf.Variable(tf.convert_to_tensor(self.x0[...,:4],dtype=tf.float32))
		self.x0_true = tf.convert_to_tensor(self.x0_true[...,:4],dtype=tf.float32)
		self.target = tf.convert_to_tensor(self.target[...,:4],dtype=tf.float32)

		#--- Append this to targets and x0 after each tweak of x0
		self.z0 = tf.zeros((self.x0.shape[0],self.x0.shape[1],self.x0.shape[2],self.N_CHANNELS-4))


		self.mode = mode
	def tb_training_loop_log_sequence(self,loss,x0,x,i):
		"""
			Helper function to format some data logging during the training loop

			Parameters
			----------
			
			loss : float32 tensor [T]
				Array of errors at each timestep (24h, 36h, 48h)
			x0 : float32 tensor [(T-1)*N_BATCHES,size,size,N_CHANNELS]

			x : float32 tensor [(T-1)*N_BATCHES,size,size,N_CHANNELS]
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
							tf.summary.image('Initial states, step '+str(j)+' RGB',
										 	x0[(j-1)*N_BATCHES:(j)*N_BATCHES,...,:3],
										 	step=i)							

							tf.summary.image('Target states, step '+str(j)+' RGB',
										 	x[(j-1)*N_BATCHES:(j)*N_BATCHES,...,:3],
										 	step=i)
						elif self.RGB_mode=="RGBA":
							tf.summary.image('Initial states, step '+str(j)+' RGBA',
										 	x0[(j-1)*N_BATCHES:(j)*N_BATCHES,...,:4],
										 	step=i)							

							tf.summary.image('Target states, step '+str(j)+' RGBA',
										 	x[(j-1)*N_BATCHES:(j)*N_BATCHES,...,:4],
										 	step=i)
			#tf.summary.scalar('Loss 1',loss[1],step=i)
			#tf.summary.scalar('Loss 2',loss[2],step=i)
			#tf.summary.scalar('Loss 3',loss[3],step=i)
			tf.summary.histogram('Loss ',loss,step=i)
	@tf.function		
	def loss_func_max(self,x,x0):
		"""
			Loss function for training to minimise. Averages over batches, returns error per time slice (sequence)
			
			Maximal initial condition perturbation, minimal final condition perturbation
			Regularise to penalise x0 outwith range [0,1]
			Parameters
			----------
			x : float32 tensor [(T-1)*N_BATCHES,size,size,N_CHANNELS]
				Current state of NCA grids, in sequence training mode
			
			Returns
			-------
			loss : float32 tensor [T]
				Array of errors at each timestep
		"""
		epsilon=0.0001
		target_err = tf.math.reduce_euclidean_norm((x[...,:4]-self.target),[-2,-3,-1])
		initial_err= tf.math.reduce_euclidean_norm((x0[...,:4]-self.x0_true),[-2,-3,-1])
		#initial_reg= tf.math.reduce_euclidean_norm(x0[...,:4])
		mask_reg = tf.reduce_sum(((tf.cast(tf.math.greater(epsilon,self.x0),self.x0.dtype)[...,:4])*x0[...,:4])**2)
		initial_reg = tf.reduce_sum(tf.nn.relu(-x0[...,:4])+tf.nn.relu(x0[...,:4]-1))
		return tf.reduce_mean(tf.reshape(target_err**2-initial_err**2,(-1,self.N_BATCHES)),-1) + initial_reg**2 + mask_reg
		
	@tf.function
	def loss_func_min(self,x,x0):
		"""
			Loss function for training to minimise. Averages over batches, returns error per time slice (sequence)
			
			Minimal initial condition perturbation, Maximal final condition perturbation

			Parameters
			----------
			x : float32 tensor [(T-1)*N_BATCHES,size,size,N_CHANNELS]
				Current state of NCA grids, in sequence training mode
			
			Returns
			-------
			loss : float32 tensor [T]
				Array of errors at each timestep
		"""
		epsilon = 0.0001
		target_err = tf.math.reduce_euclidean_norm((x[...,:4]-self.target),[-2,-3,-1])
		initial_err= tf.math.reduce_euclidean_norm((x0[...,:4]-self.x0_true),[-2,-3,-1])
		#initial_reg= tf.math.reduce_euclidean_norm(x0[...,:4])
		initial_reg = tf.reduce_sum(tf.nn.relu(-x0[...,:4])+tf.nn.relu(x0[...,:4]-1))
		mask_reg = tf.reduce_sum(((tf.cast(tf.math.greater(epsilon,self.x0),self.x0.dtype)[...,:4])*x0[...,:4])**2)
		return tf.reduce_mean(tf.reshape(initial_err**2-target_err**2,(-1,self.N_BATCHES)),-1) + initial_reg**2 + mask_reg
	
	def train_step(self,x0,iter_n,update_gradients=True):
		"""
			Training step. Runs NCA model once, calculates loss gradients and tweaks initial conditions
		"""

		
		with tf.GradientTape() as g:
			g.watch(x0)
			x0_full = tf.concat((x0,self.z0),axis=-1)
			x = tf.identity(x0_full)
			for i in range(iter_n):
				x = self.NCA_model(x)
				if self.NCA_model.ADHESION_MASK is not None:
					x = _mask*self.NCA_model.ADHESION_MASK + (1-_mask)*x
				
				
				
				
			#print(x.shape)
			if self.mode=="max":
				losses = self.loss_func_max(x,x0) 
			else:
				losses = self.loss_func_min(x,x0)
			mean_loss = tf.reduce_mean(losses)
					
		
		grads = g.gradient(mean_loss,x0)
		grads = [g/(tf.norm(g)+1e-8) for g in grads]
		self.trainer.apply_gradients(zip([grads], [x0]))
		return x0,x,mean_loss,losses

	def train_sequence(self,TRAIN_ITERS,iter_n,REG_COEFF=0):
		"""
			Trains the ca to recreate the given image sequence. Error is calculated by comparing ca grid to each image after iter_n/T steps 
			
			Parameters
			----------
			
			TRAIN_ITERS : int
				how many iterations of training
			iter_n : int
				number of NCA update steps to run ca
			REG_COEFF : float32
				Strength of intermediate state regulariser - penalises any pixels outwith [0,1]
			
			Returns
			-------
			None
		"""
		#--- Setup training algorithm

		lr = 2e-3
		#lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([TRAIN_ITERS//2], [lr, lr*0.1])
		lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(lr, TRAIN_ITERS, 0.96)
		#self.trainer = tf.keras.optimizers.Adam(lr_sched)
		self.trainer = tf.keras.optimizers.Nadam(lr)
		#trainer = tf.keras.optimizers.RMSprop(lr_sched)
		
		#--- Setup adhesion and decay masks

		if self.NCA_model.ADHESION_MASK is not None:
			_mask = np.zeros((self.x0.shape),dtype="float32")
			_mask[...,4]=1
			print(_mask.shape)
			print(self.NCA_model.ADHESION_MASK.shape)

		
		
		#--- Do training loop
		
		best_mean_loss = 100000
		N_BATCHES = self.N_BATCHES
		print(N_BATCHES)
		print(self.x0.shape)
		print(self.x0_true.shape)
		for i in tqdm(range(TRAIN_ITERS)):
			
			self.x0,x,mean_loss,losses = self.train_step(self.x0,iter_n)#,i%4==0)
			
			loss = np.hstack((mean_loss,losses))
					
			#--- Write to log
			self.tb_training_loop_log_sequence(loss,self.x0,x,i)
		print("-------- Training complete ---------")
		#--- Write resulting animation to tensorboard			
		#self.tb_write_result(iter_n)

		


class NCA_PDE_Trainer(NCA_Trainer):
	"""
		Class to train a NCA to the output of a PDE simulation
	"""

	def __init__(self,NCA_model,x0,F,N_BATCHES,T,step_mul=1,model_filename=None,directory="models/"):
		"""
			Initialiser method

			Parameters
			----------
			NCA_model : object callable - float32 tensor [N_BATCHES,size,size,N_CHANNELS],float32,float32 -> float32 tensor [N_BATCHES,size,size,N_CHANNELS]
				the NCA object to train
			x0 : float32 tensor [N_BATCHES,size,size,4]
				Initial conditions for PDE model to simulate from
			F : callable - (float32 tensor [N_BATCHES,size,size,N_CHANNELS])**4 -> float32 tensor [N_BATCHES,size,size,N_CHANNELS]
				RHS of PDE in the form dX/dt = F(X,Xdx,Xdy,Xdd)
			N_BATCHES : int
				size of training batch

			T : int
				How many steps to run the PDE model for
			step_mul : int
				How many PDE steps per NCA step
			model_filename : str
				name of directories to save tensorboard log and model parameters to.
				log at :	'logs/gradient_tape/model_filename/train'
				model at : 	'models/model_filename'
				if None, sets model_filename to current time
			RGB_mode : string
				Expects "RGBA" "RGB" or "RGB-A"
				Defines how to log image channels to tensorboard
				RGBA : 4 channel RGBA image (i.e. PNG with transparancy)
				RGB : 3 channel RGB image
				RGB-A : 3+1 channel - a 3 channel RGB image alongside a black and white image representing the alpha channel
			directory : str
				Name of directory where all models get stored, defaults to 'models/'
		"""


		#self.N_CHANNELS = NCA_model.N_CHANNELS
	
		self.T_steps = T

		PDE_model = PDE_solver(F,
							   NCA_model.OBS_CHANNELS,
							   N_BATCHES,
							   size=[x0.shape[1],x0.shape[2]],
							   PADDING=NCA_model.PADDING)
		data = PDE_model.run(iterations=T*step_mul,step_size=1.0,initial_condition=x0)[::step_mul]
		#--- Renormalise data to be between 0 and 1
		data_max = np.max(data)
		data_min = np.min(data)
		data = (data-data_min)/(data_max-data_min)

		super().__init__(NCA_model,data,N_BATCHES,model_filename,directory=directory)
		
		assert x0.shape[-1]==self.OBS_CHANNELS, "Observable channels of NCA does not match data dimensions"

		"""
		self.T = 1
		"""
		#self.data = (self.data-data_min)/(data_max-data_min)
		#self.data = self.data.reshape((-1,N_BATCHES,self.data.shape[2],self.data.shape[3],self.data.shape[4]))
		

		#--- Modifications to alternative batch training trick
		"""
		self.x0 = self.x0.reshape((-1,N_BATCHES,self.x0.shape[1],self.x0.shape[2],self.x0.shape[3]))
		self.x0 = tf.convert_to_tensor(self.x0[0])
		"""

		#self.x0_true = tf.convert_to_tensor(self.data[0])
		#self.data = tf.convert_to_tensor(self.data)
		
		print("Data shape: "+str(self.data.shape))
		print("X0 shape: "+str(self.x0.shape))
		print("Target shape: "+str(self.target.shape))
		
		
		#print(self.data.shape)
		
		self.setup_tb_log_sequence()



	def setup_tb_log_sequence(self):
		"""
			Initialises the tensorboard logging of training.
			Writes some initial information. Very similar to setup_tb_log_single, but designed for sequence modelling

		"""


		#if model_filename is None:
		#	current_time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		#	train_log_dir = "logs/gradient_tape/"+current_time+"/train"
		#else:
		train_log_dir = "logs/gradient_tape/"+self.model_filename+"/train"
		train_summary_writer = tf.summary.create_file_writer(train_log_dir)
		
		#--- Log the graph structure of the NCA
		"""
		tf.summary.trace_on(graph=True,profiler=True)
		y = self.NCA_model.perceive(self.x0)
		with train_summary_writer.as_default():
			tf.summary.trace_export(name="NCA Perception",step=0,profiler_outdir=train_log_dir)
		
		tf.summary.trace_on(graph=True,profiler=True)
		x = self.NCA_model(self.x0)
		with train_summary_writer.as_default():
			tf.summary.trace_export(name="NCA full step",step=0,profiler_outdir=train_log_dir)
		"""
		#--- Log the target image and initial condtions
		

		with train_summary_writer.as_default():
			grids = self.data
			#for b in range(self.N_BATCHES):
			for i in range(self.T_steps):
				if self.RGB_mode=="RGB":
					tf.summary.image('PDE dynamics RGB',
									 grids[i,...,:self.OBS_CHANNELS],
									 step=i)
				elif self.RGB_mode=="RGBA":
					tf.summary.image('PDE dynamics RGBA',
									 grids[i,...,:self.OBS_CHANNELS],
									 step=i)
				elif self.RGB_mode=="RGB-A":
					tf.summary.image('PDE dynamics RGB --- Alpha',
									 np.concatenate((grids[i,...,:self.OBS_CHANNELS-1],
									 				 np.repeat(grids[i,...,self.OBS_CHANNELS-1:self.OBS_CHANNELS],3,axis=-1)),
									 				axis=1),
									 step=i)
				
				#No hidden dynamics of PDE
				"""
				b=0
				if self.N_CHANNELS>=self.OBS_CHANNELS+3:
					grids_hidden = grids[...,self.OBS_CHANNELS:]
					hidden_channels = grids_hidden[i,b:(b+1),...,:3]
					for j in range((self.N_CHANNELS-self.OBS_CHANNELS)//3-1):
						
						hidden_channels = np.concatenate((hidden_channels,
														  grids_hidden[i,b:(b+1),...,((j+1)*3):((j+2)*3)]))
					
					tf.summary.image('PDE hidden dynamics',
									 hidden_channels,step=i,max_outputs=(self.N_CHANNELS-1)//3-1)
				"""
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
			tf.summary.scalar('Mean Loss',loss[0],step=i)
			tf.summary.histogram('Loss ',loss,step=i)
			if i%10==0:
				#print(ca.dense_model.layers[0].get_weights()[0])
				for n in range(self.NCA_model.N_layers):
					model_params = self.NCA_model.dense_model.layers[n].get_weights()
					tf.summary.histogram('Layer '+str(n)+' weights',model_params[0],step=i)
					try:
						tf.summary.histogram('Layer '+str(n)+' biases',model_params[1],step=i)
					except Exception as e:
						pass
				weight_matrix_image = []
				for n in range(self.NCA_model.N_layers):
					model_params = self.NCA_model.dense_model.layers[n].get_weights()
					tf.summary.histogram('Layer '+str(n)+' weights',model_params[0],step=i)
					
					figure = plt.figure(figsize=(5,5))
					plt.imshow(model_params[0][0,0])
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

	def train_step(self,
				   x,
				   iter_n,
				   REG_COEFF=0,
				   update_gradients=True,
				   LOSS_FUNC=None,
				   BATCH_SIZE=64,
				   TRAIN_MODE="full",
				   NORM_GRADS=True):
		"""
			Training step. Runs NCA model once, calculates loss gradients and tweaks model

			Differs from superclass by looping over batches to avoid OOM errors
			
			Parameters
			----------
			x : float32 tensor [N_BATCHES,X,Y,N_CHANNELS]
				input states
			iter_n : int
				number of NCA update steps to run ca

			REG_COEFF : float32 optional
				Strength of intermediate state regulariser - penalises any pixels outwith [0,1]
			
			update_gradients : bool optional
				Controls whether model weights are updated at this step

			LOSS_FUNC : (float32 tensor [N_BATCHES,X,Y,N_CHANNELS])**2 -> float32 tensor [N_BATCHES] optional
				Alternative loss function
			BATCH_SIZE : int optional
				How many NCA steps to do in parallel in a gradient loop. Too big leads to OOM errors


		"""
		
		#--- If alternative loss function is provided, use that, otherwise use default
		if LOSS_FUNC is None:
			loss_func = self.loss_func
		else:
			loss_func = lambda x,x_true: LOSS_FUNC(x[...,:self.OBS_CHANNELS],x_true)
		
		
		x_a = np.array(x)
		t_a = np.array(self.target)

		#if TRAIN_MODE=="differential":
		#	print("======== Debug ========")
		#	print(t_a.shape)
			#print(np.diff(t_a,axis=-1).shape)

		losses_b = []
		BATCH_IND = np.arange(x.shape[0])
		np.random.shuffle(BATCH_IND)
		
		for b in range(x.shape[0]//BATCH_SIZE):
			#--- Slice randomly shuffled subsets of x

			X_b = x_a[BATCH_IND[b*BATCH_SIZE:(b+1)*BATCH_SIZE]]
			X_b = tf.convert_to_tensor(X_b)
			T_b = t_a[BATCH_IND[b*BATCH_SIZE:(b+1)*BATCH_SIZE]]
			T_b = tf.convert_to_tensor(T_b)

			with tf.GradientTape() as g:
				reg_log = []
				X_original = tf.identity(X_b)
				for i in range(iter_n):
					X_b = self.NCA_model(X_b)
					reg_log.append(tf.reduce_sum(tf.nn.relu(-X_b)+tf.nn.relu(X_b-1)))
					if self.NCA_model.ADHESION_MASK is not None:
						X_b = _mask*self.NCA_model.ADHESION_MASK + (1-_mask)*X_b
					
				if TRAIN_MODE=="differential":
					losses = loss_func((X_b-X_original),T_b)	
				elif TRAIN_MODE=="full":
					losses = loss_func(X_b,T_b) 

				reg_loss = tf.cast(tf.reduce_mean(reg_log),tf.float32)
				mean_loss = tf.reduce_mean(losses) + REG_COEFF*reg_loss
				
			losses_b.append(losses)
			

			if update_gradients:
				grads = g.gradient(mean_loss,self.NCA_model.dense_model.weights)
				if NORM_GRADS:
					grads = [g/(tf.norm(g)+1e-8) for g in grads]
				self.trainer.apply_gradients(zip(grads, self.NCA_model.dense_model.weights))
			x_a[BATCH_IND[b*BATCH_SIZE:(b+1)*BATCH_SIZE]] = X_b
			
		
		losses = tf.concat(losses_b,axis=0)
		mean_loss = tf.reduce_mean(losses)
		return x_a, mean_loss,losses


	def train_sequence(self,
					   TRAIN_ITERS,
					   iter_n,
					   UPDATE_RATE=1,
					   REG_COEFF=0,
					   LOSS_FUNC=None,
					   LEARN_RATE=1e-3,
					   OPTIMIZER="Nadam",
					   BATCH_SIZE=64,
					   TRAIN_MODE="full",
					   NORM_GRADS=True):
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
			BATCH_SIZE : int optional
				How many NCA steps to do in parallel in a gradient loop. Too big leads to OOM errors
			TRAIN_MODE : string optional {"full","differential"}
				full : fit resultant NCA grid to PDE state
				differential: fit differential update of NCA and PDE

			Returns
			-------
			None
		"""
		
		
		#--- Setup training algorithm

		lr = LEARN_RATE
		#lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([TRAIN_ITERS//2], [lr, lr*0.1])
		lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(lr, TRAIN_ITERS, 0.96)
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
		if TRAIN_MODE not in ["full","differential"]:
			print("TRAIN_MODE should be 'full' or 'differential' ")
		
		#--- Setup adhesion and decay masks

		if self.NCA_model.ADHESION_MASK is not None:
			_mask = np.zeros((self.x0.shape),dtype="float32")
			_mask[...,self.OBS_CHANNELS]=1
			print(_mask.shape)
			print(self.NCA_model.ADHESION_MASK.shape)

		
		#--- If using differential training
		if TRAIN_MODE=="differential":
			print(self.target.shape)
			target = self.target.reshape((self.T-1,
										  self.N_BATCHES,
										  self.target.shape[1],
										  self.target.shape[2],
										  self.OBS_CHANNELS))
			target_diff = np.diff(target,axis=0)
			print(target_diff.shape)
			self.target = target_diff.reshape(((self.T-2)*self.N_BATCHES,
											  self.target.shape[1],
											  self.target.shape[2],
											  self.OBS_CHANNELS))

			
			x0 = self.x0
			x0 = x0.reshape((self.T-1),
							self.N_BATCHES,
							self.target.shape[1],
							self.target.shape[2],
							self.N_CHANNELS)
			
			x0 = x0[:-1]
			self.x0 = x0.reshape((self.T-2)*self.N_BATCHES,
								 self.target.shape[1],
								 self.target.shape[2],
								 self.N_CHANNELS)
			self.x0_true = self.x0
			
		#--- Do training loop
		
		best_mean_loss = 1e10
		previous_mean_loss = 1e10
		self.time_of_best_model = 0
		N_BATCHES = self.N_BATCHES
		print(N_BATCHES)
		print(self.x0.shape)
		print(self.x0_true.shape)
		start_time = time()
		for i in tqdm(range(TRAIN_ITERS)):
			R = np.random.uniform()<=UPDATE_RATE
			x,mean_loss,losses = self.train_step(self.x0,
												 iter_n,
												 REG_COEFF,
												 update_gradients=R,
												 LOSS_FUNC=LOSS_FUNC,
												 BATCH_SIZE=BATCH_SIZE,
												 TRAIN_MODE=TRAIN_MODE,
												 NORM_GRADS=NORM_GRADS)#,i%4==0)
			
			self.x0[N_BATCHES:] = x[:-N_BATCHES] # updates each initial condition to be final condition of previous chunk of timesteps			
			assert not tf.math.reduce_any(tf.math.is_nan(x)), "|-|-|-|-|-|-  X reached NaN  -|-|-|-|-|-|"
			
			if N_BATCHES>1:
				self.x0[::N_BATCHES][1:] = self.x0_true[::N_BATCHES][1:] # update one batch to contain the true initial conditions
			


			#--- Save model each time it is better than previous best model (and after 1% of training iterations are done)
			
			if (mean_loss<best_mean_loss) and (mean_loss < previous_mean_loss and (i>10)):
				if self.model_filename is not None:
					self.NCA_model.save_wrapper(self.model_filename,self.directory)
					tqdm.write("--- Model saved at "+str(i)+" epochs ---")
				
				self.BEST_TRAJECTORY = self.NCA_model.run(self.x0,iter_n*self.T*2,N_BATCHES=self.N_BATCHES).numpy()
				self.time_of_best_model = i
				best_mean_loss = mean_loss

			loss = np.hstack((mean_loss,losses))
			previous_mean_loss = mean_loss
			
			#--- Write to log
			self.tb_training_loop_log_sequence(loss,x,i)
		print("-------- Training complete ---------")
		print("Time taken (seconds): "+str(time()-start_time))
		#--- Write resulting best animation to tensorboard			
		self.tb_write_result(iter_n)

		#--- If a filename is provided, save the trained NCA model.
		#if model_filename is not None:
		#	ca.save_wrapper(model_filename)
	

	