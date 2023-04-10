from NCA.trainer.NCA_trainer import *


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

		train_summary_writer = tf.summary.create_file_writer(self.LOG_DIR)
		
		#--- Log the graph structure of the NCA
		tf.summary.trace_on(graph=True,profiler=True)
		y = self.NCA_model.perceive(self.x0)
		with train_summary_writer.as_default():
			tf.summary.trace_export(name="NCA Perception",step=0,profiler_outdir=self.LOG_DIR)
		
		tf.summary.trace_on(graph=True,profiler=True)
		x = self.NCA_model(self.x0)
		with train_summary_writer.as_default():
			tf.summary.trace_export(name="NCA full step",step=0,profiler_outdir=self.LOG_DIR)
		
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


