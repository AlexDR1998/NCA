from NCA.trainer.NCA_trainer import *
from NCA.PDE_solver import PDE_solver



	


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
		
		train_summary_writer = tf.summary.create_file_writer(self.LOG_DIR)
		
		#--- Log the graph structure of the NCA
		"""
		tf.summary.trace_on(graph=True,profiler=True)
		y = self.NCA_model.perceive(self.x0)
		with train_summary_writer.as_default():
			tf.summary.trace_export(name="NCA Perception",step=0,profiler_outdir=self.LOG_DIR)
		
		tf.summary.trace_on(graph=True,profiler=True)
		x = self.NCA_model(self.x0)
		with train_summary_writer.as_default():
			tf.summary.trace_export(name="NCA full step",step=0,profiler_outdir=self.LOG_DIR)
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
		#print(N_BATCHES)
		#print(self.x0.shape)
		#print(self.x0_true.shape)
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
			assert not tf.math.reduce_any(tf.math.is_nan(x)), "|-|-|-|-|-|-  X reached NaN  -|-|-|-|-|-|"
			self.x0 = self.x0.numpy()
			self.x0[N_BATCHES:] = x[:-N_BATCHES] # updates each initial condition to be final condition of previous chunk of timesteps			
			
			if N_BATCHES>1:
				self.x0[::N_BATCHES][1:] = self.x0_true[::N_BATCHES][1:] # update one batch to contain the true initial conditions
			self.x0 = tf.convert_to_tensor(self.x0)


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
	

	