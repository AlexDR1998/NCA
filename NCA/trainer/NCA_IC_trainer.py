from NCA.trainer.NCA_trainer import *




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

	