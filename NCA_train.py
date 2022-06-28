import numpy as np
import silence_tensorflow.auto # shuts up tensorflow's spammy warning messages
import tensorflow as tf
#import scipy as sp
from tqdm import tqdm
import datetime

"""
	Bunch of methods to train NCA to data, as well as logging the process via tensorboard
	
"""





def setup_tb_log_single(ca,target,x0,model_filename=None):
	"""
		Initialises the tensorboard logging of training.
		Writes some initial information (initial grid condition, target, NCA computation graph)

		Parameters
		----------
		ca : object callable - float32 tensor [batches,size,size,N_CHANNELS],float32,float32 -> float32 tensor [batches,size,size,N_CHANNELS]
			the NCA object to train
		target : float32 tensor [batches,size,size,4]
			the target image to be grown by the NCA.
		x0 : float32 tensor [batches,size,size,k<=N_CHANNELS]
			the initial condition of NCA. If it has less channels than the NCA, pad with zeros. If none, is set to zeros with one 'seed' of 1s in the middle
		model_filename : str
			name of directories to save tensorboard log and model parameters to.
			log at :	'logs/gradient_tape/model_filename/train'
			model at : 	'models/model_filename'
			if None, doesn't save model but still saves log to 'logs/gradient_tape/*current_time*/train'

		Returns
		-------
		train_summary_writer : tf.summary.file_writer object
	"""


	if model_filename is None:
		current_time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		train_log_dir = "logs/gradient_tape/"+current_time+"/train"
	else:
		train_log_dir = "logs/gradient_tape/"+model_filename+"/train"
	train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	
	#--- Log the graph structure of the NCA
	tf.summary.trace_on(graph=True,profiler=True)
	y = ca.perceive(x0)
	with train_summary_writer.as_default():
		tf.summary.trace_export(name="NCA Perception",step=0,profiler_outdir=train_log_dir)
	
	tf.summary.trace_on(graph=True,profiler=True)
	x = ca(x0)
	with train_summary_writer.as_default():
		tf.summary.trace_export(name="NCA full step",step=0,profiler_outdir=train_log_dir)
	
	#--- Log the target image and initial condtions
	with train_summary_writer.as_default():
		tf.summary.image('Initial GSC - Brachyury T - SOX2 --- Lamina B',
						 np.concatenate((x0[:1,...,:3],np.repeat(x0[:1,...,3:4],3,axis=-1)),axis=0),
						 step=0)
	
		tf.summary.image('Target GSC - Brachyury T - SOX2 --- Lamina B',
						 np.concatenate((target[:1,...,:3],np.repeat(target[:1,...,3:4],3,axis=-1)),axis=0),
						 step=0)
	return train_summary_writer

def setup_tb_log_sequence(ca,x0,data,model_filename=None):
	"""
		Initialises the tensorboard logging of training.
		Writes some initial information. Very similar to setup_tb_log_single, but designed for sequence modelling

		Parameters
		----------
		ca : object callable - float32 tensor [batches,size,size,N_CHANNELS],float32,float32 -> float32 tensor [batches,size,size,N_CHANNELS]
			the NCA object to train
		x0 : float32 tensor [batches,size,size,N_CHANNELS]
			correctly formatter initial condition for NCA function trace
		data : float32 tensor [T,batches,size,size,4]
			the image sequence to be modelled by the NCA.
		model_filename : str
			name of directories to save tensorboard log and model parameters to.
			log at :	'logs/gradient_tape/model_filename/train'
			model at : 	'models/model_filename'
			if None, doesn't save model but still saves log to 'logs/gradient_tape/*current_time*/train'

		Returns
		-------
		train_summary_writer : tf.summary.file_writer object
	"""


	if model_filename is None:
		current_time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		train_log_dir = "logs/gradient_tape/"+current_time+"/train"
	else:
		train_log_dir = "logs/gradient_tape/"+model_filename+"/train"
	train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	
	#--- Log the graph structure of the NCA
	tf.summary.trace_on(graph=True,profiler=True)
	y = ca.perceive(x0)
	with train_summary_writer.as_default():
		tf.summary.trace_export(name="NCA Perception",step=0,profiler_outdir=train_log_dir)
	
	tf.summary.trace_on(graph=True,profiler=True)
	x = ca(x0)
	with train_summary_writer.as_default():
		tf.summary.trace_export(name="NCA full step",step=0,profiler_outdir=train_log_dir)
	
	#--- Log the target image and initial condtions
	with train_summary_writer.as_default():
		tf.summary.image('Sequence GSC - Brachyury T - SOX2',data[:,0,...,:3],step=0,max_outputs=data.shape[0])
		
		
	return train_summary_writer


def tb_training_loop_log_sequence(train_summary_writer,loss,ca,x,i,N_BATCHES):
	"""
		Helper function to format some data logging during the training loop

		Parameters
		----------
		train_summary_writer : tf.summary.file_writer object
		
		ca : object callable - float32 tensor [batches,size,size,N_CHANNELS],float32,float32 -> float32 tensor [batches,size,size,N_CHANNELS]
			the NCA object being trained, so that it's weights can be logged
		x : float32 tensor [N_BATCHES,size,size,N_CHANNELS]
			final output of NCA
		i : int
			current step in training loop - useful for logging something every n steps

	"""
	with train_summary_writer.as_default():
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
			model_params_0 = ca.dense_model.layers[0].get_weights()
			model_params_1 = ca.dense_model.layers[1].get_weights()
			model_params_2 = ca.dense_model.layers[2].get_weights()
			tf.summary.histogram('Layer 0 weights',model_params_0[0],step=i)
			tf.summary.histogram('Layer 1 weights',model_params_1[0],step=i)
			tf.summary.histogram('Layer 2 weights',model_params_2[0],step=i)
			tf.summary.histogram('Layer 0 biases',model_params_0[1],step=i)
			tf.summary.histogram('Layer 1 biases',model_params_1[1],step=i)
			tf.summary.histogram('Layer 2 biases',model_params_2[1],step=i)


def tb_training_loop_log_single(train_summary_writer,loss,ca,x,i):
	"""
		Helper function to format some data logging during the training loop

		Parameters
		----------
		train_summary_writer : tf.summary.file_writer object
		
		ca : object callable - float32 tensor [batches,size,size,N_CHANNELS],float32,float32 -> float32 tensor [batches,size,size,N_CHANNELS]
			the NCA object being trained, so that it's weights can be logged
		x : float32 tensor [N_BATCHES,size,size,N_CHANNELS]
			final output of NCA
		i : int
			current step in training loop - useful for logging something every n steps

	"""
	with train_summary_writer.as_default():
		#for j in range(len(loss)):
		tf.summary.scalar('Loss',loss,step=i)
		if i%10==0:
			tf.summary.image('Final state GSC - Brachyury T - SOX2 --- Lamina B',
							 np.concatenate((x[:1,...,:3],np.repeat(x[:1,...,3:4],3,axis=-1)),axis=0),
							 step=i)
			#print(ca.dense_model.layers[0].get_weights()[0])
			model_params_0 = ca.dense_model.layers[0].get_weights()
			model_params_1 = ca.dense_model.layers[1].get_weights()
			model_params_2 = ca.dense_model.layers[2].get_weights()
			tf.summary.histogram('Layer 0 weights',model_params_0[0],step=i)
			tf.summary.histogram('Layer 1 weights',model_params_1[0],step=i)
			tf.summary.histogram('Layer 2 weights',model_params_2[0],step=i)
			tf.summary.histogram('Layer 0 biases',model_params_0[1],step=i)
			tf.summary.histogram('Layer 1 biases',model_params_1[1],step=i)
			tf.summary.histogram('Layer 2 biases',model_params_2[1],step=i)




def tb_write_result(train_summary_writer,ca,x0):
	with train_summary_writer.as_default():
		grids = ca.run(x0,200,1)
		grids[...,4:] = (1+np.tanh(grids[...,4:]))/2.0
		for i in range(200):
			tf.summary.image('Trained NCA dynamics GSC - Brachyury T - SOX2 --- Lamina B',
							 np.concatenate((grids[i,:1,...,:3],
							 				 np.repeat(grids[i,:1,...,3:4],3,axis=-1)),
							 				axis=1),
							 step=i)
			if (ca.N_CHANNELS>=7) and (ca.N_CHANNELS <10):
				tf.summary.image('Trained NCA hidden dynamics (tanh limited)',
								 grids[i,:1,...,4:7],step=i,
							 	max_outputs=4)
			elif ((ca.N_CHANNELS>=10) and (ca.N_CHANNELS <13)):
				tf.summary.image('Trained NCA hidden dynamics (tanh limited)',
								 np.concatenate((grids[i,:1,...,4:7],
												 grids[i,:1,...,7:10]),axis=1),
								 step=i,
							 	 max_outputs=4)
			elif ((ca.N_CHANNELS>=13) and (ca.N_CHANNELS <16)):
				tf.summary.image('Trained NCA hidden dynamics (tanh limited)',
								 np.concatenate((grids[i,:1,...,4:7],
												 grids[i,:1,...,7:10],
								 				 grids[i,:1,...,10:13]),axis=1),
								 step=i,
							 	 max_outputs=4)
			elif (ca.N_CHANNELS>=16):
				tf.summary.image('Trained NCA hidden dynamics (tanh limited)',
								 np.concatenate((grids[i,:1,...,4:7],
												 grids[i,:1,...,7:10],
								 				 grids[i,:1,...,10:13],
								 				 grids[i,:1,...,13:16]),axis=1),
								 step=i,
							 	 max_outputs=4)				
								
							 
	


def train(ca,target,N_BATCHES,TRAIN_ITERS,x0=None,iter_n=50,model_filename=None):
	"""
		Trains the ca to recreate target image given an initial condition
		
		Parameters
		----------
		ca : object callable - float32 tensor [batches,size,size,N_CHANNELS],float32,float32 -> float32 tensor [batches,size,size,N_CHANNELS]
			the NCA object to train
		target : float32 tensor [batches,size,size,4]
			the target image to be grown by the NCA.
		N_BATCHES : int
			size of training batch
		TRAIN_ITERS : int
			how many iterations of training
		x0 : float32 tensor [batches,size,size,k<=N_CHANNELS]
			the initial condition of NCA. If it has less channels than the NCA, pad with zeros. If none, is set to zeros with one 'seed' of 1s in the middle
		iter_n : int
			number of NCA update steps to run from x0 - i.e. train the NCA to recreate target with iter_n steps from x0
		model_filename : str
			name of directories to save tensorboard log and model parameters to.
			log at :	'logs/gradient_tape/model_filename/train'
			model at : 	'models/model_filename'
			if None, doesn't save model but still saves log to 'logs/gradient_tape/*current_time*/train'

		
		Returns
		-------
		None
	"""
	#TRAIN_ITERS = 1000
	loss_log = []
	TARGET_SIZE = target.shape[1]
	N_CHANNELS = ca.N_CHANNELS

	lr = 2e-3
	lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([TRAIN_ITERS//2], [lr, lr*0.1])
	trainer = tf.keras.optimizers.Adam(lr_sched)
	
	
	#--- Setup initial condition
	if x0 is None:
		x0 = np.zeros((N_BATCHES,TARGET_SIZE,TARGET_SIZE,N_CHANNELS),dtype="float32")
		x0[:,TARGET_SIZE//2,TARGET_SIZE//2,:]=1
	else:
		z0 = np.zeros((x0.shape[0],x0.shape[1],x0.shape[2],N_CHANNELS-x0.shape[3]))
		x0 = np.concatenate((x0,z0),axis=-1).astype("float32")
		if x0.shape[0]==1:
			x0 = np.repeat(x0,N_BATCHES,axis=0).astype("float32")
	
	if ca.ADHESION_MASK is not None:
		_mask = np.zeros((x0.shape),dtype="float32")
		_mask[...,4]=1




	def loss_f(x):
		#return tf.reduce_mean(tf.square(x[...,:4]-target),[-2, -3, -1])
		#return tf.reduce_max(tf.square(x[...,:4]-target),[-2, -3, -1])
		return tf.math.reduce_euclidean_norm(x[...,:4]-target,[-2,-3,-1])
	
	def train_step(x):
		with tf.GradientTape() as g:
			for i in range(iter_n):
				x = ca(x)
				if ca.ADHESION_MASK is not None:
					x = _mask*ca.ADHESION_MASK + (1-_mask)*x
			loss = tf.reduce_mean(loss_f(x))
		grads = g.gradient(loss,ca.weights)
		grads = [g/(tf.norm(g)+1e-8) for g in grads]
		trainer.apply_gradients(zip(grads, ca.weights))
		return x, loss

	#--- Setup tensorboard logging
	train_summary_writer = setup_tb_log_single(ca,target,x0,model_filename)

	#--- Do training loop
	for i in tqdm(range(TRAIN_ITERS)):
		x,loss = train_step(x0)
		loss_log.append(loss)
		
		#--- Write to log
		tb_training_loop_log_single(train_summary_writer,loss,ca,x,i)
		
	#--- Write resulting animation to tensorboard			
	tb_write_result(train_summary_writer,ca,x0)

	#--- If a filename is provided, save the trained NCA model.
	if model_filename is not None:
		ca.save_wrapper(model_filename)



def train_sequence(ca,data,N_BATCHES,TRAIN_ITERS,iter_n,model_filename=None):
	"""
		Trains the ca to recreate the given image sequence. Error is calculated by comparing ca grid to each image after iter_n/T steps 
		
		Parameters
		----------
		ca : object callable - float32 tensor [batches,size,size,N_CHANNELS],float32,float32 -> float32 tensor [batches,size,size,N_CHANNELS]
			the NCA object to train
		data : float32 tensor [T,batches,size,size,4]
			The image sequence being modelled. data[0] is treated as the initial condition
		N_BATCHES : int
			size of training batch
		TRAIN_ITERS : int
			how many iterations of training
		iter_n : int
			number of NCA update steps to run ca
		model_filename : str
			name of directories to save tensorboard log and model parameters to.
			log at :	'logs/gradient_tape/model_filename/train'
			model at : 	'models/model_filename'
			if None, doesn't save model but still saves log to 'logs/gradient_tape/*current_time*/train'

		
		Returns
		-------
		None
	"""
	data = data.astype("float32")
	N_CHANNELS = ca.N_CHANNELS
	#print(np.max(data))
	#print(np.min(data))
	lr = 2e-3
	#lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([TRAIN_ITERS//2], [lr, lr*0.1])
	lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(lr, TRAIN_ITERS, 0.96)
	trainer = tf.keras.optimizers.Adam(lr_sched)
	#trainer = tf.keras.optimizers.RMSprop(lr_sched)
	
	#--- Setup initial condition
	
	if data.shape[1]==1:
		data = np.repeat(data,N_BATCHES,axis=1).astype("float32")
	
	x0 = np.copy(data[:])
	x0[1:] = data[:-1] # Including 1 extra time slice to account for hidden 12h time
	target = data[1:]
	#for i in range(target.shape[0]):
	#	plt.imshow(target[i,0,...,:3])
	#	plt.show()
	#	plt.imshow(x0[i+1,0,...,:3])
	#	plt.show()
	#print(x0.shape)
	#print(target.shape)

	T = data.shape[0]
	z0 = np.zeros((x0.shape[0],x0.shape[1],x0.shape[2],x0.shape[3],N_CHANNELS-x0.shape[4]))
	x0 = np.concatenate((x0,z0),axis=-1).astype("float32")
	
	x0 = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
	#print(x0.shape)
	target = target.reshape((-1,target.shape[2],target.shape[3],target.shape[4]))
	#print(target.shape)
	x0_true = np.copy(x0)

	if ca.ADHESION_MASK is not None:
		_mask = np.zeros((x0.shape),dtype="float32")
		_mask[...,4]=1




	#loss_mask = np.ones(x0.shape[0]).astype(bool)
	#loss_mask[:N_BATCHES] = 0
	#print(loss_mask)
	#print()

	def loss_f(x):
		#return tf.reduce_mean(tf.square(x[...,:4]-target),[-2, -3, -1])
		#return tf.reduce_max(tf.square(x[...,:4]-target),[-2, -3, -1])
		#return tf.math.reduce_euclidean_norm(tf.boolean_mask(x[...,:4],loss_mask)-target,[-2,-3,-1])
		return tf.math.reduce_euclidean_norm((x[N_BATCHES:,...,:4]-target),[-2,-3,-1])
		#return tf.reduce_max(tf.square(x[N_BATCHES:,...,:4]-target),[-2,-3,-1])
		#return -tf.reduce_sum(tf.math.l2_normalize(x[N_BATCHES:,...,:4])*(target),[-2,-3,-1])
	#print(loss_f(x0))

	def train_step(x,update_gradients=True):
		state_log = []
		with tf.GradientTape() as g:
			#times = [24,36,48]
			#times = np.linspace(0,iter_n,num=5,endpoint=True,dtype=int)[2:]
			#g.watch(state_log)
			#g.watch(x)
			#print(times)
			for i in range(iter_n):
				x = ca(x)
				if ca.ADHESION_MASK is not None:
					x = _mask*ca.ADHESION_MASK + (1-_mask)*x
			losses = loss_f(x) 
			mean_loss = tf.reduce_mean(losses)
					
					
			
			
			
		if update_gradients:
			grads = g.gradient(mean_loss,ca.weights)
			grads = [g/(tf.norm(g)+1e-8) for g in grads]
			trainer.apply_gradients(zip(grads, ca.weights))
		return x, mean_loss,losses

	#--- Setup tensorboard logging
	train_summary_writer = setup_tb_log_sequence(ca,x0,data,model_filename)



	#--- Do training loop
	best_mean_loss = 100000

	for i in tqdm(range(TRAIN_ITERS)):
		
		x,mean_loss,losses = train_step(x0)#,i%4==0)
		#if i>10:
		#x0[N_BATCHES:2*N_BATCHES] = x[:N_BATCHES] # hidden 12h timeslice
		x0[N_BATCHES:] = x[:-N_BATCHES] # updates each initial condition to be final condition of previous chunk of timesteps
		if N_BATCHES>1:
			x0[::N_BATCHES][2:] = x0_true[::N_BATCHES][2:] # update one batch to contain the true initial conditions
		
		#--- Save model each time it is better than previous best model (and after 10% of training iterations are done)
		if (mean_loss<best_mean_loss) and (i>TRAIN_ITERS//10):
			if model_filename is not None:
				ca.save_wrapper(model_filename)
				print("--- Model saved at "+str(i)+" epochs ---")
			best_mean_loss = mean_loss

		loss = np.hstack((mean_loss,losses))
		
		
		#--- Write to log
		tb_training_loop_log_sequence(train_summary_writer,loss,ca,x,i,N_BATCHES)
	print("-------- Training complete ---------")
	#--- Write resulting animation to tensorboard			
	tb_write_result(train_summary_writer,ca,x0)

	#--- If a filename is provided, save the trained NCA model.
	#if model_filename is not None:
	#	ca.save_wrapper(model_filename)