from NCA_utils import *
import matplotlib.pyplot as plt
from matplotlib import image
import tensorflow as tf
import scipy as sp
from tqdm import tqdm
import datetime
#import keras.applications.vgg16 as vgg16
#--- Setup definitions



class NCA(tf.keras.Model):
	#--- Neural Cellular Automata class
	def __init__(self,N_CHANNELS,FIRE_RATE=0.5,ADHESION_MASK=None):
		super(NCA,self).__init__()
		self.N_CHANNELS=N_CHANNELS # RGBA +hidden layers
		self.FIRE_RATE=FIRE_RATE # controls stochastic updates - i.e. grid isn't globaly synchronised
		if ADHESION_MASK is not None:
			self.ADHESION_MASK=np.repeat(ADHESION_MASK[...,np.newaxis],N_CHANNELS,axis=-1) # Signals where cells can adhere to the micropattern
		else:
			self.ADHESION_MASK=None
		#--- Set up dense nn for perception vector
		self.dense_model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(4*self.N_CHANNELS,1,activation=tf.nn.tanh),
			tf.keras.layers.Conv2D(2*self.N_CHANNELS,1,activation=tf.nn.tanh),
			tf.keras.layers.Conv2D(self.N_CHANNELS,1,activation=None,kernel_initializer=tf.zeros_initializer)])
		self(tf.zeros([1,3,3,N_CHANNELS])) # Dummy call to build the model
		
		print(self.dense_model.summary())
	
	@tf.function
	def perceive(self,x):
		"""
			Constructs a field y where each "pixel" represents the perception of local environments.
			Choice of kernels here is important.

			Parameters
			----------
			x : float32 tensor [batches,size,size,N_CHANNELS]
				state space of NCA, first 4 channels are typically visualised as RGBA,
				rest are hidden channels

			Returns
			-------
			y : float32 tensor [batches,size,size,depth]
				perception field, where each coordinate [batch,size,size] is a vector encoding
				local structure of x at [batch,size,size]. Used as input to self.dense_model
		"""
		_i = np.array([0,1,0],dtype=np.float32)
		I  = np.outer(_i,_i)
		dx = (np.outer([1,2,1],[-1,0,1])/8.0).astype(np.float32)
		dy = dx.T
		kernel = tf.stack([I,dx,dy],-1)[:,:,None,:]
		kernel = tf.repeat(kernel,self.N_CHANNELS,2)
		y = tf.nn.depthwise_conv2d(x,kernel,[1,1,1,1],"SAME")
		
		return y
	
	@tf.function
	def call(self,x,step_size=1.0,fire_rate=None):
		"""
			Applies a neural network (the trainable part of the model) to the perception field.
			
			Parameters
			----------
			x : float32 tensor [batches,size,size,N_CHANNELS]
				state space of NCA, first 4 channels are typically visualised as RGBA,
				rest are hidden channels
			step_size : float=1.0
				scale size of updates
			fire_rate : float=None
				controls probability of each pixel updating

			Returns
			-------
			x_new : float32 tensor
				new state space of NCA, with (stochastically masked) update applied across all channels and batches
		"""
		print(x.shape)
		y = self.perceive(x)
		print(y.shape)
		dx = self.dense_model(y)*step_size
		if fire_rate is None:
			fire_rate = self.FIRE_RATE
		update_mask = tf.random.normal(tf.shape(x[:,:,:,:1])) <= fire_rate
		x_new = x + dx*tf.cast(update_mask,tf.float32)
		return x_new

	def run(self,x0,T,N_BATCHES=1,ADHESION_MASK=None):
		"""
			Iterates self.call several times to perform a NCA simulation.
			
			Parameters
			----------
			x0 : float32 array [batches,size,size,channels]
				Initial condition for NCA simulation
			T : int 
				number of timesteps to run for		
			N_BATCHES : int=1
				number of batches of simulations to run in parallel
			
			Returns	
			-------
			trajectory : float32 array [T,batches,size,size,channels]
				time series resulting from running NCA for T steps starting at x0
		"""

		TARGET_SIZE = x0.shape[1]
		x0 = x0[0:N_BATCHES] # If initial condition is too wide in batches dimension, reduce it
		trajectory = np.zeros((T,N_BATCHES,TARGET_SIZE,TARGET_SIZE,self.N_CHANNELS),dtype="float32")
		
		print("Trajectory shape: "+str(trajectory.shape))
		print("x0 shape: "+str(x0.shape))
		if x0.shape[-1]<self.N_CHANNELS: # If x0 has less channels than the NCA, pad zeros to x0
			z0 = np.zeros((N_BATCHES,TARGET_SIZE,TARGET_SIZE,self.N_CHANNELS-x0.shape[-1]),dtype="float32")
			x0 = np.concatenate((x0,z0),axis=-1)
		assert trajectory[0].shape == x0.shape
		print("x0 shape: "+str(x0.shape))
		
		if ADHESION_MASK is not None:
			print("Adhesion mask shape: "+str(ADHESION_MASK.shape))
			#self.ADHESION_MASK = ADHESION_MASK
			self.ADHESION_MASK=np.repeat(ADHESION_MASK[...,np.newaxis],self.N_CHANNELS,axis=-1)
		if self.ADHESION_MASK is not None:
			_mask = np.zeros((x0.shape),dtype="float32")
			_mask[...,4]=1
			print("Adhesion channel select mask shape: "+str(_mask.shape))
			x0 = _mask*self.ADHESION_MASK[:N_BATCHES] + (1-_mask)*x0

		print("Trajectory shape: "+str(trajectory.shape))
		print("x0 shape: "+str(x0.shape))
		trajectory[0] = x0
		
		for t in range(1,T):
			trajectory[t] = self.call(trajectory[t-1])
			if self.ADHESION_MASK is not None:
				trajectory[t] = _mask*self.ADHESION_MASK[:N_BATCHES] + (1-_mask)*trajectory[t]
		#print(trajectory.shape)
		return trajectory


	
	def get_config(self):
		return {"N_CHANNELS":self.N_CHANNELS,
				"FIRE_RATE": self.FIRE_RATE}
				#"ADHESION_MASK":self.ADHESION_MASK,
				#"dense_model":self.dense_model}
	
	@classmethod
	def from_config(cls,config):
		return cls(**config)

	
	def save_wrapper(self,filename=None):
		"""
			Saves the trainable part of the model - the dense nn trained on the perception field.
			Wrapper for keras.models.save function, puts things in the right directory etc.

			Parameters
			----------
			filename : str
				Name of directory where keras SavedModel files are contained
		"""

		if filename is None:
			filename=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		
		self.save("models/"+filename)

def load_wrapper(filename):
	"""
		Loads the trainable part of the model - the dense nn trained on the perception field

		Parameters
		----------
		filename : str
			Name of directory where keras SavedModel files are contained
	"""
	return tf.keras.models.load_model("models/"+filename,custom_objects={"NCA":NCA})
	



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
	TARGET_SIZE = target.shape[0]
	N_CHANNELS = ca.N_CHANNELS

	lr = 2e-3
	lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([TRAIN_ITERS//2], [lr, lr*0.1])
	trainer = tf.keras.optimizers.Adam(lr_sched)
	
	
	#--- Setup initial condition
	if x0 is None:
		x0 = np.zeros((N_BATCHES,TARGET_SIZE,TARGET_SIZE,N_CHANNELS),dtype="float32")
		x0[:,TARGET_SIZE//2,TARGET_SIZE//2,:]=1
	else:
		#print(x0.shape)
		z0 = np.zeros((x0.shape[0],x0.shape[1],x0.shape[2],16-x0.shape[3]))
		#print(z0.shape)
		x0 = np.concatenate((x0,z0),axis=-1).astype("float32")
		#print(x0.shape)
		#x0 = x0[np.newaxis]
		if x0.shape[0]==1:
			x0 = np.repeat(x0,N_BATCHES,axis=0).astype("float32")
		#print(x0.shape)

	if ca.ADHESION_MASK is not None:
		_mask = np.zeros((x0.shape),dtype="float32")
		_mask[...,4]=1

	



	def loss_f(x):
		#return tf.reduce_mean(tf.square(x[...,:4]-target),[-2, -3, -1])
		#return tf.reduce_max(tf.square(x[...,:4]-target),[-2, -3, -1])
		return tf.math.reduce_euclidean_norm(x[...,:4]-target,[-2,-3,-1])
	def train_step(x):
		#iter_n=np.random.randint(50,70)
		#iter_n = 50
		with tf.GradientTape() as g:
			for i in range(iter_n):
				x = ca(x)
				if ca.ADHESION_MASK is not None:
					x = _mask*ca.ADHESION_MASK + (1-_mask)*x
			#x = ca.run(x,iter_n,N_BATCHES)[-1]
			#print(x)
			loss = tf.reduce_mean(loss_f(x))
		grads = g.gradient(loss,ca.weights)
		grads = [g/(tf.norm(g)+1e-8) for g in grads]
		trainer.apply_gradients(zip(grads, ca.weights))


		return x, loss




	#--- Setup tensorboard logging stuff
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
	
	#--- Log the target image
	with train_summary_writer.as_default():
		tf.summary.image('Target image',target,step=0)

	#--- Do training loop
	for i in tqdm(range(TRAIN_ITERS)):
		x,loss = train_step(x0)
		loss_log.append(loss)
		
		#--- Write to log
		with train_summary_writer.as_default():
			tf.summary.scalar('Loss',loss,step=i)
			if i%10==0:
				tf.summary.image('Final state',x[...,:4],step=i)
		#if i%100==0:
		#	plt.imshow(x[0,:,:,:4])
		#	plt.show()
	with train_summary_writer.as_default():
		grids = ca.run(x0,200,N_BATCHES)
		#grids = run(ca,200,1,TARGET_SIZE)
		
		grids[...,4:] = (1+np.tanh(grids[...,4:]))/2.0
		for i in range(200):
			tf.summary.image('Trained NCA dynamics (RGBA)',grids[i,...,:4],step=i)
			tf.summary.image('Trained NCA hidden dynamics (tanh limited) 1',grids[i,...,4:8],step=i)
			tf.summary.image('Trained NCA hidden dynamics (tanh limited) 2',grids[i,...,8:12],step=i)
			tf.summary.image('Trained NCA hidden dynamics (tanh limited) 3',grids[i,...,12:],step=i)
	
	#--- If a filename is provided, save the trained NCA model.
	if model_filename is not None:
		ca.save_wrapper(model_filename)

""" - disused
def run(model,T=100,N_BATCHES=1,TARGET_SIZE=40):
	N_CHANNELS = model.N_CHANNELS
	grids = np.zeros((T,N_BATCHES,TARGET_SIZE,TARGET_SIZE,N_CHANNELS),dtype="float32")
	grids[0,:,TARGET_SIZE//2,TARGET_SIZE//2,:]=1
	for t in range(1,T):
		grids[t]=(model(grids[t-1]))
		if model.ADHESION_MASK is not None:
			grids[t,...,4] = model.ADHESION_MASK
	return grids
"""






def main():

	T=0
	N_CHANNELS=16 # Must be greater than 4
	N_BATCHES=4
	

	#a = load_sequence_A("A1_F11")
	a = load_sequence_batch(N_BATCHES)[:,:,::2,::2]
	#for i in range(5):
	#	plt.imshow(a[T,i,...,3])
	#	plt.show()
	
	print(a.shape)
	mask = adhesion_mask_batch(a)
	print(mask.shape)
	
	for i in range(N_BATCHES):
		plt.imshow(a[T,i,...,:3])
		plt.show()
		plt.imshow(mask[i])
		plt.show()
	


	ca = NCA(N_CHANNELS,ADHESION_MASK=mask)


	#print(np.max(target[...,3]))
	train(ca,a[2],N_BATCHES,1000,a[0],48,"euclidean_norm_error_tanh_activation")

	print("Training complete")
	#t1 = datetime.datetime.now()

	
	#ca.save_wrapper("save_test_2")
	#t2 = datetime.datetime.now()
	#print(t2-t1)
	#ca2 = tf.keras.models.load_model("save_test",custom_objects={"NCA":NCA})
	ca2 = load_wrapper("euclidean_norm_error_tanh_activation")
	#print(a[0].shape)

	#grids = run(ca,1000,1,100)[:,0]
	grids = ca.run(a[0],100,1,mask)[:,0]
	print(np.max(grids))
	print(np.min(grids))
	my_animate(grids[...,:4])

	grids2 = ca2.run(a[0],100,1,mask)[:,0]
	my_animate(grids2[...,:4])


	grids[...,4:] = (1+np.tanh(grids[...,4:]))/2.0
	#my_animate(np.abs(grids[...,:4]-target))
	my_animate(grids[...,4:8])
	my_animate(grids[...,8:12])
	my_animate(grids[...,12:])
	plt.imshow(grids[50,...,:4])
	plt.show()
	#my_animate((grids[...,:4]+1)/2.0)

	#error = np.sqrt(np.sum((target-grids[50,:,:,:4])**2))
	#print(error)
	#grids = (grids+np.abs(np.min(grids)))/(2*np.max(grids))
	
	# Visualise different subsets of channels 
	
	#my_animate(grids[...,3:6])
	#my_animate(grids[...,6:9])
	
if __name__=="__main__":
	main()