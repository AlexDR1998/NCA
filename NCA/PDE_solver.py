import numpy as np
#import scipy as sp
import tensorflow as tf
from tqdm import tqdm
from NCA.NCA_utils import periodic_padding
#import matplotlib.pyplot as plt
"""
A class for the simulation of arbitrary systems of time dependent PDEs with 2 spatial dimensions

"""

class PDE_solver(object):
	#Runs numerical solutions PDEs of the form dX/dt = F(X,gradX,grad^2X)

	def __init__(self,F,N_CHANNELS,N_BATCHES,size=[128,128],PADDING="periodic"):
		"""
		#Expects F to be of 4 2D fields of N_CHANNEL channels/dimensions
			Parameters
			----------
			F : float32 tensor [size[0],size[1],N_CHANNELS]**4 -> float32 tensor [size[0],size[1],N_CHANNELS]


			PADDING : string optional
				zero, flat or periodic boundary conditions
		
		"""

		self.F = F
		self.X = np.zeros((N_BATCHES,size[0],size[1],N_CHANNELS))
		self.N_CHANNELS=N_CHANNELS
		self.N_BATCHES=N_BATCHES
		self.PADDING=PADDING

		dx = (np.outer([1,2,1],[-1,0,1])/8.0).astype(np.float32)
		dy = dx.T
		lap = np.array([[0.25,0.5,0.25],
						[0.5,-3,0.5],
						[0.25,0.5,0.25]]).astype(np.float32)
		#lap = np.array([[0.05,0.2,0.05],
		#				[0.2,-1,0.2],
		#				[0.05,0.2,0.05]]).astype(np.float32)

		kernel = tf.stack([dx,dy,lap],-1)[:,:,None,:]
		#kernel = tf.stack([lap],-1)[:,:,None,:]
		print(kernel.shape)

		self.KERNEL = tf.repeat(kernel,self.N_CHANNELS,2)
		
		print(self.KERNEL.shape)

	
	@tf.function
	def calculate_derivatives(self,X,KERNEL):
		"""
			Calculates spatial derivates of X

			Parameters
			----------
			X : float32 tensor [N_BATCHES,size,size,N_CHANNELS]

			KERNEL : float32 tensor [1,3,3,3]

			Returns
			-------
			_X : float32 tensor [N_BATCHES,size,size,3*N_CHANNELS]
				_X[...,::3] 			-- Gradients in x
				_X[...,1::3]			-- Gradients in y
				_X[...,2::3]			-- Laplacian
		"""
		return tf.nn.depthwise_conv2d(X,KERNEL,[1,1,1,1],"VALID")
	
	def update(self,step_size):
		
		if self.PADDING=="periodic":
			X_pad = periodic_padding(self.X,1)
		elif self.PADDING=="flat":
			X_pad = tf.pad(self.X,tf.constant([[0,0],[1,1],[1,1],[0,0]]),"SYMMETRIC")
		else:
			X_pad = tf.pad(self.X,tf.constant([[0,0],[1,1],[1,1],[0,0]]),"CONSTANT")
			
		_X = self.calculate_derivatives(X_pad,self.KERNEL)

		#if self.PADDING!="zero":
		#	_X = _X[:,1:-1,1:-1]

		

		Xdx = _X[...,::3]
		Xdy = _X[...,1::3]
		Xdd = _X[...,2::3]


		#gradX_x = sp.signal.convolve2d(self.X,dx,boundary='wrap',mode='same') 
		#gradX_y = sp.signal.convolve2d(self.X,dy,boundary='wrap',mode='same') 

		self.X = self.X + step_size*self.F(self.X,Xdx,Xdy,Xdd)
		return self.X

	def run(self,iterations,step_size=0.1,initial_condition=None):
		#Apply update iteratively and output full solution
		trajectory = np.zeros((iterations+1,self.N_BATCHES,self.X.shape[1],self.X.shape[2],self.N_CHANNELS))
		if initial_condition is None:
			initial_condition = 2*np.random.uniform(size=self.X.shape)-1
		trajectory[0] = initial_condition
		self.X = initial_condition.astype(np.float32)

		for i in tqdm(range(1,iterations+1)):
			trajectory[i] = self.update(step_size)
		return trajectory


@tf.function
def lap(X):
	#Helper function to calculate the laplacian of a given vector field
	lap = np.array([[0.25,0.5,0.25],
					[0.5,-3,0.5],
					[0.25,0.5,0.25]]).astype(np.float32)
	kernel = tf.stack([lap],-1)[:,:,None,:]
	KERNEL = tf.repeat(kernel,X.shape[-1],2)
	return tf.nn.depthwise_conv2d(X,KERNEL,[1,1,1,1],"SAME")

