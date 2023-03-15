#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
import tensorflow as tf
class GOL_solver(object):
	def __init__(self,N_BATCHES,SIZE=[128,128],PADDING="periodic"):
		self.N_BATCHES=N_BATCHES
		self.PADDING=PADDING
		self.SIZE = SIZE
		
	def run(self,N_STEPS):
		trajectory = np.zeros(shape=(N_STEPS,self.N_BATCHES,self.SIZE[0],self.SIZE[1]),dtype=np.int32)
		trajectory[0,:] = np.random.choice([0,1],size=(self.N_BATCHES,self.SIZE[0],self.SIZE[1]))
		for i in range(N_STEPS-1):
			"""
			lattice = trajectory[i]
			_u = np.roll(lattice,[1,0],axis=[0,1])
			_d = np.roll(lattice,[-1,0],axis=[0,1])
			_l = np.roll(lattice,[0,1],axis=[0,1])
			_r = np.roll(lattice,[0,-1],axis=[0,1])
			_ul = np.roll(lattice,[1,1],axis=[0,1])
			_ur = np.roll(lattice,[1,-1],axis=[0,1])
			_dl = np.roll(lattice,[-1,1],axis=[0,1])
			_dr = np.roll(lattice,[-1,-1],axis=[0,1])
			ns = _u + _d + _l + _r + _ul + _ur + _dl + _dr
			print(ns)
			alive = np.logical_or(np.logical_and(ns==2,lattice==1),ns==3)
			trajectory[i+1] = alive.astype(np.int32)#tf.cast(alive,dtype=tf.int32)
			"""
			for j in range(self.N_BATCHES):
				trajectory[i+1,j] = self.gol_update(trajectory[i,j])
		return trajectory
	
			
	def gol_update(self,grid):
	#Performs one update step on total grid
		k = np.array([[1,1,1],
					  [1,9,1],
					  [1,1,1]])
		rule = np.array([0,0,0,1,0,0,0,0,0,
						 0,0,1,1,0,0,0,0,0,0])
		#kernel = tf.stack([k],-1)[None,:,:]
		#return rule[tf.cast(tf.nn.depthwise_conv2d(grid,kernel,[1,1,1,1],"SAME"),tf.int32)]
		
		return rule[signal.convolve2d(grid,k,boundary='wrap',mode='same').astype(int)]
