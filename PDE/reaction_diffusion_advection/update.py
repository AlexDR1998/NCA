#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:46:20 2023

@author: s1605376
"""
import jax
import equinox as eqx
import jax.numpy as jnp
import time
from PDE.reaction_diffusion_advection.advection import V
from PDE.reaction_diffusion_advection.reaction import R
from PDE.reaction_diffusion_advection.diffusion import D

class F(eqx.Module):
	f_v: V
	f_r: R
	f_d: D
	N_CHANNELS: int
	PERIODIC: bool
	dx: float
	def __init__(self,N_CHANNELS,PERIODIC,dx,key=jax.random.PRNGKey(int(time.time()))):
		self.N_CHANNELS = N_CHANNELS
		self.PERIODIC = PERIODIC
		key1,key2,key3 = jax.random.split(key,3)
		self.dx = dx
		self.f_r = R(N_CHANNELS,key=key1)
		self.f_v = V(N_CHANNELS,PERIODIC,dx=self.dx,key=key2)
		self.f_d = D(N_CHANNELS,PERIODIC,dx=self.dx,key=key3)

	@eqx.filter_jit
	def __call__(self,t,X,args):
		"""

		Parameters
		----------
		t : float32
			timestep
		
		X : float32 [N_CHANNELS,_,_]
			input PDE lattice state.
		
		args : None
			Required for format of diffrax.ODETerm

		Returns
		-------
		X_update : float32 [N_CHANNELS,_,_]
			update to PDE lattice state state.

		"""
		return self.f_d(X) - self.f_v(X) + self.f_r(X)
	
	def partition(self):
		r_diff,r_static = self.f_r.partition()
		v_diff,v_static = self.f_v.partition()
		d_diff,d_static = self.f_d.partition()
		total_diff,total_static = eqx.partition(self,eqx.is_array)
		#print(total_diff)
		#print(total_static)
		where_r = lambda m:m.f_r
		where_v = lambda m:m.f_v
		where_d = lambda m:m.f_d
		
		total_diff = eqx.tree_at(where_r,total_diff,r_diff)
		total_diff = eqx.tree_at(where_v,total_diff,v_diff)
		total_diff = eqx.tree_at(where_d,total_diff,d_diff)
		
		total_static = eqx.tree_at(where_r,total_static,r_static)
		total_static = eqx.tree_at(where_v,total_static,v_static)
		total_static = eqx.tree_at(where_d,total_static,d_static)
		return total_diff,total_static
	
	def combine(self,diff,static):
		self = eqx.combine(diff,static)