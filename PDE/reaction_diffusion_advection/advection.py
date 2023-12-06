#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:17:42 2023

@author: Alex Richardson

Function that handles the advection bit. Function call does div(V(X)*X), trainable parameters are attached to V only.
V: Re^c->Re^(2c)
"""
import jax
import equinox as eqx
import jax.numpy as jnp
import time

class V(eqx.Module):
    layers: list
    sobel_x_layers: list
    sobel_y_layers: list
    N_CHANNELS: int
    PERIODIC: bool
    DIM: int
    def __init__(self,N_CHANNELS,PERIODIC=True,DIM=2,dx=0.1,ZERO_INIT=True,key=jax.random.PRNGKey(int(time.time()))):
        key1,key2 = jax.random.split(key,2)
        self.N_CHANNELS = N_CHANNELS
        self.DIM = DIM
        self.PERIODIC = PERIODIC
        @jax.jit
        def periodic_pad(x):
            if self.PERIODIC:
                return jnp.pad(x, ((0,0),(1,1),(1,1)), mode='wrap')
            else:
                return x

        @jax.jit
        def periodic_unpad(x):
            if self.PERIODIC:
                return x[:,1:-1,1:-1]
            else:
                return x
        
        self.layers = [eqx.nn.Conv2d(in_channels=self.N_CHANNELS,
                                     out_channels=self.N_CHANNELS,
                                     kernel_size=1,
                                     use_bias=False,
                                     key=key1),
                       jax.nn.relu,
                       eqx.nn.Conv2d(in_channels=self.N_CHANNELS,
                                     out_channels=self.DIM*self.N_CHANNELS,
                                     kernel_size=1,
                                     use_bias=False,
                                     key=key2)]
        if ZERO_INIT:               
            w_zeros = jnp.zeros((self.DIM*self.N_CHANNELS,self.N_CHANNELS,1,1))
            w_where = lambda l: l.weight
            self.layers[-1] = eqx.tree_at(w_where,self.layers[-1],w_zeros)
        
        grad_x = jnp.outer(jnp.array([1.0,2.0,1.0]),jnp.array([-1.0,0.0,1.0])) /dx
        grad_y = grad_x.T
        
        kernel_dx = jnp.expand_dims(grad_x,(0,1))
        kernel_dx = jnp.repeat(kernel_dx,self.N_CHANNELS,axis=0)
        kernel_dy = jnp.expand_dims(grad_y,(0,1))
        kernel_dy = jnp.repeat(kernel_dy,self.N_CHANNELS,axis=0)
        
        self.sobel_x_layers = [periodic_pad,
                               eqx.nn.Conv2d(in_channels=self.N_CHANNELS,
                                             out_channels=self.N_CHANNELS,
                                             kernel_size=3,
                                             use_bias=False,
                                             key=key1,
                                             padding=1,
                                             groups=self.N_CHANNELS),
                               periodic_unpad]
        self.sobel_y_layers = [periodic_pad,
                               eqx.nn.Conv2d(in_channels=self.N_CHANNELS,
                                             out_channels=self.N_CHANNELS,
                                             kernel_size=3,
                                             use_bias=False,
                                             key=key1,
                                             padding=1,
                                             groups=self.N_CHANNELS),
                               periodic_unpad]
        w_where = lambda l: l.weight
        self.sobel_x_layers[1] = eqx.tree_at(w_where,self.sobel_x_layers[1],kernel_dx)
        self.sobel_y_layers[1] = eqx.tree_at(w_where,self.sobel_y_layers[1],kernel_dy)
        
        
    @eqx.filter_jit
    def f(self,X):
        for L in self.layers:
            X = L(X)
        return X
    @eqx.filter_jit
    def __call__(self,X):
        vx = self.f(X)
        vxx = jnp.tile(X,(self.DIM,1,1))*vx
        vxx_x = vxx[:self.N_CHANNELS]
        vxx_y = vxx[self.N_CHANNELS:]
        
        for L in self.sobel_x_layers:
            vxx_x = L(vxx_x)
        for L in self.sobel_y_layers:
            vxx_y = L(vxx_y)
    
        return vxx_x + vxx_y
        
    def partial(self,X,i):
        return self.f(X)[i*self.N_CHANNELS:(i+1)*self.N_CHANNELS]
    
    def partition(self):
        where_x = lambda m: m.sobel_x_layers[1].weight
        where_y = lambda m: m.sobel_y_layers[1].weight
        sobel_x = self.sobel_x_layers[1].weight
        sobel_y = self.sobel_y_layers[1].weight
        diff,static = eqx.partition(self,eqx.is_array)
        diff = eqx.tree_at(where_x,diff,None)
        diff = eqx.tree_at(where_y,diff,None)
        static = eqx.tree_at(where_x,static,sobel_x,is_leaf=lambda x: x is None)
        static = eqx.tree_at(where_y,static,sobel_y,is_leaf=lambda x: x is None)
        return diff, static
    def combine(self,diff,static):
        self = eqx.combine(diff,static)
        