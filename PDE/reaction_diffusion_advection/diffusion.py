#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:27:01 2023

@author: s1605376
"""
import jax
import equinox as eqx
import jax.numpy as jnp
import time

class D(eqx.Module):
    layers: list
    N_CHANNELS: int
    PERIODIC: bool
    def __init__(self,N_CHANNELS,PERIODIC,dx=0.1,key=jax.random.PRNGKey(int(time.time()))):
        self.N_CHANNELS = N_CHANNELS
        self.PERIODIC = PERIODIC
        key1,key2 = jax.random.split(key,2)
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
        
        KERNEL = jnp.array([[0.25,0.5,0.25],[0.5,-3,0.5],[0.25,0.5,0.25]]) / (dx*dx)
        KERNEL = jnp.expand_dims(KERNEL,(0,1))
        KERNEL = jnp.repeat(KERNEL,self.N_CHANNELS,axis=0)
        #print(KERNEL.shape)
        self.layers = [
            periodic_pad,
            eqx.nn.Conv2d(in_channels=self.N_CHANNELS,
                          out_channels=self.N_CHANNELS,
                          kernel_size=3,
                          use_bias=False,
                          key=key1,
                          padding=1,
                          groups=self.N_CHANNELS),
            periodic_unpad,
            eqx.nn.Conv2d(in_channels=self.N_CHANNELS,
                          out_channels=self.N_CHANNELS,
                          kernel_size=1,
                          use_bias=False,
                          key=key1,
                          groups=self.N_CHANNELS),
        ]
        w_where = lambda l: l.weight
        self.layers[1] = eqx.tree_at(w_where,self.layers[1],KERNEL)
        self.layers[-1]= eqx.tree_at(w_where,self.layers[-1],jnp.abs(self.layers[-1].weight)) # Initialise diffusion constants as positive
    @eqx.filter_jit
    def __call__(self,X):
        self.layers[-1]= eqx.tree_at(lambda l: l.weight,self.layers[-1],jnp.abs(self.layers[-1].weight))
        for L in self.layers:
            X = L(X)
        return X

    def partition(self):
        where = lambda m: m.layers[1].weight
        kernel = self.layers[1].weight
        diff,static = eqx.partition(self,eqx.is_array)
        diff = eqx.tree_at(where,diff,None)
        static = eqx.tree_at(where,static,kernel,is_leaf=lambda x: x is None)
        return diff, static
    
    def combine(self,diff,static):
        self = eqx.combine(diff,static)
