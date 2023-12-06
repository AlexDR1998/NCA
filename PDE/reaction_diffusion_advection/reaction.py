#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:25:27 2023

@author: Alex Richardson

Function that handles the reaction bit. R: Re^c->Re^c
"""
import jax
import equinox as eqx
import jax.numpy as jnp
import time


class R(eqx.Module):
    layers: list
    N_CHANNELS: int
    
    def __init__(self,N_CHANNELS,ZERO_INIT=True,key=jax.random.PRNGKey(int(time.time()))):
        key1,key2 = jax.random.split(key,2)
        self.N_CHANNELS = N_CHANNELS
        self.layers = [eqx.nn.Conv2d(in_channels=self.N_CHANNELS,
                                     out_channels=self.N_CHANNELS,
                                     kernel_size=1,
                                     use_bias=False,
                                     key=key1),
                       jax.nn.relu,
                       eqx.nn.Conv2d(in_channels=self.N_CHANNELS,
                                     out_channels=self.N_CHANNELS,
                                     kernel_size=1,
                                     use_bias=False,
                                     key=key2)]
        
        if ZERO_INIT:               
            w_zeros = jnp.zeros((self.N_CHANNELS,self.N_CHANNELS,1,1))
            w_where = lambda l: l.weight
            self.layers[-1] = eqx.tree_at(w_where,self.layers[-1],w_zeros)
    @eqx.filter_jit
    def __call__(self,X):
        for L in self.layers:
            X = L(X)
        return X
    def partition(self): 
        return eqx.partition(self,eqx.is_array)
    def combine(self,diff,static):
        self = eqx.combine(diff,static)
        