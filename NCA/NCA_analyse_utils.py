import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import numpy as np 
from NCA_utils import *
from tqdm import tqdm
import tensorflow as tf 

"""
	Helper functions for plotting / analysing stuff in NCA_analyse

"""


def plot_weight_matrices(NCA):
	# Given an NCA, return heatmaps of its weight matrices
	weights = NCA.dense_model.get_weights()
	L = len(weights)-1
	fig,ax = plt.subplots(1,L)
	print(weights)
	print(L)
	for i in range(L):
		print(weights[i].shape)
		ax[i].imshow(weights[i][0,0])
	plt.show()