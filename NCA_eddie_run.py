from NCA_class import *
from NCA_train import *
from NCA_utils import load_sequence_ensemble_average
import numpy as np
import sys
"""
	General purpose python script for training NCA on eddie.

	System Arguments
	----------------
	training_iters : int
		How many training iterations
	N_BATCHES : int
		How many batches for training
	N_CHANNELS : int 
		How many channels for the NCA (must be at least 4/5 without/with mask)
	FIRE_RATE : int
		Stochastic fire rate - give nubmer between 1 and 100, is then divided by 100
		Basically just to avoid doing maths in bash script

"""
training_iters = int(sys.argv[1])
N_BATCHES = int(sys.argv[2])
N_CHANNELS= int(sys.argv[3])
FIRE_RATE = float(sys.argv[4])/20.0
filename = sys.argv[5]

data,mask = load_sequence_ensemble_average()#[:,:,:,::2,::2]
data = data[:,:,::2,::2]
mask = mask[:,::2,::2]

#cfg = tf.ConfigProto()
#cfg.gpu_options.allow_growth = True
#with tf.Session(config=cfg) as sess:
ca = NCA(N_CHANNELS,FIRE_RATE=FIRE_RATE,ADHESION_MASK=mask)
train_sequence(ca,data,N_BATCHES,training_iters,24,filename)