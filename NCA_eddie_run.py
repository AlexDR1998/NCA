from NCA_class import *
from NCA_utils import load_sequence_ensemble_average
import numpy as np
import sys
"""
	General purpose python script for training NCA on eddie.
"""
training_iters = int(sys.argv[1])
N_BATCHES = int(sys.argv[2])
N_CHANNELS= int(sys.argv[3])
filename = sys.argv[4]

data,mask = load_sequence_ensemble_average()#[:,:,:,::2,::2]
data = data[:,:,::2,::2]
mask = mask[:,::2,::2]

cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth = True
with tf.Session(config=cfg) as sess:
	ca = NCA(N_CHANNELS,ADHESION_MASK=mask)
	train_sequence(ca,data,N_BATCHES,training_iters,24,filename)