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
		Stochastic fire rate - give nubmer between 1 and 20, is then divided by 20
		Basically just to avoid doing maths in bash script
	DECAY_FACTOR : int
		Hidden channel decay rate - similar to above, give int between 1 and 20
		as cba doing bash maths
"""
training_iters = int(sys.argv[1])
N_BATCHES = int(sys.argv[2])
N_CHANNELS= int(sys.argv[3])
FIRE_RATE = float(sys.argv[4])/20.0
DECAY_FACTOR = float(sys.argv[5])/20.0
N_MODELS = 4 # How many models to train for given parameters
filename = sys.argv[6]

data,mask = load_sequence_ensemble_average()#[:,:,:,::2,::2]
data = data[:,:,::2,::2]
mask = mask[:,::2,::2]

#cfg = tf.ConfigProto()
#cfg.gpu_options.allow_growth = True
#with tf.Session(config=cfg) as sess:
print("Training NCA with parameters:")
ca = NCA(N_CHANNELS,FIRE_RATE=FIRE_RATE,DECAY_FACTOR=DECAY_FACTOR,ADHESION_MASK=mask)
print(ca)
print("Saving to "+str(filename))
for i in range(N_MODELS):
	trainer = NCA_Trainer(ca,data,N_BATCHES,filename+"_b"+str(i))
	trainer.train_sequence(training_iters,24,filename+"_b"+str(i))
#train_sequence(ca,data,N_BATCHES,training_iters,24,filename)