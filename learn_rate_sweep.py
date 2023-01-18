from NCA_class import *
from NCA_train import *
from NCA_utils import *
import numpy as np
import os 
import sys

"""
	Thoroughly check each combination of loss function and training algorithm, on both emoji and heat equation
"""


index=int(sys.argv[1])-1

#LEARN_RATE,LEARN_RATE_STRING,OPTIMIZER,TRAIN_MODE,NORM_GRADS = index_to_learnrate_parameters(index)
#LEARN_RATE,LEARN_RATE_STRING,RATIO,NORM_GRADS = index_to_Nadam_parameters(index)
LOSS_FUNC,LOSS_FUNC_STRING,SAMPLING,NORM_GRADS = index_to_mitosis_parameters(index)
LEARN_RATE = 1e-3
order=1
N_CHANNELS = 4
N_CHANNELS_PDE = 4
N_BATCHES = 4
OBS_CHANNELS=2
TRAIN_ITERS = 8000
OPTIMIZER="Nadam"
TRAIN_MODE="full"
BATCH_SIZE=64 # Split gradient updates into batches - computing gradient across all steps (~1000 timesteps) causes OOM errors on Eddie
NCA_WEIGHT_REG = 0.01
if NORM_GRADS:
	readif_filename="training_exploration/PDE_readif_"+OPTIMIZER+"_"+LOSS_FUNC_STRING+"_sampling_"+str(SAMPLING)+"_grad_norm"
else:
	readif_filename="training_exploration/PDE_readif_"+OPTIMIZER+"_"+LOSS_FUNC_STRING+"_sampling_"+str(SAMPLING)


#--- Reaction Diffusion equation ------------------------------------------------------------------------------

def F_readif_2(X,Xdx,Xdy,Xdd,D=[0.1,0.05],f=0.0367,k=0.0649):
	# Reaction diffusion as described in https://www.karlsims.com/rd.html

	ch_1 = D[0]*Xdd[...,0] - X[...,1]**2*X[...,0] + f*(1-X[...,0])
	ch_2 = D[1]*Xdd[...,1] + X[...,1]**2*X[...,0] - (k+f)*X[...,1]
	return tf.stack([ch_1,ch_2],-1)

ca_readif =NCA(N_CHANNELS_PDE,
			   FIRE_RATE=1,
			   ACTIVATION="swish",
			   OBS_CHANNELS=2,
			   REGULARIZER=NCA_WEIGHT_REG,
			   PADDING="periodic",
			   LAYERS=2,
			   KERNEL_TYPE="ID_LAP",
			   ORDER=order)

print(ca_readif)




x0 = np.ones((N_BATCHES,64,64,2)).astype(np.float32)

#x0[1,:32]=0
x0[0,24:40,24:40]=0
x0[1,16:24,16:24]=0
x0[1,48:56,48:56]=0
x0[1,10:30,34:54]=0
x0[1,34:54,10:30]=0
x0[2,30:34]=0
x0[2,40:44,30:34]=0
x0[2,20:24,24:40]=0


x0[3,4:24,16:24]=0
x0[3,42:46,40:60]=0
x0[3,16:24,40:48]=0
x0[3,40:48,16:24]=0

x0[...,1] = 1-x0[...,0]
trainer = NCA_PDE_Trainer(ca_readif,x0,F_readif_2,N_BATCHES,40*(16//SAMPLING),step_mul=SAMPLING,model_filename=readif_filename)
trainer.train_sequence(TRAIN_ITERS,
					   SAMPLING,
					   REG_COEFF=0.01,
					   LEARN_RATE=LEARN_RATE,
					   OPTIMIZER=OPTIMIZER,
					   TRAIN_MODE=TRAIN_MODE,
					   NORM_GRADS=NORM_GRADS,
					   LOSS_FUNC=LOSS_FUNC)