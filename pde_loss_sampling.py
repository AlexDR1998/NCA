from NCA.NCA_class import *
from NCA.trainer.NCA_trainer import *
from NCA.trainer.NCA_PDE_trainer import NCA_PDE_Trainer
from NCA.NCA_utils import *
from NCA.GOL_solver import GOL_solver
import numpy as np
import os 
import sys

"""
	Explore variations of NCA model for PDE and emoji problem, with training setup that works
"""

index=int(sys.argv[1])-1
N_CHANNELS_PDE = 8
N_BATCHES = 4
OBS_CHANNELS_PDE=2
TRAIN_ITERS = 4000
LEARN_RATE = 1e-3
BATCH_SIZE=64 # Split gradient updates into batches - computing gradient across all steps (~1000 timesteps) causes OOM errors on Eddie
NCA_WEIGHT_REG = 0.001
OPTIMIZER="Nadam"
KERNEL_TYPE="ID_LAP"
ACTIVATION = "relu"
LOSS_FUNC,LOSS_FUNC_STRING,PDE_SAMPLING = index_to_mitosis_parameters(index)	
PDE_STEPS=1024//PDE_SAMPLING

#FILENAME = "model_exploration/Nadam_euclidean_emoji_32_channels_32_sampling_"+KERNEL_TYPE


def F_readif_chem_mitosis(X,Xdx,Xdy,Xdd,D=[0.1,0.05],f=0.0367,k=0.0649):
	# Reaction diffusion as described in https://www.karlsims.com/rd.html
	ch_1 = D[0]*Xdd[...,0] - X[...,1]**2*X[...,0] + f*(1-X[...,0])
	ch_2 = D[1]*Xdd[...,1] + X[...,1]**2*X[...,0] - (k+f)*X[...,1]
	return tf.stack([ch_1,ch_2],-1)

def F_readif_chem_coral(X,Xdx,Xdy,Xdd,D=[0.1,0.05],f=0.06230,k=0.06258):
	# Reaction diffusion as described in https://www.karlsims.com/rd.html
	ch_1 = D[0]*Xdd[...,0] - X[...,1]**2*X[...,0] + f*(1-X[...,0])
	ch_2 = D[1]*Xdd[...,1] + X[...,1]**2*X[...,0] - (k+f)*X[...,1]
	return tf.stack([ch_1,ch_2],-1)

   


for i in range(4):
	
# 	#------------------------------------ Mitosis patterning experiment -------------------------------------------
# 	FILENAME = "trainer_validation/Nadam_mitosis_relu_8ch_ID_LAP_"+LOSS_FUNC_STRING+"_"+str(PDE_SAMPLING)+"_sampling_instance_"+str(i)
# 	ca_readif =NCA(N_CHANNELS_PDE,
# 	  			   FIRE_RATE=1,              
# 	  			   OBS_CHANNELS=2,
# 	   			   REGULARIZER=NCA_WEIGHT_REG,
# 	   			   KERNEL_TYPE=KERNEL_TYPE,
# 				   ACTIVATION=ACTIVATION)
# 	
# 	print(ca_readif)
# 	
# 	x0 = np.ones((N_BATCHES,64,64,2)).astype(np.float32)
# 	
# 	#x0[1,:32]=0
# 	x0[0,24:40,24:40]=0
# 	x0[1,16:24,16:24]=0
# 	x0[1,48:56,48:56]=0
# 	x0[1,10:30,34:54]=0
# 	x0[1,34:54,10:30]=0
# 	x0[2,30:34]=0
# 	x0[2,40:44,30:34]=0
# 	x0[2,20:24,24:40]=0
# 	
# 	
# 	x0[3,4:24,16:24]=0
# 	x0[3,42:46,40:60]=0
# 	x0[3,16:24,40:48]=0
# 	x0[3,40:48,16:24]=0
# 	
# 	x0[...,1] = 1-x0[...,0]
# 		
# 		
# 	trainer = NCA_PDE_Trainer(ca_readif, x0, F_readif_chem_mitosis, N_BATCHES, PDE_STEPS, step_mul=PDE_SAMPLING, model_filename=FILENAME)
# 	trainer.train_sequence(TRAIN_ITERS, PDE_SAMPLING, REG_COEFF=0.01, LEARN_RATE=LEARN_RATE, OPTIMIZER=OPTIMIZER, LOSS_FUNC=LOSS_FUNC)

	#---------------------------------- Coral patterning experiment ---------------------------------
	
	FILENAME = "trainer_validation/Nadam_coral_relu_8ch_ID_LAP_"+LOSS_FUNC_STRING+"_"+str(PDE_SAMPLING)+"_sampling_instance_"+str(i)
	ca_readif =NCA(N_CHANNELS_PDE,
	  			   FIRE_RATE=1,              
	  			   OBS_CHANNELS=2,
	   			   REGULARIZER=NCA_WEIGHT_REG,
	   			   KERNEL_TYPE=KERNEL_TYPE,
				   ACTIVATION=ACTIVATION)
	
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
		
		
	trainer = NCA_PDE_Trainer(ca_readif, x0, F_readif_chem_coral, N_BATCHES, PDE_STEPS, step_mul=PDE_SAMPLING, model_filename=FILENAME)
	trainer.train_sequence(TRAIN_ITERS, PDE_SAMPLING, REG_COEFF=0.01, LEARN_RATE=LEARN_RATE, OPTIMIZER=OPTIMIZER, LOSS_FUNC=LOSS_FUNC)


