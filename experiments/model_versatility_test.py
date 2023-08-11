from NCA.NCA_class import *
from NCA.trainer.NCA_trainer import *
from NCA.trainer.NCA_PDE_trainer import NCA_PDE_Trainer
from NCA.NCA_utils import *
from NCA.GOL_solver import GOL_solver
import numpy as np
import os 
import sys
os.chdir("..")
"""
	Check if the ideal model / training parameter combination for reaction diffusion works on other PDEs / emoji problem
"""

index=int(sys.argv[1])-1
N_CHANNELS_EMOJI = 16
N_CHANNELS_PDE = 8
N_CHANNELS_GOL = 16
N_BATCHES = 4
OBS_CHANNELS_PDE=2
TRAIN_ITERS = 8000
LEARN_RATE = 1e-3
BATCH_SIZE=64 # Split gradient updates into batches - computing gradient across all steps (~1000 timesteps) causes OOM errors on Eddie
NCA_WEIGHT_REG = 0.001
OPTIMIZER="Nadam"
ACTIVATION ="swish"
TRAIN_MODE="full"
LOSS_FUNC,LOSS_FUNC_STRING,SAMPLING,TASK = index_to_generalise_test_2()
PDE_STEPS=1024//SAMPLING
FILENAME = "trainer_validation/Nadam_"+LOSS_FUNC_STRING+"_sampling_"+str(SAMPLING)+"_"+TASK+"_v3"



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

def F_readif_act(X,Xdx,Xdy,Xdd,D=[0.014,0.4],s=[0.01,0.01],k=[0,1],kd=[0.001,0.008]):
	# Reaction diffusion equaiton of signalin activator/inhibitor dynamics in Chhabra et al 2019
	a_sq = X[...,0]**2
	ch_1 = D[0]*Xdd[...,0] + s[0]*a_sq / ((1+k[1]*X[...,1])*(1+k[0]*a_sq)) - kd[0]*X[...,0]
	ch_2 = D[1]*Xdd[...,1] + s[1]*a_sq - kd[1]*X[...,1]
	return tf.stack([ch_1,ch_2],-1)

def F_heat(X,Xdx,Xdy,Xdd,D=0.33):
	return D*Xdd




if TASK=="heat":
	#--------------------------------- Heat Equation ---------------------------------------------
    ca_heat = NCA(N_CHANNELS_PDE,
				  FIRE_RATE=1,
				  OBS_CHANNELS=OBS_CHANNELS_PDE,
				  REGULARIZER=NCA_WEIGHT_REG,
				  KERNEL_TYPE="ID_LAP",
				  ACTIVATION=ACTIVATION)
    print(ca_heat)    
    x0 = np.random.uniform(size=(N_BATCHES,64,64,OBS_CHANNELS_PDE)).astype(np.float32)
    x0[0,24:40,24:40]=1
    x0[1,30:34]=1
    x0[2,30:34]=1
    x0[2,40:44,30:34]=0
    x0[2,20:24,24:40]=0

    x0[3,4:24,16:24]=0
    x0[3,42:46,40:60]=0
    x0[3,16:24,40:48]=1
    x0[3,40:48,16:24]=1
    trainer = NCA_PDE_Trainer(ca_heat, x0, F_heat, N_BATCHES, T=PDE_STEPS, step_mul=SAMPLING, model_filename=FILENAME)
    trainer.train_sequence(TRAIN_ITERS, SAMPLING, REG_COEFF=0.01, LEARN_RATE=LEARN_RATE, OPTIMIZER=OPTIMIZER, TRAIN_MODE=TRAIN_MODE, NORM_GRADS=True, LOSS_FUNC=LOSS_FUNC)	
    
elif TASK=="mitosis":
    ca_readif =NCA(N_CHANNELS_PDE,
    			   FIRE_RATE=1,
    			   OBS_CHANNELS=2,
    			   REGULARIZER=NCA_WEIGHT_REG,
    			   KERNEL_TYPE="ID_LAP",
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
    trainer = NCA_PDE_Trainer(ca_readif, x0, F_readif_chem_mitosis, N_BATCHES, PDE_STEPS, step_mul=SAMPLING, model_filename=FILENAME)
    trainer.train_sequence(TRAIN_ITERS, SAMPLING, REG_COEFF=0.01, LEARN_RATE=LEARN_RATE, OPTIMIZER=OPTIMIZER, TRAIN_MODE=TRAIN_MODE, NORM_GRADS=True, LOSS_FUNC=LOSS_FUNC)

elif TASK=="coral":
    ca_readif =NCA(N_CHANNELS_PDE,
    			   FIRE_RATE=1,
    			   OBS_CHANNELS=2,
    			   REGULARIZER=NCA_WEIGHT_REG,
    			   KERNEL_TYPE="ID_LAP",
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
    trainer = NCA_PDE_Trainer(ca_readif, x0, F_readif_chem_coral, N_BATCHES, PDE_STEPS, step_mul=SAMPLING, model_filename=FILENAME)
    trainer.train_sequence(TRAIN_ITERS, SAMPLING, REG_COEFF=0.01, LEARN_RATE=LEARN_RATE, OPTIMIZER=OPTIMIZER, TRAIN_MODE=TRAIN_MODE, NORM_GRADS=True, LOSS_FUNC=LOSS_FUNC)

elif TASK=="gol":
	
	#--- Note that time sampling is different for GoL. Instead of matching one
	#    NCA update with one GoL update, we match 1 GoL to n NCA updates
	gol = GOL_solver(N_BATCHES,[64,64])
	data = gol.run(SAMPLING)[...,None]
	ca_gol = NCA(N_CHANNELS_GOL,
			     FIRE_RATE=1,
				 REGULARIZER=0,
				 OBS_CHANNELS=1,
				 KERNEL_TYPE="GOL",
				 ACTIVATION=ACTIVATION)
	trainer = NCA_Trainer(ca_gol,
					      data, 
						  N_BATCHES, 
						  model_filename=FILENAME)
	trainer.train_sequence(TRAIN_ITERS, 
						   SAMPLING, 
						   REG_COEFF=0.01, 
						   LEARN_RATE=LEARN_RATE, 
						   OPTIMIZER=OPTIMIZER, 						            
						   NORM_GRADS=True, 
						   LOSS_FUNC=LOSS_FUNC)
elif TASK=="emoji":
	data = load_emoji_sequence(["alien_monster.png","butterfly.png","rooster_1f413.png","rooster_1f413.png"],downsample=2)
	print(data)
	ca = NCA(N_CHANNELS_EMOJI,
		     ACTIVATION=ACTIVATION,
			 REGULARIZER=NCA_WEIGHT_REG,
			 LAYERS=2,
			 KERNEL_TYPE="ID_LAP_AV",
			 PADDING="zero")
	trainer = NCA_Trainer(ca,data,N_BATCHES,model_filename=FILENAME)
	trainer.data_pad_augment(2,10)
	trainer.data_noise_augment(0.001)
	print(ca)
	trainer.train_sequence(TRAIN_ITERS,SAMPLING,LOSS_FUNC=LOSS_FUNC,OPTIMIZER=OPTIMIZER,LEARN_RATE=LEARN_RATE)