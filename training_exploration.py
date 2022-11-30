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
LOSS_FUNC,OPTIMIZER,LOSS_FUNC_STRING = index_to_trainer_parameters(index)

N_CHANNELS = 16
N_CHANNELS_PDE = 8
N_BATCHES = 4
OBS_CHANNELS=1
TRAIN_ITERS = 8000
multiplier=20 # Compare PDE and NCA every 20 steps - if compared at every step, too much ram is used

emoji_filename ="training_exploration/emoji_alien_monster_rooster_stable_"+OPTIMIZER+"_"+LOSS_FUNC_STRING
heat_filename = "training_exploration/PDE_heat_eq_"+OPTIMIZER+"_"+LOSS_FUNC_STRING
readif_filename="training_exploration/PDE_readif_"+OPTIMIZER+"_"+LOSS_FUNC_STRING


#--- Emoji morph alien->rooster stable ------------------------------------------------------------------------

ca_emoji = NCA(N_CHANNELS,
			   ACTIVATION="swish",
			   REGULARIZER=0.1,
			   LAYERS=2,
			   KERNEL_TYPE="ID_LAP")

print(ca_emoji)

emoji_data = load_emoji_sequence(["alien_monster.png","rooster_1f413.png","rooster_1f413.png"],downsample=2)
trainer_emoji = NCA_Trainer(ca_emoji,emoji_data,N_BATCHES,model_filename=emoji_filename)
trainer_emoji.data_pad_augment(2,10)
trainer_emoji.data_noise_augment(0.001)
trainer_emoji.train_sequence(TRAIN_ITERS,60,LOSS_FUNC=LOSS_FUNC,OPTIMIZER=OPTIMIZER)






#--- Heat equation -----------------------------------------------------------------------------------------------


def F_heat(X,Xdx,Xdy,Xdd,D=0.33):
	return D*Xdd

ca_heat =NCA(N_CHANNELS_PDE,
			 FIRE_RATE=1,
			 ACTIVATION="swish",
			 OBS_CHANNELS=OBS_CHANNELS,
			 REGULARIZER=0.1,
			 PADDING="periodic",
			 LAYERS=2,
			 KERNEL_TYPE="ID_LAP")

print(ca_heat)

x0 = np.random.uniform(size=(N_BATCHES,64,64,OBS_CHANNELS)).astype(np.float32)
x0[0,24:40,24:40]=1
x0[1,30:34]=1
x0[2,30:34]=1
x0[2,40:44,30:34]=0
x0[2,20:24,24:40]=0

x0[3,4:24,16:24]=0
x0[3,42:46,40:60]=0
x0[3,16:24,40:48]=1
x0[3,40:48,16:24]=1
trainer = NCA_PDE_Trainer(ca_heat,x0,F_heat,N_BATCHES,100,step_mul=multiplier,model_filename=heat_filename)
trainer.train_sequence(TRAIN_ITERS,multiplier,LOSS_FUNC=LOSS_FUNC,OPTIMIZER=OPTIMIZER)



#--- Reaction Diffusion equation ------------------------------------------------------------------------------

def F_readif_2(X,Xdx,Xdy,Xdd,D=[0.2,0.05],f=0.061,k=0.06264):
	# Reaction diffusion as described in https://www.karlsims.com/rd.html

	ch_1 = D[0]*Xdd[...,0] - X[...,1]**2*X[...,0] + f*(1-X[...,0])
	ch_2 = D[1]*Xdd[...,1] + X[...,1]**2*X[...,0] - (k+f)*X[...,1]
	return tf.stack([ch_1,ch_2],-1)

ca_readif =NCA(N_CHANNELS_PDE,
			   FIRE_RATE=1,
			   ACTIVATION="swish",
			   OBS_CHANNELS=2,
			   REGULARIZER=0.1,
			   PADDING="periodic",
			   LAYERS=2,
			   KERNEL_TYPE="ID_LAP")

print(ca_readif)




x0 = np.ones((N_BATCHES,64,64,2)).astype(np.float32)

#x0[1,:32]=0
x0[0,24:40,24:40]=0
x0[1,30:34]=0
x0[2,30:34]=0
x0[2,40:44,30:34]=0
x0[2,20:24,24:40]=0

x0[3,4:24,16:24]=0
x0[3,42:46,40:60]=0
x0[3,16:24,40:48]=0
x0[3,40:48,16:24]=0

x0[...,1] = 1-x0[...,0]
trainer = NCA_PDE_Trainer(ca_readif,x0,F_readif_2,N_BATCHES,100,step_mul=multiplier,model_filename=readif_filename)
trainer.train_sequence(TRAIN_ITERS,multiplier,LOSS_FUNC=LOSS_FUNC,OPTIMIZER=OPTIMIZER)