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
N_BATCHES = 4
OBS_CHANNELS=1
TRAIN_ITERS = 8000

emoji_filename = "training_exploration/emoji_alien_monster_rooster_stable_"+OPTIMIZER+"_"+LOSS_FUNC_STRING
heat_filename = "training_exploration/PDE_heat_eq_"+OPTIMIZER+"_"+LOSS_FUNC_STRING



#--- Emoji morph alien->rooster stable ------------------------------------------------------------------------

ca_emoji = NCA(N_CHANNELS,
			   ACTIVATION="swish",
			   REGULARIZER=0.1,
			   LAYERS=2,
			   KERNEL_TYPE="ID_LAP")

print(ca_emoji)

emoji_data = load_emoji_sequence(["alien_monster.png","rooster_1f413.png","rooster_1f413.png"],downsample=2)
trainer_emoji = NCA_Trainer(ca_emoji,emoji_data,N_BATCHES,model_filename=emoji_filename)
trainer_emoji.data_pad_augment(2,2)
trainer_emoji.data_noise_augment(0.001)
trainer_emoji.train_sequence(TRAIN_ITERS,60,LOSS_FUNC=LOSS_FUNC,OPTIMIZER=OPTIMIZER)






#--- Heat equation -----------------------------------------------------------------------------------------------

def F_heat(X,Xdx,Xdy,Xdd,D=1):
	return D*Xdd

ca_heat =NCA(N_CHANNELS,
			 ACTIVATION="swish",
			 OBS_CHANNELS=OBS_CHANNELS,
			 REGULARIZER=0.1,
			 PADDING="periodic",
			 LAYERS=2,
			 KERNEL_TYPE="ID_LAP")

print(ca_heat)

x0 = np.random.uniform(size=(N_BATCHES,64,64,OBS_CHANNELS)).astype(np.float32)
x0[1,:32]=0
mask=np.zeros((64,64,OBS_CHANNELS)).astype(int)
mask[16:48,16:48]=1
x0[2]*=mask
x0[3]*=(1-mask)
trainer = NCA_PDE_Trainer(ca_heat,x0,F_heat,N_BATCHES,100,model_filename=heat_filename)
trainer.train_sequence(TRAIN_ITERS,1,LOSS_FUNC=LOSS_FUNC,OPTIMIZER=OPTIMIZER)