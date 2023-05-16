from NCA.NCA_class import *
from NCA.trainer.NCA_trainer import *
from NCA.trainer.NCA_PDE_trainer import NCA_PDE_Trainer
from NCA.NCA_utils import *
from NCA.GOL_solver import GOL_solver
import numpy as np
import os 
import sys
from scipy.ndimage import rotate as sp_rotate
"""
	Explore variations of NCA model for PDE and emoji problem, with training setup that works
"""

index=int(sys.argv[1])-1
N_CHANNELS_EMOJI = 32
N_BATCHES = 8
TRAIN_ITERS = 8000
LEARN_RATE = 1e-3
NCA_WEIGHT_REG = 0.001
OPTIMIZER="Nadam"
TRAIN_MODE="full"
KERNEL_TYPE,TASK= index_to_emoji_symmetry_parameters(index)
EMOJI_SAMPLING = 32
ACTIVATION="relu"
LOSS_FUNC=None # euclidean
FILENAME = "model_exploration/Nadam_euclidean_emoji_"+TASK+"_2_layer_"+KERNEL_TYPE+"_relu_v1"
#FILENAME = "model_exploration/Nadam_euclidean_emoji_32_channels_32_sampling_"+KERNEL_TYPE



if TASK=="normal":
		
	data = load_emoji_sequence(["alien_monster.png","microbe.png","rooster_1f413.png","rooster_1f413.png"],downsample=2)
	ca = NCA(N_CHANNELS_EMOJI,
		     ACTIVATION=ACTIVATION,
			 REGULARIZER=NCA_WEIGHT_REG,
			 LAYERS=2,
			 KERNEL_TYPE=KERNEL_TYPE,
			 PADDING="zero")
	trainer = NCA_Trainer(ca,data,N_BATCHES,model_filename=FILENAME)
	trainer.data_pad_augment(10)
	trainer.data_shift_augment()
	trainer.data_noise_augment(0.0001)
	print(ca)
	trainer.train_sequence(TRAIN_ITERS,EMOJI_SAMPLING,LOSS_FUNC=LOSS_FUNC,OPTIMIZER=OPTIMIZER,LEARN_RATE=LEARN_RATE)
	
elif TASK=="rotated_data":
	data = load_emoji_sequence(["alien_monster.png","microbe.png","rooster_1f413.png","rooster_1f413.png"],downsample=2)
	WIDTH = 5
	padwidth = ((0,0),(0,0),(WIDTH,WIDTH),(WIDTH,WIDTH),(0,0))
	data = np.pad(data,padwidth)
	ca = NCA(N_CHANNELS_EMOJI,
		     ACTIVATION=ACTIVATION,
			 REGULARIZER=NCA_WEIGHT_REG,
			 LAYERS=2,
			 KERNEL_TYPE=KERNEL_TYPE,
			 PADDING="zero")
	trainer = NCA_Trainer(ca,data,N_BATCHES,model_filename=FILENAME)
	trainer.data_pad_augment(5)
	trainer.data_flip_augment()
	trainer.data_rotate_augment()
	trainer.data_shift_augment()
	trainer.data_noise_augment(0.0001)
	print(ca)
	trainer.train_sequence(TRAIN_ITERS,EMOJI_SAMPLING,LOSS_FUNC=LOSS_FUNC,OPTIMIZER=OPTIMIZER,LEARN_RATE=LEARN_RATE)
	
elif TASK=="rotation_task":
	data = load_emoji_sequence(["mushroom_1f344.png","mushroom_1f344.png","mushroom_1f344.png"])
	WIDTH = 10
	padwidth = ((0,0),(0,0),(WIDTH,WIDTH),(WIDTH,WIDTH),(0,0))
	data = np.pad(data,padwidth)
	data[1] = sp_rotate(data[1],angle=45,axes=(1,2),reshape = False)
	data[2] = sp_rotate(data[2],angle=90,axes=(1,2),reshape = False)
	data = data[:,:,::2,::2]
	ca = NCA(N_CHANNELS_EMOJI,
		     ACTIVATION=ACTIVATION,
			 REGULARIZER=NCA_WEIGHT_REG,
			 LAYERS=2,
			 KERNEL_TYPE=KERNEL_TYPE,
			 PADDING="zero")
	trainer = NCA_Trainer(ca,data,N_BATCHES,model_filename=FILENAME)
	trainer.data_pad_augment(10)
	trainer.data_rotate_augment()
	trainer.data_shift_augment()
	trainer.data_noise_augment(0.0001)
	print(ca)
	trainer.train_sequence(TRAIN_ITERS,EMOJI_SAMPLING,LOSS_FUNC=LOSS_FUNC,OPTIMIZER=OPTIMIZER,LEARN_RATE=LEARN_RATE)
	
elif TASK=="translation_task":
	data = load_emoji_sequence(["rooster_1f413.png","rooster_1f413.png","rooster_1f413.png"],downsample=2)
	WIDTH = 20
	padwidth = ((0,0),(0,0),(WIDTH,WIDTH),(WIDTH,WIDTH),(0,0))
	data = np.pad(data,padwidth)
	data[0] = np.roll(data[0],16,axis=2)
	data[2] = np.roll(data[2],-16,axis=2)
	ca = NCA(N_CHANNELS_EMOJI,
		     ACTIVATION=ACTIVATION,
			 REGULARIZER=NCA_WEIGHT_REG,
			 LAYERS=2,
			 KERNEL_TYPE=KERNEL_TYPE,
			 PADDING="periodic")
	trainer = NCA_Trainer(ca,data,N_BATCHES,model_filename=FILENAME)
	trainer.data_pad_augment(0)
	trainer.data_rotate_augment()
	trainer.data_noise_augment(0.0001)
	print(ca)
	trainer.train_sequence(TRAIN_ITERS,EMOJI_SAMPLING,LOSS_FUNC=LOSS_FUNC,OPTIMIZER=OPTIMIZER,LEARN_RATE=LEARN_RATE)