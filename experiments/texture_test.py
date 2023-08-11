from NCA.NCA_class import *
from NCA.trainer.NCA_trainer import *
from NCA.NCA_utils import *
from NCA.NCA_visualise import *
import numpy as np
import os 
import sys
#import matplotlib.pyplot as plt
os.chdir("..")
N_CHANNELS=16
N_BATCHES=2


def train_texture(filename_sequence,model_filename,downsample=8,LOSS_FUNC=loss_vgg,OPTIMIZER="Nadam",KERNEL_TYPE="ID_DIFF_LAP"):
	data = load_emoji_sequence(filename_sequence,impath_emojis="../Data/dtd/images/",downsample=downsample,crop_square=True)
	print(data)
	ca = NCA(N_CHANNELS,ACTIVATION="relu",REGULARIZER=0.1,LAYERS=2,KERNEL_TYPE=KERNEL_TYPE,OBS_CHANNELS=3)
	trainer = NCA_Trainer(ca,data,N_BATCHES,CYCLIC=True,model_filename=model_filename)
	#trainer.data_pad_augment(10)
	#trainer.data_shift_augment()
	trainer.data_noise_augment(1)
	
	print(ca)
	trainer.train_sequence(200,32,LOSS_FUNC=LOSS_FUNC,OPTIMIZER=OPTIMIZER,LEARN_RATE=2e-3,INJECT_TRUE=False)


train_texture(["striped/striped_0001.jpg","striped/striped_0001.jpg"], "texture_test")