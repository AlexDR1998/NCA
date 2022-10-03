from NCA_class import *
from NCA_train import *
from NCA_utils import *
import numpy as np
import os 
import sys


N_CHANNELS=16
N_BATCHES=4
index=int(sys.argv[1])


def train_emoji_sequence(filename_sequence,model_filename,downsample=2):
	data = load_emoji_sequence(filename_sequence,downsample)
	ca = NCA(N_CHANNELS,ACTIVATION="swish",REGULARIZER=0.1)
	trainer = NCA_Trainer(ca,data,N_BATCHES,model_filename=model_filename)
	trainer.data_pad_augment(1,15)
	trainer.data_noise_augment()
	trainer.train_sequence(8000,60)


def main():
	if index==1:
		train_emoji_sequence(["mushroom_1f344.png",
							  "lizard_1f98e.png",
							  "rooster_1f413.png"],
							  "emoji_sequence_swish_noise_pad_augment_mushroom_lizard_rooster_eddie")
	
	if index==2:
		train_emoji_sequence(["crab.png",
							  "alien_monster.png",
							  "butterfly.png"],
							  "emoji_sequence_swish_noise_pad_augment_crab_alien_butterfly_eddie")
	
	if index==3:
		train_emoji_sequence(["butterfly.png",
							  "microbe.png",
							  "eye.png"],
							  "emoji_sequence_swish_noise_pad_augment_butterfly_microbe_eye_eddie")

	#make_video_file("emoji_sequence_1layer_skull_rainbow_skull_eddie")
	#visualise_distance_to_target("emoji_sequence_mushroom_lizard_rooster_eddie")



	
main()