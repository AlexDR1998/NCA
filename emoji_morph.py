from NCA_class import *
from NCA_train import *
from NCA_utils import *
#from NCA_visualise import *
import numpy as np
import os 
import sys


N_CHANNELS=16
N_BATCHES=4
index=int(sys.argv[1])


def train_emoji_sequence(filename_sequence,model_filename,downsample=2):
	data = load_emoji_sequence(filename_sequence,downsample)
	ca = NCA(N_CHANNELS)
	trainer = NCA_Trainer(ca,data,N_BATCHES,model_filename=model_filename)
	trainer.train_sequence(4000,60)


def main():
	if index==1:
		train_emoji_sequence(["mushroom_1f344.png",
							  "lizard_1f98e.png",
							  "rooster_1f413.png"],
							  "emoji_sequence_mushroom_lizard_rooster_eddie")
	if index==2:
		train_emoji_sequence(["skull_1f480.png",
							  "rainbow_1f308.png",
							  "rainbow_1f308.png"],
							  "emoji_sequence_skull_rainbow_stable_eddie")
	if index==3:
		train_emoji_sequence(["skull_1f480.png",
							  "rainbow_1f308.png",
							  "skull_1f480.png"],
							  "emoji_sequence_skull_rainbow_skull_eddie")

main()