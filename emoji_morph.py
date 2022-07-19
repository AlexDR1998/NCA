from NCA_class import *
from NCA_train import *
from NCA_utils import *
#from NCA_visualise import *
import numpy as np
import os 
import sys


N_CHANNELS=16
N_BATCHES=4
suffix=str(sys.argv[1])


def train_emoji_sequence(filename_sequence,model_filename):
	data = load_emoji_sequence(filename_sequence,1)
	ca = NCA(N_CHANNELS)
	trainer = NCA_Trainer(ca,data,N_BATCHES,model_filename=model_filename)
	trainer.train_sequence(4000,120)


def main():
	train_emoji_sequence(["mushroom_1f344.png",
						  "lizard_1f98e.png",
						  "rooster_1f413.png"],
						  "emoji_sequence_mushroom_lizard_highres_"+suffix)


main()