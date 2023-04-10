from NCA.NCA_class import *
from NCA.trainer.NCA_trainer import *
from NCA.NCA_utils import *
from NCA.NCA_visualise import *
import numpy as np
import os 
import sys
#import matplotlib.pyplot as plt

N_CHANNELS=14
N_BATCHES=2
index=int(sys.argv[1])


def train_emoji_sequence(filename_sequence,model_filename,downsample=2,LOSS_FUNC=None,OPTIMIZER="Nadam",KERNEL_TYPE="ID_LAP_AV"):
	data = load_emoji_sequence(filename_sequence,downsample=downsample)
	print(data)
	ca = NCA(N_CHANNELS,ACTIVATION="swish",REGULARIZER=0.1,LAYERS=2,KERNEL_TYPE=KERNEL_TYPE,PADDING="zero")
	trainer = NCA_Trainer(ca,data,N_BATCHES,model_filename=model_filename)
	trainer.data_pad_augment(2,10)
	trainer.data_noise_augment(0.001)
	print(ca)
	trainer.train_sequence(10,60,LOSS_FUNC=LOSS_FUNC,OPTIMIZER=OPTIMIZER,LEARN_RATE=2e-3)


def train_emoji_sequence_masked(filename_sequence,model_filename,downsample=2,LOSS_FUNC=None,OPTIMIZER="Nadam",KERNEL_TYPE="ID_LAP_AV"):
	data = load_emoji_sequence(filename_sequence,downsample=downsample)
	mask = np.zeros((N_BATCHES*(data.shape[0]-1),data.shape[2],data.shape[3],2))
	mask[:,::4,:,1]=1
	mask[:,:,::4,0]=1
	ca = NCA(N_CHANNELS,ACTIVATION="swish",REGULARIZER=0.1,LAYERS=2,KERNEL_TYPE=KERNEL_TYPE,PADDING="zero",ENV_CHANNEL_DATA=mask)
	
	trainer = NCA_Trainer(ca,data,N_BATCHES,model_filename=model_filename)
	#trainer.data_pad_augment(2,10)
	trainer.data_noise_augment(0.001)
	print(ca)
	trainer.train_sequence(5,60,LOSS_FUNC=LOSS_FUNC,OPTIMIZER=OPTIMIZER,LEARN_RATE=2e-3)
	
	ca2 = load_wrapper(model_filename)
	tr = ca2.run(data[0],T=360)
	my_animate(tr[:,0,...,:4])

def prune_emoji_sequence(filename_sequence,model_filename,downsample=2,LOSS_FUNC=None,OPTIMIZER="Nadam"):
	data = load_emoji_sequence(filename_sequence,downsample=downsample)
	ca = load_wrapper(model_filename)
	trainer = NCA_Trainer(ca,data,N_BATCHES,model_filename=model_filename+"_pruned")
	trainer.data_pad_augment(2,10)
	trainer.data_noise_augment(0.001)
	trainer.train_sequence(100,60,LOSS_FUNC=LOSS_FUNC,OPTIMIZER=OPTIMIZER,LEARN_RATE=2e-3,PRUNE_MODEL=True,SPARSITY=0.7)

def train_emoji_pairs(initial_filenames,target_filenames,model_filename,downsample=2):
	"""
		Trains one NCA that maps each entry in initial_filenames to the corresponding entry in target_filenames
	"""
	x0 = load_emoji_sequence(initial_filenames,downsample)
	xT = load_emoji_sequence(target_filenames,downsample)
	x0 = np.swapaxes(x0,0,1)
	xT = np.swapaxes(xT,0,1)
	data = np.concatenate((x0,xT),axis=0)

	ca = NCA(N_CHANNELS,ACTIVATION="swish",REGULARIZER=0.1)
	trainer = NCA_Trainer(ca,data,N_BATCHES=data.shape[1],model_filename=model_filename)
	trainer.data_pad_augment(4,15)
	trainer.data_noise_augment()
	trainer.train_sequence(8000,60)	

def train_denoise(filenames,AMOUNT,model_filename,downsample=2):
	"""
		Incrementaly denoises images
		
		Parameters
		----------
		
		AMOUNT : float (0,1)
			How much noise to remove at each iteration

	"""
	noise_steps = int(1/AMOUNT)
	xT = load_emoji_sequence(filenames,downsample)
	xT = np.swapaxes(xT,0,1)
	data = np.zeros((noise_steps,xT.shape[1],xT.shape[2],xT.shape[3],xT.shape[4]))
	for i in range(noise_steps):
		noise = np.random.uniform(size=xT.shape)
		signal_strength = AMOUNT*i
		data[i] = (1-signal_strength)*noise + signal_strength*xT
		#plt.imshow(data[i,1])
		#plt.show()

	ca = NCA(N_CHANNELS,ACTIVATION="swish",REGULARIZER=0.1)
	trainer = NCA_Trainer(ca,data,N_BATCHES=data.shape[1],model_filename=model_filename)
	trainer.train_sequence(8000,24)

def main():
	if index==1:
		train_emoji_sequence(["mushroom_1f344.png",
							  "lizard_1f98e.png",
							  "rooster_1f413.png"],
							  "emoji_sequence_euclidean_mushroom_lizard_rooster")
	
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
	if index==4:
		train_emoji_pairs(["butterfly.png","microbe.png"],
						  ["lizard_1f98e.png","alien_monster.png"],
						  "emoji_pairs_swish_butterfly_microbe")
	
	if index==5:
		train_emoji_pairs(["butterfly.png","rooster_1f413.png"],
						  ["lizard_1f98e.png","mushroom_1f344.png"],
						  "emoji_pairs_swish_butterfly_rooster")

	
	if index==6:
		train_emoji_pairs(["microbe.png","rooster_1f413.png"],
						  ["alien_monster.png","mushroom_1f344.png"],
						  "emoji_pairs_swish_microbe_rooster")
	if index==7:
		train_emoji_sequence(["lizard_1f98e.png","rooster_1f413.png"],
							 "emoji_lizard_rooster_euclidean_adagrad_ID_LAP_AV")
	
	if index==8:
		train_emoji_sequence(["alien_monster.png","rooster_1f413.png","rooster_1f413.png"],
							 "emoji_alien_monster_rooster_wasserstein_channels_adam",
							 LOSS_FUNC=loss_sliced_wasserstein_channels)

	if index==9:
		train_emoji_sequence(["alien_monster.png","rooster_1f413.png","rooster_1f413.png"],
							 "emoji_alien_monster_rooster_wasserstein_grids_adam",
							 LOSS_FUNC=loss_sliced_wasserstein_grid)

	if index==10:
		train_emoji_sequence(["alien_monster.png","rooster_1f413.png","rooster_1f413.png"],
							 "emoji_alien_monster_rooster_wasserstein_rotations_adam",
							 LOSS_FUNC=loss_sliced_wasserstein_rotate)
	if index==11:
		train_emoji_sequence(["alien_monster.png","rooster_1f413.png","rooster_1f413.png"],
							 "emoji_alien_monster_rooster_spectral_adam",
							 LOSS_FUNC=loss_spectral)
	if index==12:
		train_emoji_sequence(["alien_monster.png","rooster_1f413.png","rooster_1f413.png"],
							 "emoji_alien_monster_rooster_bhattacharyya_adam",
							 LOSS_FUNC=loss_bhattacharyya)
	if index==13:
		train_emoji_sequence_masked(["alien_monster.png","rooster_1f413.png","rooster_1f413.png"],
							 "emoji_masked_debug")
	if index==14:
		prune_emoji_sequence(["alien_monster.png","rooster_1f413.png","rooster_1f413.png"],
					         "emoji_alien_monster_rooster_euclidean_nadam_stable")
	






	"""
	if index==6:
		train_emoji_sequence(["crab.png",
							  "alien_monster.png",
							  "butterfly.png",
							  "butterfly.png"],
							  "emoji_sequence_swish_noise_pad_augment_crab_alien_butterfly_stable_eddie")
	"""
	#make_video_file("emoji_sequence_1layer_skull_rainbow_skull_eddie")
	#visualise_distance_to_target("emoji_sequence_mushroom_lizard_rooster_eddie")



	
main()