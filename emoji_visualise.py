from NCA_class import *
from NCA_train import *
from NCA_utils import *
from NCA_visualise import *
import numpy as np
import os 
import sys


N_CHANNELS=16
N_BATCHES=4












def make_video_file(filename):
	ca = load_wrapper(filename)
	print(str(ca))
	x0 = (load_emoji_sequence(["butterfly.png"])[0])
	#w = x0.shape[2]
	#x0 = np.flip(x0,axis=1)
	x0 = np.rot90(x0,1,axes=(1,2))
	#x0[:,w//2-10:w//2+10,w//2-10:w//2+10]=0# = np.flip(x0,axis=2)
	K=20
	#x0 = np.pad(x0,((0,0),(K,K),(K,K),(0,0)))
	print(x0.shape)
	grid = ca.run(x0,60*4)
	my_animate(grid[:,0,...,:3])
	vis = NCA_Visualiser()
	#vis.save_image_sequence((np.tanh(grid[...,4:])+1)/2.0,"figures/mushroom_lizard_rooster/hidden_")
	#vis.save_image_sequence(grid[...,:3],"figures/mushroom_lizard_rooster/padded/")


def visualise_training_loss():
	steps,losses = load_loss_log("logs/eddie_runs/emoji_sequence_skull_rainbow_skull_eddie/train")
	plt.plot(steps,losses,label="2 hidden layers")

	steps,losses = load_loss_log("logs/eddie_runs/emoji_sequence_1layer_skull_rainbow_skull_eddie/train")
	plt.plot(steps,losses,label="1 hidden layer")
	plt.ylabel("Loss")
	plt.xlabel("Training iterations")
	plt.legend()
	plt.yscale("log")
	plt.show()



def visualise_distance_to_target(filename):

	def pad_helper(array,K):
		return np.pad(array,((0,0),(K,K),(K,K),(0,0)))

	ca = load_wrapper(filename)
	x0 = (load_emoji_sequence(["mushroom_1f344.png"])[0])
	x1 = (load_emoji_sequence(["lizard_1f98e.png"])[0])
	x2 = (load_emoji_sequence(["rooster_1f413.png"])[0])

	w = x0.shape[2]
	
	#x0[:,w//2-5:w//2+5,w//2-5:w//2+5]=0
	grid1 = ca.run(pad_helper(x0,5),60*4)
	
	#x0 = (load_emoji_sequence(["mushroom_1f344.png"])[0])
	#x0[:,w//2-10:w//2+10,w//2-10:w//2+10]=0
	grid2 = ca.run(pad_helper(x0,10),60*4)

	#x0 = (load_emoji_sequence(["mushroom_1f344.png"])[0])
	#x0[:,w//2-15:w//2+15,w//2-15:w//2+15]=0
	grid3 = ca.run(pad_helper(x0,20),60*4)


	#x0 = (load_emoji_sequence(["mushroom_1f344.png"])[0])
	my_animate(grid1[:,0,...,:3])
	my_animate(grid2[:,0,...,:3])
	my_animate(grid3[:,0,...,:3])
	vis = NCA_Visualiser()
	#print(grid.shape)
	#print(x0.shape)
	dist0_1 = vis.distance_from_target(grid1[...,:4],pad_helper(x0,5))
	dist1_1 = vis.distance_from_target(grid1[...,:4],pad_helper(x1,5))
	dist2_1 = vis.distance_from_target(grid1[...,:4],pad_helper(x2,5))

	dist0_2 = vis.distance_from_target(grid2[...,:4],pad_helper(x0,10))
	dist1_2 = vis.distance_from_target(grid2[...,:4],pad_helper(x1,10))
	dist2_2 = vis.distance_from_target(grid2[...,:4],pad_helper(x2,10))

	dist0_3 = vis.distance_from_target(grid3[...,:4],pad_helper(x0,20))
	dist1_3 = vis.distance_from_target(grid3[...,:4],pad_helper(x1,20))
	dist2_3 = vis.distance_from_target(grid3[...,:4],pad_helper(x2,20))

	#dist1_flipped = vis.distance_from_target(grid[...,:4],np.flip(x1,axis=2))
	#dist2_flipped = vis.distance_from_target(grid[...,:4],np.flip(x2,axis=2))

	#np.save("figures/mushroom_lizard_rooster/1layer/dist0.npy",dist0)
	#np.save("figures/mushroom_lizard_rooster/1layer/dist1.npy",dist1)
	#np.save("figures/mushroom_lizard_rooster/1layer/dist2.npy",dist2)
	#dist0_1 = np.load("figures/mushroom_lizard_rooster/1layer/dist0.npy")
	#dist1_1 = np.load("figures/mushroom_lizard_rooster/1layer/dist1.npy")
	#dist2_1 = np.load("figures/mushroom_lizard_rooster/1layer/dist2.npy")

	#m = np.max([np.max(dist0),np.max(dist1),np.max(dist2)])
	m = 60#np.max([np.max(dist0),np.max(dist1)])
	fig, ax = plt.subplots()
	ax.set_xticks([0,60,120], minor=False)
	ax.xaxis.grid(True, which='major')
	

	plt.plot(dist0_1,color="red",label="Mushroom")
	plt.plot(dist0_2,color="red",ls="--")#,label="Mushroom 20*20 hole")
	plt.plot(dist0_3,color="red",ls=":")#,label="Mushroom 30*30 hole")
	#plt.plot(dist0_1,color="red",ls="--",label="Skull 1 layer")
	
	plt.plot(dist1_1,color="green",label="Lizard")
	plt.plot(dist1_2,color="green",ls="--")#,label="Lizard 20*20 hole")
	plt.plot(dist1_3,color="green",ls=":")#,label="Lizard 30*30 hole")


	plt.plot(dist2_1,color="blue",label="Rooster")
	plt.plot(dist2_2,color="blue",ls="--")#,label="Rooster 20*20 hole")
	plt.plot(dist2_3,color="blue",ls=":")#,label="Rooster 30*30 hole")

	#plt.plot(dist1,color="green",label="Lizard")
	#plt.plot(dist1_flipped,color="green",ls="--",label="Lizard flipped")

	#plt.plot(dist2,color="blue",label="Rooster")
	#plt.plot(dist2_flipped,color="blue",ls="--",label="Rooster flipped")
	#plt.grid([60,120],axis="x")
	plt.xlabel("NCA iterations")
	plt.ylabel("Euclidean Distance")
	plt.legend()
	plt.show()



def main():
	#visualise_distance_to_target("emoji_sequence_mushroom_lizard_rooster_eddie")
	make_video_file("emoji_sequence_sigmoid_2layer_isotropic_butterfly_microbe_eye_eddie")
	#visualise_training_loss()
main()