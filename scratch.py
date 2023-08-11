from NCA.NCA_class import *
from NCA.trainer.NCA_trainer import *
from NCA.trainer.NCA_PDE_trainer import NCA_PDE_Trainer
from NCA.trainer.NCA_IC_trainer import NCA_IC_Trainer
from NCA.NCA_utils import *
from NCA.NCA_visualise import *
from NCA.NCA_analyse import NCA_Perturb
from NCA.GOL_solver import GOL_solver
from NCA.PDE_solver import *
import numpy as np
import os 
import scipy as sp
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
	The messy / cluttered bit where I combine the clean bits
"""




def initial_condition_perturations():

	ca = load_wrapper("emoji_alien_monster_rooster_euclidean_nadam_stable")
	ca.PADDING="zero"
	vis = NCA_Visualiser([ca])
	
	data = load_emoji_sequence(["alien_monster.png","rooster_1f413.png","rooster_1f413.png"])
	print(data.shape)
	
	
	
	trainer_max = NCA_IC_Trainer(ca,data,4,"emoji_alien_monster_rooster_stable_ic_morph_maximal",mode="max")
	trainer_max.train_sequence(200,60)
	
	trainer_min = NCA_IC_Trainer(ca,data,4,"emoji_alien_monster_rooster_stable_ic_morph_minimal",mode="min")
	trainer_min.train_sequence(200,60)
	
	x0_max = trainer_max.x0
	x0_min = trainer_min.x0
	
	
	x0_max = np.pad(x0_max,((0,0),(20,20),(20,20),(0,0)))
	x0_min = np.pad(x0_min,((0,0),(20,20),(20,20),(0,0)))
	data = np.pad(data,((0,0),(0,0),(20,20),(20,20),(0,0)))
	
	trajectory = ca.run(data[0],200)
	my_animate(trajectory[:,0,...,:4])
	vis.save_image_sequence_RGBA(np.array(trajectory), "figures/alien_rooster/unperturbed/")
	
	
	#print(x0.shape)
	plt.imshow(x0_max[0])
	plt.show()
	trajectory = ca.run(x0_max[0:1],200)
	my_animate(trajectory[:,0,...,:4])
	vis.save_image_sequence_RGBA(np.array(trajectory),"figures/alien_rooster_stable/max_perturbation/")
	
	
	
	plt.imshow(x0_min[0])
	plt.show()
	trajectory = ca.run(x0_min[0:1],200)
	my_animate(trajectory[:,0,...,:4])
	vis.save_image_sequence_RGBA(np.array(trajectory),"figures/alien_rooster_stable/min_perturbation/")
	#ca_vis.activation_heatmap_2()
	#ca_vis.activation_heatmap_1()

def stable_model_demo():

	
	ca = load_wrapper("emoji_alien_monster_rooster_euclidean_nadam_stable")
	#ca = load_wrapper("training_exploration/emoji_alien_monster_rooster_stable_Adam_euclidean")
	data = load_emoji_sequence(["alien_monster.png"],downsample=2)
	ca.PADDING="zero"
	print(data.shape)
	#data = sp.ndimage.zoom(data,zoom=[1,1,2,2,1])
	plt.imshow(data[0,0])
	plt.show()
	#ca.upscale_kernel()
	x0 = np.zeros((1,80,80,4))
	x0[:,10:70,10:70] = data[0]
	
	trajectory = ca.run(x0,120)
	my_animate(trajectory[:,0,...,:4])
	
	#x0[:,:,:x0.shape[1]//2] = data[1,:,:,:x0.shape[1]//2]
	
	
	size = data.shape[2]
	x0 = np.zeros((1,100,100,4))
	x0[:,30:30+size,:10+size//2] = data[0,:,::-1,:size//2+10]
	x0[:,:size,20+size//2:30+size] = data[0,:,:,:size//2+10][:,:,::-1]
	
	trajectory_sliced = ca.run(x0,120)
	my_animate(trajectory_sliced[:,0,...,:4])
	
	vis = NCA_Visualiser([ca])
	#vis.gradient_plot()
	print(trajectory.shape)
	vis.save_image_sequence_RGBA(trajectory,"figures/alien_rooster_stable/normal/")
	vis.save_image_sequence_RGBA(trajectory_sliced,"figures/alien_rooster_stable/perturbed/")
	"""
	ana = NCA_Perturb([ca])
	ana.zero_threshold(0.01)
	trajectory = ca.run(x0,1000)

	my_animate(trajectory[::2,0,...,:4])
	vis = NCA_Visualiser([ca])
	vis.space_average(trajectory)
	vis.gradient_plot()
	vis.plot_weight_matrices()
	"""
	#x0[:,100:100+size,100:100+size] = data[1]
	#plt.imshow(x0[0])
	#plt.show()
	

def stable_model_analyse():
	ca = load_wrapper("emoji_alien_monster_rooster_stable_high_quality")
	print(ca)
	x0 = load_emoji_sequence(["alien_monster.png"],downsample=2)[0]
	ana = NCA_Perturb([ca])
	grads = ana.update_gradients(x0,100,L=0)
	plt.plot(np.sum(grads,axis=0))
	plt.show()
	grads = grads / np.max(grads[...,:4])
	my_animate(grads[...,:4])

	"""
	x0 = ca.run(x0,2)[0]
	x0 = tf.convert_to_tensor(x0)
	print(x0.shape)
	grads = np.zeros((100,48,64,16))
	for t in tqdm(range(100)):
		for i in range(16):
			with tf.GradientTape() as g:
				x1 = ca(x0)
				diff = (x0-x1)[0,...,i]
				#print(diff.shape)
				grad = g.gradient(diff,ca.dense_model.weights)
				grads[t,...,i] = grad[0]
			x0 = tf.identity(x1)
			
		#print(grad[0].shape)
		#plt.imshow(grad[0][0,0])
		#plt.show()
	"""
def wasserstein_training_demos():
	"""
		Demonstrate the different behaviours learned by the same NCA on alien->rooster with
		different variants of the sliced wasserstein loss
	"""

	ca_chan = load_wrapper("training_exploration/emoji_alien_monster_rooster_stable_Adam_sliced_wasserstein_channels")
	#ca_grid = load_wrapper("training_exploration/emoji_alien_monster_rooster_stable_Adam_sliced_wasserstein_grid")
	ca_rott = load_wrapper("training_exploration/emoji_alien_monster_rooster_stable_Adam_sliced_wasserstein_rotate")
	x0 = load_emoji_sequence(["alien_monster.png"],downsample=2)[0]

	tra_chan = ca_chan.run(x0,180)
	#tra_grid = ca_grid.run(x0,180)
	tra_rott = ca_rott.run(x0,180)

	my_animate(tra_chan[:,0,...,:4])
	#my_animate(tra_grid[:,0,...,:4])
	my_animate(tra_rott[:,0,...,:4])

def main():

	

	
	#plt.show()
	#iter_n = 2
	#gol = GOL_solver(4,[64,64])
	#data = gol.run(16)[...,None]
	#nca = NCA(8,1,OBS_CHANNELS=1)
	#trainer = NCA_Trainer(nca, data, 4,model_filename="GOL_NCA_test1")
	#trainer.train_sequence(4000, iter_n,OPTIMIZER="Adam")
	
	#print(data.shape)
	#my_animate((data[:,0]))
	#my_animate((data[:,1]))
	#initial_condition_perturations()
	#wasserstein_training_demos()
	#stable_model_demo()
	#stable_model_analyse()
	#solver = PDE_solver(F_ch,1)
	#solver.X = 2*np.random.uniform(size=(1,128,128,1)).astype(np.float32)-1
	#plt.imshow(solver.X[0])
	#plt.show()
	#for i in range(400):
	#	solver.update(0.1)
	#trajectory = solver.run(1000)
	#my_animate(trajectory[::10])
	#plt.imshow(solver.X[0])
	#plt.show()


	
	#ca2 = load_wrapper("grid_search/_ch6_fr18_dr18_b0")
	#ca = load_wrapper("grid_search/optimal_hyperparameters_correct_2layer_ch_fr_b1")
	#ca = load_wrapper("emoji_sequence_2layer_isotropic_nobias_butterfly_microbe_eye_eddie")
	"""
	ca = load_wrapper("NCA_PDE_test_2")
	x0 = 2*np.random.uniform(size=(1,128,128,8))-1
	x0[...,1:]=0
	plt.imshow(x0[0,...,0])
	plt.show()
	trajectory = ca.run(x0,400)[:,0,...,0]
	trajectory = 0.5*(trajectory+1)
	print(trajectory.shape)
	my_animate(trajectory)
	"""
	
	
	
	#stable_model_demo()











	

main()