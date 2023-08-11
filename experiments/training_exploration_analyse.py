from NCA.NCA_class import *
from NCA.trainer.NCA_trainer import *
from NCA.NCA_utils import *
from NCA.NCA_visualise import *
import numpy as np
import os 
import sys
os.chdir("..")


"""
	Analyse and visualise each combination of loss function and training algorithm, on both emoji and heat equation
"""
def F_heat(X,Xdx,Xdy,Xdd,D=0.33):
	return D*Xdd
def F_readif_chem_coral(X,Xdx,Xdy,Xdd,D=[0.1,0.05],f=0.06230,k=0.06258):
	# Reaction diffusion as described in https://www.karlsims.com/rd.html
	ch_1 = D[0]*Xdd[...,0] - X[...,1]**2*X[...,0] + f*(1-X[...,0])
	ch_2 = D[1]*Xdd[...,1] + X[...,1]**2*X[...,0] - (k+f)*X[...,1]
	return tf.stack([ch_1,ch_2],-1)

def plot_training_error():
	for j in range(6):
		for index in tqdm(np.arange(4)*6+j):
			try:
				_,OPTIMIZER,LOSS_FUNC_STRING = index_to_trainer_parameters(index)
				#emoji_filename = "logs/eddie_runs/training_exploration/emoji_alien_monster_rooster_stable_"+OPTIMIZER+"_"+LOSS_FUNC_STRING+"/train"
				#heat_filename = "logs/eddie_runs/training_exploration/PDE_heat_eq_"+OPTIMIZER+"_"+LOSS_FUNC_STRING+"_order_1/train"
				readif_filename = "logs/eddie_runs/training_exploration/PDE_readif_"+OPTIMIZER+"_"+LOSS_FUNC_STRING+"_order_1_v4/train"
				steps,losses = load_loss_log(readif_filename)
				#print(np.array(steps).shape)
				#print(np.array(losses).shape)
				plt.semilogy(steps[:4000],losses[:4000],label=OPTIMIZER,alpha=0.5)
			except:
				pass
		plt.title(LOSS_FUNC_STRING)
		plt.legend()
		plt.xlabel("Training iterations")
		plt.ylabel("Loss")
		plt.show()



def run_PDE_models():
	S = 64
	I=2000
	OBS_CHANNELS=1
	k=3
	x0 = np.random.uniform(size=(1,S,S,OBS_CHANNELS)).astype(np.float32)
	x0[:,:32,:32]=0
	x0[:,:16,:16]=1
	x0[:,32:,32:]=1
	x0[:,48:,48:]=0
	PDE_model = PDE_solver(F_heat,OBS_CHANNELS,1,size=[S,S],PADDING="periodic")
	trajectory_true = PDE_model.run(iterations=I,step_size=1.0,initial_condition=x0)
	for index in tqdm(np.arange(7)+7*k):
		try:
			_,OPTIMIZER,LOSS_FUNC_STRING = index_to_trainer_parameters(index)
			ca = load_wrapper("training_exploration/PDE_heat_eq_"+OPTIMIZER+"_"+LOSS_FUNC_STRING+"_order_1")
			

			ca.PADDING="periodic"
			trajectory = ca.run(x0,I)
			#vis = NCA_Visualiser([ca])
			#vis.space_average(trajectory)

			
			#my_animate(trajectory[:,0,...,0])
			#my_animate(trajectory_true[:,0,...,0])
			plt.plot(np.sum(np.abs(trajectory[...,:OBS_CHANNELS]-trajectory_true[:I]),axis=(1,2,3,4))/(S*S),label=LOSS_FUNC_STRING)
			plt.title("Average absolute difference between PDE and NCA - "+OPTIMIZER)
		except:
			pass
	plt.xlabel("Time steps")
	plt.ylabel("Error")
	plt.legend()
	plt.show()

def run_and_save_PDE_trajectory():
	
	S = 64
	I=300
	OBS_CHANNELS=2
	k=3
	x0 = np.random.uniform(size=(1,S,S,OBS_CHANNELS)).astype(np.float32)
	x0[:,:32,:32]=0
	x0[:,:16,:16]=1
	x0[:,32:,32:]=1
	x0[:,48:,48:]=0
	PDE_model = PDE_solver(F_readif_chem_coral,OBS_CHANNELS,1,size=[S,S],PADDING="periodic")
	#PDE_model = PDE_solver(F_heat,OBS_CHANNELS,1,size=[S,S],PADDING="periodic")
	trajectory_true = PDE_model.run(iterations=I,step_size=1.0,initial_condition=x0)
	#ca = load_wrapper("training_exploration/PDE_heat_eq_Adagrad_euclidean_order_1")
	#ca = load_wrapper("trainer_validation/Nadam_spectral_sampling_16_coral_v2")
	ca = load_wrapper("model_exploration/Nadam_euclidean_coral_2_layer_ID_LAP_relu_v1")
	ca.PADDING="periodic"
	trajectory = ca.run(x0,I)
	my_animate(trajectory[:,0,...,:1])
	my_animate(trajectory_true[:,0])
	traj_diff = np.abs(trajectory_true[:-1]-trajectory[...,:1])
	my_animate(traj_diff[:,0]/np.max(traj_diff))
	vis = NCA_Visualiser([ca])
	vis.save_image_sequence(np.array(trajectory[...,:1]),"figures/readif/coral/relu/NCA/")
	vis.save_image_sequence(np.array(trajectory_true),"figures/readif/coral/relu/PDE/")
	vis.save_image_sequence(traj_diff,"figures/readif/coral/relu/ERR/")
#run_PDE_models()
plot_training_error()
#run_and_save_PDE_trajectory()